//! Minimal AVX2/FMA single-precision GEMM for the DA3 hot shapes.
//!
//! This module exists to bypass candle's per-op overhead for the matmuls that
//! dominate the ViT backbone and DPT head: the QKV/FC1/FC2/Proj linears, and
//! the attention Q·Kᵀ and A·V batched matmuls. It operates on raw `&[f32]`
//! buffers (no candle `Tensor` wrapping) and parallelizes with rayon over the
//! N-axis column blocks.
//!
//! # Layout convention
//!
//! All matrices are **row-major** `f32`:
//! - `A` is `[M, K]`
//! - `B` is `[K, N]` for `matmul_nn`, `[N, K]` for `matmul_nt`
//! - `C`/result is `[M, N]`
//!
//! # Microkernel
//!
//! The AVX2/FMA microkernel covers an `MR × NR = 6 × 16` output tile
//! (6 rows by 2 AVX2 lanes of 8 floats), holding 12 `__m256` accumulators.
//! That register pressure fits comfortably in the 16 ymm registers without
//! spilling, and matches the tile ggml's `simd-gemm.h` uses on AVX2.
//!
//! # Correctness
//!
//! All public entry points are validated by the unit tests at the bottom of
//! this file against a naive reference implementation across a range of
//! shapes including non-multiples of MR/NR.

use rayon::prelude::*;

/// Microkernel output tile: rows.
const MR: usize = 6;
/// Microkernel output tile: columns (in floats, = 2 AVX2 vectors).
pub(crate) const NR: usize = 16;
/// Row-block granularity for rayon parallelism. Each parallel task processes
/// `MC` rows (≈ `MC / MR` microkernels). The default is MR (6) because rayon's
/// work-stealing makes fine-grained tasks essentially free, and the fine
/// granularity gives perfect load balance (no thread ever idles waiting for a
/// straggler on a coarser task). Empirically, MC=48/96 regressed total
/// inference by ~10-30% due to worse tail balance on small-M shapes.
///
/// Override at runtime via `DA_TINYBLAS_MC` for experimentation (must be a
/// multiple of MR=6); invalid values fall back to the default.
const MC_DEFAULT: usize = MR;

#[inline]
fn mc() -> usize {
    use std::sync::OnceLock;
    static MC: OnceLock<usize> = OnceLock::new();
    *MC.get_or_init(|| match std::env::var("DA_TINYBLAS_MC").as_deref() {
        Ok(s) => s
            .parse::<usize>()
            .ok()
            .filter(|&v| v >= MR && v % MR == 0)
            .unwrap_or(MC_DEFAULT),
        _ => MC_DEFAULT,
    })
}

/// Parallelization axis. Default `'auto'` selects per-call based on shape
/// (see [`should_use_n_axis`] and [`should_use_k_blocking`]). `'m'` forces
/// M-row-block parallelisation; `'n'` forces N-column-block parallelisation;
/// `'k'` forces K-blocking.
///
/// Override at runtime via `DA_TINYBLAS_AXIS` (`'m'`, `'n'`, `'k'`, or `'auto'`).
#[inline]
fn par_axis() -> char {
    use std::sync::OnceLock;
    static AXIS: OnceLock<char> = OnceLock::new();
    *AXIS.get_or_init(|| match std::env::var("DA_TINYBLAS_AXIS").as_deref() {
        Ok(s) if s.starts_with('n') || s.starts_with('N') => 'n',
        Ok(s) if s.starts_with('m') || s.starts_with('M') => 'm',
        Ok(s) if s.starts_with('k') || s.starts_with('K') => 'k',
        _ => 'a', // 'auto' (default)
    })
}

/// Column-block granularity for `'n'`-axis parallelism (`gemm_nn_avx2_par_n`).
/// Each parallel task processes all M rows × `NC` columns. NC must be a
/// multiple of NR=16 and large enough to amortise rayon dispatch overhead
/// but small enough that the B panel (`K * NC * 4` bytes) fits in L2.
///
/// Default 32 (2 NR-tiles). For the QKV GEMM (N=2304, K=768) this gives
/// 72 N-blocks (3 per thread at 24 threads) and a per-task B panel of
/// 96 KiB (fits L1/L2 comfortably). The previous default of 64 gave only
/// 36 N-blocks and a 192 KiB B panel; the smaller NC improves both
/// parallelism and B-panel cache residency for the attention GEMMs.
///
/// Override at runtime via `DA_TINYBLAS_NC` (multiple of NR=16).
const NC_DEFAULT: usize = 32;

#[inline]
fn nc() -> usize {
    use std::sync::OnceLock;
    static NC: OnceLock<usize> = OnceLock::new();
    *NC.get_or_init(|| match std::env::var("DA_TINYBLAS_NC").as_deref() {
        Ok(s) => s
            .parse::<usize>()
            .ok()
            .filter(|&v| v >= NR && v % NR == 0)
            .unwrap_or(NC_DEFAULT),
        _ => NC_DEFAULT,
    })
}

/// Returns `true` if the CPU supports the AVX2 + FMA microkernel path.
#[inline]
pub fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// C[M,N] = A[M,K] @ B[K,N], all row-major. Returns a freshly-allocated C.
pub fn matmul_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), m.checked_mul(k).unwrap_or(0));
    debug_assert_eq!(b.len(), k.checked_mul(n).unwrap_or(0));
    let mut c = vec![0.0f32; m * n];
    gemm_nn_into(m, n, k, a, b, &mut c);
    c
}

/// C[M,N] = A[M,K] @ B[N,K]ᵀ, all row-major. Returns a freshly-allocated C.
///
/// This is the matmul shape used by `nn.Linear`: `y = x @ Wᵀ`, where the
/// weight `W` is stored row-major as `[out, in] = [N, K]`.
pub fn matmul_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), m.checked_mul(k).unwrap_or(0));
    debug_assert_eq!(b.len(), n.checked_mul(k).unwrap_or(0));
    // For NT we transpose B into the NN layout and call gemm_nn_into. The
    // transpose is a single cache-friendly pass; for repeated calls with the
    // same weight (the common Linear case) prefer pre-packing via
    // [`pack_b_nt`].
    let mut b_packed = vec![0.0f32; k * n];
    transpose_b_nt(k, n, b, &mut b_packed);
    let mut c = vec![0.0f32; m * n];
    gemm_nn_into(m, n, k, a, &b_packed, &mut c);
    c
}

/// Linear forward: `y = x @ Wᵀ + b`, returning a freshly-allocated `y`.
///
/// - `x`: `[M, K]`
/// - `w`: `[N, K]` (PyTorch `Linear` weight layout, `out_features, in_features`)
/// - `b`: `[N]` or empty/none
pub fn linear_fwd(
    m: usize,
    n: usize,
    k: usize,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut y = matmul_nt(m, n, k, x, w);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), n);
        for i in 0..m {
            let row = &mut y[i * n..(i + 1) * n];
            for (v, &bv) in row.iter_mut().zip(b.iter()) {
                *v += bv;
            }
        }
    }
    y
}

/// In-place `C += A @ B` (both A and B row-major, NN). `c` must be `[M*N]`.
///
/// This is the low-level building block; prefer [`matmul_nn`] / [`matmul_nt`]
/// unless you are managing the output buffer yourself (e.g. fused epilogues).
pub fn gemm_nn_into(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // Safety: feature-checked above.
        unsafe { gemm_nn_avx2(m, n, k, a, b, c) };
        return;
    }
    gemm_nn_scalar(m, n, k, a, b, c);
}

/// In-place `C = A @ B + bias` (both A and B row-major, NN). `c` must be
/// `[M*N]`, `bias` must be `[N]` (broadcast across all M rows).
///
/// This is the bias-fused variant of [`gemm_nn_into`]. Instead of pre-copying
/// `bias` into each row of `C` and then accumulating `C += A@B`, it initializes
/// the GEMM accumulators with the bias and writes `C = A@B + bias` directly.
/// This eliminates the bias pre-copy pass (M*N float writes + a rayon dispatch)
/// and converts the GEMM store path from load-add-store to pure-store.
///
/// For K-blocked shapes (where K is split into L2-resident chunks), only the
/// first K-chunk uses the bias-init microkernel; subsequent chunks use the
/// regular accumulate path, which correctly carries the bias through.
///
/// Use this for all DA3 Linear layers (QKV, proj, fc1, fc2) where the bias is
/// broadcast across the token dimension. For attention QK^T and A·V (no bias)
/// continue to use [`gemm_nn_into`] / [`gemm_nn_into_serial`].
pub fn gemm_nn_bias_into(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    bias: &[f32],
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    debug_assert_eq!(bias.len(), n);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        // A@B is zero, so C = bias (broadcast across rows).
        c.par_chunks_mut(n).for_each(|row| {
            row.copy_from_slice(bias);
        });
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // Safety: feature-checked above.
        unsafe { gemm_nn_avx2_bias(m, n, k, a, b, c, bias) };
        return;
    }
    gemm_nn_scalar_bias(m, n, k, a, b, c, bias);
}

/// Serial (single-threaded) variant of [`gemm_nn_into`].
///
/// Use this inside an already-parallel context (e.g. when the caller parallelises
/// across a higher-level axis such as attention heads) to avoid nested rayon
/// contention. For the DA3 per-head attention GEMMs this is ~3-5× faster than
/// the parallel `gemm_nn_into` because the latter spawns many tiny tasks that
/// fight the outer parallel loop for threads and thrash the cache.
pub fn gemm_nn_into_serial(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // Safety: feature-checked above.
        unsafe { gemm_nn_avx2_serial(m, n, k, a, b, c) };
        return;
    }
    gemm_nn_scalar(m, n, k, a, b, c);
}

/// Pre-pack a `[N,K]` weight matrix into the `[K,N]` row-major layout the NN
/// kernel consumes. Use when calling `gemm_nn_into` with a transposed weight
/// many times (e.g. the `Linear` weights loaded once at engine load).
pub fn pack_b_nt(k: usize, n: usize, b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(b.len(), n * k);
    let mut packed = vec![0.0f32; k * n];
    transpose_b_nt(k, n, b, &mut packed);
    packed
}

// ---------------------------------------------------------------------------
// Internal: AVX2 kernel
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
unsafe fn gemm_nn_avx2(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    // Three parallelization strategies, selectable at runtime:
    //
    // * `'m'` (default for small-N): parallelise over M-row blocks of height
    //   `MC`. Each task reads `A[MC, K]` (small, stays in L1/L2) and the *entire*
    //   `B[K, N]`. With `M/MC` tasks per forward, B is re-read `M/MC` times. This
    //   is fine when B fits in L2, but for large-B shapes (B > 2 MiB L2) every B
    //   read goes to L3, burning shared L3 bandwidth and becoming the bottleneck.
    //
    // * `'n'`: parallelise over N-column blocks of width `NC`. Each task reads
    //   all of `A[M, K]` (shared via L3 across the N-tasks) and only
    //   `B[K, NC]` (sized to fit in L2). This drastically cuts B traffic for
    //   large-B shapes; the trade-off is increased A traffic, but A is naturally
    //   reused in L3 because every N-task reads the same A.
    //
    // * `'k'` (K-blocking): when B is too big for L2 AND N is too small for
    //   enough N-blocks (e.g. conv3x3_stride2 with N=216), split K into chunks
    //   of `KC` rows so each `B[KC, N]` fits L2. The K-chunk stays L2-resident
    //   across the sequential M-blocks a core processes, drastically reducing
    //   L3 traffic vs the plain M-axis path.
    //
    // * `'auto'` (default): pick per-call based on shape. Prefers `'n'` when B
    //   is too big for L2 **and** there are enough N-blocks (`N / NC ≥ rayon
    //   threads`); falls back to `'k'` when B is too big for L2 but N is too
    //   small for `'n'`; otherwise uses `'m'`.
    //
    // Override at runtime via `DA_TINYBLAS_AXIS` (`'m'`, `'n'`, `'k'`, or `'auto'`).
    let use_n;
    let use_k;
    match par_axis() {
        'n' => {
            use_n = n >= NR;
            use_k = false;
        }
        'k' => {
            use_n = false;
            use_k = true;
        }
        'm' => {
            use_n = false;
            use_k = false;
        }
        _ => {
            // Check K-blocking first: for shapes where N ≥ 3·K (e.g. FFN fc1
            // with N=3072, K=768), K-blocking beats N-axis because it keeps
            // the tiny A tile in L1 instead of re-reading the full A from L3
            // for every NC-wide task. For all other shapes, prefer N-axis →
            // K-blocking → M-axis as before.
            use_k = should_use_k_blocking(m, n, k);
            use_n = !use_k && should_use_n_axis(m, n, k);
        }
    }
    if use_n {
        unsafe { gemm_nn_avx2_par_n(m, n, k, a, b, c) }
    } else if use_k {
        unsafe { gemm_nn_avx2_kblocked(m, n, k, a, b, c) }
    } else {
        unsafe { gemm_nn_avx2_par_m(m, n, k, a, b, c) }
    }
}

/// L2 cache budget (bytes) used by [`should_use_n_axis`] and [`auto_nc`].
///
/// Defaults to 1280 KiB, not the full 2 MiB per-core L2 of the i7-13700K's
/// P-cores. The reason: with 24 rayon threads sharing the 30 MiB L3, each
/// core's effective L3 share is ~1.25 MiB. A B-chunk sized for the full 2 MiB
/// L2 exceeds this, causing the per-core K-blocking tasks to spill B-chunks
/// to L3 and re-read them (the M-block tasks that run sequentially on the
/// same core no longer find B in L2). A B-chunk target of ~1.25 MiB leaves
/// headroom for the A tile and accumulators while staying within the
/// per-core L3 share.
///
/// Measured improvement on DA3-BASE q5_k (24 threads): ~2-4 ms end-to-end
/// vs the previous 2 MiB default (median ~304 ms → ~301 ms).
///
/// Override at runtime via `DA_TINYBLAS_L2_KIB` (in KiB).
#[cfg(target_arch = "x86_64")]
fn l2_budget_bytes() -> usize {
    use std::sync::OnceLock;
    static L2_KIB: OnceLock<usize> = OnceLock::new();
    *L2_KIB.get_or_init(|| {
        std::env::var("DA_TINYBLAS_L2_KIB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1280)
            * 1024
    })
}

/// Heuristic for `'auto'` axis selection: use N-axis parallelisation when both
/// (a) the B panel `B[K, N]` exceeds the L2 budget, so M-axis would re-read it
///    from L3 `M/MC` times; and (b) there are enough N-blocks (at the default
///    `NC=32`) to keep all rayon threads busy.
#[cfg(target_arch = "x86_64")]
fn should_use_n_axis(_m: usize, n: usize, k: usize) -> bool {
    // (a) B must not fit in L2 — otherwise M-axis is already optimal.
    let b_bytes = k.checked_mul(n).unwrap_or(usize::MAX) * 4;
    if b_bytes <= l2_budget_bytes() {
        return false;
    }
    // (b) need enough N-blocks for parallelism at the default NC.
    let n_blocks = n.div_ceil(NC_DEFAULT);
    n_blocks >= rayon::current_num_threads().max(1)
}

/// Pick `NC` (N-column-block width) for N-axis parallelisation.
///
/// Currently just returns the env-var-configured default (NC=64). Kept as a
/// seam for future per-shape auto-tuning — the experiments showed that NC=64
/// is near-optimal across all DA3 GEMM shapes because:
/// - For small-K shapes (conv1x1 with K=128): B-strip = 32 KiB fits L1, so
///   larger NC gives no cache benefit and only reduces task count.
/// - For large-K shapes (FFN with K=768): B-strip = 192 KiB fits L2 but not
///   L1, and larger NC would push it past L2.
#[cfg(target_arch = "x86_64")]
fn auto_nc(_n: usize, _k: usize) -> usize {
    nc()
}

/// K-blocking chunk size (in K dimension) for the K-blocked GEMM path.
///
/// When the B panel `B[K, N]` does not fit L2 and there are too few N-blocks
/// for N-axis parallelism (e.g. `conv3x3_stride2` with M=768, N=216, K=6912,
/// B=5.7 MiB), we split the K dimension into chunks of `KC` rows. The chunk's
/// B panel `B[KC, N]` is reused across M-axis tasks running sequentially on
/// the same core, cutting L3 traffic.
///
/// Override at runtime via `DA_TINYBLAS_KC` (must be a multiple of 8);
/// invalid values fall back to the per-call shape-tuned default (see
/// [`auto_kc`]).
const KC_DEFAULT: usize = 0; // 0 = use shape-tuned auto_kc

#[cfg(target_arch = "x86_64")]
#[inline]
fn kc_env() -> usize {
    use std::sync::OnceLock;
    static KC: OnceLock<usize> = OnceLock::new();
    *KC.get_or_init(|| match std::env::var("DA_TINYBLAS_KC").as_deref() {
        Ok(s) => s
            .parse::<usize>()
            .ok()
            .filter(|&v| v >= 8 && v % 8 == 0)
            .unwrap_or(KC_DEFAULT),
        _ => KC_DEFAULT,
    })
}

/// Per-call shape-tuned KC. The optimal KC depends on N:
/// - For small N (≤ 256, e.g. conv3x3_stride2 with N=216): prefer `k/2`
///   (only 2 K-chunks). Empirically this beats smaller KC because each task
///   does little work per N-tile (only 14 tiles for N=216), so dispatch
///   overhead dominates if KC is too small.
/// - For large N (> 256, e.g. FFN fc2 with N=768): target B-chunk ~ L2 budget
///   (1.25 MiB by default; see [`l2_budget_bytes`]). This gives KC ≈ 426 for
///   N=768, which empirically beats k/2 because B-chunk L2 residency matters
///   more than dispatch overhead when each task has many N-tiles (48 for N=768).
///
/// Note: an L1-based cap (KC ≤ ~340 to keep the 6×KC×4 A tile under 8 KiB)
/// was tested and found to regress `lat3 resize` by ~10% — the extra K-chunk
/// dispatch overhead (21 chunks vs 2) outweighs the A-cache benefit for
/// small-N shapes. The current heuristic prioritises dispatch efficiency for
/// small N and B L2 residency for large N.
///
/// Empirically validated on the DA3 hot shapes (24 threads, L2 budget 1280 KiB):
/// - conv3x3_stride2 (N=216, K=6912): KC=3456 → small-N path (k/2)
/// - FFN fc1 (N=3072, K=768): KC≈104 → large-N path (L2-budget-driven)
/// - FFN fc2 (N=768, K=3072): KC≈416 → large-N path (L2-budget-driven)
#[cfg(target_arch = "x86_64")]
fn auto_kc(n: usize, k: usize) -> usize {
    // Honor explicit env-var override.
    let env = kc_env();
    if env > 0 {
        return env.min(k);
    }
    // Round-helper: round down to a multiple of 8.
    let round8 = |x: usize| (x / 8) * 8;
    if n <= 256 {
        // Small N: prefer few K-chunks (k/2). Each task has few N-tiles, so
        // dispatch overhead matters more than B-chunk L2 residency.
        round8(k / 2).max(8)
    } else {
        // Large N: target B-chunk ~ L2 budget. Each task has many N-tiles,
        // so per-task work is large enough to amortize dispatch.
        let kc_l2 = l2_budget_bytes() / (n * 4);
        round8(kc_l2).max(round8(k / 16)).min(round8(k / 2))
    }
}

/// Heuristic for K-blocking: use it when (a) the B panel exceeds L2 (so M-axis
/// alone is L3-bound) AND (b) either N is small enough that N-axis would starve
/// threads, OR N is much larger than K (N ≥ 3·K) so that the N-axis strategy
/// re-reads the full `A[M,K]` from L3 for every narrow `NC`-wide task whereas
/// K-blocking shrinks `A` to a tiny `MC×KC` tile that lives in L1.
#[cfg(target_arch = "x86_64")]
fn should_use_k_blocking(m: usize, n: usize, k: usize) -> bool {
    // Need K large enough to split into at least 2 chunks; otherwise K-blocking
    // is just the regular path.
    if k < 32 {
        return false;
    }
    // B must exceed L2 budget; otherwise M-axis already keeps B L2-resident.
    let b_bytes = k.checked_mul(n).unwrap_or(usize::MAX) * 4;
    if b_bytes <= l2_budget_bytes() {
        return false;
    }
    // Need enough M-row blocks for parallelism at MC granularity.
    let m_blocks = m.div_ceil(MR);
    if m_blocks < rayon::current_num_threads().max(1) {
        return false;
    }
    // If N-axis is viable AND N is not much larger than K, prefer N-axis.
    // The exception is shapes like FFN fc1 (N=3072, K=768): even though N-axis
    // gives 48 tasks at NC=64, each task re-reads the full 2.5 MiB A panel
    // from L3 (~120 MiB total), and the per-task B-strip (192 KiB) is dwarfed
    // by the A traffic. K-blocking with KC≈168 keeps A in L1 (64 KiB) and B in
    // L2 (1.97 MiB), which empirically beats N-axis by ~15% on that shape.
    if should_use_n_axis(m, n, k) && n <= 3 * k {
        return false;
    }
    true
}

/// M-axis parallelisation. Each rayon task processes `MC` rows × all N columns.
#[cfg(target_arch = "x86_64")]
unsafe fn gemm_nn_avx2_par_m(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    // Parallelize over row-blocks of height `MC` (a multiple of MR). Each task's
    // C slice is `MC * n` contiguous floats, so rayon's `par_chunks_mut` gives
    // us disjoint mutable slices without any unsafe splitting. Within each task
    // we walk the MR-row microkernels serially. Using MC >> MR (96 vs 6) keeps
    // the number of rayon tasks small (≈ M/96) so dispatch overhead stays low —
    // important for the attention GEMMs which are already nested inside a
    // per-head rayon loop.
    let mc = mc();
    let chunk_rows = if m >= mc { mc } else { MR };
    let chunk = chunk_rows * n;
    c.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(i_blk, c_row)| unsafe {
            let i_base = i_blk * chunk_rows;
            let rows_in_chunk = c_row.len() / n; // ≤ chunk_rows (tail may be smaller)
            let a_row = &a[i_base * k..(i_base + rows_in_chunk) * k];
            gemm_nn_rows_chunk_avx2(rows_in_chunk, n, k, a_row, b, c_row);
        });
}

/// N-axis parallelisation. Each rayon task processes all M rows × `NC` columns.
/// Within a task we walk M in MR-row microkernels and N in NR-wide tiles.
///
/// `NC` is chosen per-call by [`auto_nc`] to balance parallelism (enough
/// N-blocks for all rayon threads) against B-panel L2 residency (NC small
/// enough that `B[K, NC]` fits in L2). The env-var override `DA_TINYBLAS_NC`
/// forces a fixed NC when set.
///
/// # Safety
///
/// Each rayon task owns a disjoint range `[j_start, j_end)` of columns. Tasks
/// only write `C[i*n + j_start .. i*n + j_end]` for `i in 0..m`, which are
/// disjoint across tasks, so the concurrent mutable access via raw pointers is
/// safe.
#[cfg(target_arch = "x86_64")]
unsafe fn gemm_nn_avx2_par_n(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let nc = auto_nc(n, k);
    let n_blocks = n.div_ceil(nc);
    // Wrap raw pointers as `usize` so they can be moved into the rayon closure
    // (raw pointers are not `Send + Sync`). We reconstruct them inside the task
    // body. Each task only writes C[i*n + j_start .. i*n + j_end] for its own
    // (j_start, j_end) range, which is disjoint across tasks.
    let b_addr = b.as_ptr() as usize;
    let c_addr = c.as_mut_ptr() as usize;
    (0..n_blocks).into_par_iter().for_each(|jb| unsafe {
        let j_start = jb * nc;
        let j_end = (j_start + nc).min(n);
        let b_ptr = b_addr as *const f32;
        let c_ptr = c_addr as *mut f32;
        // For each M-row block, walk N tiles within [j_start, j_end).
        let mut i = 0;
        while i < m {
            let r = (m - i).min(MR);
            let a_row = &a[i * k..(i + r) * k];
            let c_row_ptr = c_ptr.add(i * n);
            // B panel for this column block: base = b_ptr + j_start, row stride = n.
            // The microkernel loads 16 contiguous floats per B row, which exactly
            // matches B[k, j_start..j_start+16] since B is row-major.
            gemm_nn_rows_cols_avx2(
                r,
                j_end - j_start, // number of columns to process in this block
                k,
                a_row,
                b_ptr.add(j_start),     // B base for column j_start of row 0
                n,                      // B row stride (full N)
                c_row_ptr.add(j_start), // C base for column j_start of row 0
                n,                      // C row stride (full N)
            );
            i += r;
        }
    });
}

/// K-blocking GEMM: split K into chunks of `KC` rows so each B-chunk `B[KC, N]`
/// is reused across M-axis tasks running sequentially on the same core,
/// cutting L3 traffic. Outer serial loop over K-chunks; inner rayon parallel
/// over M-row blocks (each task processes `MC` rows for one K-chunk).
///
/// The optimal `KC` is shape-dependent (see [`auto_kc`]). For small N it may
/// be larger than the L2 budget (e.g. lateral-3 `conv3x3_stride2` uses KC=K/2
/// giving a 2.85 MiB B-chunk that exceeds L2) because dispatch overhead
/// dominates when each task has few N-tiles; for large N it targets the L2
/// budget to maximize per-task L2 residency.
///
/// Per-task A is `A[i_base..i_base+rows, kc_start..kc_start+kc_len]`. The
/// microkernel needs A with row stride = `k` (full K) but K-loop count =
/// `kc_len`, so we use a strided-A microkernel variant.
///
/// # Safety
///
/// Each rayon task owns a disjoint `MC × n` slice of C (via `par_chunks_mut`).
/// The raw A pointer is read-only and shared; we shift it to point at
/// `A[i_base, kc_start]` and pass `k` as the row stride.
#[cfg(target_arch = "x86_64")]
unsafe fn gemm_nn_avx2_kblocked(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let mc = mc();
    let chunk_rows = if m >= mc { mc } else { MR };
    let chunk = chunk_rows * n;
    let kc = auto_kc(n, k).min(k);
    let a_addr = a.as_ptr() as usize;
    // Outer serial loop over K-chunks. Within each K-chunk, all M-block tasks
    // read the SAME B-chunk (L3-resident or warmer), and per-core L2 retains
    // it across sequential tasks.
    for kc_start in (0..k).step_by(kc) {
        let kc_len = (kc).min(k - kc_start);
        // B-chunk: B[kc_start..kc_start+kc_len, 0..n] is contiguous in [K, N]
        // row-major layout, so a single slice.
        let b_chunk = &b[kc_start * n..(kc_start + kc_len) * n];
        c.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(i_blk, c_row)| unsafe {
                let i_base = i_blk * chunk_rows;
                let rows_in_chunk = c_row.len() / n;
                // A pointer: shift to row i_base, column kc_start.
                let a_ptr = (a_addr as *const f32).add(i_base * k + kc_start);
                gemm_nn_rows_chunk_kc_avx2(
                    rows_in_chunk,
                    n,
                    kc_len,
                    a_ptr,
                    k, // A row stride (full K)
                    b_chunk,
                    c_row,
                );
            });
    }
}

/// Serial (no rayon) version of [`gemm_nn_avx2`]. Walks MC-row blocks in a
/// plain loop. Identical per-microkernel work; only the outer dispatch differs.
/// Use when the caller already provides parallelism (e.g. per-head attention).
#[cfg(target_arch = "x86_64")]
unsafe fn gemm_nn_avx2_serial(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    gemm_nn_rows_chunk_avx2(m, n, k, a, b, c);
}

/// Compute `C[rows, n] += A[rows, k] @ B[k, n]` for `rows` rows (any size),
/// serially looping over MR-row microkernels. Shared between the parallel and
/// serial entry points.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn gemm_nn_rows_chunk_avx2(
    rows: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    let mut i = 0;
    while i < rows {
        let r = (rows - i).min(MR);
        let a_row = &a[i * k..(i + r) * k];
        let c_row = &mut c[i * n..(i + r) * n];
        gemm_nn_rows_avx2(r, n, k, a_row, b, c_row, n);
        i += r;
    }
}

/// Like [`gemm_nn_rows_chunk_avx2`] but takes A as a raw pointer with
/// independent row stride (`a_stride`) and K-loop count (`kc_len`). This is
/// used by [`gemm_nn_avx2_kblocked`]: A points at `A[i_base, kc_start]`, with
/// `a_stride = k` (full K), and we iterate the K-loop only over `kc_len`
/// elements (the K-chunk width).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn gemm_nn_rows_chunk_kc_avx2(
    rows: usize,
    n: usize,
    kc_len: usize,
    a_ptr: *const f32,
    a_stride: usize,
    b: &[f32],     // [kc_len, n] contiguous
    c: &mut [f32], // [rows, n] contiguous
) {
    let mut i = 0;
    while i < rows {
        let r = (rows - i).min(MR);
        let a_row_ptr = a_ptr.add(i * a_stride);
        let c_row = &mut c[i * n..(i + r) * n];
        gemm_nn_rows_kc_avx2(r, n, kc_len, a_row_ptr, a_stride, b, c_row);
        i += r;
    }
}

/// Compute C[rows, cols] += A[rows, k] @ B[k, cols] for a single row-block,
/// where `cols` may be a contiguous column-range inside a larger row stride.
/// `1 <= rows <= MR`. Walks columns in NR-wide tiles, falling back to scalar
/// for the tail columns. Full-width tiles dispatch to the MR-wide microkernel
/// when `rows == MR`, otherwise they fall through to per-row 1x16 microkernels
/// (so partial-row tails stay vectorized).
///
/// `b_ptr` / `c_ptr` point at the first element of the column block (row 0,
/// start col); both B and C rows are spaced by `n_stride`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nn_rows_cols_avx2(
    rows: usize,
    cols: usize,
    k: usize,
    a: &[f32],         // [rows, k]
    b_ptr: *const f32, // points at B[0, j_start]
    n_stride: usize,   // B row stride (== N of the full B matrix)
    c_ptr: *mut f32,   // points at C[0, j_start]
    c_stride: usize,   // C row stride (== N of the full C matrix)
) {
    let mut j = 0;
    while j + NR <= cols {
        if rows == MR {
            microkernel_6x16_ptr(
                a.as_ptr(),
                b_ptr.add(j),
                c_ptr.add(j),
                k,
                n_stride,
                c_stride,
            );
        } else {
            for ii in 0..rows {
                microkernel_1x16_ptr(
                    a.as_ptr().add(ii * k),
                    b_ptr.add(j),
                    c_ptr.add(ii * c_stride).add(j),
                    k,
                    n_stride,
                );
            }
        }
        j += NR;
    }
    // Scalar tail columns.
    while j < cols {
        for ii in 0..rows {
            let mut s = *c_ptr.add(ii * c_stride + j);
            let arow = &a[ii * k..(ii + 1) * k];
            for kk in 0..k {
                s += arow[kk] * *b_ptr.add(kk * n_stride + j);
            }
            *c_ptr.add(ii * c_stride + j) = s;
        }
        j += 1;
    }
}

/// Strided-output panel GEMM: accumulates `C[i, j_start..j_start+n_panel] +=
/// A[i, 0..k] @ B_panel[0..k, 0..n_panel]` for all rows `i in 0..m`, where the
/// output `C` has row stride `c_stride` (the full output width, typically `HW`).
///
/// `B_panel` is a fully-materialised `[k, n_panel]` row-major tile (e.g. an
/// upsampled spatial strip computed on-the-fly from a smaller input). The
/// caller is responsible for materialising it; this function just runs the
/// tuned microkernel against it.
///
/// Used by [`crate::fast_conv::conv1x1_upsample`] to fuse the bilinear upsample
/// into the conv1x1 GEMM: instead of writing the full `[k, HW]` upsampled
/// activation to DRAM and reading it back, each parallel task materialises a
/// narrow `[k, n_panel]` strip in L1/L2 and immediately consumes it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_nn_panel_strided(
    m: usize,
    n_panel: usize,
    k: usize,
    a: &[f32],
    b_panel: &[f32],
    c: &mut [f32],
    c_stride: usize,
    j_start: usize,
) {
    debug_assert!(a.len() >= m * k, "a.len()={}, need {}*{}", a.len(), m, k);
    debug_assert!(
        b_panel.len() >= k * n_panel,
        "b_panel.len()={}, need {}*{}",
        b_panel.len(),
        k,
        n_panel
    );
    debug_assert!(
        c.len() >= m * c_stride,
        "c.len()={}, need {}*{}",
        c.len(),
        m,
        c_stride
    );
    if m == 0 || n_panel == 0 || k == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // Safety: feature-checked above; slices are bounds-checked above and
        // the microkernel stays within `[0, m) × [j_start, j_start+n_panel)`.
        unsafe {
            let b_ptr = b_panel.as_ptr();
            let c_ptr = c.as_mut_ptr().add(j_start);
            gemm_nn_rows_cols_avx2(m, n_panel, k, a, b_ptr, n_panel, c_ptr, c_stride);
        }
        return;
    }
    // Scalar fallback: same strided-output semantics.
    for ii in 0..m {
        let arow = &a[ii * k..(ii + 1) * k];
        for jj in 0..n_panel {
            let mut s = c[ii * c_stride + j_start + jj];
            for kk in 0..k {
                s += arow[kk] * b_panel[kk * n_panel + jj];
            }
            c[ii * c_stride + j_start + jj] = s;
        }
    }
}

/// Compute C[rows, n] += A[rows, k] @ B[k, n] for a single row-block.
/// `1 <= rows <= MR`. Walks columns in NR-wide tiles, falling back to
/// scalar for the tail columns. Full-width tiles dispatch to the MR-wide
/// microkernel when `rows == MR`, otherwise they fall through to per-row
/// 1x16 microkernels (so partial-row tails stay vectorized).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nn_rows_avx2(
    rows: usize,
    n: usize,
    k: usize,
    a: &[f32],     // [rows, k]
    b: &[f32],     // [k, n]
    c: &mut [f32], // [rows, n]
    n_stride: usize,
) {
    let mut j = 0;
    while j + NR <= n {
        if rows == MR {
            // Fast path: full 6-row microkernel.
            microkernel_6x16(&a[0..], &b[j..], &mut c[j..], k, n_stride);
        } else {
            // Tail-row block (1..=5 rows): call the 1-row microkernel per row.
            for ii in 0..rows {
                let a_row = &a[ii * k..(ii + 1) * k];
                let c_row = &mut c[ii * n_stride + j..];
                microkernel_1x16(a_row, &b[j..], c_row, k, n_stride);
            }
        }
        j += NR;
    }
    // Scalar tail columns.
    while j < n {
        for ii in 0..rows {
            let mut s = c[ii * n_stride + j];
            let arow = &a[ii * k..(ii + 1) * k];
            for kk in 0..k {
                s += arow[kk] * b[kk * n_stride + j];
            }
            c[ii * n_stride + j] = s;
        }
        j += 1;
    }
}

/// Like [`gemm_nn_rows_avx2`] but takes A as a raw pointer with independent
/// row stride (`a_stride`) instead of contiguous `[rows, kc_len]`. Used by
/// [`gemm_nn_avx2_kblocked`] where A points at `A[i_base, kc_start]` in the
/// full `[M, K]` layout (so row stride is `k`, the full K).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nn_rows_kc_avx2(
    rows: usize,
    n: usize,
    kc_len: usize,
    a_ptr: *const f32, // points at A[0, 0] of this row block, K-chunk start
    a_stride: usize,   // A row stride (typically full K)
    b: &[f32],         // [kc_len, n] contiguous
    c: &mut [f32],     // [rows, n] contiguous
) {
    let n_stride = n;
    let mut j = 0;
    while j + NR <= n {
        if rows == MR {
            microkernel_6x16_kc_ptr(
                a_ptr,
                b.as_ptr().add(j),
                c.as_mut_ptr().add(j),
                kc_len,
                a_stride,
                n_stride,
                n_stride,
            );
        } else {
            for ii in 0..rows {
                microkernel_1x16_kc_ptr(
                    a_ptr.add(ii * a_stride),
                    b.as_ptr().add(j),
                    c.as_mut_ptr().add(ii * n_stride).add(j),
                    kc_len,
                    n_stride,
                );
            }
        }
        j += NR;
    }
    // Scalar tail columns.
    while j < n {
        for ii in 0..rows {
            let mut s = c[ii * n_stride + j];
            let arow_ptr = a_ptr.add(ii * a_stride);
            for kk in 0..kc_len {
                s += *arow_ptr.add(kk) * b[kk * n_stride + j];
            }
            c[ii * n_stride + j] = s;
        }
        j += 1;
    }
}

/// 6 rows × 16 cols microkernel (2 AVX2 vectors per column group).
///
/// Computes `C[i, 0..16] += A[i, 0..k] * B[0..k, 0..16]` for i in 0..6.
/// `b_ptr` points at B row 0, col 0; stride between B rows is `b_stride`.
/// `c_ptr` points at C[i_row_base, 0]; stride between C rows is `c_stride`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn microkernel_6x16(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n_stride: usize) {
    microkernel_6x16_ptr(
        a.as_ptr(),
        b.as_ptr(),
        c.as_mut_ptr(),
        k,
        n_stride,
        n_stride,
    )
}

/// Raw-pointer variant of [`microkernel_6x16`] supporting independent B and C
/// row strides. Used by the N-axis parallel GEMM where B and C share the same
/// full-N row stride but start at a column offset.
///
/// The K-loop is unrolled by 2 to amortise the address-computation ALU ops
/// (6 `leaq`/`addq` per K iteration that compute the A row pointers) across
/// twice as many FMA ops, reducing port-0/1 contention with the 12 FMAs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn microkernel_6x16_ptr(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    k: usize,
    b_stride: usize,
    c_stride: usize,
) {
    use std::arch::x86_64::*;

    // 12 accumulators: acc[i][v] for i in 0..6, v in 0..2.
    let mut acc = [[_mm256_setzero_ps(); 2]; 6];

    // K-unrolled by 2: process two consecutive K rows per loop body. The second
    // K iteration's A broadcasts use the same base addresses with a +4 byte
    // offset, so the compiler reuses the `leaq` results from the first half.
    let k2 = k & !1; // largest even ≤ k
    let mut kk = 0;
    while kk < k2 {
        // --- First K element (kk) ---
        let bp = b_ptr.add(kk * b_stride);
        let bv0a = _mm256_loadu_ps(bp);
        let bv1a = _mm256_loadu_ps(bp.add(8));

        let a0a = _mm256_broadcast_ss(&*a_ptr.add(kk));
        let a1a = _mm256_broadcast_ss(&*a_ptr.add(k + kk));
        let a2a = _mm256_broadcast_ss(&*a_ptr.add(2 * k + kk));
        let a3a = _mm256_broadcast_ss(&*a_ptr.add(3 * k + kk));
        let a4a = _mm256_broadcast_ss(&*a_ptr.add(4 * k + kk));
        let a5a = _mm256_broadcast_ss(&*a_ptr.add(5 * k + kk));

        acc[0][0] = _mm256_fmadd_ps(a0a, bv0a, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0a, bv1a, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1a, bv0a, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1a, bv1a, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2a, bv0a, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2a, bv1a, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3a, bv0a, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3a, bv1a, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4a, bv0a, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4a, bv1a, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5a, bv0a, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5a, bv1a, acc[5][1]);

        // --- Second K element (kk+1) ---
        let bp2 = bp.add(b_stride);
        let bv0b = _mm256_loadu_ps(bp2);
        let bv1b = _mm256_loadu_ps(bp2.add(8));

        // A[i, kk+1] = A[i, kk] + 1 element → base+4 bytes. The compiler reuses
        // the row base addresses computed above.
        let a0b = _mm256_broadcast_ss(&*a_ptr.add(kk + 1));
        let a1b = _mm256_broadcast_ss(&*a_ptr.add(k + kk + 1));
        let a2b = _mm256_broadcast_ss(&*a_ptr.add(2 * k + kk + 1));
        let a3b = _mm256_broadcast_ss(&*a_ptr.add(3 * k + kk + 1));
        let a4b = _mm256_broadcast_ss(&*a_ptr.add(4 * k + kk + 1));
        let a5b = _mm256_broadcast_ss(&*a_ptr.add(5 * k + kk + 1));

        acc[0][0] = _mm256_fmadd_ps(a0b, bv0b, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0b, bv1b, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1b, bv0b, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1b, bv1b, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2b, bv0b, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2b, bv1b, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3b, bv0b, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3b, bv1b, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4b, bv0b, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4b, bv1b, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5b, bv0b, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5b, bv1b, acc[5][1]);

        kk += 2;
    }
    // Tail: odd K.
    if kk < k {
        let bp = b_ptr.add(kk * b_stride);
        let bv0 = _mm256_loadu_ps(bp);
        let bv1 = _mm256_loadu_ps(bp.add(8));

        let a0 = _mm256_broadcast_ss(&*a_ptr.add(kk));
        let a1 = _mm256_broadcast_ss(&*a_ptr.add(k + kk));
        let a2 = _mm256_broadcast_ss(&*a_ptr.add(2 * k + kk));
        let a3 = _mm256_broadcast_ss(&*a_ptr.add(3 * k + kk));
        let a4 = _mm256_broadcast_ss(&*a_ptr.add(4 * k + kk));
        let a5 = _mm256_broadcast_ss(&*a_ptr.add(5 * k + kk));

        acc[0][0] = _mm256_fmadd_ps(a0, bv0, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0, bv1, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1, bv0, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1, bv1, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2, bv0, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2, bv1, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3, bv0, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3, bv1, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4, bv0, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4, bv1, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5, bv0, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5, bv1, acc[5][1]);
    }

    // Store. We add into C (the caller pre-zeroes or accumulates).
    for i in 0..6 {
        let crow = c_ptr.add(i * c_stride);
        let cur0 = _mm256_loadu_ps(crow);
        let cur1 = _mm256_loadu_ps(crow.add(8));
        _mm256_storeu_ps(crow, _mm256_add_ps(acc[i][0], cur0));
        _mm256_storeu_ps(crow.add(8), _mm256_add_ps(acc[i][1], cur1));
    }
}

/// 1 row × 16 cols microkernel (tail rows).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn microkernel_1x16(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n_stride: usize) {
    microkernel_1x16_ptr(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), k, n_stride)
}

/// Raw-pointer variant of [`microkernel_1x16`] supporting independent B and C
/// row strides.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn microkernel_1x16_ptr(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    k: usize,
    b_stride: usize,
) {
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for kk in 0..k {
        let bv0 = _mm256_loadu_ps(b_ptr.add(kk * b_stride));
        let bv1 = _mm256_loadu_ps(b_ptr.add(kk * b_stride + 8));
        let av = _mm256_broadcast_ss(&*a_ptr.add(kk));
        acc0 = _mm256_fmadd_ps(av, bv0, acc0);
        acc1 = _mm256_fmadd_ps(av, bv1, acc1);
    }

    let cur0 = _mm256_loadu_ps(c_ptr);
    let cur1 = _mm256_loadu_ps(c_ptr.add(8));
    _mm256_storeu_ps(c_ptr, _mm256_add_ps(acc0, cur0));
    _mm256_storeu_ps(c_ptr.add(8), _mm256_add_ps(acc1, cur1));
}

/// Strided-A variant of [`microkernel_6x16_ptr`] for the K-blocked path:
/// the K-loop iterates `kc_len` times, but A rows are spaced by `a_stride`
/// (typically the full K, not kc_len). B and C row strides remain independent.
///
/// K-loop unrolled by 2 for the same reason as [`microkernel_6x16_ptr`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn microkernel_6x16_kc_ptr(
    a_ptr: *const f32, // points at A[0, kc_start]
    b_ptr: *const f32, // points at B[kc_start, j_tile]
    c_ptr: *mut f32,   // points at C[0, j_tile]
    kc_len: usize,     // K-loop iteration count
    a_stride: usize,   // A row stride (typically full K)
    b_stride: usize,   // B row stride (== N of the full B matrix)
    c_stride: usize,   // C row stride (== N of the full C matrix)
) {
    use std::arch::x86_64::*;

    let mut acc = [[_mm256_setzero_ps(); 2]; 6];

    // K-unrolled by 2.
    let k2 = kc_len & !1;
    let mut kk = 0;
    while kk < k2 {
        // --- First K element (kk) ---
        let bp = b_ptr.add(kk * b_stride);
        let bv0a = _mm256_loadu_ps(bp);
        let bv1a = _mm256_loadu_ps(bp.add(8));

        let a0a = _mm256_broadcast_ss(&*a_ptr.add(kk));
        let a1a = _mm256_broadcast_ss(&*a_ptr.add(a_stride + kk));
        let a2a = _mm256_broadcast_ss(&*a_ptr.add(2 * a_stride + kk));
        let a3a = _mm256_broadcast_ss(&*a_ptr.add(3 * a_stride + kk));
        let a4a = _mm256_broadcast_ss(&*a_ptr.add(4 * a_stride + kk));
        let a5a = _mm256_broadcast_ss(&*a_ptr.add(5 * a_stride + kk));

        acc[0][0] = _mm256_fmadd_ps(a0a, bv0a, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0a, bv1a, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1a, bv0a, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1a, bv1a, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2a, bv0a, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2a, bv1a, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3a, bv0a, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3a, bv1a, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4a, bv0a, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4a, bv1a, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5a, bv0a, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5a, bv1a, acc[5][1]);

        // --- Second K element (kk+1) ---
        let bp2 = bp.add(b_stride);
        let bv0b = _mm256_loadu_ps(bp2);
        let bv1b = _mm256_loadu_ps(bp2.add(8));

        let a0b = _mm256_broadcast_ss(&*a_ptr.add(kk + 1));
        let a1b = _mm256_broadcast_ss(&*a_ptr.add(a_stride + kk + 1));
        let a2b = _mm256_broadcast_ss(&*a_ptr.add(2 * a_stride + kk + 1));
        let a3b = _mm256_broadcast_ss(&*a_ptr.add(3 * a_stride + kk + 1));
        let a4b = _mm256_broadcast_ss(&*a_ptr.add(4 * a_stride + kk + 1));
        let a5b = _mm256_broadcast_ss(&*a_ptr.add(5 * a_stride + kk + 1));

        acc[0][0] = _mm256_fmadd_ps(a0b, bv0b, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0b, bv1b, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1b, bv0b, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1b, bv1b, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2b, bv0b, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2b, bv1b, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3b, bv0b, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3b, bv1b, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4b, bv0b, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4b, bv1b, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5b, bv0b, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5b, bv1b, acc[5][1]);

        kk += 2;
    }
    // Tail: odd kc_len.
    if kk < kc_len {
        let bp = b_ptr.add(kk * b_stride);
        let bv0 = _mm256_loadu_ps(bp);
        let bv1 = _mm256_loadu_ps(bp.add(8));

        let a0 = _mm256_broadcast_ss(&*a_ptr.add(kk));
        let a1 = _mm256_broadcast_ss(&*a_ptr.add(a_stride + kk));
        let a2 = _mm256_broadcast_ss(&*a_ptr.add(2 * a_stride + kk));
        let a3 = _mm256_broadcast_ss(&*a_ptr.add(3 * a_stride + kk));
        let a4 = _mm256_broadcast_ss(&*a_ptr.add(4 * a_stride + kk));
        let a5 = _mm256_broadcast_ss(&*a_ptr.add(5 * a_stride + kk));

        acc[0][0] = _mm256_fmadd_ps(a0, bv0, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0, bv1, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1, bv0, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1, bv1, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2, bv0, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2, bv1, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3, bv0, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3, bv1, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4, bv0, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4, bv1, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5, bv0, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5, bv1, acc[5][1]);
    }

    for i in 0..6 {
        let crow = c_ptr.add(i * c_stride);
        let cur0 = _mm256_loadu_ps(crow);
        let cur1 = _mm256_loadu_ps(crow.add(8));
        _mm256_storeu_ps(crow, _mm256_add_ps(acc[i][0], cur0));
        _mm256_storeu_ps(crow.add(8), _mm256_add_ps(acc[i][1], cur1));
    }
}

/// Strided-A variant of [`microkernel_1x16_ptr`] for the K-blocked path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn microkernel_1x16_kc_ptr(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    kc_len: usize,
    b_stride: usize,
) {
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for kk in 0..kc_len {
        let bv0 = _mm256_loadu_ps(b_ptr.add(kk * b_stride));
        let bv1 = _mm256_loadu_ps(b_ptr.add(kk * b_stride + 8));
        let av = _mm256_broadcast_ss(&*a_ptr.add(kk));
        acc0 = _mm256_fmadd_ps(av, bv0, acc0);
        acc1 = _mm256_fmadd_ps(av, bv1, acc1);
    }

    let cur0 = _mm256_loadu_ps(c_ptr);
    let cur1 = _mm256_loadu_ps(c_ptr.add(8));
    _mm256_storeu_ps(c_ptr, _mm256_add_ps(acc0, cur0));
    _mm256_storeu_ps(c_ptr.add(8), _mm256_add_ps(acc1, cur1));
}

// ---------------------------------------------------------------------------
// Internal: bias-fused GEMM (C = A @ B + bias)
// ---------------------------------------------------------------------------
//
// These are the bias-init counterparts of the microkernels and dispatchers
// above. The key difference: instead of zero-initializing the accumulators and
// doing load-add-store in the epilogue, we initialize the accumulators WITH the
// bias vector (broadcast across the MR rows of a tile) and do a pure store.
// This eliminates the bias pre-copy into C and the load of C in the store path.
//
// For K-blocked shapes, only the first K-chunk uses these bias-init
// microkernels. Subsequent K-chunks use the regular accumulate microkernels
// (which read C back and add to it), so the bias is correctly carried through.

/// 6×16 bias-init microkernel: computes `C[0..6, tile] = A[0..6, 0..k] @
/// B[0..k, tile] + bias[tile]` and stores (pure store, no accumulate).
///
/// `bias_ptr` points at `bias[tile_col_start]` and must have at least 16
/// elements. The bias is loaded once and broadcast to all 6 rows' accumulators.
///
/// The K-loop is identical to [`microkernel_6x16_ptr`]; only the accumulator
/// init and store epilogue differ.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn microkernel_6x16_ptr_bias(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    k: usize,
    b_stride: usize,
    c_stride: usize,
    bias_ptr: *const f32,
) {
    use std::arch::x86_64::*;

    // Zero-init accumulators (same as the non-bias microkernel). The bias is
    // added in the store epilogue, which avoids adding load latency to the
    // critical path of the first FMA.
    let mut acc = [[_mm256_setzero_ps(); 2]; 6];

    let k2 = k & !1;
    let mut kk = 0;
    while kk < k2 {
        let bp = b_ptr.add(kk * b_stride);
        let bv0a = _mm256_loadu_ps(bp);
        let bv1a = _mm256_loadu_ps(bp.add(8));

        let a0a = _mm256_broadcast_ss(&*a_ptr.add(kk));
        let a1a = _mm256_broadcast_ss(&*a_ptr.add(k + kk));
        let a2a = _mm256_broadcast_ss(&*a_ptr.add(2 * k + kk));
        let a3a = _mm256_broadcast_ss(&*a_ptr.add(3 * k + kk));
        let a4a = _mm256_broadcast_ss(&*a_ptr.add(4 * k + kk));
        let a5a = _mm256_broadcast_ss(&*a_ptr.add(5 * k + kk));

        acc[0][0] = _mm256_fmadd_ps(a0a, bv0a, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0a, bv1a, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1a, bv0a, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1a, bv1a, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2a, bv0a, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2a, bv1a, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3a, bv0a, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3a, bv1a, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4a, bv0a, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4a, bv1a, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5a, bv0a, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5a, bv1a, acc[5][1]);

        let bp2 = bp.add(b_stride);
        let bv0b = _mm256_loadu_ps(bp2);
        let bv1b = _mm256_loadu_ps(bp2.add(8));

        let a0b = _mm256_broadcast_ss(&*a_ptr.add(kk + 1));
        let a1b = _mm256_broadcast_ss(&*a_ptr.add(k + kk + 1));
        let a2b = _mm256_broadcast_ss(&*a_ptr.add(2 * k + kk + 1));
        let a3b = _mm256_broadcast_ss(&*a_ptr.add(3 * k + kk + 1));
        let a4b = _mm256_broadcast_ss(&*a_ptr.add(4 * k + kk + 1));
        let a5b = _mm256_broadcast_ss(&*a_ptr.add(5 * k + kk + 1));

        acc[0][0] = _mm256_fmadd_ps(a0b, bv0b, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0b, bv1b, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1b, bv0b, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1b, bv1b, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2b, bv0b, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2b, bv1b, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3b, bv0b, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3b, bv1b, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4b, bv0b, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4b, bv1b, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5b, bv0b, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5b, bv1b, acc[5][1]);

        kk += 2;
    }
    if kk < k {
        let bp = b_ptr.add(kk * b_stride);
        let bv0 = _mm256_loadu_ps(bp);
        let bv1 = _mm256_loadu_ps(bp.add(8));

        let a0 = _mm256_broadcast_ss(&*a_ptr.add(kk));
        let a1 = _mm256_broadcast_ss(&*a_ptr.add(k + kk));
        let a2 = _mm256_broadcast_ss(&*a_ptr.add(2 * k + kk));
        let a3 = _mm256_broadcast_ss(&*a_ptr.add(3 * k + kk));
        let a4 = _mm256_broadcast_ss(&*a_ptr.add(4 * k + kk));
        let a5 = _mm256_broadcast_ss(&*a_ptr.add(5 * k + kk));

        acc[0][0] = _mm256_fmadd_ps(a0, bv0, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0, bv1, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1, bv0, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1, bv1, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2, bv0, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2, bv1, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3, bv0, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3, bv1, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4, bv0, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4, bv1, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5, bv0, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5, bv1, acc[5][1]);
    }

    // Store: load bias (same for all 6 rows, hoisted out of loop), add to each
    // row's accumulator, and store. This replaces the non-bias load-from-C with
    // a load-from-bias, which is L1-hot (the bias vector is ≤ 12 KB).
    let bv0 = _mm256_loadu_ps(bias_ptr);
    let bv1 = _mm256_loadu_ps(bias_ptr.add(8));
    for i in 0..6 {
        let crow = c_ptr.add(i * c_stride);
        _mm256_storeu_ps(crow, _mm256_add_ps(acc[i][0], bv0));
        _mm256_storeu_ps(crow.add(8), _mm256_add_ps(acc[i][1], bv1));
    }
}

/// 1×16 bias-init microkernel (tail rows). See [`microkernel_6x16_ptr_bias`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn microkernel_1x16_ptr_bias(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    k: usize,
    b_stride: usize,
    bias_ptr: *const f32,
) {
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for kk in 0..k {
        let bv0 = _mm256_loadu_ps(b_ptr.add(kk * b_stride));
        let bv1 = _mm256_loadu_ps(b_ptr.add(kk * b_stride + 8));
        let av = _mm256_broadcast_ss(&*a_ptr.add(kk));
        acc0 = _mm256_fmadd_ps(av, bv0, acc0);
        acc1 = _mm256_fmadd_ps(av, bv1, acc1);
    }

    let bv0 = _mm256_loadu_ps(bias_ptr);
    let bv1 = _mm256_loadu_ps(bias_ptr.add(8));
    _mm256_storeu_ps(c_ptr, _mm256_add_ps(acc0, bv0));
    _mm256_storeu_ps(c_ptr.add(8), _mm256_add_ps(acc1, bv1));
}

/// 6×16 strided-A bias-init microkernel for the K-blocked path (first chunk
/// only). See [`microkernel_6x16_kc_ptr`] and [`microkernel_6x16_ptr_bias`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn microkernel_6x16_kc_ptr_bias(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    kc_len: usize,
    a_stride: usize,
    b_stride: usize,
    c_stride: usize,
    bias_ptr: *const f32,
) {
    use std::arch::x86_64::*;

    let mut acc = [[_mm256_setzero_ps(); 2]; 6];

    let k2 = kc_len & !1;
    let mut kk = 0;
    while kk < k2 {
        let bp = b_ptr.add(kk * b_stride);
        let bv0a = _mm256_loadu_ps(bp);
        let bv1a = _mm256_loadu_ps(bp.add(8));

        let a0a = _mm256_broadcast_ss(&*a_ptr.add(kk));
        let a1a = _mm256_broadcast_ss(&*a_ptr.add(a_stride + kk));
        let a2a = _mm256_broadcast_ss(&*a_ptr.add(2 * a_stride + kk));
        let a3a = _mm256_broadcast_ss(&*a_ptr.add(3 * a_stride + kk));
        let a4a = _mm256_broadcast_ss(&*a_ptr.add(4 * a_stride + kk));
        let a5a = _mm256_broadcast_ss(&*a_ptr.add(5 * a_stride + kk));

        acc[0][0] = _mm256_fmadd_ps(a0a, bv0a, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0a, bv1a, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1a, bv0a, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1a, bv1a, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2a, bv0a, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2a, bv1a, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3a, bv0a, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3a, bv1a, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4a, bv0a, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4a, bv1a, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5a, bv0a, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5a, bv1a, acc[5][1]);

        let bp2 = bp.add(b_stride);
        let bv0b = _mm256_loadu_ps(bp2);
        let bv1b = _mm256_loadu_ps(bp2.add(8));

        let a0b = _mm256_broadcast_ss(&*a_ptr.add(kk + 1));
        let a1b = _mm256_broadcast_ss(&*a_ptr.add(a_stride + kk + 1));
        let a2b = _mm256_broadcast_ss(&*a_ptr.add(2 * a_stride + kk + 1));
        let a3b = _mm256_broadcast_ss(&*a_ptr.add(3 * a_stride + kk + 1));
        let a4b = _mm256_broadcast_ss(&*a_ptr.add(4 * a_stride + kk + 1));
        let a5b = _mm256_broadcast_ss(&*a_ptr.add(5 * a_stride + kk + 1));

        acc[0][0] = _mm256_fmadd_ps(a0b, bv0b, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0b, bv1b, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1b, bv0b, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1b, bv1b, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2b, bv0b, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2b, bv1b, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3b, bv0b, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3b, bv1b, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4b, bv0b, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4b, bv1b, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5b, bv0b, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5b, bv1b, acc[5][1]);

        kk += 2;
    }
    if kk < kc_len {
        let bp = b_ptr.add(kk * b_stride);
        let bv0 = _mm256_loadu_ps(bp);
        let bv1 = _mm256_loadu_ps(bp.add(8));

        let a0 = _mm256_broadcast_ss(&*a_ptr.add(kk));
        let a1 = _mm256_broadcast_ss(&*a_ptr.add(a_stride + kk));
        let a2 = _mm256_broadcast_ss(&*a_ptr.add(2 * a_stride + kk));
        let a3 = _mm256_broadcast_ss(&*a_ptr.add(3 * a_stride + kk));
        let a4 = _mm256_broadcast_ss(&*a_ptr.add(4 * a_stride + kk));
        let a5 = _mm256_broadcast_ss(&*a_ptr.add(5 * a_stride + kk));

        acc[0][0] = _mm256_fmadd_ps(a0, bv0, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(a0, bv1, acc[0][1]);
        acc[1][0] = _mm256_fmadd_ps(a1, bv0, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(a1, bv1, acc[1][1]);
        acc[2][0] = _mm256_fmadd_ps(a2, bv0, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(a2, bv1, acc[2][1]);
        acc[3][0] = _mm256_fmadd_ps(a3, bv0, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(a3, bv1, acc[3][1]);
        acc[4][0] = _mm256_fmadd_ps(a4, bv0, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(a4, bv1, acc[4][1]);
        acc[5][0] = _mm256_fmadd_ps(a5, bv0, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(a5, bv1, acc[5][1]);
    }

    let bv0 = _mm256_loadu_ps(bias_ptr);
    let bv1 = _mm256_loadu_ps(bias_ptr.add(8));
    for i in 0..6 {
        let crow = c_ptr.add(i * c_stride);
        _mm256_storeu_ps(crow, _mm256_add_ps(acc[i][0], bv0));
        _mm256_storeu_ps(crow.add(8), _mm256_add_ps(acc[i][1], bv1));
    }
}

/// 1×16 strided-A bias-init microkernel for the K-blocked path (first chunk
/// only). See [`microkernel_1x16_kc_ptr`] and [`microkernel_1x16_ptr_bias`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn microkernel_1x16_kc_ptr_bias(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    kc_len: usize,
    b_stride: usize,
    bias_ptr: *const f32,
) {
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for kk in 0..kc_len {
        let bv0 = _mm256_loadu_ps(b_ptr.add(kk * b_stride));
        let bv1 = _mm256_loadu_ps(b_ptr.add(kk * b_stride + 8));
        let av = _mm256_broadcast_ss(&*a_ptr.add(kk));
        acc0 = _mm256_fmadd_ps(av, bv0, acc0);
        acc1 = _mm256_fmadd_ps(av, bv1, acc1);
    }

    let bv0 = _mm256_loadu_ps(bias_ptr);
    let bv1 = _mm256_loadu_ps(bias_ptr.add(8));
    _mm256_storeu_ps(c_ptr, _mm256_add_ps(acc0, bv0));
    _mm256_storeu_ps(c_ptr.add(8), _mm256_add_ps(acc1, bv1));
}

/// Bias-init variant of [`gemm_nn_rows_chunk_avx2`]. Passes `bias_ptr` (pointing
/// at `bias[0]`, length `n`) through to the row-walker.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn gemm_nn_rows_chunk_avx2_bias(
    rows: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    bias_ptr: *const f32,
) {
    let mut i = 0;
    while i < rows {
        let r = (rows - i).min(MR);
        let a_row = &a[i * k..(i + r) * k];
        let c_row = &mut c[i * n..(i + r) * n];
        gemm_nn_rows_avx2_bias(r, n, k, a_row, b, c_row, n, bias_ptr);
        i += r;
    }
}

/// Bias-init variant of [`gemm_nn_rows_avx2`]. Same dispatch structure; calls
/// the `_bias` microkernels and uses `bias[j]` as the starting value for scalar
/// tail columns.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nn_rows_avx2_bias(
    rows: usize,
    n: usize,
    k: usize,
    a: &[f32],     // [rows, k]
    b: &[f32],     // [k, n]
    c: &mut [f32], // [rows, n]
    n_stride: usize,
    bias_ptr: *const f32, // points at bias[0], length n
) {
    let mut j = 0;
    while j + NR <= n {
        if rows == MR {
            microkernel_6x16_ptr_bias(
                a.as_ptr(),
                b.as_ptr().add(j),
                c.as_mut_ptr().add(j),
                k,
                n_stride,
                n_stride,
                bias_ptr.add(j),
            );
        } else {
            for ii in 0..rows {
                microkernel_1x16_ptr_bias(
                    a.as_ptr().add(ii * k),
                    b.as_ptr().add(j),
                    c.as_mut_ptr().add(ii * n_stride).add(j),
                    k,
                    n_stride,
                    bias_ptr.add(j),
                );
            }
        }
        j += NR;
    }
    // Scalar tail columns: out = bias[j] + sum_k(a[k] * b[k*n+j]).
    while j < n {
        for ii in 0..rows {
            let arow = &a[ii * k..(ii + 1) * k];
            let mut s = *bias_ptr.add(j);
            for kk in 0..k {
                s += arow[kk] * b[kk * n_stride + j];
            }
            c[ii * n_stride + j] = s;
        }
        j += 1;
    }
}

/// Bias-init variant of [`gemm_nn_rows_cols_avx2`] (N-axis parallelism).
/// `bias_ptr` points at `bias[j_block_start]`, and the microkernel for column
/// tile `j` (within the block) receives `bias_ptr.add(j)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nn_rows_cols_avx2_bias(
    rows: usize,
    cols: usize,
    k: usize,
    a: &[f32],
    b_ptr: *const f32,
    n_stride: usize,
    c_ptr: *mut f32,
    c_stride: usize,
    bias_ptr: *const f32, // points at bias[j_block_start]
) {
    let mut j = 0;
    while j + NR <= cols {
        if rows == MR {
            microkernel_6x16_ptr_bias(
                a.as_ptr(),
                b_ptr.add(j),
                c_ptr.add(j),
                k,
                n_stride,
                c_stride,
                bias_ptr.add(j),
            );
        } else {
            for ii in 0..rows {
                microkernel_1x16_ptr_bias(
                    a.as_ptr().add(ii * k),
                    b_ptr.add(j),
                    c_ptr.add(ii * c_stride).add(j),
                    k,
                    n_stride,
                    bias_ptr.add(j),
                );
            }
        }
        j += NR;
    }
    // Scalar tail columns.
    while j < cols {
        for ii in 0..rows {
            let mut s = *bias_ptr.add(j);
            let arow = &a[ii * k..(ii + 1) * k];
            for kk in 0..k {
                s += arow[kk] * *b_ptr.add(kk * n_stride + j);
            }
            *c_ptr.add(ii * c_stride + j) = s;
        }
        j += 1;
    }
}

/// Bias-init variant of [`gemm_nn_rows_chunk_kc_avx2`] for the first K-chunk.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn gemm_nn_rows_chunk_kc_avx2_bias(
    rows: usize,
    n: usize,
    kc_len: usize,
    a_ptr: *const f32,
    a_stride: usize,
    b: &[f32],
    c: &mut [f32],
    bias_ptr: *const f32,
) {
    let mut i = 0;
    while i < rows {
        let r = (rows - i).min(MR);
        let a_row_ptr = a_ptr.add(i * a_stride);
        let c_row = &mut c[i * n..(i + r) * n];
        gemm_nn_rows_kc_avx2_bias(r, n, kc_len, a_row_ptr, a_stride, b, c_row, bias_ptr);
        i += r;
    }
}

/// Bias-init variant of [`gemm_nn_rows_kc_avx2`] for the first K-chunk.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_nn_rows_kc_avx2_bias(
    rows: usize,
    n: usize,
    kc_len: usize,
    a_ptr: *const f32,
    a_stride: usize,
    b: &[f32],
    c: &mut [f32],
    bias_ptr: *const f32,
) {
    let n_stride = n;
    let mut j = 0;
    while j + NR <= n {
        if rows == MR {
            microkernel_6x16_kc_ptr_bias(
                a_ptr,
                b.as_ptr().add(j),
                c.as_mut_ptr().add(j),
                kc_len,
                a_stride,
                n_stride,
                n_stride,
                bias_ptr.add(j),
            );
        } else {
            for ii in 0..rows {
                microkernel_1x16_kc_ptr_bias(
                    a_ptr.add(ii * a_stride),
                    b.as_ptr().add(j),
                    c.as_mut_ptr().add(ii * n_stride).add(j),
                    kc_len,
                    n_stride,
                    bias_ptr.add(j),
                );
            }
        }
        j += NR;
    }
    // Scalar tail columns.
    while j < n {
        for ii in 0..rows {
            let mut s = *bias_ptr.add(j);
            let arow_ptr = a_ptr.add(ii * a_stride);
            for kk in 0..kc_len {
                s += *arow_ptr.add(kk) * b[kk * n_stride + j];
            }
            c[ii * n_stride + j] = s;
        }
        j += 1;
    }
}

/// M-axis parallelisation with bias fusion. Each rayon task processes `MC`
/// rows × all N columns, calling [`gemm_nn_rows_chunk_avx2_bias`].
#[cfg(target_arch = "x86_64")]
unsafe fn gemm_nn_avx2_par_m_bias(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    bias: &[f32],
) {
    let mc = mc();
    let chunk_rows = if m >= mc { mc } else { MR };
    let chunk = chunk_rows * n;
    // Wrap as usize so the closure is Send+Sync (raw pointers are neither).
    let bias_addr = bias.as_ptr() as usize;
    c.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(i_blk, c_row)| unsafe {
            let i_base = i_blk * chunk_rows;
            let rows_in_chunk = c_row.len() / n;
            let a_row = &a[i_base * k..(i_base + rows_in_chunk) * k];
            let bias_ptr = bias_addr as *const f32;
            gemm_nn_rows_chunk_avx2_bias(rows_in_chunk, n, k, a_row, b, c_row, bias_ptr);
        });
}

/// N-axis parallelisation with bias fusion. Each rayon task processes all M
/// rows × `NC` columns, calling [`gemm_nn_rows_cols_avx2_bias`] with a bias
/// pointer shifted to the task's column block start.
#[cfg(target_arch = "x86_64")]
unsafe fn gemm_nn_avx2_par_n_bias(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    bias: &[f32],
) {
    let nc = auto_nc(n, k);
    let n_blocks = n.div_ceil(nc);
    let b_addr = b.as_ptr() as usize;
    let c_addr = c.as_mut_ptr() as usize;
    let bias_addr = bias.as_ptr() as usize;
    (0..n_blocks).into_par_iter().for_each(|jb| unsafe {
        let j_start = jb * nc;
        let j_end = (j_start + nc).min(n);
        let b_ptr = b_addr as *const f32;
        let c_ptr = c_addr as *mut f32;
        let bias_ptr = (bias_addr as *const f32).add(j_start);
        let mut i = 0;
        while i < m {
            let r = (m - i).min(MR);
            let a_row = &a[i * k..(i + r) * k];
            let c_row_ptr = c_ptr.add(i * n);
            gemm_nn_rows_cols_avx2_bias(
                r,
                j_end - j_start,
                k,
                a_row,
                b_ptr.add(j_start),
                n,
                c_row_ptr.add(j_start),
                n,
                bias_ptr,
            );
            i += r;
        }
    });
}

/// K-blocked parallelisation with bias fusion. The first K-chunk uses the
/// bias-init microkernels (pure store, C = bias + partial GEMM); subsequent
/// K-chunks use the regular accumulate microkernels (C += partial GEMM),
/// correctly carrying the bias through.
#[cfg(target_arch = "x86_64")]
unsafe fn gemm_nn_avx2_kblocked_bias(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    bias: &[f32],
) {
    let mc = mc();
    let chunk_rows = if m >= mc { mc } else { MR };
    let chunk = chunk_rows * n;
    let kc = auto_kc(n, k).min(k);
    let a_addr = a.as_ptr() as usize;
    // Wrap as usize so the closure is Send+Sync (raw pointers are neither).
    let bias_addr = bias.as_ptr() as usize;
    for (kc_idx, kc_start) in (0..k).step_by(kc).enumerate() {
        let kc_len = kc.min(k - kc_start);
        let b_chunk = &b[kc_start * n..(kc_start + kc_len) * n];
        if kc_idx == 0 {
            // First K-chunk: bias-init (pure store). C = bias + A@B[:kc].
            c.par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(i_blk, c_row)| unsafe {
                    let i_base = i_blk * chunk_rows;
                    let rows_in_chunk = c_row.len() / n;
                    let a_ptr = (a_addr as *const f32).add(i_base * k + kc_start);
                    let bias_ptr = bias_addr as *const f32;
                    gemm_nn_rows_chunk_kc_avx2_bias(
                        rows_in_chunk,
                        n,
                        kc_len,
                        a_ptr,
                        k,
                        b_chunk,
                        c_row,
                        bias_ptr,
                    );
                });
        } else {
            // Subsequent K-chunks: regular accumulate. C += A@B[kc_start..].
            c.par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(i_blk, c_row)| unsafe {
                    let i_base = i_blk * chunk_rows;
                    let rows_in_chunk = c_row.len() / n;
                    let a_ptr = (a_addr as *const f32).add(i_base * k + kc_start);
                    gemm_nn_rows_chunk_kc_avx2(rows_in_chunk, n, kc_len, a_ptr, k, b_chunk, c_row);
                });
        }
    }
}

/// AVX2 dispatcher for bias-fused GEMM. Same axis-selection heuristic as
/// [`gemm_nn_avx2`].
#[cfg(target_arch = "x86_64")]
unsafe fn gemm_nn_avx2_bias(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    bias: &[f32],
) {
    let use_n;
    let use_k;
    match par_axis() {
        'n' => {
            use_n = n >= NR;
            use_k = false;
        }
        'k' => {
            use_n = false;
            use_k = true;
        }
        'm' => {
            use_n = false;
            use_k = false;
        }
        _ => {
            use_k = should_use_k_blocking(m, n, k);
            use_n = !use_k && should_use_n_axis(m, n, k);
        }
    }
    if use_n {
        gemm_nn_avx2_par_n_bias(m, n, k, a, b, c, bias)
    } else if use_k {
        gemm_nn_avx2_kblocked_bias(m, n, k, a, b, c, bias)
    } else {
        gemm_nn_avx2_par_m_bias(m, n, k, a, b, c, bias)
    }
}

// ---------------------------------------------------------------------------
// Internal: scalar reference
// ---------------------------------------------------------------------------

fn gemm_nn_scalar(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    // IKJ order: cache-friendly for row-major C, vectorizable by the compiler.
    for i in 0..m {
        let arow = &a[i * k..(i + 1) * k];
        let crow = &mut c[i * n..(i + 1) * n];
        for kk in 0..k {
            let av = arow[kk];
            if av == 0.0 {
                continue;
            }
            let brow = &b[kk * n..(kk + 1) * n];
            for j in 0..n {
                crow[j] += av * brow[j];
            }
        }
    }
}

/// Scalar reference for [`gemm_nn_bias_into`]. Computes `C = A@B + bias`.
fn gemm_nn_scalar_bias(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    bias: &[f32],
) {
    for i in 0..m {
        let arow = &a[i * k..(i + 1) * k];
        let crow = &mut c[i * n..(i + 1) * n];
        for j in 0..n {
            let mut s = bias[j];
            for kk in 0..k {
                s += arow[kk] * b[kk * n + j];
            }
            crow[j] = s;
        }
    }
}

/// Transpose B from [N,K] row-major to [K,N] row-major (the NN layout).
fn transpose_b_nt(k: usize, n: usize, src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), n * k);
    debug_assert_eq!(dst.len(), k * n);
    // For each (ki, nj), dst[ki*n + nj] = src[nj*k + ki].
    for nj in 0..n {
        let src_row = &src[nj * k..(nj + 1) * k];
        for ki in 0..k {
            dst[ki * n + nj] = src_row[ki];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn naive_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        // b is [n, k]; result[i,j] = sum_k a[i,k] * b[j,k]
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for kk in 0..k {
                    s += a[i * k + kk] * b[j * k + kk];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32) / (1u64 << 31) as f32 - 1.0
            })
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn nn_matches_naive_on_small_multiples() {
        // Multiples of MR/NR.
        let (m, n, k) = (12, 32, 16);
        let a = pseudo_random(1, m * k);
        let b = pseudo_random(2, k * n);
        let expected = naive_nn(m, n, k, &a, &b);
        let got = matmul_nn(m, n, k, &a, &b);
        assert!(
            max_abs_diff(&expected, &got) < 1e-3,
            "nn small multiples failed"
        );
    }

    #[test]
    fn nn_matches_naive_with_tail_rows_and_cols() {
        // 9 rows (tail of 3 after 6), 20 cols (tail of 4 after 16).
        let (m, n, k) = (9, 20, 17);
        let a = pseudo_random(3, m * k);
        let b = pseudo_random(4, k * n);
        let expected = naive_nn(m, n, k, &a, &b);
        let got = matmul_nn(m, n, k, &a, &b);
        assert!(max_abs_diff(&expected, &got) < 1e-3, "nn tails failed");
    }

    #[test]
    fn nn_matches_naive_on_attention_shape() {
        // Real DA3 attention QK^T shape (one head).
        let (m, n, k) = (673, 673, 64);
        let a = pseudo_random(5, m * k);
        let b = pseudo_random(6, k * n);
        let expected = naive_nn(m, n, k, &a, &b);
        let got = matmul_nn(m, n, k, &a, &b);
        let rel =
            max_abs_diff(&expected, &got) / expected.iter().cloned().fold(0.0f32, f32::max).abs();
        assert!(rel < 1e-4, "nn attention shape failed: rel={rel}");
    }

    #[test]
    fn nn_matches_naive_on_qkv_shape() {
        let (m, n, k) = (673, 2304, 768);
        let a = pseudo_random(7, m * k);
        let b = pseudo_random(8, k * n);
        let expected = naive_nn(m, n, k, &a, &b);
        let got = matmul_nn(m, n, k, &a, &b);
        let rel =
            max_abs_diff(&expected, &got) / expected.iter().cloned().fold(0.0f32, f32::max).abs();
        assert!(rel < 1e-4, "nn qkv shape failed: rel={rel}");
    }

    #[test]
    fn nt_matches_naive_on_small() {
        let (m, n, k) = (12, 32, 16);
        let a = pseudo_random(11, m * k);
        let b = pseudo_random(12, n * k); // [n, k]
        let expected = naive_nt(m, n, k, &a, &b);
        let got = matmul_nt(m, n, k, &a, &b);
        assert!(max_abs_diff(&expected, &got) < 1e-3, "nt small failed");
    }

    #[test]
    fn nt_matches_naive_on_linear_shape() {
        // Linear: x[m,k] @ W[n,k]^T -> [m, n] (DA3 FC1 shape)
        let (m, n, k) = (673, 3072, 768);
        let a = pseudo_random(13, m * k);
        let w = pseudo_random(14, n * k);
        let expected = naive_nt(m, n, k, &a, &w);
        let got = matmul_nt(m, n, k, &a, &w);
        let rel =
            max_abs_diff(&expected, &got) / expected.iter().cloned().fold(0.0f32, f32::max).abs();
        assert!(rel < 1e-4, "nt linear shape failed: rel={rel}");
    }

    #[test]
    fn linear_fwd_applies_bias() {
        let (m, n, k) = (4, 8, 4);
        let x = pseudo_random(21, m * k);
        let w = pseudo_random(22, n * k);
        let bias = pseudo_random(23, n);
        let got = linear_fwd(m, n, k, &x, &w, Some(&bias));

        let mut expected = naive_nt(m, n, k, &x, &w);
        for i in 0..m {
            for j in 0..n {
                expected[i * n + j] += bias[j];
            }
        }
        assert!(max_abs_diff(&expected, &got) < 1e-4, "linear bias failed");
    }

    #[test]
    fn gemm_nn_into_accumulates() {
        let (m, n, k) = (6, 16, 8);
        let a = pseudo_random(31, m * k);
        let b = pseudo_random(32, k * n);
        let mut c = pseudo_random(33, m * n);
        let mut expected = c.clone();
        // expected += naive a@b
        let ab = naive_nn(m, n, k, &a, &b);
        for i in 0..m * n {
            expected[i] += ab[i];
        }
        gemm_nn_into(m, n, k, &a, &b, &mut c);
        assert!(max_abs_diff(&expected, &c) < 1e-4, "accumulate failed");
    }

    /// Validate the K-blocked GEMM path by directly invoking
    /// `gemm_nn_avx2_kblocked` (bypassing the heuristic) and comparing to the
    /// naive reference. Uses a small shape to keep the test fast, but exercises
    /// the full K-chunk outer loop + strided-A microkernel.
    #[test]
    fn kblocked_gemm_matches_naive() {
        // Small shape: M=18, N=32, K=32. KC default would be auto_kc(32, 32):
        // n ≤ 256 → k/2 = 16. So 2 K-chunks of 16 each.
        let (m, n, k) = (18, 32, 32);
        let a = pseudo_random(51, m * k);
        let b = pseudo_random(52, k * n);
        let expected = naive_nn(m, n, k, &a, &b);
        let mut c = vec![0.0f32; m * n];
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // Safety: feature-checked above; sizes match the inputs.
            unsafe { gemm_nn_avx2_kblocked(m, n, k, &a, &b, &mut c) };
        } else {
            gemm_nn_scalar(m, n, k, &a, &b, &mut c);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            gemm_nn_scalar(m, n, k, &a, &b, &mut c);
        }
        assert!(max_abs_diff(&expected, &c) < 1e-3, "kblocked path failed");
    }

    /// Larger shape that exercises K-blocking with multiple M-blocks and
    /// multiple K-chunks, including tail handling in both dimensions.
    #[test]
    fn kblocked_gemm_large_shape_matches_naive() {
        // M=48, N=40, K=64. KC = auto_kc(40, 64) = 32 (k/2). 2 K-chunks.
        // M=48 → 8 M-blocks of MC=6 (no tail).
        let (m, n, k) = (48, 40, 64);
        let a = pseudo_random(61, m * k);
        let b = pseudo_random(62, k * n);
        let expected = naive_nn(m, n, k, &a, &b);
        let mut c = vec![0.0f32; m * n];
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            unsafe { gemm_nn_avx2_kblocked(m, n, k, &a, &b, &mut c) };
        } else {
            gemm_nn_scalar(m, n, k, &a, &b, &mut c);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            gemm_nn_scalar(m, n, k, &a, &b, &mut c);
        }
        assert!(
            max_abs_diff(&expected, &c) < 1e-3,
            "kblocked large shape failed"
        );
    }

    /// Shape with M tail (not a multiple of MR=6) to exercise the partial-row
    /// microkernel path within K-blocking.
    #[test]
    fn kblocked_gemm_with_m_tail_matches_naive() {
        // M=20 (not multiple of 6; tail of 2), N=32, K=32.
        let (m, n, k) = (20, 32, 32);
        let a = pseudo_random(71, m * k);
        let b = pseudo_random(72, k * n);
        let expected = naive_nn(m, n, k, &a, &b);
        let mut c = vec![0.0f32; m * n];
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            unsafe { gemm_nn_avx2_kblocked(m, n, k, &a, &b, &mut c) };
        } else {
            gemm_nn_scalar(m, n, k, &a, &b, &mut c);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            gemm_nn_scalar(m, n, k, &a, &b, &mut c);
        }
        assert!(max_abs_diff(&expected, &c) < 1e-3, "kblocked m-tail failed");
    }

    #[test]
    fn empty_inputs_are_noop() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];
        gemm_nn_into(0, 0, 0, &a, &b, &mut c);
        // Just must not panic.
    }

    #[test]
    fn pack_b_nt_round_trips_via_nt() {
        // matmul_nt with raw W vs matmul_nn with packed W must agree.
        let (m, n, k) = (12, 32, 16);
        let a = pseudo_random(41, m * k);
        let w = pseudo_random(42, n * k);
        let direct = matmul_nt(m, n, k, &a, &w);
        let packed = pack_b_nt(k, n, &w);
        let mut via_packed = vec![0.0f32; m * n];
        gemm_nn_into(m, n, k, &a, &packed, &mut via_packed);
        assert!(
            max_abs_diff(&direct, &via_packed) < 1e-4,
            "pack_b_nt disagrees"
        );
    }

    /// Verify that the K-blocking heuristic fires for FFN fc1-like shapes
    /// (large N, moderate K: N > 3·K) even when N-axis would otherwise be viable.
    /// This is the shape that motivated the `n <= 3 * k` carve-out in
    /// [`should_use_k_blocking`].
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn k_blocking_preferred_for_fcn_fc1_shape() {
        // FFN fc1: M=864, N=3072, K=768. B = 9.0 MiB > L2. N = 4·K.
        // N-axis would be viable (96 N-blocks at NC=32 ≥ 24 threads) but
        // K-blocking wins empirically because N > 3·K (the carve-out condition
        // in should_use_k_blocking).
        assert!(should_use_k_blocking(864, 3072, 768));
        // QKV proj: M=864, N=2304, K=768. N = 3·K exactly. N-axis is preferred
        // here (K-blocking was 30% slower in benchmarks).
        assert!(!should_use_k_blocking(864, 2304, 768));
        // FFN fc2: M=864, N=768, K=3072. With NC=32, N-axis is viable
        // (768/32 = 24 N-blocks ≥ 24 threads) and N ≪ 3·K, so N-axis wins
        // over K-blocking. (With the previous NC=64, only 12 N-blocks were
        // available, which undersaturated the 24-thread pool, so K-blocking
        // was preferred. The NC=32 default makes N-axis viable for fc2.)
        assert!(!should_use_k_blocking(864, 768, 3072));
        // Small B that fits L2 (256 KiB ≤ 1.25 MiB): M-axis should be used, not
        // K-blocking. E.g. attn per-head QK^T shape.
        assert!(!should_use_k_blocking(864, 64, 864));
    }

    // -------------------------------------------------------------------------
    // Bias-fused GEMM tests
    // -------------------------------------------------------------------------

    /// Naive reference for `C = A@B + bias` (bias broadcast across rows).
    fn naive_nn_bias(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], bias: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = bias[j];
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    #[test]
    fn bias_matches_naive_on_small_multiples() {
        let (m, n, k) = (12, 32, 16);
        let a = pseudo_random(101, m * k);
        let b = pseudo_random(102, k * n);
        let bias = pseudo_random(103, n);
        let expected = naive_nn_bias(m, n, k, &a, &b, &bias);
        let mut got = vec![0.0f32; m * n];
        gemm_nn_bias_into(m, n, k, &a, &b, &mut got, &bias);
        assert!(
            max_abs_diff(&expected, &got) < 1e-3,
            "bias small multiples failed"
        );
    }

    #[test]
    fn bias_matches_naive_with_tails() {
        // 9 rows (MR tail of 3), 20 cols (NR tail of 4).
        let (m, n, k) = (9, 20, 17);
        let a = pseudo_random(104, m * k);
        let b = pseudo_random(105, k * n);
        let bias = pseudo_random(106, n);
        let expected = naive_nn_bias(m, n, k, &a, &b, &bias);
        let mut got = vec![0.0f32; m * n];
        gemm_nn_bias_into(m, n, k, &a, &b, &mut got, &bias);
        assert!(max_abs_diff(&expected, &got) < 1e-3, "bias tails failed");
    }

    #[test]
    fn bias_matches_naive_on_qkv_shape() {
        // Real DA3 QKV shape (N-axis parallel path).
        let (m, n, k) = (673, 2304, 768);
        let a = pseudo_random(107, m * k);
        let b = pseudo_random(108, k * n);
        let bias = pseudo_random(109, n);
        let expected = naive_nn_bias(m, n, k, &a, &b, &bias);
        let mut got = vec![0.0f32; m * n];
        gemm_nn_bias_into(m, n, k, &a, &b, &mut got, &bias);
        let rel =
            max_abs_diff(&expected, &got) / expected.iter().cloned().fold(0.0f32, f32::max).abs();
        assert!(rel < 1e-4, "bias qkv shape failed: rel={rel}");
    }

    #[test]
    fn bias_matches_naive_on_fc1_shape() {
        // FFN fc1 shape (K-blocked path: N=3072 > 3·K=2304).
        let (m, n, k) = (673, 3072, 768);
        let a = pseudo_random(110, m * k);
        let b = pseudo_random(111, k * n);
        let bias = pseudo_random(112, n);
        let expected = naive_nn_bias(m, n, k, &a, &b, &bias);
        let mut got = vec![0.0f32; m * n];
        gemm_nn_bias_into(m, n, k, &a, &b, &mut got, &bias);
        let rel =
            max_abs_diff(&expected, &got) / expected.iter().cloned().fold(0.0f32, f32::max).abs();
        assert!(rel < 1e-4, "bias fc1 shape failed: rel={rel}");
    }

    #[test]
    fn bias_matches_naive_on_fc2_shape() {
        // FFN fc2 shape (K-blocked path: N < threads·NC).
        let (m, n, k) = (673, 768, 3072);
        let a = pseudo_random(113, m * k);
        let b = pseudo_random(114, k * n);
        let bias = pseudo_random(115, n);
        let expected = naive_nn_bias(m, n, k, &a, &b, &bias);
        let mut got = vec![0.0f32; m * n];
        gemm_nn_bias_into(m, n, k, &a, &b, &mut got, &bias);
        let rel =
            max_abs_diff(&expected, &got) / expected.iter().cloned().fold(0.0f32, f32::max).abs();
        assert!(rel < 1e-4, "bias fc2 shape failed: rel={rel}");
    }

    #[test]
    fn bias_matches_naive_on_proj_shape() {
        // Attention proj shape (K-blocked path: N=K=768).
        let (m, n, k) = (673, 768, 768);
        let a = pseudo_random(116, m * k);
        let b = pseudo_random(117, k * n);
        let bias = pseudo_random(118, n);
        let expected = naive_nn_bias(m, n, k, &a, &b, &bias);
        let mut got = vec![0.0f32; m * n];
        gemm_nn_bias_into(m, n, k, &a, &b, &mut got, &bias);
        let rel =
            max_abs_diff(&expected, &got) / expected.iter().cloned().fold(0.0f32, f32::max).abs();
        assert!(rel < 1e-4, "bias proj shape failed: rel={rel}");
    }

    #[test]
    fn bias_equivalent_to_precopy_plus_accumulate() {
        // The bias-fused path must produce the same result as the old pattern:
        // pre-copy bias into C, then gemm_nn_into (C += A@B).
        let (m, n, k) = (673, 2304, 768);
        let a = pseudo_random(119, m * k);
        let b = pseudo_random(120, k * n);
        let bias = pseudo_random(121, n);

        // Old pattern.
        let mut old = vec![0.0f32; m * n];
        old.par_chunks_mut(n)
            .for_each(|row| row.copy_from_slice(&bias));
        gemm_nn_into(m, n, k, &a, &b, &mut old);

        // New fused path.
        let mut new = vec![0.0f32; m * n];
        gemm_nn_bias_into(m, n, k, &a, &b, &mut new, &bias);

        assert!(
            max_abs_diff(&old, &new) < 1e-4,
            "bias fuse disagrees with pre-copy+accumulate"
        );
    }
}
