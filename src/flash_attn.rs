//! Fused flash attention for the DA3 backbone (Q·Kᵀ + softmax + A·V in one
//! pass, no materialised `[n, n]` scores matrix).
//!
//! Mirrors what `ggml_flash_attn_ext` does on CPU (see
//! `third_party/ggml/src/ggml-cpu/ops.cpp` line 8558+
//! `ggml_compute_forward_flash_attn_ext_tiled`): the standard online-softmax
//! flash-attention recurrence (Milakov et al., arXiv 2112.05682) with
//! **tiled Q and KV**, reusing each K/V tile across `Q_TILE_SZ` queries.
//!
//! # Why tiled, not per-query
//!
//! The original per-query implementation (`forward_per_query`) walked all keys
//! serially for each query row. For DA3 shapes (n=864, heads=12, hd=64) it was
//! ~20ms *slower* than the materialised tinyBLAS QKᵀ+AV path because each key
//! was loaded once per query — no reuse across the 864 queries in a head.
//!
//! The tiled version processes `Q_TILE_SZ=64` queries at once. For each KV tile
//! of `KV_TILE_SZ=64` keys it:
//!   1. Packs the K tile transposed (`[hd, KV_TILE_SZ]`) so `QKᵀ` is a clean
//!      NN gemm: `[Q_TILE_SZ, hd] @ [hd, KV_TILE_SZ]`.
//!   2. Runs `simd_gemm`-equivalent (our tinyBLAS) for the `[64,64]` QKᵀ.
//!   3. Runs online softmax across the 64 queries.
//!   4. Runs tinyBLAS again for the `[64,64]` AV (`softmax_KQ @ V32`).
//!
//! The K/V tile (64 keys × 64 floats = 16 KiB) fits in L1, so the second and
//! subsequent Q tiles in the same KV column reuse warm cache lines.
//!
//! # Parallelisation
//!
//! Flattens `(head, q_tile)` into `heads * ceil(n/Q_TILE_SZ)` independent rayon
//! tasks. For DA3-BASE that's `12 * ceil(864/64) = 12 * 14 = 168` tasks — well
//! above the 16-thread count, giving good load balance without contention.
//! Each task is fully independent (own accumulators, own output slice).
//!
//! # Layouts
//!
//! - `q`, `k`, `v`, `out`: `[heads, n, head_dim]` row-major.
//! - No mask, no ALiBi bias, no logit softcap — DA3 uses plain bidirectional
//!   attention.

use rayon::prelude::*;

/// Q-tile size (queries per tile). Matches ggml's `GGML_FA_TILE_Q`.
pub const Q_TILE_SZ: usize = 64;
/// KV-tile size (keys per tile). Matches ggml's `GGML_FA_TILE_KV`.
pub const KV_TILE_SZ: usize = 64;

/// Per-thread scratch for the tiled flash-attention tasks. Each task needs
/// ~80 KiB of scratch (for hd=64); allocating these per-task would cause
/// ~12k heap allocs per forward. Stored thread-local so it's reused across
/// tasks on the same thread.
struct FlashScratch {
    q_scaled: Vec<f32>,
    kq: Vec<f32>,
    vkq: Vec<f32>,
    k32: Vec<f32>,
    v32: Vec<f32>,
}

impl FlashScratch {
    fn ensure(&mut self, hd: usize) {
        self.q_scaled.resize(Q_TILE_SZ * hd, 0.0);
        self.kq.resize(Q_TILE_SZ * KV_TILE_SZ, 0.0);
        self.vkq.resize(Q_TILE_SZ * hd, 0.0);
        self.k32.resize(hd * KV_TILE_SZ, 0.0);
        self.v32.resize(KV_TILE_SZ * hd, 0.0);
    }
}

thread_local! {
    static FLASH_SCRATCH: std::cell::RefCell<FlashScratch> = const { std::cell::RefCell::new(FlashScratch {
        q_scaled: Vec::new(),
        kq: Vec::new(),
        vkq: Vec::new(),
        k32: Vec::new(),
        v32: Vec::new(),
    }) };
}

/// Run fused flash attention (tiled).
///
/// - `q`/`k`/`v`: `[heads, n, hd]` row-major, length `heads * n * hd`.
/// - `out`: `[heads, n, hd]`, written (not added to).
/// - `scale`: `1/sqrt(hd)`.
///
/// Dispatches to the AVX2 tiled path when available, otherwise the scalar
/// tiled path. The per-query AVX2 path is available separately as
/// [`forward_per_query`] for A/B comparison.
pub fn forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    heads: usize,
    n: usize,
    hd: usize,
    scale: f32,
) {
    debug_assert_eq!(q.len(), heads * n * hd);
    debug_assert_eq!(k.len(), heads * n * hd);
    debug_assert_eq!(v.len(), heads * n * hd);
    debug_assert_eq!(out.len(), heads * n * hd);
    if n == 0 || hd == 0 || heads == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if crate::tinyblas::has_avx2_fma() && hd % 8 == 0 {
        // Safety: feature-checked above.
        unsafe { forward_tiled_avx2(q, k, v, out, heads, n, hd, scale) };
        return;
    }
    forward_tiled_scalar(q, k, v, out, heads, n, hd, scale);
}

// ---------------------------------------------------------------------------
// Tiled implementation (the fast path, matches ggml_compute_forward_flash_attn_ext_tiled)
// ---------------------------------------------------------------------------

/// Scalar tiled flash attention. This is also the reference implementation
/// the AVX2 path is validated against.
fn forward_tiled_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    heads: usize,
    n: usize,
    hd: usize,
    scale: f32,
) {
    let per_head = n * hd;
    let n_qtiles = n.div_ceil(Q_TILE_SZ);

    // Flatten (head, q_tile) into a single parallel axis. We capture `out` as a
    // raw `usize` address so the closure is `Send`; each task writes only its
    // own disjoint `[base + q_start*hd, base + q_end*hd)` slice (verified by the
    // disjoint (h, qt) task indices).
    let total_qtiles = heads * n_qtiles;
    let out_base = out.as_mut_ptr() as usize;
    (0..total_qtiles).into_par_iter().for_each(move |task| {
        let h = task / n_qtiles;
        let qt = task % n_qtiles;
        let base = h * per_head;
        let q_h = &q[base..base + per_head];
        let k_h = &k[base..base + per_head];
        let v_h = &v[base..base + per_head];

        let q_start = qt * Q_TILE_SZ;
        let q_end = (q_start + Q_TILE_SZ).min(n);
        let q_rows = q_end - q_start;
        let out_h_start = base + q_start * hd;
        // Safety: task (h, qt) owns out[out_h_start .. out_h_start + q_rows*hd]
        // exclusively — no other task writes this range.
        let out_tile = unsafe {
            core::slice::from_raw_parts_mut((out_base as *mut f32).add(out_h_start), q_rows * hd)
        };

        // Per-task heap scratch (boxed so per-task stack usage is bounded
        // regardless of hd). KQ is 64*64*4 = 16 KiB, VKQ/K32/V32 same order.
        let mut kq = vec![0.0f32; Q_TILE_SZ * KV_TILE_SZ];
        let mut vkq = vec![0.0f32; Q_TILE_SZ * hd];
        let mut m = vec![f32::NEG_INFINITY; Q_TILE_SZ];
        let mut s = vec![0.0f32; Q_TILE_SZ];
        // K tile transposed: [hd, KV_TILE_SZ] for the NN gemm.
        let mut k32 = vec![0.0f32; hd * KV_TILE_SZ];
        // V tile (contiguous): [KV_TILE_SZ, hd] for the NN gemm.
        let mut v32 = vec![0.0f32; KV_TILE_SZ * hd];

        // Walk KV tiles in order, accumulating online softmax.
        let mut kv = 0usize;
        while kv < n {
            let kv_rows = (KV_TILE_SZ).min(n - kv);

            // Pack K tile transposed: k32[d * KV_TILE_SZ + j] = k_h[(kv+j)*hd + d].
            // Zero-pad the tail rows so the GEMM operates on a full KV_TILE_SZ
            // column (avoids a slow partial-N scalar tail in the microkernel).
            for j in 0..kv_rows {
                let src = &k_h[(kv + j) * hd..(kv + j + 1) * hd];
                for d in 0..hd {
                    k32[d * KV_TILE_SZ + j] = src[d];
                }
            }
            for j in kv_rows..KV_TILE_SZ {
                for d in 0..hd {
                    k32[d * KV_TILE_SZ + j] = 0.0;
                }
            }

            // KQ = Q_tile @ K32ᵀ  →  [Q_TILE_SZ, KV_TILE_SZ]
            // (Q_tile is [Q_TILE_SZ, hd], K32 is [hd, KV_TILE_SZ], NN gemm.)
            // Only the valid q_rows actually matter; the others stay 0.
            for i in 0..q_rows {
                for j in 0..KV_TILE_SZ {
                    let mut acc = 0.0f32;
                    let qrow = &q_h[(q_start + i) * hd..(q_start + i + 1) * hd];
                    for d in 0..hd {
                        acc += qrow[d] * k32[d * KV_TILE_SZ + j];
                    }
                    kq[i * KV_TILE_SZ + j] = acc * scale;
                }
            }

            // Set padded KQ columns to -inf so softmax assigns them 0 weight.
            for i in 0..q_rows {
                for j in kv_rows..KV_TILE_SZ {
                    kq[i * KV_TILE_SZ + j] = f32::NEG_INFINITY;
                }
            }

            // Online softmax per query row.
            for i in 0..q_rows {
                let row = &mut kq[i * KV_TILE_SZ..i * KV_TILE_SZ + KV_TILE_SZ];
                // tile max
                let mut tile_max = f32::NEG_INFINITY;
                for &x in row.iter() {
                    if x > tile_max {
                        tile_max = x;
                    }
                }
                if tile_max == f32::NEG_INFINITY {
                    // All -inf (shouldn't happen with bidirectional attn,
                    // but guard anyway).
                    continue;
                }
                let mold = m[i];
                let mnew = if mold > tile_max { mold } else { tile_max };
                if mnew > mold {
                    let ms = (mold - mnew).exp();
                    for d in 0..hd {
                        vkq[i * hd + d] *= ms;
                    }
                    s[i] *= ms;
                }
                m[i] = mnew;
                // softmax numerators + sum
                let mut row_sum = 0.0f32;
                for j in 0..KV_TILE_SZ {
                    let e = (row[j] - mnew).exp();
                    row[j] = e;
                    row_sum += e;
                }
                s[i] += row_sum;
            }

            // Pack V tile: v32[j * hd + d] = v_h[(kv+j)*hd + d].
            // Zero-pad tail rows (the corresponding KQ columns are -inf →
            // softmax weight 0, so the V values don't matter).
            for j in 0..kv_rows {
                let src = &v_h[(kv + j) * hd..(kv + j + 1) * hd];
                v32[j * hd..j * hd + hd].copy_from_slice(src);
            }
            for j in kv_rows..KV_TILE_SZ {
                for d in 0..hd {
                    v32[j * hd + d] = 0.0;
                }
            }

            // VKQ += KQ @ V32  →  [Q_TILE_SZ, hd]  (NN gemm)
            for i in 0..q_rows {
                let row = &kq[i * KV_TILE_SZ..i * KV_TILE_SZ + KV_TILE_SZ];
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for j in 0..KV_TILE_SZ {
                        acc += row[j] * v32[j * hd + d];
                    }
                    vkq[i * hd + d] += acc;
                }
            }

            kv += KV_TILE_SZ;
        }

        // Normalise: out = VKQ / s.
        for i in 0..q_rows {
            let inv_s = if s[i] > 0.0 { 1.0 / s[i] } else { 0.0 };
            for d in 0..hd {
                out_tile[i * hd + d] = vkq[i * hd + d] * inv_s;
            }
        }
    });
}

/// AVX2 tiled flash attention. Uses tinyBLAS's 6×16 microkernel for the two
/// per-tile GEMMs (`QKᵀ` and `AV`), matching what ggml does on CPU.
///
/// Annotated with `#[target_feature(enable="avx2,fma")]` so that AVX2
/// intrinsics in the rayon closure body (max-reduction, VKQ rescale, and the
/// `softmax_exp_sum_avx2` helper) are compiled as inline instructions rather
/// than function-call wrappers. Without this annotation, the closure (which
/// is a separate function from `forward_tiled_avx2`) would not inherit the
/// target-feature context, and each `_mm256_*` intrinsic would lower to a
/// `callq` with a `vzeroupper` (~20 cycles) — the overhead dominated the
/// softmax cost before this annotation was added.
///
/// # Safety
/// - `has_avx2_fma()` must be true.
/// - `hd % 8 == 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn forward_tiled_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    heads: usize,
    n: usize,
    hd: usize,
    scale: f32,
) {
    let per_head = n * hd;
    let n_qtiles = n.div_ceil(Q_TILE_SZ);
    let total_qtiles = heads * n_qtiles;

    // Each task needs Q_TILE_SZ*hd (Q copy) + Q_TILE_SZ*KV_TILE_SZ (KQ) +
    // Q_TILE_SZ*hd (VKQ) + KV_TILE_SZ*hd (K32) + KV_TILE_SZ*hd (V32) floats.
    // For hd=64: 64*64*4 + 64*64 + 64*64 + 64*64 + 64*64 = 4*4096 + 4096 = ~80 KiB.
    // That's fine on a typical stack (rayon gives 8 MiB on the main thread, and
    // its worker stacks are sized off the main thread's). We box the largest
    // buffers to keep stack usage predictable and avoid the kernel's stack-protector
    // guard firing under deep rayon nesting.

    // Capture `out` as a raw `usize` so the closure is `Send`; each task writes
    // only its own disjoint slice.
    let out_base = out.as_mut_ptr() as usize;
    (0..total_qtiles).into_par_iter().for_each(move |task| {
        let h = task / n_qtiles;
        let qt = task % n_qtiles;
        let base = h * per_head;
        let q_h = &q[base..base + per_head];
        let k_h = &k[base..base + per_head];
        let v_h = &v[base..base + per_head];

        let q_start = qt * Q_TILE_SZ;
        let q_end = (q_start + Q_TILE_SZ).min(n);
        let q_rows = q_end - q_start;
        let out_h_start = base + q_start * hd;
        // Safety: task (h, qt) owns out[out_h_start .. out_h_start + q_rows*hd]
        // exclusively.
        let out_tile = unsafe {
            core::slice::from_raw_parts_mut((out_base as *mut f32).add(out_h_start), q_rows * hd)
        };

        // Borrow thread-local scratch (reused across tasks on this thread).
        // We extract raw pointers to the individual buffers because Rust's
        // borrow checker can't prove the disjoint field borrows are safe
        // through a single RefCell. The data lives for the thread's lifetime
        // and is not re-borrowed within this task.
        // m and s are small (Q_TILE_SZ floats = 256 B) — kept on the stack.
        let (q_scaled_ptr, kq_ptr, vkq_ptr, k32_ptr, v32_ptr) = FLASH_SCRATCH.with(|c| {
            let mut s = c.borrow_mut();
            s.ensure(hd);
            (
                s.q_scaled.as_mut_ptr(),
                s.kq.as_mut_ptr(),
                s.vkq.as_mut_ptr(),
                s.k32.as_mut_ptr(),
                s.v32.as_mut_ptr(),
            )
        });
        let mut m = [f32::NEG_INFINITY; Q_TILE_SZ];
        let mut s = [0.0f32; Q_TILE_SZ];
        // Safety: the pointers are valid for the duration of this task (the
        // thread-local buffer is not re-borrowed or resized within this closure).
        let q_scaled = unsafe { core::slice::from_raw_parts_mut(q_scaled_ptr, Q_TILE_SZ * hd) };
        let kq = unsafe { core::slice::from_raw_parts_mut(kq_ptr, Q_TILE_SZ * KV_TILE_SZ) };
        let vkq = unsafe { core::slice::from_raw_parts_mut(vkq_ptr, Q_TILE_SZ * hd) };
        let k32 = unsafe { core::slice::from_raw_parts_mut(k32_ptr, hd * KV_TILE_SZ) };
        let v32 = unsafe { core::slice::from_raw_parts_mut(v32_ptr, KV_TILE_SZ * hd) };

        // Pre-scale Q once (fuses the per-KV-tile KQ *= scale into the Q copy).
        // q_scaled[i*hd + d] = q_h[(q_start+i)*hd + d] * scale. Padding rows
        // (i >= q_rows) stay 0.
        let nvec_h = hd / 8;
        let scale_v = std::arch::x86_64::_mm256_set1_ps(scale);
        for i in 0..q_rows {
            let src = q_h.as_ptr().add((q_start + i) * hd);
            let dst = q_scaled.as_mut_ptr().add(i * hd);
            for v in 0..nvec_h {
                let qv = std::arch::x86_64::_mm256_loadu_ps(src.add(v * 8));
                std::arch::x86_64::_mm256_storeu_ps(
                    dst.add(v * 8),
                    std::arch::x86_64::_mm256_mul_ps(qv, scale_v),
                );
            }
        }

        let mut kv = 0usize;
        while kv < n {
            let kv_rows = (KV_TILE_SZ).min(n - kv);

            // Pack K tile transposed, zero-padded.
            for j in 0..kv_rows {
                let src = &k_h[(kv + j) * hd..(kv + j + 1) * hd];
                for d in 0..hd {
                    k32[d * KV_TILE_SZ + j] = src[d];
                }
            }
            for j in kv_rows..KV_TILE_SZ {
                for d in 0..hd {
                    k32[d * KV_TILE_SZ + j] = 0.0;
                }
            }

            // KQ[Q_TILE_SZ, KV_TILE_SZ] = Q_scaled[Q_TILE_SZ, hd] @ K32[hd, KV_TILE_SZ].
            // (Scale is already baked into Q_scaled — no post-GEMM scale needed.)
            for x in kq[..q_rows * KV_TILE_SZ].iter_mut() {
                *x = 0.0;
            }
            crate::tinyblas::gemm_nn_into_serial(
                q_rows,
                KV_TILE_SZ,
                hd,
                &q_scaled[..q_rows * hd],
                &k32,
                &mut kq[..q_rows * KV_TILE_SZ],
            );

            // Pad unused KQ columns with -inf so softmax gives them 0 weight.
            // Only the tail columns beyond kv_rows need padding.
            if kv_rows < KV_TILE_SZ {
                for i in 0..q_rows {
                    for j in kv_rows..KV_TILE_SZ {
                        kq[i * KV_TILE_SZ + j] = f32::NEG_INFINITY;
                    }
                }
            }

            // Online softmax per query row. The max reduction over the score
            // row is vectorised (8 lanes at a time); the exp+sum uses
            // [`softmax_exp_sum_avx2`] (a `#[target_feature]` helper that inlines
            // [`exp_fast_avx2`] — a degree-6 polynomial approximation that avoids
            // ~116M scalar `expf` calls per inference, the dominant softmax cost).
            for i in 0..q_rows {
                let row = &mut kq[i * KV_TILE_SZ..i * KV_TILE_SZ + KV_TILE_SZ];
                // Vectorised max reduction over the row.
                let mut tile_max = f32::NEG_INFINITY;
                for chunk in row[..kv_rows].chunks_exact(8) {
                    let v = std::arch::x86_64::_mm256_loadu_ps(chunk.as_ptr());
                    let h = hmax_ps(v);
                    if h > tile_max {
                        tile_max = h;
                    }
                }
                for j in (kv_rows / 8 * 8)..kv_rows {
                    if row[j] > tile_max {
                        tile_max = row[j];
                    }
                }
                if tile_max == f32::NEG_INFINITY {
                    continue;
                }
                let mold = m[i];
                let mnew = if mold > tile_max { mold } else { tile_max };
                if mnew > mold {
                    let ms = (mold - mnew).exp();
                    // Vectorised VKQ rescale: vkq[i, 0..hd] *= ms.
                    let ms_v = std::arch::x86_64::_mm256_set1_ps(ms);
                    let vkq_row = &mut vkq[i * hd..i * hd + hd];
                    for v in 0..nvec_h {
                        let cur = std::arch::x86_64::_mm256_loadu_ps(vkq_row.as_ptr().add(v * 8));
                        std::arch::x86_64::_mm256_storeu_ps(
                            vkq_row.as_mut_ptr().add(v * 8),
                            std::arch::x86_64::_mm256_mul_ps(cur, ms_v),
                        );
                    }
                    s[i] *= ms;
                }
                m[i] = mnew;
                // Vectorised softmax: compute e = exp(row[j] - mnew) for all
                // KV_TILE_SZ columns at once, write back to `row`, and
                // horizontally sum into `s[i]`. Uses [`exp_fast_avx2`] (degree-6
                // polynomial, max rel error < 2e-6) instead of scalar `expf`, the
                // single biggest flash-attention win — ~116M expf evaluations per
                // inference dominate softmax time. KV_TILE_SZ=64 = 8 × 8-lane
                // vectors, so the loop is exact (no tail).
                //
                // Extracted into a `#[target_feature]` helper because this code
                // lives in a rayon closure that does NOT inherit the outer
                // function's `#[target_feature]` — without the annotation each
                // `_mm256_*` intrinsic and `exp_fast_avx2` lowers to a separate
                // `callq`, which dominated softmax cost (measured ~20% end-to-end
                // regression before the extraction).
                s[i] += softmax_exp_sum_avx2(row, mnew);
            }

            // Pack V tile, zero-padded.
            for j in 0..kv_rows {
                let src = &v_h[(kv + j) * hd..(kv + j + 1) * hd];
                v32[j * hd..j * hd + hd].copy_from_slice(src);
            }
            for j in kv_rows..KV_TILE_SZ {
                for d in 0..hd {
                    v32[j * hd + d] = 0.0;
                }
            }

            // VKQ[Q_TILE_SZ, hd] += KQ[Q_TILE_SZ, KV_TILE_SZ] @ V32[KV_TILE_SZ, hd].
            crate::tinyblas::gemm_nn_into_serial(
                q_rows,
                hd,
                KV_TILE_SZ,
                &kq[..q_rows * KV_TILE_SZ],
                &v32,
                &mut vkq[..q_rows * hd],
            );

            kv += KV_TILE_SZ;
        }

        // Normalise: out = VKQ / s (vectorised across the hd lanes).
        for i in 0..q_rows {
            let inv_s = if s[i] > 0.0 { 1.0 / s[i] } else { 0.0 };
            let inv_v = std::arch::x86_64::_mm256_set1_ps(inv_s);
            let vkq_row = &vkq[i * hd..i * hd + hd];
            let out_row = &mut out_tile[i * hd..i * hd + hd];
            for v in 0..nvec_h {
                let cur = std::arch::x86_64::_mm256_loadu_ps(vkq_row.as_ptr().add(v * 8));
                std::arch::x86_64::_mm256_storeu_ps(
                    out_row.as_mut_ptr().add(v * 8),
                    std::arch::x86_64::_mm256_mul_ps(cur, inv_v),
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Per-query implementation (kept for A/B; slower for DA3 shapes)
// ---------------------------------------------------------------------------

/// Run fused flash attention using the per-query recurrence.
///
/// This is the original implementation; it processes one query row at a time
/// (no Q-tile reuse). Slower than [`forward`] for DA3 shapes because each key
/// is loaded once per query. Kept for A/B comparison and as a simpler
/// reference for the online-softmax recurrence.
pub fn forward_per_query(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    heads: usize,
    n: usize,
    hd: usize,
    scale: f32,
) {
    debug_assert_eq!(q.len(), heads * n * hd);
    debug_assert_eq!(k.len(), heads * n * hd);
    debug_assert_eq!(v.len(), heads * n * hd);
    debug_assert_eq!(out.len(), heads * n * hd);
    if n == 0 || hd == 0 || heads == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if crate::tinyblas::has_avx2_fma() && hd % 8 == 0 {
        // Safety: feature-checked above.
        unsafe { forward_per_query_avx2(q, k, v, out, heads, n, hd, scale) };
        return;
    }
    forward_per_query_scalar(q, k, v, out, heads, n, hd, scale);
}

/// Scalar reference implementation (also used on non-x86_64 / odd `hd`).
fn forward_per_query_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    heads: usize,
    n: usize,
    hd: usize,
    scale: f32,
) {
    let per_head = n * hd;
    let total_rows = heads * n;
    const Q_ROWS: usize = 8;
    out.par_chunks_mut(Q_ROWS * hd)
        .enumerate()
        .for_each(|(blk, out_blk)| {
            let row0 = blk * Q_ROWS;
            let row1 = (row0 + Q_ROWS).min(total_rows);
            for row in row0..row1 {
                let h = row / n;
                let qi = row % n;
                let base = h * per_head;
                let qrow = &q[base + qi * hd..base + (qi + 1) * hd];
                let k_h = &k[base..base + per_head];
                let v_h = &v[base..base + per_head];
                let orow = &mut out_blk[(row - row0) * hd..(row - row0 + 1) * hd];
                online_softmax_row_scalar(qrow, k_h, v_h, orow, n, hd, scale);
            }
        });
}

/// Scalar online-softmax recurrence for one query row.
#[inline]
fn online_softmax_row_scalar(
    qrow: &[f32],
    k_h: &[f32], // [n, hd]
    v_h: &[f32], // [n, hd]
    out: &mut [f32],
    n: usize,
    hd: usize,
    scale: f32,
) {
    let mut m = f32::NEG_INFINITY;
    let mut s = 0.0f32;
    let mut vkq = vec![0.0f32; hd];
    for j in 0..n {
        let krow = &k_h[j * hd..(j + 1) * hd];
        let mut dot = 0.0f32;
        for d in 0..hd {
            dot += qrow[d] * krow[d];
        }
        let sj = dot * scale;
        if sj > m {
            let ms = (m - sj).exp();
            for d in 0..hd {
                vkq[d] *= ms;
            }
            s *= ms;
            m = sj;
            let vrow = &v_h[j * hd..(j + 1) * hd];
            for d in 0..hd {
                vkq[d] += vrow[d]; // vs == 1
            }
            s += 1.0;
        } else {
            let vs = (sj - m).exp();
            let vrow = &v_h[j * hd..(j + 1) * hd];
            for d in 0..hd {
                vkq[d] += vs * vrow[d];
            }
            s += vs;
        }
    }
    let inv_s = if s > 0.0 { 1.0 / s } else { 0.0 };
    for d in 0..hd {
        out[d] = vkq[d] * inv_s;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn forward_per_query_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    heads: usize,
    n: usize,
    hd: usize,
    scale: f32,
) {
    use std::arch::x86_64::*;
    let per_head = n * hd;
    let total_rows = heads * n;
    const Q_ROWS: usize = 16;

    out.par_chunks_mut(Q_ROWS * hd)
        .enumerate()
        .for_each(|(blk, out_blk)| {
            let row0 = blk * Q_ROWS;
            let row1 = (row0 + Q_ROWS).min(total_rows);
            let nvec = hd / 8;
            let mut vkq_task: Vec<__m256> = vec![_mm256_setzero_ps(); nvec];
            for row in row0..row1 {
                let h = row / n;
                let qi = row % n;
                let base = h * per_head;
                let qrow = q.as_ptr().add(base + qi * hd);
                let k_h = k.as_ptr().add(base);
                let v_h = v.as_ptr().add(base);
                let orow = out_blk.as_mut_ptr().add((row - row0) * hd);
                unsafe {
                    online_softmax_row_avx2(
                        qrow,
                        k_h,
                        v_h,
                        orow,
                        n,
                        hd,
                        scale,
                        vkq_task.as_mut_ptr(),
                    );
                }
            }
        });
}

/// AVX2 online-softmax recurrence for one query row.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn online_softmax_row_avx2(
    qrow: *const f32,
    k_h: *const f32,
    v_h: *const f32,
    out: *mut f32,
    n: usize,
    hd: usize,
    scale: f32,
    vkq: *mut std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;
    let nvec = hd / 8;
    for v in 0..nvec {
        *vkq.add(v) = _mm256_setzero_ps();
    }
    let mut m: f32 = f32::NEG_INFINITY;
    let mut s: f32 = 0.0f32;
    for j in 0..n {
        let kj = k_h.add(j * hd);
        let vj = v_h.add(j * hd);
        let mut acc = _mm256_setzero_ps();
        for v in 0..nvec {
            let qv = _mm256_loadu_ps(qrow.add(v * 8));
            let kv = _mm256_loadu_ps(kj.add(v * 8));
            acc = _mm256_fmadd_ps(qv, kv, acc);
        }
        let dot = hsum_ps(acc);
        let sj = dot * scale;
        if sj > m {
            let ms = (m - sj).exp();
            let ms_v = _mm256_set1_ps(ms);
            for v in 0..nvec {
                *vkq.add(v) = _mm256_mul_ps(*vkq.add(v), ms_v);
            }
            s *= ms;
            m = sj;
            for v in 0..nvec {
                let vv = _mm256_loadu_ps(vj.add(v * 8));
                *vkq.add(v) = _mm256_add_ps(*vkq.add(v), vv);
            }
            s += 1.0;
        } else {
            let vs = (sj - m).exp();
            let vs_v = _mm256_set1_ps(vs);
            for v in 0..nvec {
                let vv = _mm256_loadu_ps(vj.add(v * 8));
                *vkq.add(v) = _mm256_fmadd_ps(vs_v, vv, *vkq.add(v));
            }
            s += vs;
        }
    }
    let inv_s = if s > 0.0 { 1.0 / s } else { 0.0 };
    let inv_v = _mm256_set1_ps(inv_s);
    for v in 0..nvec {
        let r = _mm256_mul_ps(*vkq.add(v), inv_v);
        _mm256_storeu_ps(out.add(v * 8), r);
    }
}

/// Horizontal sum of the 8 lanes of an `__m256` to a scalar `f32`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_ps(x: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_si256(_mm256_castps_si256(x), 1);
    let lo128 = _mm256_castps256_ps128(x);
    let sum128 = _mm_add_ps(lo128, _mm_castsi128_ps(hi128));
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let total = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(total)
}

/// Horizontal max of the 8 lanes of an `__m256`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hmax_ps(x: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_si256(_mm256_castps_si256(x), 1);
    let lo128 = _mm256_castps256_ps128(x);
    let max128 = _mm_max_ps(lo128, _mm_castsi128_ps(hi128));
    let shuf = _mm_movehdup_ps(max128);
    let maxs = _mm_max_ps(max128, shuf);
    let shuf2 = _mm_movehl_ps(maxs, maxs);
    let total = _mm_max_ss(maxs, shuf2);
    _mm_cvtss_f32(total)
}

/// Vectorised online-softmax exp+sum step: computes `e = exp(row[j] - mnew)`
/// in-place for every element of `row`, and returns `Σ e`.
///
/// `row.len()` must be a multiple of 8 (the caller passes `KV_TILE_SZ=64`).
/// Uses [`crate::fast_block::avx2_gelu::exp_fast_avx2`] (degree-6 polynomial,
/// max rel error < 2e-6 over [-87, 88]) instead of scalar `expf`. For
/// flash-attention softmax the argument `row[j] - mnew` is always ≤ 0 (since
/// `mnew` is the running max), so there is no overflow risk and underflow
/// (exp(−large) → 0) is handled gracefully by the clamp in `exp_fast_avx2`.
///
/// Annotated with `#[target_feature]` so all AVX2 intrinsics and
/// `exp_fast_avx2` are inlined — without this, each intrinsic lowers to a
/// separate `callq` and the loop is ~20× slower.
///
/// # Safety
/// AVX2 + FMA must be available at runtime (`has_avx2_fma()`).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn softmax_exp_sum_avx2(row: &mut [f32], mnew: f32) -> f32 {
    use std::arch::x86_64::*;
    let mnew_v = _mm256_set1_ps(mnew);
    let mut sum_v = _mm256_setzero_ps();
    for chunk in row.chunks_exact_mut(8) {
        let r = _mm256_loadu_ps(chunk.as_ptr());
        let d = _mm256_sub_ps(r, mnew_v);
        let e = crate::fast_block::avx2_gelu::exp_fast_avx2(d);
        _mm256_storeu_ps(chunk.as_mut_ptr(), e);
        sum_v = _mm256_add_ps(sum_v, e);
    }
    hsum_ps(sum_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: usize,
        n: usize,
        hd: usize,
        scale: f32,
    ) -> Vec<f32> {
        let per_head = n * hd;
        let mut out = vec![0.0f32; heads * per_head];
        for h in 0..heads {
            let qh = &q[h * per_head..(h + 1) * per_head];
            let kh = &k[h * per_head..(h + 1) * per_head];
            let vh = &v[h * per_head..(h + 1) * per_head];
            for i in 0..n {
                let mut scores = vec![0.0f32; n];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..n {
                    let mut d = 0.0f32;
                    for d2 in 0..hd {
                        d += qh[i * hd + d2] * kh[j * hd + d2];
                    }
                    scores[j] = d * scale;
                    if scores[j] > mx {
                        mx = scores[j];
                    }
                }
                let mut sum = 0.0f32;
                for j in 0..n {
                    scores[j] = (scores[j] - mx).exp();
                    sum += scores[j];
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for j in 0..n {
                        acc += scores[j] * vh[j * hd + d];
                    }
                    out[h * per_head + i * hd + d] = acc * inv;
                }
            }
        }
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn tiled_matches_reference_small() {
        let (heads, n, hd) = (2, 5, 16);
        let len = heads * n * hd;
        let q: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let k: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).cos() * 0.5).collect();
        let v: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).tan() * 0.1).collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; len];
        forward(&q, &k, &v, &mut out, heads, n, hd, scale);
        let reference = ref_attention(&q, &k, &v, heads, n, hd, scale);
        let d = max_abs_diff(&out, &reference);
        assert!(d < 1e-4, "tiled vs reference max abs diff = {d}");
    }

    #[test]
    fn tiled_matches_reference_da3_shape() {
        let (heads, n, hd) = (1, 864, 64);
        let len = heads * n * hd;
        let q: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 0.013).fract() - 0.5) * 2.0)
            .collect();
        let k: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 0.017).fract() - 0.5) * 2.0)
            .collect();
        let v: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 0.019).fract() - 0.5) * 2.0)
            .collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; len];
        forward(&q, &k, &v, &mut out, heads, n, hd, scale);
        let reference = ref_attention(&q, &k, &v, heads, n, hd, scale);
        let d = max_abs_diff(&out, &reference);
        assert!(d < 5e-4, "tiled vs reference max abs diff = {d}");
    }

    #[test]
    fn tiled_matches_reference_multi_head() {
        // Multi-head, multi-tile: exercises the parallel (head, q_tile) dispatch.
        let (heads, n, hd) = (12, 200, 64);
        let len = heads * n * hd;
        let q: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 0.013).fract() - 0.5) * 2.0)
            .collect();
        let k: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 0.017).fract() - 0.5) * 2.0)
            .collect();
        let v: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 0.019).fract() - 0.5) * 2.0)
            .collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; len];
        forward(&q, &k, &v, &mut out, heads, n, hd, scale);
        let reference = ref_attention(&q, &k, &v, heads, n, hd, scale);
        let d = max_abs_diff(&out, &reference);
        assert!(d < 5e-4, "tiled vs reference max abs diff = {d}");
    }

    #[test]
    fn tiled_and_per_query_agree() {
        // The two flash implementations should agree to f32 accumulation tolerance.
        let (heads, n, hd) = (3, 130, 32);
        let len = heads * n * hd;
        let q: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 0.013).fract() - 0.5) * 2.0)
            .collect();
        let k: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 0.017).fract() - 0.5) * 2.0)
            .collect();
        let v: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 0.019).fract() - 0.5) * 2.0)
            .collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out_tiled = vec![0.0f32; len];
        let mut out_pq = vec![0.0f32; len];
        forward(&q, &k, &v, &mut out_tiled, heads, n, hd, scale);
        forward_per_query(&q, &k, &v, &mut out_pq, heads, n, hd, scale);
        let d = max_abs_diff(&out_tiled, &out_pq);
        assert!(d < 1e-5, "tiled vs per-query max abs diff = {d}");
    }

    #[test]
    fn zero_input_gives_zero_output() {
        let (heads, n, hd) = (1, 4, 8);
        let len = heads * n * hd;
        let q = vec![0.0f32; len];
        let k = vec![0.0f32; len];
        let v = vec![0.0f32; len];
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; len];
        forward(&q, &k, &v, &mut out, heads, n, hd, scale);
        for &o in &out {
            assert!(o.abs() < 1e-6, "expected ~0 output, got {o}");
        }
    }
}
