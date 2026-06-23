//! Fast convolutions for the DPT head, bypassing candle's per-op overhead.
//!
//! Implements two paths:
//! - **Winograd F(2x2,3x3)** for 3×3 stride-1 pad-1 convs, ported 1:1 from
//!   `src/winograd.cpp` (the C++ engine's AVX-512 implementation), adapted here
//!   for AVX2/FMA (8 floats per vector instead of 16). Reduces multiplications
//!   by 2.25× vs direct convolution (16 mults per 4 outputs vs 9 per 1).
//! - **tinyBLAS GEMM** for 1×1 convs (which are pure matrix multiplies).
//!
//! # Env var
//!
//! Enabled by `DA_FAST_HEAD=1` (see [`fast_head_enabled`]). When set, the DPT
//! head's [`conv_fwd`](crate::dpt_head) dispatches stride-1 convs to these
//! kernels instead of candle's `Conv2d::forward`.

use crate::tinyblas;
use rayon::prelude::*;

// --- Winograd F(2x2,3x3) constants ----------------------------------------
//
// Input tile:  IT × IT = 4 × 4  (covers a 2×2 output tile with a 3×3 kernel)
// Output tile: OT × OT = 2 × 2
// Positions:   P = IT * IT = 16 (winograd-domain matrices per filter/input)
// Block:       TB = 8 tiles per winograd-domain GEMM (amortizes U-row loads)
//
// U layout: [P, IC, OC]   (transformed filters, OC innermost for vectorization)
// V layout: [P, IC, TB]   (transformed input tiles, TB innermost for strided
//                          broadcast in the GEMM microkernel)
// M layout: [P, TB, OC]   (winograd-domain products)
const IT: usize = 4;
const OT: usize = 2;
const P: usize = 16;
/// Tiles per block for the Winograd conv. Larger TB amortises per-block
/// overhead (input/output transforms, buffer setup) over more tiles, but each
/// tile adds one accumulator to the wino GEMM microkernel — TB=8 with the
/// 16-OC fast path fills all 16 ymm registers and forces the compiler to spill.
/// TB=4 with 16-OC fits in 8 accumulators (no spill) but doubles the block
/// count.
const TB: usize = 4;

/// Pixel-chunk granularity for the conv1x1 single-output-channel fast path.
/// Each parallel task processes this many output pixels. Chosen large enough
/// to keep rayon dispatch overhead negligible, small enough for good load
/// balance across 32 threads (hw/PX_CHUNK tasks).
const PX_CHUNK: usize = 1024;

/// Wrapper around `*mut f32` that implements `Send + Sync`, for use as a
/// captured variable in rayon closures. The safety obligation is that writes
/// through copies of this pointer target disjoint memory across parallel
/// tasks (which is true in [`conv3x3_pad1`]'s tile-block dispatch).
#[derive(Copy, Clone)]
struct SyncPtr(*mut f32);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

impl SyncPtr {
    /// Write `val` to `self[offset]`. Caller must ensure `offset` is in bounds
    /// and that no other thread writes to the same offset.
    #[inline]
    unsafe fn write(self, offset: usize, val: f32) {
        std::ptr::write(self.0.add(offset), val);
    }
}

// Thread-local scratch buffers for [`conv3x3_pad1_wino`], reused across
// tile-blocks on the same thread to avoid per-block allocation churn. Grown
// lazily; never shrunk (so a transient large conv doesn't cause reallocation
// churn later).
//
// VBLK/MBLK hold the transformed input (V) and winograd-domain product (M)
// matrices for the whole block (sized P*IC*TB and P*TB*OC respectively).
//
// PATCHES holds the per-tile patch buffers (dpatch/vpatch/mpatch/ypatch)
// used inside the input/output transforms. Grouped in a single struct so
// they can be borrowed together from one thread_local entry. Hoisting them
// here avoids 4 Vec allocations per block (thousands of blocks per conv).
thread_local! {
    static VBLK: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static MBLK: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static PATCHES: std::cell::RefCell<WinoPatches> = const { std::cell::RefCell::new(WinoPatches::new()) };
}

/// Per-tile scratch buffers, grouped in a single struct so they can be borrowed
/// together from one thread_local entry (avoids nested `with` calls).
struct WinoPatches {
    dpatch: Vec<f32>, // [IT*IT] input tile
    vpatch: Vec<f32>, // [P] transformed input
    mpatch: Vec<f32>, // [P] winograd-domain product
    ypatch: Vec<f32>, // [OT*OT] output tile
}

impl WinoPatches {
    const fn new() -> Self {
        Self {
            dpatch: Vec::new(),
            vpatch: Vec::new(),
            mpatch: Vec::new(),
            ypatch: Vec::new(),
        }
    }
}

/// Whether the candle-free DPT-head conv path is enabled (`DA_FAST_HEAD=1`).
/// Read once and cached.
pub fn fast_head_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("DA_FAST_HEAD").as_deref(),
            Ok("1") | Ok("on") | Ok("true") | Ok("ON") | Ok("TRUE") | Ok("True")
        )
    })
}

// --- Winograd F(2,3) transforms (exact ports from winograd.cpp F2Policy) ---

/// Filter transform: U = G g Gᵀ, 3×3 → 4×4.
/// `g` is the 3×3 filter (row-major, ky outer / kx inner).
/// `u` receives the 16 transformed values (row-major 4×4).
fn wino_filt_f2(g: &[f32], u: &mut [f32]) {
    debug_assert_eq!(g.len(), 9);
    debug_assert_eq!(u.len(), P);
    let mut gg = [[0.0f32; 3]; 4];
    // Apply G on the 3 columns of g: g[ky*3 + j] for ky in 0..3.
    for j in 0..3 {
        let c0 = g[j];
        let c1 = g[3 + j];
        let c2 = g[6 + j];
        gg[0][j] = c0;
        gg[1][j] = 0.5 * (c0 + c1 + c2);
        gg[2][j] = 0.5 * (c0 - c1 + c2);
        gg[3][j] = c2;
    }
    // Apply G on the 3 columns of each Gg row.
    for i in 0..4 {
        let c0 = gg[i][0];
        let c1 = gg[i][1];
        let c2 = gg[i][2];
        u[i * 4] = c0;
        u[i * 4 + 1] = 0.5 * (c0 + c1 + c2);
        u[i * 4 + 2] = 0.5 * (c0 - c1 + c2);
        u[i * 4 + 3] = c2;
    }
}

/// Input transform: V = Bᵀ d B, 4×4 → 4×4.
/// `d` is the 4×4 input tile (row-major).
/// `v` receives the 16 transformed values.
fn wino_inp_f2(d: &[f32], v: &mut [f32]) {
    debug_assert_eq!(d.len(), P);
    debug_assert_eq!(v.len(), P);
    let mut m = [0.0f32; P];
    // Bᵀ on columns: d[row*4 + j] for row in 0..4.
    for j in 0..4 {
        let r0 = d[j];
        let r1 = d[4 + j];
        let r2 = d[8 + j];
        let r3 = d[12 + j];
        m[j] = r0 - r2;
        m[4 + j] = r1 + r2;
        m[8 + j] = r2 - r1;
        m[12 + j] = r1 - r3;
    }
    // Bᵀ on rows.
    for i in 0..4 {
        let c0 = m[i * 4];
        let c1 = m[i * 4 + 1];
        let c2 = m[i * 4 + 2];
        let c3 = m[i * 4 + 3];
        v[i * 4] = c0 - c2;
        v[i * 4 + 1] = c1 + c2;
        v[i * 4 + 2] = c2 - c1;
        v[i * 4 + 3] = c1 - c3;
    }
}

/// Output transform: Y = Aᵀ m A, 4×4 → 2×2.
/// `m` is the 4×4 winograd-domain product (row-major).
/// `y` receives the 4 output values (row-major 2×2).
fn wino_outp_f2(m: &[f32], y: &mut [f32]) {
    debug_assert_eq!(m.len(), P);
    debug_assert_eq!(y.len(), OT * OT);
    let mut p = [0.0f32; 8];
    // Aᵀ on columns: m[row*4 + j] for row in 0..4.
    for j in 0..4 {
        let r0 = m[j];
        let r1 = m[4 + j];
        let r2 = m[8 + j];
        let r3 = m[12 + j];
        p[j] = r0 + r1 + r2;
        p[4 + j] = r1 - r2 - r3;
    }
    // Aᵀ on rows.
    for i in 0..2 {
        let c0 = p[i * 4];
        let c1 = p[i * 4 + 1];
        let c2 = p[i * 4 + 2];
        let c3 = p[i * 4 + 3];
        y[i * 2] = c0 + c1 + c2;
        y[i * 2 + 1] = c1 - c2 - c3;
    }
}

// --- Winograd F(4x4,3x3) transforms (1:1 port from winograd.cpp F4Policy) ---
//
// F(4,3) uses 6×6 input tiles → 4×4 output tiles. Theoretical mult reduction:
//   F(2,3): 4.0 mults/output   F(4,3): 2.25 mults/output   → 1.78× fewer mults.
// The filter/input transforms use fractions 1/6, 1/12, 1/24 (vs F(2,3)'s clean
// 0.5 halves), so precision is ~1e-4 (vs F(2,3)'s ~1e-5). Acceptable for the
// downstream ReLU + final depth projection.

/// F(4,3) constants. (See [`WinoPolicy`] trait for the shared dispatch.)
const IT4: usize = 6;
const OT4: usize = 4;
const P4: usize = 36;

/// F(4,3) input row/column transform B^T (one 6-vector → 6-vector).
/// Mirrors C++ `F4Policy::Brow`.
#[inline]
fn brow_f4(x: &[f32], r: &mut [f32]) {
    r[0] = 4.0 * x[0] - 5.0 * x[2] + x[4];
    r[1] = -4.0 * x[1] - 4.0 * x[2] + x[3] + x[4];
    r[2] = 4.0 * x[1] - 4.0 * x[2] - x[3] + x[4];
    r[3] = -2.0 * x[1] - x[2] + 2.0 * x[3] + x[4];
    r[4] = 2.0 * x[1] - x[2] - 2.0 * x[3] + x[4];
    r[5] = 4.0 * x[1] - 5.0 * x[3] + x[5];
}

/// F(4,3) filter row/column transform G (one 3-vector → 6-vector).
/// Mirrors C++ `F4Policy::Grow`.
#[inline]
fn grow_f4(y: &[f32], u: &mut [f32]) {
    let a = y[0];
    let b = y[1];
    let c = y[2];
    u[0] = 0.25 * a;
    u[1] = -(a + b + c) * (1.0f32 / 6.0);
    u[2] = (-a + b - c) * (1.0f32 / 6.0);
    u[3] = a * (1.0f32 / 24.0) + b * (1.0f32 / 12.0) + c * (1.0f32 / 6.0);
    u[4] = a * (1.0f32 / 24.0) - b * (1.0f32 / 12.0) + c * (1.0f32 / 6.0);
    u[5] = c;
}

/// F(4,3) output row/column transform A^T (one 6-vector → 4-vector).
/// Mirrors C++ `F4Policy::Arow`.
#[inline]
fn arow_f4(m: &[f32], o: &mut [f32]) {
    o[0] = m[0] + m[1] + m[2] + m[3] + m[4];
    o[1] = m[1] - m[2] + 2.0 * m[3] - 2.0 * m[4];
    o[2] = m[1] + m[2] + 4.0 * m[3] + 4.0 * m[4];
    o[3] = m[1] - m[2] + 8.0 * m[3] - 8.0 * m[4] + m[5];
}

/// F(4,3) filter transform: U = G g Gᵀ, 3×3 → 6×6.
/// `g` is the 3×3 filter (row-major). `u` receives 36 values (row-major 6×6).
fn wino_filt_f4(g: &[f32], u: &mut [f32]) {
    debug_assert_eq!(g.len(), 9);
    debug_assert_eq!(u.len(), P4);
    let mut gg = [[0.0f32; 3]; 6];
    let mut col = [0.0f32; 3];
    let mut out = [0.0f32; 6];
    // Apply G on the 3 columns of g.
    for j in 0..3 {
        col[0] = g[j];
        col[1] = g[3 + j];
        col[2] = g[6 + j];
        grow_f4(&col, &mut out);
        for i in 0..6 {
            gg[i][j] = out[i];
        }
    }
    // Apply G on the 3 columns of each Gg row.
    for i in 0..6 {
        grow_f4(&gg[i], &mut out);
        for k in 0..6 {
            u[i * 6 + k] = out[k];
        }
    }
}

/// F(4,3) input transform: V = Bᵀ d B, 6×6 → 6×6.
/// `d` is the 6×6 input tile (row-major). `v` receives 36 values.
fn wino_inp_f4(d: &[f32], v: &mut [f32]) {
    debug_assert_eq!(d.len(), P4);
    debug_assert_eq!(v.len(), P4);
    let mut m = [0.0f32; P4];
    let mut col = [0.0f32; 6];
    let mut out = [0.0f32; 6];
    // B^T on columns.
    for j in 0..6 {
        col[0] = d[j];
        col[1] = d[6 + j];
        col[2] = d[12 + j];
        col[3] = d[18 + j];
        col[4] = d[24 + j];
        col[5] = d[30 + j];
        brow_f4(&col, &mut out);
        for i in 0..6 {
            m[i * 6 + j] = out[i];
        }
    }
    // B^T on rows.
    for i in 0..6 {
        brow_f4(&m[i * 6..i * 6 + 6], &mut out);
        for k in 0..6 {
            v[i * 6 + k] = out[k];
        }
    }
}

/// F(4,3) output transform: Y = Aᵀ m A, 6×6 → 4×4.
/// `m` is the 6×6 winograd-domain product (row-major).
/// `y` receives the 16 output values (row-major 4×4).
fn wino_outp_f4(m: &[f32], y: &mut [f32]) {
    debug_assert_eq!(m.len(), P4);
    debug_assert_eq!(y.len(), OT4 * OT4);
    let mut p = [0.0f32; 24]; // 4×6 (A^T applied on columns)
    let mut col = [0.0f32; 6];
    let mut out = [0.0f32; 4];
    // A^T on columns.
    for j in 0..6 {
        col[0] = m[j];
        col[1] = m[6 + j];
        col[2] = m[12 + j];
        col[3] = m[18 + j];
        col[4] = m[24 + j];
        col[5] = m[30 + j];
        arow_f4(&col, &mut out);
        for i in 0..4 {
            p[i * 6 + j] = out[i];
        }
    }
    // A^T on rows.
    for i in 0..4 {
        arow_f4(&p[i * 6..i * 6 + 6], &mut out);
        for k in 0..4 {
            y[i * 4 + k] = out[k];
        }
    }
}

/// Winograd policy trait: monomorphizes the conv implementation over F(2,3)
/// vs F(4,3). Each impl supplies the tile sizes (IT/OT/P) and the three
/// transform functions. The transforms are static methods on a ZST, so the
/// compiler resolves them at monomorphization time and inlines them into the
/// tile-block loop.
trait WinoPolicy {
    const IT: usize;
    const OT: usize;
    const P: usize;
    fn filt(g: &[f32], u: &mut [f32]);
    fn inp(d: &[f32], v: &mut [f32]);
    fn outp(m: &[f32], y: &mut [f32]);
}

/// F(2×2, 3×3) Winograd — exact halves-only transform, max|d| ~ 1e-5.
enum F2 {}
impl WinoPolicy for F2 {
    const IT: usize = IT;
    const OT: usize = OT;
    const P: usize = P;
    #[inline]
    fn filt(g: &[f32], u: &mut [f32]) {
        wino_filt_f2(g, u)
    }
    #[inline]
    fn inp(d: &[f32], v: &mut [f32]) {
        wino_inp_f2(d, v)
    }
    #[inline]
    fn outp(m: &[f32], y: &mut [f32]) {
        wino_outp_f2(m, y)
    }
}

/// F(4×4, 3×3) Winograd — 1.78× fewer mults than F(2,3); uses 1/6, 1/24
/// fractions so max|d| ~ 1e-4..1e-3. Used for large output spatial sizes where
/// the arithmetic reduction dominates the slight precision loss.
enum F4 {}
impl WinoPolicy for F4 {
    const IT: usize = IT4;
    const OT: usize = OT4;
    const P: usize = P4;
    #[inline]
    fn filt(g: &[f32], u: &mut [f32]) {
        wino_filt_f4(g, u)
    }
    #[inline]
    fn inp(d: &[f32], v: &mut [f32]) {
        wino_inp_f4(d, v)
    }
    #[inline]
    fn outp(m: &[f32], y: &mut [f32]) {
        wino_outp_f4(m, y)
    }
}

/// Build the transformed filter bank U from the PyTorch-layout weights.
///
/// - `w`: `[OC, IC, 3, 3]` row-major (PyTorch `Conv2d.weight`)
/// - Returns `U[P, IC, OC]` (OC innermost so the winograd GEMM vectorizes
///   cleanly over OC — matches the C++ layout `U[pos*IC*OC + ic*OC + oc]`).
fn build_u<W: WinoPolicy>(w: &[f32], ic: usize, oc: usize) -> Vec<f32> {
    let p = W::P;
    let mut u = vec![0.0f32; p * ic * oc];
    let mut tile = vec![0.0f32; p];
    for oc_i in 0..oc {
        for ic_i in 0..ic {
            let g_base = (oc_i * ic + ic_i) * 9;
            W::filt(&w[g_base..g_base + 9], &mut tile);
            for pos in 0..p {
                u[pos * ic * oc + ic_i * oc + oc_i] = tile[pos];
            }
        }
    }
    u
}

// Cache of transformed filter banks U, keyed by the weight data pointer.
// Since the DPT head's weights are loaded once and never change, the pointer
// is stable across forwards — the U transform is computed once on the first
// forward and reused thereafter.
//
// Key: (pointer, ic, oc, P) — ic/oc guard against pointer reuse with different
// shapes; P (the winograd position count, 16 for F(2,3) and 36 for F(4,3))
// distinguishes policies for the same weight pointer. The same conv can be
// dispatched through both policies during A/B benchmarking, so both must be
// cached independently.
//
// Values are stored in an `Arc<Vec<f32>>` so the lookup returns a reference-
// counted clone (cheap — just an atomic increment) rather than deep-copying
// the (often 1 MB+) U bank on every conv call.
type UCache = std::collections::HashMap<(usize, usize, usize, usize), std::sync::Arc<Vec<f32>>>;
static U_CACHE: std::sync::Mutex<Option<UCache>> = std::sync::Mutex::new(None);

/// Get the transformed filter bank U for `w`, using the cache when the weight
/// pointer has been seen before. The cache is keyed by `(w.as_ptr() as usize,
/// ic, oc, P)`; since DPT head weights are immutable after load, the pointer is
/// stable across forward passes and the U transform runs only once per conv
/// (per policy).
fn get_or_build_u<W: WinoPolicy>(w: &[f32], ic: usize, oc: usize) -> std::sync::Arc<Vec<f32>> {
    let key = (w.as_ptr() as usize, ic, oc, W::P);
    let mut guard = U_CACHE.lock().expect("U_CACHE poisoned");
    if let Some(cache) = guard.as_mut() {
        if let Some(u) = cache.get(&key) {
            return u.clone();
        }
    }
    let u = std::sync::Arc::new(build_u::<W>(w, ic, oc));
    let cache = guard.get_or_insert_with(std::collections::HashMap::new);
    cache.insert(key, u.clone());
    u
}

// --- Winograd-domain GEMM microkernel -------------------------------------
//
// For each winograd position `pos`:
//   M[t][oc] = Σ_ic U[ic][oc] · V[ic][t],   t ∈ [0, TB), oc ∈ [0, OC)
//
// U: [IC][OC] row-major (OC innermost).
// V: [IC][TB] row-major (TB innermost).
// M: [TB][OC] row-major (OC innermost per tile row).
//
// The AVX2 kernel processes 8 OC at a time, holding TB=8 __m256 accumulators.
// Each U-row load (8 floats) is reused across all TB tiles → high arithmetic
// intensity (8·8 = 64 FMA per 8+8 = 16 loads = 4 FLOP/byte).

fn wino_gemm(u: &[f32], v: &[f32], m: &mut [f32], ic: usize, oc: usize, tb_cur: usize) {
    #[cfg(target_arch = "x86_64")]
    if tinyblas::has_avx2_fma() {
        // Safety: feature-checked above.
        unsafe { wino_gemm_avx2(u, v, m, ic, oc, tb_cur) };
        return;
    }
    wino_gemm_scalar(u, v, m, ic, oc, tb_cur);
}

/// Winograd-domain batched GEMM microkernel (AVX2/FMA).
///
/// Computes, for each winograd position independently:
///   M[t][oc] = Σ_ic U[ic][oc] · V[ic][t],   t ∈ [0, TB), oc ∈ [0, OC)
///
/// # Loop structure and port pressure
///
/// The hot loop processes **16 OC per iteration** (two 8-lane u-loads) and for
/// each of the `tb_cur` tiles broadcasts V[ic, t] once and FMAs it into both
/// 8-lane halves. Per ic-iteration this issues:
///   - 2 u-loads  (ports 2/3)  → 1 cycle
///   - tb_cur broadcasts (port 5) → tb_cur cycles
///   - 2·tb_cur FMAs (ports 0/1) → tb_cur cycles
///
/// For tb_cur = 8 that's 8 cycles for 16·8 = 128 FLOP-loads ⇒ **16 FLOP/cycle**
/// (theoretical AVX2 peak with two FMA units). The previous 8-OC loop achieved
/// only 8 FLOP/cycle because port 5 (broadcasts) was the bottleneck — every
/// broadcast fed only 8 FMA lanes instead of 16.
///
/// Register pressure: tb_cur=8 with 16-OC accumulators = 16 ymm regs (all of
/// them); the two u-vec temporaries are reloaded inside the tile loop. The
/// compiler schedules this without spilling to memory in practice (verified via
/// `cargo asm`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn wino_gemm_avx2(
    u: &[f32],     // [IC, OC]
    v: &[f32],     // [IC, TB]
    m: &mut [f32], // [TB, OC]  (only [tb_cur, OC] is written/read)
    ic: usize,
    oc: usize,
    tb_cur: usize,
) {
    use std::arch::x86_64::*;
    // Zero the M region we'll write to.
    for i in 0..tb_cur * oc {
        m[i] = 0.0;
    }
    // Special case OC=1: the OC dimension can't be vectorised (only 1 lane),
    // so vectorise across the tile dimension instead. For TB=4, v[ic, 0..4] is
    // 4 contiguous floats — a __m128 load. Broadcast u[ic] as a __m128 and
    // FMA into a single 4-wide accumulator. This is 4× faster than the scalar
    // tail (which processes 1 tile per FMA instead of 4).
    if oc == 1 && tb_cur > 1 {
        let mut acc = _mm_setzero_ps();
        for ic_i in 0..ic {
            // v[ic_i, 0..tb_cur] is contiguous (TB entries per ic row).
            let v_vec = _mm_loadu_ps(v.as_ptr().add(ic_i * TB));
            let u_b = _mm_set1_ps(u[ic_i]);
            acc = _mm_fmadd_ps(u_b, v_vec, acc);
        }
        // Store tb_cur floats (upper lanes are garbage if tb_cur < 4).
        let mut tmp = [0.0f32; 4];
        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
        for t in 0..tb_cur {
            m[t] = tmp[t];
        }
        return;
    }
    // Fast path: 16 OC at a time. Reaches 16 FLOP/cycle (peak) by amortising
    // each V broadcast across two u-vector loads (16 FMA lanes per broadcast).
    //
    // On x86_64 with only 16 ymm registers, tb_cur=8 × 2 halves = 16 accumulator
    // registers fills the entire file; the u-vec temporaries must then be
    // reloaded inside the tile loop (compiler emits a few stack spills). For
    // tb_cur < 8 or oc < 16 we fall through to the 8-OC path below, which fits
    // comfortably in 8 accumulators and runs without spills. Empirically the
    // spill cost is small relative to the 2× FMA improvement at tb_cur=8.
    let prefer_16oc = true; // TB=4 → 8 accs, no spilling
    let mut oc_i = 0;
    while prefer_16oc && oc_i + 16 <= oc {
        // 2·tb_cur accumulators (two 8-lane halves per tile).
        let mut acc_lo = [_mm256_setzero_ps(); TB];
        let mut acc_hi = [_mm256_setzero_ps(); TB];
        for ic_i in 0..ic {
            let u_lo = _mm256_loadu_ps(u.as_ptr().add(ic_i * oc + oc_i));
            let u_hi = _mm256_loadu_ps(u.as_ptr().add(ic_i * oc + oc_i + 8));
            let vp = v.as_ptr().add(ic_i * TB);
            for t in 0..tb_cur {
                // Safety: `vp.add(t)` points into the `v` slice (bounds checked
                // by ic_i < IC and t < tb_cur <= TB). The intrinsic wants `&f32`;
                // we form the reference from the raw pointer.
                let v_b = _mm256_set1_ps(std::ptr::read(vp.add(t)));
                acc_lo[t] = _mm256_fmadd_ps(u_lo, v_b, acc_lo[t]);
                acc_hi[t] = _mm256_fmadd_ps(u_hi, v_b, acc_hi[t]);
            }
        }
        for t in 0..tb_cur {
            _mm256_storeu_ps(m.as_mut_ptr().add(t * oc + oc_i), acc_lo[t]);
            _mm256_storeu_ps(m.as_mut_ptr().add(t * oc + oc_i + 8), acc_hi[t]);
        }
        oc_i += 16;
    }
    // Fallback: 8 OC at a time (same broadcast cost but only 8 FMA lanes).
    while oc_i + 8 <= oc {
        let mut acc = [_mm256_setzero_ps(); TB];
        for ic_i in 0..ic {
            let u_vec = _mm256_loadu_ps(u.as_ptr().add(ic_i * oc + oc_i));
            let vp = v.as_ptr().add(ic_i * TB);
            for t in 0..tb_cur {
                let v_b = _mm256_set1_ps(std::ptr::read(vp.add(t)));
                acc[t] = _mm256_fmadd_ps(u_vec, v_b, acc[t]);
            }
        }
        for t in 0..tb_cur {
            _mm256_storeu_ps(m.as_mut_ptr().add(t * oc + oc_i), acc[t]);
        }
        oc_i += 8;
    }
    // Scalar tail for the remaining OC lanes.
    while oc_i < oc {
        for ic_i in 0..ic {
            let u_val = u[ic_i * oc + oc_i];
            if u_val == 0.0 {
                continue;
            }
            let vp = &v[ic_i * TB..];
            for t in 0..tb_cur {
                m[t * oc + oc_i] += u_val * vp[t];
            }
        }
        oc_i += 1;
    }
}

fn wino_gemm_scalar(u: &[f32], v: &[f32], m: &mut [f32], ic: usize, oc: usize, tb_cur: usize) {
    for i in 0..tb_cur * oc {
        m[i] = 0.0;
    }
    for ic_i in 0..ic {
        let up = &u[ic_i * oc..(ic_i + 1) * oc];
        let vp = &v[ic_i * TB..(ic_i + 1) * TB];
        for t in 0..tb_cur {
            let vv = vp[t];
            if vv == 0.0 {
                continue;
            }
            let mt = &mut m[t * oc..(t + 1) * oc];
            for oc_i in 0..oc {
                mt[oc_i] += up[oc_i] * vv;
            }
        }
    }
}

/// Winograd F(2x2,3x3) convolution: 3×3 stride-1 pad-1.
///
/// - `x`: `[N, IC, H, W]` NCHW row-major
/// - `w`: `[OC, IC, 3, 3]` PyTorch `Conv2d.weight`
/// - `b`: `[OC]` bias
/// - Returns `[N, OC, H, W]` (same spatial dims as input for pad-1 stride-1).
///
/// Selects between the Winograd kernel (default) and a direct im2col+GEMM
/// kernel. Winograd minimises FLOP (≈44% of direct) but pays transform
/// overhead and runs small per-tile GEMMs that underutilise the FMA units for
/// small IC or OC. For large spatial sizes with small channels, the direct
/// path's large well-shaped GEMM wins despite the extra arithmetic. Override
/// the selection with `DA_FAST_CONV3X3=direct` or `=wino`.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_pad1(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n: usize,
    ic: usize,
    h: usize,
    wid: usize,
    oc: usize,
) -> Vec<f32> {
    conv3x3_pad1_impl(x, w, b, n, ic, h, wid, oc, false, None, false)
}

/// Same as [`conv3x3_pad1`] but applies `relu(x)` to input pixels as they are
/// loaded into the Winograd input transform. This avoids materialising a
/// separate `relu(x)` buffer when the caller only needs `conv(relu(x))` —
/// saving one full read+write pass over the (often multi-MiB) input tensor.
///
/// Border zero-padding is unaffected (max(0.0, 0.0) == 0.0).
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_pad1_relu_in(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n: usize,
    ic: usize,
    h: usize,
    wid: usize,
    oc: usize,
) -> Vec<f32> {
    conv3x3_pad1_impl(x, w, b, n, ic, h, wid, oc, true, None, false)
}

/// Like [`conv3x3_pad1_relu_in`], but also adds `residual` to the output
/// (`out = conv(relu(x)) + bias + residual`). The residual must have the same
/// shape as the output `[N, OC, H, W]`. Fusing the add into the output scatter
/// avoids a separate read+write pass over the output tensor.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_pad1_relu_in_res_out(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n: usize,
    ic: usize,
    h: usize,
    wid: usize,
    oc: usize,
    residual: &[f32],
) -> Vec<f32> {
    conv3x3_pad1_impl(x, w, b, n, ic, h, wid, oc, true, Some(residual), false)
}

/// Same as [`conv3x3_pad1`] but applies `relu` to the output (`max(0, conv(x) + bias)`).
/// Used to fuse the trailing ReLU of `out2a` into the conv output scatter,
/// avoiding a separate read+write pass over the output tensor.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_pad1_relu_out(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n: usize,
    ic: usize,
    h: usize,
    wid: usize,
    oc: usize,
) -> Vec<f32> {
    conv3x3_pad1_impl(x, w, b, n, ic, h, wid, oc, false, None, true)
}

/// Fused `relu(out2a(upsample_bilinear_ac(x, h_lo, w_lo, h, w)))`.
///
/// Performs a 3×3 stride-1 pad-1 conv with `relu` applied to the output, where
/// each input pixel is computed on-the-fly via `align_corners=true` bilinear
/// interpolation from the un-upsampling input `x` at `[n, ic, h_lo, w_lo]`.
/// This eliminates the materialisation of the full upsampled input tensor
/// (~43 MiB at the DA3-BASE output stage), saving one DRAM write+read
/// round-trip.
///
/// If `upsample_add` is `Some`, its `[ic, h, w]` values are added to the
/// upsampled input before the conv (used to fold the UV positional embed into
/// the fusion).
///
/// Returns `[n, oc, h, wid]`.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_pad1_relu_out_upsample(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n: usize,
    ic: usize,
    h_lo: usize,
    w_lo: usize,
    oc: usize,
    h: usize,
    wid: usize,
    upsample_add: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(x.len(), n * ic * h_lo * w_lo, "input size mismatch");
    assert_eq!(w.len(), oc * ic * 9, "weight size mismatch");
    assert_eq!(b.len(), oc, "bias size mismatch");
    if let Some(add) = upsample_add {
        assert_eq!(add.len(), ic * h * wid, "upsample_add size mismatch");
    }

    let mut out = vec![0.0f32; n * oc * h * wid];
    if h == 0 || wid == 0 || oc == 0 || ic == 0 {
        add_bias(&mut out, b, n, oc, h * wid);
        return out;
    }

    let spec = UpsampleSpec::new_ac(h_lo, w_lo, h, wid);
    let fusion = UpsampleFusion {
        spec: &spec,
        add: upsample_add,
    };

    let policy = select_wino_policy(ic, oc, h, wid);
    match policy {
        WinoChoice::F2 => conv3x3_pad1_wino::<F2>(
            x,
            w,
            b,
            n,
            ic,
            h_lo,
            w_lo,
            oc,
            h,
            wid,
            false,
            None,
            true,
            Some(&fusion),
            &mut out,
        ),
        WinoChoice::F4 => conv3x3_pad1_wino::<F4>(
            x,
            w,
            b,
            n,
            ic,
            h_lo,
            w_lo,
            oc,
            h,
            wid,
            false,
            None,
            true,
            Some(&fusion),
            &mut out,
        ),
    }
    out
}

/// Panel width (`NC`) for [`conv1x1_upsample`]. Each parallel task
/// materialises a `[ic, NC]` B-strip from the un-upsampling input, then runs
/// the strided-output panel GEMM against it.
///
/// Tuned to keep the B-strip + A weight matrix + C panel under ~1.5 MiB so the
/// whole working set fits in L2, while producing enough panels (`hw / NC`) to
/// saturate the rayon thread pool. For the DA3-BASE rn1 fusion stage
/// (ic=256, oc=256, hw=55 296) this gives `NC = 512` → 108 panels, with a
/// 512 KiB B-strip.
fn conv1x1_upsample_nc(ic: usize, hw: usize) -> usize {
    // Target B-strip size ≈ 512 KiB: `ic * NC * 4 ≈ 512 KiB` → NC ≈ 128 KiB / ic.
    const TARGET_B_STRIP_BYTES: usize = 512 * 1024;
    let nr = tinyblas::NR;
    let nc_from_budget = (TARGET_B_STRIP_BYTES / (ic.max(1) * 4)).max(nr);
    // Round down to a multiple of NR (16) for full-width microkernel tiles.
    let nc = (nc_from_budget / nr) * nr;
    // Ensure enough panels for thread parallelism: aim for ≥ 2× rayon threads.
    let nthreads = rayon::current_num_threads().max(1);
    let min_nc_for_hw = hw.saturating_sub(1) / (2 * nthreads) + 1;
    nc.max(min_nc_for_hw).min(hw)
}

/// Fused `conv1x1(upsample_bilinear_ac(x_lo, h, w))`.
///
/// Performs a 1×1 conv (`out[oc, h, w] = W[oc, ic] @ x_up[ic, h, w] + bias`)
/// where each upsampled input pixel `x_up[ic_i, oy, ox]` is computed on-the-fly
/// via `align_corners=true` bilinear interpolation from the un-upsampling
/// input `x_lo` at `[n, ic, h_lo, w_lo]`.
///
/// This eliminates the materialisation of the full `[ic, h, w]` upsampled
/// tensor (e.g. ~56 MiB at the DA3-BASE rn1 fusion stage: `96×144 → 192×288`,
/// 256 channels). Each rayon task materialises a narrow `[ic, NC]` strip in L2
/// (≈ 512 KiB, computed from the L3-resident 14 MiB input) and immediately
/// consumes it via the tuned panel GEMM.
///
/// Used by the fusion stage `rn1` of [`crate::fast_dpt`]. Returns `[n, oc, h, w]`.
#[allow(clippy::too_many_arguments)]
pub fn conv1x1_upsample(
    x_lo: &[f32],
    w: &[f32],
    b: &[f32],
    n: usize,
    ic: usize,
    h_lo: usize,
    w_lo: usize,
    oc: usize,
    h: usize,
    wid: usize,
) -> Vec<f32> {
    assert_eq!(x_lo.len(), n * ic * h_lo * w_lo, "input size mismatch");
    assert_eq!(w.len(), oc * ic, "weight size mismatch");
    assert_eq!(b.len(), oc, "bias size mismatch");

    let hw = h * wid;
    let mut out = vec![0.0f32; n * oc * hw];
    if hw == 0 || oc == 0 || ic == 0 {
        add_bias(&mut out, b, n, oc, hw);
        return out;
    }

    let spec = UpsampleSpec::new_ac(h_lo, w_lo, h, wid);
    let nc = conv1x1_upsample_nc(ic, hw);

    for n_i in 0..n {
        let x_batch = &x_lo[n_i * ic * h_lo * w_lo..(n_i + 1) * ic * h_lo * w_lo];
        let out_batch = &mut out[n_i * oc * hw..(n_i + 1) * oc * hw];

        // Fuse bias into GEMM epilogue: pre-fill each channel row with its bias
        // so the panel GEMM's accumulate yields `bias + W @ x_up`.
        out_batch
            .par_chunks_mut(hw)
            .enumerate()
            .for_each(|(oc_i, row)| {
                let bv = b[oc_i];
                for v in row.iter_mut() {
                    *v = bv;
                }
            });

        // Parallelise over NC-wide column panels of the output. Each task:
        //   1. Materialises b_strip[ic, NC] by bilinear-interpolating x_batch
        //      (the L3-resident un-upsampling input) at the panel's output
        //      pixels. b_strip stays in L1/L2.
        //   2. Runs the tuned panel GEMM (`out[oc, NC] += W[oc, ic] @ b_strip`).
        // The output C has row stride = hw (full width); each task writes a
        // disjoint `[j_start, j_end)` column range across all oc rows.
        //
        // Raw pointers are wrapped as `usize` so they can be moved into the
        // rayon closure (raw pointers are not `Send + Sync`). Each task only
        // touches `out[i*hw + j_start .. i*hw + j_end]` across rows i, which is
        // disjoint across tasks, so the concurrent mutable access is safe.
        let x_addr = x_batch.as_ptr() as usize;
        let out_addr = out_batch.as_mut_ptr() as usize;
        let out_len = out_batch.len();
        let n_panels = hw.div_ceil(nc);
        (0..n_panels).into_par_iter().for_each(|jb| unsafe {
            let j_start = jb * nc;
            let j_end = (j_start + nc).min(hw);
            let panel_w = j_end - j_start;
            let x_ptr = x_addr as *const f32;
            let out_ptr = out_addr as *mut f32;
            // b_strip[ic, panel_w] — materialised on-the-fly from x_batch via
            // the upsample spec. Indices are bounds-checked implicitly: oy/ox
            // are within the output spatial dims, and y0/y1/x0/x1 are within
            // h_lo/w_lo by spec construction.
            let mut b_strip = vec![0.0f32; ic * panel_w];
            let ytab = &spec.ytab;
            let xtab = &spec.xtab;
            for ic_i in 0..ic {
                let ic_base_lo = ic_i * h_lo * w_lo;
                let ic_base_up = ic_i * panel_w;
                for px in 0..panel_w {
                    let op = j_start + px;
                    let oy = op / wid;
                    let ox = op % wid;
                    let (y0, y1, wy) = ytab[oy];
                    let (x0, x1, wx) = xtab[ox];
                    let y0 = y0 as usize;
                    let y1 = y1 as usize;
                    let x0 = x0 as usize;
                    let x1 = x1 as usize;
                    let p00 = *x_ptr.add(ic_base_lo + y0 * w_lo + x0);
                    let p01 = *x_ptr.add(ic_base_lo + y0 * w_lo + x1);
                    let p10 = *x_ptr.add(ic_base_lo + y1 * w_lo + x0);
                    let p11 = *x_ptr.add(ic_base_lo + y1 * w_lo + x1);
                    let wyx0 = (1.0 - wy) * (1.0 - wx);
                    let wyx1 = (1.0 - wy) * wx;
                    let wyx2 = wy * (1.0 - wx);
                    let wyx3 = wy * wx;
                    *b_strip.as_mut_ptr().add(ic_base_up + px) =
                        p00 * wyx0 + p01 * wyx1 + p10 * wyx2 + p11 * wyx3;
                }
            }
            // Panel GEMM: out_batch[oc_i, j_start..j_end] += W[oc_i, ic] @ b_strip[ic, panel_w].
            let out_slice = std::slice::from_raw_parts_mut(out_ptr, out_len);
            tinyblas::gemm_nn_panel_strided(oc, panel_w, ic, w, &b_strip, out_slice, hw, j_start);
        });
    }
    out
}

/// Selects which Winograd policy to use for a given conv shape.
///
/// F(4,3) reduces muls/output by 1.78× (2.25 vs 4.0) and reduces tile count
/// by 4×, at the cost of slightly worse precision (1e-4 vs 1e-5). F(4,3) wins
/// for most shapes but loses badly for very small spatial sizes where the
/// reduced tile-block count undersaturates the rayon thread pool.
///
/// Override with `DA_FAST_CONV3X3_WINO=f2` or `=f4`.
fn select_wino_policy(_ic: usize, _oc: usize, hout: usize, wout: usize) -> WinoChoice {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<&str> = OnceLock::new();
    let mode = *OVERRIDE.get_or_init(|| match std::env::var("DA_FAST_CONV3X3_WINO").as_deref() {
        Ok("f2") => "f2",
        Ok("f4") => "f4",
        _ => "auto",
    });
    match mode {
        "f2" => WinoChoice::F2,
        "f4" => WinoChoice::F4,
        _ => {
            // Auto: pick per-shape based on microbenchmarks (32 threads).
            //
            // F(4,3) reduces mults by 1.78× but produces 4× fewer tile-blocks
            // (each tile covers 16 output pixels vs 4). For small spatial sizes,
            // the reduced block count undersaturates the rayon thread pool —
            // F(4,3) is ~2× slower for hw=216 (only 4 blocks for 32 threads).
            //
            // Microbench threshold (32 threads, TB=4):
            //   hw=216:   F4 loses 2.0× ( 4 blocks, severe undersaturation)
            //   hw=864:   F4 loses 5%   (14 blocks, mild undersaturation)
            //   hw=3456:  F4 roughly even (54 blocks, enough for 32 threads)
            //   hw=13824: F4 wins 7-35%
            //   hw=55296: F4 wins 24-38%
            //   hw=169k:  F4 wins 36-55%
            //
            // The threshold of 2048 picks F(4,3) for hw=3456+ (enough blocks)
            // and F(2,3) below (avoid undersaturation). This may need adjustment
            // for different thread counts — a more adaptive heuristic would
            // scale with rayon's thread pool size, but the current shapes are
            // fixed at inference time.
            let hw = hout * wout;
            if hw >= 2048 {
                WinoChoice::F4
            } else {
                WinoChoice::F2
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WinoChoice {
    F2,
    F4,
}

// --- Upsample fusion ------------------------------------------------------
//
// The output stage of the DPT head runs:
//   1. conv3x3_pad1 (out1) at 192×288 → 64 channels
//   2. bilinear upsample 192×288 → 336×504 (writes 43 MiB)
//   3. conv3x3_pad1_relu_out (out2a) at 336×504, reading the 43 MiB upsampled
//      tensor
//
// Steps 2+3 can be fused: out2a's Winograd input transform reads a 6×6 patch
// of *upsampled* values per tile. Instead of materialising the full 43 MiB
// upsampled tensor and then reading it back, we compute each upsampled value
// on-the-fly from the 14 MiB un-upsampling input (which fits entirely in L2).
//
// This eliminates 86 MiB of DRAM round-trip (43 MiB write + 43 MiB read) at
// the cost of 4 input reads + 3 FMA per upsampled value — a clear win because
// the 14 MiB input is L2-resident.
//
// The fusion is only valid when the upsampled tensor is consumed by exactly
// one reader (out2a). When a sky head is present it also reads the upsampled
// tensor, so the caller must fall back to the materialised path.

/// Precomputed `align_corners=true` bilinear upsample tables from
/// `(h_lo, w_lo)` to `(up_h, up_w)`. For each output position `o`, stores the
/// `(lo, hi, weight)` triple such that `value = input[lo]*(1-w) + input[hi]*w`.
/// Built once per output-stage call; shared across all tiles and threads.
pub(crate) struct UpsampleSpec {
    /// `(x0, x1, wx)` for each output column in `[0, up_w)`.
    xtab: Vec<(u32, u32, f32)>,
    /// `(y0, y1, wy)` for each output row in `[0, up_h)`.
    ytab: Vec<(u32, u32, f32)>,
    up_h: u32,
    up_w: u32,
}

impl UpsampleSpec {
    /// Build tables for `align_corners=true` bilinear upsample
    /// `(h_lo, w_lo) → (up_h, up_w)`. Identical math to
    /// [`crate::fast_dpt::upsample_bilinear_ac`].
    pub(crate) fn new_ac(h_lo: usize, w_lo: usize, up_h: usize, up_w: usize) -> Self {
        let scale_y = if up_h > 1 {
            (h_lo - 1) as f32 / (up_h - 1) as f32
        } else {
            0.0
        };
        let scale_x = if up_w > 1 {
            (w_lo - 1) as f32 / (up_w - 1) as f32
        } else {
            0.0
        };
        let xtab = (0..up_w)
            .map(|ox| {
                let fx = ox as f32 * scale_x;
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(w_lo - 1);
                let wx = fx - x0 as f32;
                (x0 as u32, x1 as u32, wx)
            })
            .collect();
        let ytab = (0..up_h)
            .map(|oy| {
                let fy = oy as f32 * scale_y;
                let y0 = fy.floor() as usize;
                let y1 = (y0 + 1).min(h_lo - 1);
                let wy = fy - y0 as f32;
                (y0 as u32, y1 as u32, wy)
            })
            .collect();
        Self {
            xtab,
            ytab,
            up_h: up_h as u32,
            up_w: up_w as u32,
        }
    }
}

/// Active upsample fusion parameters for a single conv call. Borrows the
/// precomputed [`UpsampleSpec`] plus an optional per-channel add tensor
/// (e.g. the UV positional embed) in upsampled `[IC, up_h, up_w]` layout.
pub(crate) struct UpsampleFusion<'a> {
    pub spec: &'a UpsampleSpec,
    pub add: Option<&'a [f32]>,
}

#[allow(clippy::too_many_arguments)]
fn conv3x3_pad1_impl(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n: usize,
    ic: usize,
    h: usize,
    wid: usize,
    oc: usize,
    relu_input: bool,
    residual: Option<&[f32]>,
    relu_output: bool,
) -> Vec<f32> {
    assert_eq!(x.len(), n * ic * h * wid, "input size mismatch");
    assert_eq!(w.len(), oc * ic * 9, "weight size mismatch");
    assert_eq!(b.len(), oc, "bias size mismatch");

    let pad: i32 = 1;
    let wout = wid as i32 + 2 * pad - 2; // = wid for pad=1
    let hout = h as i32 + 2 * pad - 2; // = h for pad=1
    debug_assert!(wout > 0 && hout > 0);
    let wout = wout as usize;
    let hout = hout as usize;

    let mut out = vec![0.0f32; n * oc * hout * wout];
    if hout == 0 || wout == 0 || oc == 0 || ic == 0 {
        // Nothing to compute; just add bias.
        add_bias(&mut out, b, n, oc, hout * wout);
        return out;
    }

    // Kernel selection. The direct path (im2col + GEMM) tends to win for large
    // spatial sizes where the GEMM is well-shaped (large N=HW) and the per-tile
    // Winograd transforms/overhead dominate. Winograd remains better for small
    // spatial sizes with many channels (fusion-stage convs).
    //
    // The heuristic threshold was tuned on the DA3-BASE DPT head shapes:
    // the direct path helps the output convs at 336×504 but hurts the fusion
    // convs at 24×36..192×288.
    use std::sync::OnceLock;
    static CONV3X3_MODE: OnceLock<&str> = OnceLock::new();
    let mode = *CONV3X3_MODE.get_or_init(|| match std::env::var("DA_FAST_CONV3X3").as_deref() {
        Ok("direct") => "direct",
        Ok("wino") => "wino",
        _ => "auto",
    });
    let use_direct = match mode {
        "direct" => true,
        "wino" => false,
        _ => {
            // Auto: currently Winograd wins on all DA3-BASE DPT head shapes.
            // The direct im2col path materialises an IC·9 × Hout·Wout buffer
            // that for large HW (169k) becomes hundreds of MiB and is dominated
            // by allocation + memory bandwidth, not compute. A tiled-im2col
            // variant would change this trade-off but isn't implemented yet.
            false
        }
    };
    if use_direct {
        conv3x3_pad1_direct_into(x, w, b, n, ic, h, wid, oc, hout, wout, &mut out, relu_input);
        if let Some(res) = residual {
            // Fuse residual add (only used with Winograd path in practice,
            // but handle direct path for correctness).
            out.par_iter_mut()
                .zip_eq(res.par_iter())
                .for_each(|(o, r)| *o += r);
        }
        if relu_output {
            out.par_iter_mut().for_each(|v| *v = v.max(0.0));
        }
        return out;
    }

    // Select Winograd policy (F(2,3) vs F(4,3)) based on conv shape.
    let policy = select_wino_policy(ic, oc, hout, wout);
    match policy {
        WinoChoice::F2 => conv3x3_pad1_wino::<F2>(
            x,
            w,
            b,
            n,
            ic,
            h,
            wid,
            oc,
            hout,
            wout,
            relu_input,
            residual,
            relu_output,
            None,
            &mut out,
        ),
        WinoChoice::F4 => conv3x3_pad1_wino::<F4>(
            x,
            w,
            b,
            n,
            ic,
            h,
            wid,
            oc,
            hout,
            wout,
            relu_input,
            residual,
            relu_output,
            None,
            &mut out,
        ),
    }

    out
}

/// Generic Winograd F(2,3) / F(4,3) convolution core, parameterised over a
/// [`WinoPolicy`]. Monomorphisation produces two specialisations (F2, F4) with
/// the transforms inlined and the IT/OT/P constants folded to immediates.
///
/// Writes directly into `out` (caller-allocated), fusing bias + optional
/// residual add + optional relu_output into the output scatter.
///
/// If `upsample` is `Some`, `x` is treated as the *un-upsampling* input at
/// `[N, IC, h, wid]` and each patch value is computed on-the-fly via
/// `align_corners=true` bilinear interpolation to the output grid
/// `[hout, wout]`. In that case `relu_input` is ignored (the upsample fusion
/// is only used from the output stage, which doesn't apply relu-input).
#[allow(clippy::too_many_arguments)]
fn conv3x3_pad1_wino<W: WinoPolicy>(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n: usize,
    ic: usize,
    h: usize,
    wid: usize,
    oc: usize,
    hout: usize,
    wout: usize,
    relu_input: bool,
    residual: Option<&[f32]>,
    relu_output: bool,
    upsample: Option<&UpsampleFusion>,
    out: &mut [f32],
) {
    let pad: i32 = 1;
    let it = W::IT;
    let ot = W::OT;
    let p = W::P;

    let tiles_x = wout.div_ceil(ot);
    let tiles_y = hout.div_ceil(ot);
    let ntiles = tiles_x * tiles_y;
    debug_assert!(ntiles > 0 && oc > 0 && ic > 0);

    // Transform filters once (cached by weight pointer — the U transform is
    // expensive for large IC and the weights don't change between forwards).
    let u = get_or_build_u::<W>(w, ic, oc);

    let total = n * ntiles;
    let nblocks = total.div_ceil(TB);

    // Parallelize over tile-blocks. Each block writes to a disjoint set of
    // output pixels (each tile covers a unique OT×OT region), so writes from
    // different threads never alias.
    //
    // Safety: `out_ptr` is the base of the output buffer. Each rayon task
    // writes only to its own tiles' output locations, which are disjoint
    // across tasks. The raw pointer is never used to create overlapping
    // &mut references.
    //
    // `*mut f32` is not `Sync` by default because raw pointers aren't. We
    // wrap it in a newtype that asserts `Sync`, which is sound here because
    // each parallel task writes to disjoint output locations (no data races).
    //
    // Per-thread scratch buffers (vblk, mblk) are stored in thread_local so
    // they're reused across blocks on the same thread, avoiding per-block
    // heap allocation churn (each conv has hundreds to thousands of blocks).
    let out_ptr = SyncPtr(out.as_mut_ptr());
    let out_len = out.len();

    (0..nblocks).into_par_iter().for_each(|blk| {
        let t0 = blk * TB;
        let tb_cur = std::cmp::min(TB, total - t0);

        VBLK.with(|cell| {
            let mut vblk = cell.borrow_mut();
            vblk.resize(p * ic * TB, 0.0);
            MBLK.with(|cell2| {
                let mut mblk = cell2.borrow_mut();
                mblk.resize(p * TB * oc, 0.0);
                PATCHES.with(|cell3| {
                    let mut wp = cell3.borrow_mut();
                    wp.dpatch.resize(it * it, 0.0);
                    wp.vpatch.resize(p, 0.0);
                    wp.mpatch.resize(p, 0.0);
                    wp.ypatch.resize(ot * ot, 0.0);
                    // Split borrow: destructure through &mut to get disjoint
                    // &mut Vec for each field. The borrow checker allows this
                    // through a direct &mut to the struct.
                    let wp_ref: &mut WinoPatches = &mut *wp;
                    let WinoPatches {
                        dpatch,
                        vpatch,
                        mpatch,
                        ypatch,
                    } = wp_ref;
                    let dpatch: &mut [f32] = dpatch;
                    let vpatch: &mut [f32] = vpatch;
                    let mpatch: &mut [f32] = mpatch;
                    let ypatch: &mut [f32] = ypatch;

                    // 1. Input transform: for each tile in the block, gather the
                    //    IT×IT patch and compute V. The patch reads with
                    //    zero-padding at borders.
                    for tb in 0..tb_cur {
                        let idx = t0 + tb;
                        let n_i = idx / ntiles;
                        let t = idx % ntiles;
                        let ty = t / tiles_x;
                        let tx = t % tiles_x;
                        let iy0 = (ty * ot) as i32 - pad;
                        let ix0 = (tx * ot) as i32 - pad;
                        let xn_off = n_i * ic * h * wid;

                        for ic_i in 0..ic {
                            let xc_off = xn_off + ic_i * h * wid;
                            if let Some(up) = upsample {
                                // Fused upsample path: compute each patch
                                // value via bilinear interpolation from the
                                // un-upsampling input. `h`/`wid` are the
                                // un-upsampling dims; `hout`/`wout` are the
                                // upsampled output dims (patch coords).
                                let xtab = &up.spec.xtab;
                                let ytab = &up.spec.ytab;
                                let up_h = up.spec.up_h as usize;
                                let up_w = up.spec.up_w as usize;
                                let add = up.add;
                                let add_off = ic_i * hout * wout;
                                for i in 0..it {
                                    let yy = iy0 + i as i32;
                                    if yy >= 0 && (yy as usize) < up_h {
                                        let yy_u = yy as usize;
                                        let (y0, y1, wy) = ytab[yy_u];
                                        let (y0, y1) = (y0 as usize, y1 as usize);
                                        let row0 = xc_off + y0 * wid;
                                        let row1 = xc_off + y1 * wid;
                                        for j in 0..it {
                                            let xx = ix0 + j as i32;
                                            dpatch[i * it + j] = if xx >= 0 && (xx as usize) < up_w
                                            {
                                                let xx_u = xx as usize;
                                                let (x0, x1, wx) = xtab[xx_u];
                                                let (x0, x1) = (x0 as usize, x1 as usize);
                                                let p00 = x[row0 + x0];
                                                let p01 = x[row0 + x1];
                                                let p10 = x[row1 + x0];
                                                let p11 = x[row1 + x1];
                                                let top = p00 * (1.0 - wx) + p01 * wx;
                                                let bot = p10 * (1.0 - wx) + p11 * wx;
                                                let v = top * (1.0 - wy) + bot * wy;
                                                if let Some(add) = add {
                                                    v + add[add_off + yy_u * wout + xx_u]
                                                } else {
                                                    v
                                                }
                                            } else {
                                                0.0
                                            };
                                        }
                                    } else {
                                        for j in 0..it {
                                            dpatch[i * it + j] = 0.0;
                                        }
                                    }
                                }
                            } else {
                                // Direct path: gather the IT×IT patch with
                                // zero-padding at borders.
                                for i in 0..it {
                                    let yy = iy0 + i as i32;
                                    if yy >= 0 && (yy as usize) < h {
                                        let row_off = xc_off + (yy as usize) * wid;
                                        for j in 0..it {
                                            let xx = ix0 + j as i32;
                                            dpatch[i * it + j] = if xx >= 0 && (xx as usize) < wid {
                                                let v = x[row_off + xx as usize];
                                                if relu_input {
                                                    v.max(0.0)
                                                } else {
                                                    v
                                                }
                                            } else {
                                                0.0
                                            };
                                        }
                                    } else {
                                        // Whole row is outside the input → all zeros.
                                        for j in 0..it {
                                            dpatch[i * it + j] = 0.0;
                                        }
                                    }
                                }
                            }
                            W::inp(dpatch, vpatch);
                            let vbase = ic_i * TB + tb;
                            for pos in 0..p {
                                vblk[pos * ic * TB + vbase] = vpatch[pos];
                            }
                        }
                    }

                    // 2. Winograd-domain GEMM: for each position, M[tb, oc] = U @ V.
                    for pos in 0..p {
                        let u_pos = &u[pos * ic * oc..(pos + 1) * ic * oc];
                        let v_pos = &vblk[pos * ic * TB..(pos + 1) * ic * TB];
                        let m_pos = &mut mblk[pos * TB * oc..(pos + 1) * TB * oc];
                        wino_gemm(u_pos, v_pos, m_pos, ic, oc, tb_cur);
                    }

                    // 3. Output transform: for each tile, gather M[oc] across
                    //    positions and scatter the OT×OT output into the
                    //    destination. Bounds-check the last row/column for tiles
                    //    that extend past Hout/Wout. Bias is fused into the
                    //    scatter (avoids a separate output pass).
                    for tb in 0..tb_cur {
                        let idx = t0 + tb;
                        let n_i = idx / ntiles;
                        let t = idx % ntiles;
                        let ty = t / tiles_x;
                        let tx = t % tiles_x;
                        let oy0 = ty * ot;
                        let ox0 = tx * ot;

                        for oc_i in 0..oc {
                            let bv = b[oc_i];
                            let mbase = tb * oc + oc_i;
                            for pos in 0..p {
                                mpatch[pos] = mblk[pos * TB * oc + mbase];
                            }
                            W::outp(mpatch, ypatch);
                            let yc_off = (n_i * oc + oc_i) * hout * wout;
                            for i in 0..ot {
                                let oy = oy0 + i;
                                if oy >= hout {
                                    continue;
                                }
                                for j in 0..ot {
                                    let ox = ox0 + j;
                                    if ox >= wout {
                                        continue;
                                    }
                                    let off = yc_off + oy * wout + ox;
                                    debug_assert!(
                                        off < out_len,
                                        "output offset {off} >= {out_len}"
                                    );
                                    unsafe {
                                        let v = ypatch[i * ot + j] + bv;
                                        let v = if let Some(res) = residual {
                                            v + res[off]
                                        } else {
                                            v
                                        };
                                        let v = if relu_output { v.max(0.0) } else { v };
                                        out_ptr.write(off, v);
                                    }
                                }
                            }
                        }
                    }
                });
            });
        });
    });
}

/// Direct 3×3 stride-1 pad-1 convolution via im2col + GEMM.
///
/// Builds the im2col matrix `[IC·9, Hout·Wout]` (each column is the flattened
/// `(ic, ky, kx)` patch for one output pixel) and calls the high-performance
/// `tinyblas::gemm_nn_into` with `M=OC, K=IC·9, N=Hout·Wout`.
///
/// This uses ~2.3× more FLOP than Winograd F(2,3) but avoids the per-tile
/// transform overhead and runs a single large well-shaped GEMM. For large
/// spatial sizes (Hout·Wout ≫ 50k) with modest IC·OC, the GEMM efficiency gain
/// outweighs the extra arithmetic. Bias is fused into the GEMM epilogue (caller
/// pre-fills `out` with bias, then the accumulate yields `bias + W @ col`).
#[allow(clippy::too_many_arguments)]
fn conv3x3_pad1_direct_into(
    x: &[f32], // [N, IC, H, W]
    w: &[f32], // [OC, IC, 3, 3] PyTorch
    b: &[f32], // [OC]
    n: usize,
    ic: usize,
    h: usize,
    wid: usize,
    oc: usize,
    hout: usize,
    wout: usize,
    out: &mut [f32], // [N, OC, Hout, Wout]
    relu_input: bool,
) {
    debug_assert_eq!(x.len(), n * ic * h * wid);
    debug_assert_eq!(w.len(), oc * ic * 9);
    debug_assert_eq!(b.len(), oc);
    debug_assert_eq!(out.len(), n * oc * hout * wout);
    let hw_out = hout * wout;
    let ksq = 9; // 3*3
    for n_i in 0..n {
        let x_n = &x[n_i * ic * h * wid..(n_i + 1) * ic * h * wid];
        let out_n = &mut out[n_i * oc * hw_out..(n_i + 1) * oc * hw_out];
        // Fuse bias: pre-fill each OC plane with its bias.
        out_n
            .par_chunks_mut(hw_out)
            .enumerate()
            .for_each(|(oc_i, row)| {
                let bv = b[oc_i];
                for v in row.iter_mut() {
                    *v = bv;
                }
            });
        // im2col: build [IC·9, Hout·Wout] where each column is the (ic, ky, kx)
        // patch for one output pixel. The (ic, ky, kx) ordering matches
        // w.weight's [OC, IC, 3, 3] = [OC, IC·9] layout, so no permutation.
        //
        // Parallelise over the IC·9 rows of the column matrix. Each row is a
        // contiguous [Hout·Wout] slice of the output column buffer.
        let mut col = vec![0.0f32; ic * ksq * hw_out];
        col.par_chunks_mut(hw_out)
            .enumerate()
            .for_each(|(kic, col_ch)| {
                // kic = (ic_i * 3 + ky) * 3 + kx.
                let ic_i = kic / ksq;
                let ky = (kic / 3) % 3;
                let kx = kic % 3;
                let row_off = ic_i * h * wid;
                for oy in 0..hout {
                    // Input y for this output pixel: pad-1 means input window
                    // starts at (oy - 1 + ky). With pad=1, valid inputs are in
                    // [0, h); outside → 0.
                    let iy = oy as i32 - 1 + ky as i32;
                    if iy < 0 || iy as usize >= h {
                        // Whole row is zero.
                        for ox in 0..wout {
                            col_ch[oy * wout + ox] = 0.0;
                        }
                        continue;
                    }
                    let iy = iy as usize;
                    for ox in 0..wout {
                        let ix = ox as i32 - 1 + kx as i32;
                        let v = if ix < 0 || ix as usize >= wid {
                            0.0
                        } else {
                            let raw = x_n[row_off + iy * wid + ix as usize];
                            if relu_input {
                                raw.max(0.0)
                            } else {
                                raw
                            }
                        };
                        col_ch[oy * wout + ox] = v;
                    }
                }
            });
        // GEMM: out[oc, hw_out] += w[oc, ic·9] @ col[ic·9, hw_out].
        // No transpose — both have ic·9 in (ic, ky, kx) order.
        tinyblas::gemm_nn_into(oc, hw_out, ic * ksq, w, &col, out_n);
    }
}

/// 1×1 stride-1 pad-0 convolution via tinyBLAS GEMM.
///
/// - `x`: `[N, IC, H, W]` NCHW
/// - `w`: `[OC, IC, 1, 1]` PyTorch layout (treated as `[OC, IC]`)
/// - `b`: `[OC]` bias
/// - Returns `[N, OC, H, W]` NCHW.
///
/// For NCHW input with N=1, the layout is already `[IC, HW]` row-major, which is
/// the `[K, N]` operand of a standard `C[M,N] = A[M,K] @ B[K,N]` GEMM with
/// `M=OC, K=IC, N=HW`. The weight `[OC, IC]` is `[M, K]`. So the 1×1 conv is a
/// **single GEMM with no transposes** — far faster than the im2col+transpose
/// approach.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::uninit_vec)] // every element is bias-filled (or written by the OC=1 path) before any read
pub fn conv1x1(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n: usize,
    ic: usize,
    h: usize,
    wid: usize,
    oc: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), n * ic * h * wid);
    assert_eq!(w.len(), oc * ic);
    assert_eq!(b.len(), oc);

    let hw = h * wid;
    let total = n * oc * hw;
    // Allocate without zero-fill — every element is written by the bias
    // pre-fill (or the OC=1 fast path) below before the GEMM reads it.
    let mut out: Vec<f32> = Vec::with_capacity(total);
    // SAFETY: `capacity >= total`. Every element in `0..total` is written
    // before being read: the OC=1 fast path writes each pixel exactly once,
    // and the general path writes every element via the bias pre-fill before
    // the GEMM accumulates onto it.
    unsafe { out.set_len(total) };
    // Fast path for single (N=1, OC=1): the GEMM would be [1, ic] @ [ic, hw]
    // which has M=1 and thus no row-parallelism in `gemm_nn_into`. Instead
    // parallelise over output pixels — each pixel's dot product across input
    // channels is independent. This is the final depth-projection conv.
    if n == 1 && oc == 1 {
        let bv = b[0];
        let w_row = &w[..ic];
        let out_ptr = SyncPtr(out.as_mut_ptr());
        (0..hw).into_par_iter().step_by(PX_CHUNK).for_each(|px0| {
            let px_n = PX_CHUNK.min(hw - px0);
            for i in 0..px_n {
                let px = px0 + i;
                let mut acc = bv;
                for ic_i in 0..ic {
                    acc += w_row[ic_i] * x[ic_i * hw + px];
                }
                unsafe {
                    out_ptr.write(px, acc);
                }
            }
        });
        return out;
    }
    for n_i in 0..n {
        let x_batch = &x[n_i * ic * hw..(n_i + 1) * ic * hw];
        let out_batch = &mut out[n_i * oc * hw..(n_i + 1) * oc * hw];
        // Fuse bias into GEMM epilogue: pre-fill each channel plane with its
        // bias value so `gemm_nn_into`'s accumulate yields `bias + A@B`.
        // This removes a separate bias-add pass over the output.
        out_batch
            .par_chunks_mut(hw)
            .enumerate()
            .for_each(|(oc_i, row)| {
                let bv = b[oc_i];
                for v in row.iter_mut() {
                    *v = bv;
                }
            });
        // out[oc, px] = Σ_ic w[oc, ic] · x[ic, px]  (NN GEMM: M=OC, K=IC, N=HW)
        tinyblas::gemm_nn_into(oc, hw, ic, w, x_batch, out_batch);
    }
    out
}

/// Add per-channel bias to a `[N, OC, HW]` buffer (OC outermost within each N).
fn add_bias(out: &mut [f32], b: &[f32], _n: usize, oc: usize, hw: usize) {
    out.par_chunks_mut(hw).enumerate().for_each(|(idx, chunk)| {
        let oc_i = idx % oc;
        let bv = b[oc_i];
        for v in chunk.iter_mut() {
            *v += bv;
        }
    });
}

// --- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive direct 3×3 stride-1 pad-1 conv, for reference.
    #[allow(clippy::too_many_arguments)]
    fn naive_conv3x3_pad1(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        n: usize,
        ic: usize,
        h: usize,
        wid: usize,
        oc: usize,
    ) -> Vec<f32> {
        let pad = 1usize;
        let mut out = vec![0.0f32; n * oc * h * wid];
        for n_i in 0..n {
            for oc_i in 0..oc {
                for y in 0..h {
                    for x_i in 0..wid {
                        let mut s = b[oc_i];
                        for ic_i in 0..ic {
                            for ky in 0..3usize {
                                for kx in 0..3usize {
                                    let iy = y as i32 + ky as i32 - pad as i32;
                                    let ix = x_i as i32 + kx as i32 - pad as i32;
                                    if iy >= 0 && iy < h as i32 && ix >= 0 && ix < wid as i32 {
                                        let xv = x[((n_i * ic + ic_i) * h + iy as usize) * wid
                                            + ix as usize];
                                        let wv = w[((oc_i * ic + ic_i) * 3 + ky) * 3 + kx];
                                        s += xv * wv;
                                    }
                                }
                            }
                        }
                        out[((n_i * oc + oc_i) * h + y) * wid + x_i] = s;
                    }
                }
            }
        }
        out
    }

    /// Naive 1×1 conv, for reference.
    #[allow(clippy::too_many_arguments)]
    fn naive_conv1x1(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        n: usize,
        ic: usize,
        h: usize,
        wid: usize,
        oc: usize,
    ) -> Vec<f32> {
        let hw = h * wid;
        let mut out = vec![0.0f32; n * oc * hw];
        for n_i in 0..n {
            for oc_i in 0..oc {
                for px in 0..hw {
                    let mut s = b[oc_i];
                    for ic_i in 0..ic {
                        s += x[((n_i * ic + ic_i) * hw) + px] * w[oc_i * ic + ic_i];
                    }
                    out[(n_i * oc + oc_i) * hw + px] = s;
                }
            }
        }
        out
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        assert_eq!(a.len(), b.len());
        let mut max_diff = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        if max_diff > tol {
            eprintln!("max_diff = {max_diff} (tol = {tol})");
            false
        } else {
            true
        }
    }

    #[test]
    fn winograd_matches_naive_small() {
        // Small case: 1×4×4 input, 3×3 conv to 2 channels.
        let n = 1;
        let ic = 3;
        let h = 4;
        let wid = 4;
        let oc = 2;
        let x: Vec<f32> = (0..n * ic * h * wid)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        let w: Vec<f32> = (0..oc * ic * 9).map(|i| (i as f32) * 0.01 - 0.4).collect();
        let b: Vec<f32> = vec![0.5, -0.3];

        let naive = naive_conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        let fast = conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        assert!(approx_eq(&naive, &fast, 1e-4), "winograd vs naive mismatch");
    }

    #[test]
    fn winograd_matches_naive_medium() {
        // DPT-head-like shape: 1×8×8, IC=4, OC=4.
        let n = 1;
        let ic = 4;
        let h = 8;
        let wid = 8;
        let oc = 4;
        let x: Vec<f32> = (0..n * ic * h * wid)
            .map(|i| (((i as f32) * 0.37) % 2.0) - 1.0)
            .collect();
        let w: Vec<f32> = (0..oc * ic * 9)
            .map(|i| (((i as f32) * 0.71) % 2.0) - 1.0)
            .collect();
        let b: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];

        let naive = naive_conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        let fast = conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        assert!(approx_eq(&naive, &fast, 1e-4));
    }

    #[test]
    fn winograd_matches_naive_odd_dims() {
        // Odd spatial dims → partial last tile.
        let n = 1;
        let ic = 2;
        let h = 5;
        let wid = 7;
        let oc = 3;
        let x: Vec<f32> = (0..n * ic * h * wid).map(|i| (i as f32) * 0.03).collect();
        let w: Vec<f32> = (0..oc * ic * 9).map(|i| (i as f32) * 0.05 - 0.5).collect();
        let b: Vec<f32> = vec![1.0, 2.0, 3.0];

        let naive = naive_conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        let fast = conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        assert_eq!(naive.len(), fast.len(), "output length mismatch");
        assert!(approx_eq(&naive, &fast, 1e-4));
    }

    #[test]
    fn winograd_matches_naive_batch2() {
        // Batch > 1: exercises the n-axis in the tile loop.
        let n = 2;
        let ic = 2;
        let h = 6;
        let wid = 6;
        let oc = 2;
        let x: Vec<f32> = (0..n * ic * h * wid).map(|i| (i as f32) * 0.07).collect();
        let w: Vec<f32> = (0..oc * ic * 9).map(|i| (i as f32) * 0.11).collect();
        let b: Vec<f32> = vec![0.0, 0.0];

        let naive = naive_conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        let fast = conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        assert!(approx_eq(&naive, &fast, 1e-4));
    }

    #[test]
    fn winograd_matches_naive_dpt_shape() {
        // Realistic DPT head shape: IC=128, OC=64, H=42, W=62.
        // (One of the fusion-pyramid lateral resolutions.)
        let n = 1;
        let ic = 16; // reduced from 128 for test speed
        let h = 42;
        let wid = 62;
        let oc = 16;
        // Pseudo-random but deterministic.
        let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Use lower 32 bits so the f32 conversion stays bounded.
            ((rng as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..n * ic * h * wid).map(|_| next()).collect();
        let w: Vec<f32> = (0..oc * ic * 9).map(|_| next()).collect();
        let b: Vec<f32> = (0..oc).map(|_| next()).collect();

        let naive = naive_conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        let fast = conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        assert!(
            approx_eq(&naive, &fast, 1e-3),
            "DPT-shape winograd mismatch"
        );
    }

    #[test]
    fn conv1x1_matches_naive() {
        let n = 1;
        let ic = 4;
        let h = 8;
        let wid = 8;
        let oc = 3;
        let x: Vec<f32> = (0..n * ic * h * wid).map(|i| (i as f32) * 0.03).collect();
        let w: Vec<f32> = (0..oc * ic).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = vec![0.1, 0.2, 0.3];

        let naive = naive_conv1x1(&x, &w, &b, n, ic, h, wid, oc);
        let fast = conv1x1(&x, &w, &b, n, ic, h, wid, oc);
        assert!(approx_eq(&naive, &fast, 1e-4));
    }

    #[test]
    fn winograd_identity_filter() {
        // Identity 3×3 filter (center=1, rest=0) → output = input.
        let n = 1;
        let ic = 1;
        let h = 6;
        let wid = 6;
        let oc = 1;
        let x: Vec<f32> = (0..n * ic * h * wid).map(|i| (i as f32) * 0.1).collect();
        let mut w = vec![0.0f32; oc * ic * 9];
        w[4] = 1.0; // center
        let b = vec![0.0];

        let fast = conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        // Interior pixels should match input exactly; border pixels are zero-
        // padded so they differ (input·0 instead of input·1 on the border).
        // Check a few interior pixels.
        for y in 1..h - 1 {
            for x_i in 1..wid - 1 {
                let off = y * wid + x_i;
                assert!(
                    (fast[off] - x[off]).abs() < 1e-5,
                    "identity mismatch at ({y},{x_i}): fast={} vs x={}",
                    fast[off],
                    x[off]
                );
            }
        }
    }

    #[test]
    fn winograd_zero_input() {
        // Zero input → output = bias.
        let n = 1;
        let ic = 4;
        let h = 8;
        let wid = 8;
        let oc = 4;
        let x = vec![0.0f32; n * ic * h * wid];
        let w: Vec<f32> = (0..oc * ic * 9).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let fast = conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        for oc_i in 0..oc {
            for px in 0..h * wid {
                let off = oc_i * h * wid + px;
                assert!(
                    (fast[off] - b[oc_i]).abs() < 1e-5,
                    "zero-input mismatch at oc={oc_i} px={px}: {} vs {}",
                    fast[off],
                    b[oc_i]
                );
            }
        }
    }

    // --- F(4,3) Winograd tests ----------------------------------------------
    //
    // These tests force the F(4,3) policy via env var, then verify it matches
    // the naive direct convolution. The env-var override is read once per
    // process via `OnceLock`, so each test runs in its own process via
    // `cargo test` (which serialises tests within a binary by default).
    //
    // IMPORTANT: `select_wino_policy` caches the env var in a `OnceLock`, so if
    // another test already triggered the cache with "auto", these tests would
    // not pick up the override. We work around this by calling `conv3x3_pad1_wino::<F4>`
    // directly (bypassing the auto-select), which is the function under test.

    /// Helper: run a 3×3 conv through the F(4,3) policy directly.
    #[allow(clippy::too_many_arguments)]
    fn conv3x3_f4(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        n: usize,
        ic: usize,
        h: usize,
        wid: usize,
        oc: usize,
    ) -> Vec<f32> {
        let hout = h;
        let wout = wid;
        let mut out = vec![0.0f32; n * oc * hout * wout];
        conv3x3_pad1_wino::<F4>(
            x, w, b, n, ic, h, wid, oc, hout, wout, false, None, false, None, &mut out,
        );
        out
    }

    #[test]
    fn winograd_f4_matches_naive_small() {
        let n = 1;
        let ic = 3;
        let h = 8;
        let wid = 8; // multiples of OT4=4 to avoid partial-tile edge cases
        let oc = 2;
        let x: Vec<f32> = (0..n * ic * h * wid)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        let w: Vec<f32> = (0..oc * ic * 9).map(|i| (i as f32) * 0.01 - 0.4).collect();
        let b: Vec<f32> = vec![0.5, -0.3];

        let naive = naive_conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        let f4 = conv3x3_f4(&x, &w, &b, n, ic, h, wid, oc);
        assert! {
            approx_eq(&naive, &f4, 1e-3),
            "F(4,3) vs naive mismatch (max_diff = {:?})",
            naive.iter().zip(f4.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
        };
    }

    #[test]
    fn winograd_f4_matches_naive_dpt_shape() {
        // Realistic DPT head shape (similar to out2a: 64→32 @ 336×504).
        // Reduced sizes for test speed but still exercises the large-HW path.
        let n = 1;
        let ic = 16;
        let h = 48;
        let wid = 64; // divisible by 4
        let oc = 16;
        let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..n * ic * h * wid).map(|_| next()).collect();
        let w: Vec<f32> = (0..oc * ic * 9).map(|_| next()).collect();
        let b: Vec<f32> = (0..oc).map(|_| next()).collect();

        let naive = naive_conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        let f4 = conv3x3_f4(&x, &w, &b, n, ic, h, wid, oc);
        // F(4,3) uses 1/6, 1/24 fractions → tolerance 1e-3 (vs F(2,3)'s 1e-4).
        assert! {
            approx_eq(&naive, &f4, 1e-3),
            "F(4,3) DPT-shape mismatch"
        };
    }

    #[test]
    fn winograd_f4_matches_f2() {
        // Cross-policy consistency: F(4,3) and F(2,3) should agree to ~1e-3
        // (both approximate the same true convolution).
        let n = 1;
        let ic = 8;
        let h = 16;
        let wid = 16;
        let oc = 8;
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..n * ic * h * wid).map(|_| next()).collect();
        let w: Vec<f32> = (0..oc * ic * 9).map(|_| next()).collect();
        let b: Vec<f32> = (0..oc).map(|_| next()).collect();

        let f2 = conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        let f4 = conv3x3_f4(&x, &w, &b, n, ic, h, wid, oc);
        // Both should agree to within their combined precision budgets.
        assert! {
            approx_eq(&f2, &f4, 1e-3),
            "F(2,3) vs F(4,3) mismatch"
        };
    }

    #[test]
    fn winograd_f4_odd_dims() {
        // Odd spatial dims → partial last tile in both x and y.
        let n = 1;
        let ic = 4;
        let h = 10; // not a multiple of OT4=4
        let wid = 14;
        let oc = 4;
        let x: Vec<f32> = (0..n * ic * h * wid).map(|i| (i as f32) * 0.03).collect();
        let w: Vec<f32> = (0..oc * ic * 9).map(|i| (i as f32) * 0.05 - 0.5).collect();
        let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let naive = naive_conv3x3_pad1(&x, &w, &b, n, ic, h, wid, oc);
        let f4 = conv3x3_f4(&x, &w, &b, n, ic, h, wid, oc);
        assert_eq!(naive.len(), f4.len(), "output length mismatch");
        assert! {approx_eq(&naive, &f4, 1e-3)};
    }

    #[test]
    fn winograd_f4_relu_in_res_out() {
        // Test the fused relu-input + residual-output path through F(4,3).
        let n = 1;
        let ic = 4;
        let h = 16;
        let wid = 16;
        let oc = 4;
        let x: Vec<f32> = (0..n * ic * h * wid)
            .map(|i| (((i as f32) * 0.37) % 2.0) - 1.0)
            .collect();
        let w: Vec<f32> = (0..oc * ic * 9)
            .map(|i| (((i as f32) * 0.71) % 2.0) - 1.0)
            .collect();
        let b: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
        let residual: Vec<f32> = (0..n * oc * h * wid)
            .map(|i| (((i as f32) * 0.13) % 1.0) - 0.5)
            .collect();

        // Reference: relu(x) → conv → +bias → +residual.
        let relu_x: Vec<f32> = x.iter().map(|v| v.max(0.0)).collect();
        let naive_conv = naive_conv3x3_pad1(&relu_x, &w, &b, n, ic, h, wid, oc);
        let expected: Vec<f32> = naive_conv
            .iter()
            .zip(residual.iter())
            .map(|(c, r)| c + r)
            .collect();

        // F(4,3) with fused relu-in and residual-out.
        let hout = h;
        let wout = wid;
        let mut out = vec![0.0f32; n * oc * hout * wout];
        conv3x3_pad1_wino::<F4>(
            x.as_slice(),
            w.as_slice(),
            b.as_slice(),
            n,
            ic,
            h,
            wid,
            oc,
            hout,
            wout,
            true,                      // relu_input
            Some(residual.as_slice()), // residual
            false,                     // relu_output
            None,                      // upsample
            &mut out,
        );

        assert! {
            approx_eq(&expected, &out, 1e-3),
            "F(4,3) relu-in res-out mismatch"
        };
    }

    // --- Upsample fusion tests ---
    //
    // The fused `conv3x3_pad1_relu_out_upsample` must produce output identical
    // (within tolerance) to the unfused path of:
    //   1. materialise the bilinear upsample of `x` (align_corners=true) to
    //      (h, w)
    //   2. optionally add a per-channel tensor `add`
    //   3. run conv3x3_pad1 + relu on the result

    /// Naive `align_corners=true` bilinear upsample `[C, h_lo, w_lo] → [C, h, w]`.
    fn naive_upsample_ac(
        x: &[f32],
        c: usize,
        h_lo: usize,
        w_lo: usize,
        h: usize,
        w: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; c * h * w];
        let sy = if h > 1 {
            (h_lo - 1) as f32 / (h - 1) as f32
        } else {
            0.0
        };
        let sx = if w > 1 {
            (w_lo - 1) as f32 / (w - 1) as f32
        } else {
            0.0
        };
        for ci in 0..c {
            let in_off = ci * h_lo * w_lo;
            let out_off = ci * h * w;
            for oy in 0..h {
                let fy = oy as f32 * sy;
                let y0 = fy.floor() as usize;
                let y1 = (y0 + 1).min(h_lo - 1);
                let wy = fy - y0 as f32;
                for ox in 0..w {
                    let fx = ox as f32 * sx;
                    let x0 = fx.floor() as usize;
                    let x1 = (x0 + 1).min(w_lo - 1);
                    let wx = fx - x0 as f32;
                    let p00 = x[in_off + y0 * w_lo + x0];
                    let p01 = x[in_off + y0 * w_lo + x1];
                    let p10 = x[in_off + y1 * w_lo + x0];
                    let p11 = x[in_off + y1 * w_lo + x1];
                    let top = p00 * (1.0 - wx) + p01 * wx;
                    let bot = p10 * (1.0 - wx) + p11 * wx;
                    out[out_off + oy * w + ox] = top * (1.0 - wy) + bot * wy;
                }
            }
        }
        out
    }

    #[test]
    fn upsample_fusion_matches_materialized() {
        // Compare fused `conv(relu(upsample(x)))` vs unfused reference.
        let n = 1;
        let ic = 4;
        let h_lo = 12;
        let w_lo = 18;
        // Non-integer scale: 21/12 = 1.75, mirroring the DA3 output stage.
        let h = 21;
        let w = 32;
        let oc = 3;
        let x: Vec<f32> = (0..n * ic * h_lo * w_lo)
            .map(|i| (((i as f32) * 0.37) % 2.0) - 1.0)
            .collect();
        let wt: Vec<f32> = (0..oc * ic * 9)
            .map(|i| (((i as f32) * 0.71) % 2.0) - 1.0)
            .collect();
        let b: Vec<f32> = vec![0.1, -0.2, 0.3];

        // Reference: materialise upsample, then conv3x3_pad1_relu_out.
        let up = naive_upsample_ac(&x, ic, h_lo, w_lo, h, w);
        let naive_conv = naive_conv3x3_pad1(&up, &wt, &b, n, ic, h, w, oc);
        let expected: Vec<f32> = naive_conv.iter().map(|v| v.max(0.0)).collect();

        // Fused path.
        let fused = conv3x3_pad1_relu_out_upsample(&x, &wt, &b, n, ic, h_lo, w_lo, oc, h, w, None);
        assert! {
            approx_eq(&expected, &fused, 1e-3),
            "upsample fusion vs materialised mismatch"
        };
    }

    #[test]
    fn upsample_fusion_with_add() {
        // Same as above but with a per-channel add folded into the upsample.
        let n = 1;
        let ic = 3;
        let h_lo = 8;
        let w_lo = 12;
        let h = 14;
        let w = 21;
        let oc = 2;
        let x: Vec<f32> = (0..n * ic * h_lo * w_lo)
            .map(|i| (((i as f32) * 0.43) % 2.0) - 1.0)
            .collect();
        let wt: Vec<f32> = (0..oc * ic * 9)
            .map(|i| (((i as f32) * 0.29) % 2.0) - 1.0)
            .collect();
        let b: Vec<f32> = vec![0.05, -0.1];
        let add: Vec<f32> = (0..ic * h * w)
            .map(|i| (((i as f32) * 0.13) % 1.0) - 0.5)
            .collect();

        // Reference: upsample, add, then conv3x3_pad1_relu_out.
        let mut up = naive_upsample_ac(&x, ic, h_lo, w_lo, h, w);
        for (a, b) in up.iter_mut().zip(add.iter()) {
            *a += *b;
        }
        let naive_conv = naive_conv3x3_pad1(&up, &wt, &b, n, ic, h, w, oc);
        let expected: Vec<f32> = naive_conv.iter().map(|v| v.max(0.0)).collect();

        let fused =
            conv3x3_pad1_relu_out_upsample(&x, &wt, &b, n, ic, h_lo, w_lo, oc, h, w, Some(&add));
        assert! {
            approx_eq(&expected, &fused, 1e-3),
            "upsample fusion with add mismatch"
        };
    }

    #[test]
    fn upsample_fusion_identity_filter() {
        // With an identity 3×3 filter (center=1), conv(upsample(x)) == upsample(x)
        // at interior pixels. Useful for localising bugs in the fusion math
        // (independent of conv correctness, which is covered elsewhere).
        let n = 1;
        let ic = 2;
        let h_lo = 8;
        let w_lo = 8;
        let h = 15;
        let w = 15;
        let oc = 2;
        let x: Vec<f32> = (0..n * ic * h_lo * w_lo)
            .map(|i| (((i as f32) * 0.19) % 2.0) - 1.0)
            .collect();
        // Identity filter for each (oc, ic) pair: only matching ic→oc passes
        // through; cross-channel terms are zero. So with oc==ic and identity
        // filters on the diagonal, output[c] = upsample(x[c]) at interior.
        let mut wt = vec![0.0f32; oc * ic * 9];
        for c in 0..ic.min(oc) {
            wt[(c * ic + c) * 9 + 4] = 1.0; // center tap
        }
        let b: Vec<f32> = vec![0.0; oc];

        let fused = conv3x3_pad1_relu_out_upsample(&x, &wt, &b, n, ic, h_lo, w_lo, oc, h, w, None);
        let up = naive_upsample_ac(&x, ic, h_lo, w_lo, h, w);

        // Check interior pixels (where the conv doesn't read zero-padding).
        // F(4,3) tolerance is 1e-3 due to the 1/6, 1/24 transform fractions.
        for c in 0..oc.min(ic) {
            for oy in 1..h - 1 {
                for ox in 1..w - 1 {
                    let idx = c * h * w + oy * w + ox;
                    let expected = up[idx].max(0.0);
                    let got = fused[idx];
                    assert! {
                        (got - expected).abs() < 1e-3,
                        "identity-filter mismatch at c={}, oy={}, ox={}: got={}, expected={}",
                        c, oy, ox, got, expected
                    };
                }
            }
        }
    }

    // --- conv1x1_upsample tests ---
    //
    // The fused `conv1x1(upsample(x))` must match the unfused reference of:
    //   1. materialise `align_corners=true` bilinear upsample to (h, w)
    //   2. run conv1x1 on the result

    #[test]
    fn conv1x1_upsample_matches_materialized() {
        // Compare fused `conv1x1(upsample(x))` vs unfused reference.
        let n = 1;
        let ic = 4;
        let h_lo = 12;
        let w_lo = 18;
        // Non-integer scale (mirrors the DA3-BASE rn1 fusion stage 96×144→192×288).
        let h = 21;
        let w = 32;
        let oc = 5;
        let x: Vec<f32> = (0..n * ic * h_lo * w_lo)
            .map(|i| (((i as f32) * 0.37) % 2.0) - 1.0)
            .collect();
        let wt: Vec<f32> = (0..oc * ic)
            .map(|i| (((i as f32) * 0.71) % 2.0) - 1.0)
            .collect();
        let b: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4, -0.5];

        let up = naive_upsample_ac(&x, ic, h_lo, w_lo, h, w);
        let expected = naive_conv1x1(&up, &wt, &b, n, ic, h, w, oc);

        let fused = conv1x1_upsample(&x, &wt, &b, n, ic, h_lo, w_lo, oc, h, w);
        assert! {
            approx_eq(&expected, &fused, 1e-4),
            "conv1x1_upsample vs materialised mismatch"
        };
    }

    #[test]
    fn conv1x1_upsample_identity() {
        // With identity weights (oc==ic, unit diagonal), fused output must
        // equal the upsampled input exactly (no conv mixing, zero bias).
        let n = 1;
        let ic = 3;
        let h_lo = 8;
        let w_lo = 8;
        let h = 15;
        let w = 15;
        let oc = 3;
        let x: Vec<f32> = (0..n * ic * h_lo * w_lo)
            .map(|i| (((i as f32) * 0.19) % 2.0) - 1.0)
            .collect();
        let mut wt = vec![0.0f32; oc * ic];
        for c in 0..ic.min(oc) {
            wt[c * ic + c] = 1.0;
        }
        let b: Vec<f32> = vec![0.0; oc];

        let fused = conv1x1_upsample(&x, &wt, &b, n, ic, h_lo, w_lo, oc, h, w);
        let expected = naive_upsample_ac(&x, ic, h_lo, w_lo, h, w);
        // 1×1 conv is exact (no Winograd transforms), so tolerance is tight.
        assert! {
            approx_eq(&expected, &fused, 1e-6),
            "conv1x1_upsample identity mismatch"
        };
    }

    #[test]
    fn conv1x1_upsample_no_scale() {
        // When `h==h_lo && w==w_lo`, the upsample is identity. The fused path
        // must still produce the correct conv1x1 result (exercises the
        // same-size branch in the spec).
        let n = 1;
        let ic = 4;
        let h = 12;
        let w = 18;
        let oc = 3;
        let x: Vec<f32> = (0..n * ic * h * w)
            .map(|i| (((i as f32) * 0.5) % 2.0) - 1.0)
            .collect();
        let wt: Vec<f32> = (0..oc * ic)
            .map(|i| (((i as f32) * 0.43) % 2.0) - 1.0)
            .collect();
        let b: Vec<f32> = vec![0.2, -0.1, 0.05];

        let expected = naive_conv1x1(&x, &wt, &b, n, ic, h, w, oc);
        let fused = conv1x1_upsample(&x, &wt, &b, n, ic, h, w, oc, h, w);
        assert! {
            approx_eq(&expected, &fused, 1e-5),
            "conv1x1_upsample no-scale mismatch"
        };
    }
}
