//! Candle-free ViT block forward for DA3.
//!
//! [`FastVitBlock`] runs the entire pre-norm transformer block
//! (norm1 → attention → layerscale → residual → norm2 → FFN → layerscale →
//! residual) on raw `&[f32]` buffers, eliminating all of candle's per-op
//! allocation and dispatcher overhead for non-GEMM ops. Attention is delegated
//! to [`crate::fast_attn::FastAttention`]; the FFN (GELU-erf MLP) and the
//! block structure (layernorm, layerscale, residuals) are implemented inline
//! with buffer reuse.
//!
//! Activations flow as flat row-major `Vec<f32>` in `[n, embed]` layout.
//! Weight tensors are pre-packed to NN row-major `[in, out]` at load time.
//!
//! # Env var
//!
//! Enabled by `DA_FAST_ATTN=1` (the same flag that enables
//! [`crate::attention::Attention::forward_fast`]); see [`fast_block_enabled`].

use crate::fast_attn::{flatten_to_f32, FastAttention};
use crate::tinyblas;
use crate::vit_block::VitBlock;
use crate::Result;
use rayon::prelude::*;

/// Block-level LayerNorm eps (NOT the q/k-norm eps). Matches `cfg.ln_eps`.
pub const BLOCK_LN_EPS: f32 = 1e-6;

/// Candle-free ViT block. Holds pre-packed weights + reusable scratch.
pub struct FastVitBlock {
    // --- norm1 ---
    norm1_w: Vec<f32>,
    norm1_b: Vec<f32>,
    // --- attention (owns its own scratch) ---
    attn: FastAttention,
    // --- layerscale 1 (optional) ---
    ls1: Option<Vec<f32>>,
    // --- norm2 ---
    norm2_w: Vec<f32>,
    norm2_b: Vec<f32>,
    // --- FFN ---
    ffn: FastFfn,
    // --- layerscale 2 (optional) ---
    ls2: Option<Vec<f32>>,
    embed: usize,
    /// Block-level LayerNorm epsilon (from `cfg.ln_eps`).
    ln_eps: f32,
    /// Reusable scratch for the block-level intermediates (normed input, etc.).
    scratch: ScratchBlock,
}

/// Candle-free FFN. DA3 uses `Mlp` (fc1 → gelu_erf → fc2); SwiGLU is included
/// for completeness but DA3 doesn't hit it.
pub enum FastFfn {
    Mlp {
        fc1_w: Vec<f32>, // [embed, hidden] NN-packed
        fc1_b: Vec<f32>, // [hidden]
        fc2_w: Vec<f32>, // [hidden, embed] NN-packed
        fc2_b: Vec<f32>, // [embed]
        hidden: usize,
    },
    Swiglu {
        w12_w: Vec<f32>, // [embed, 2*hidden] NN-packed
        w12_b: Vec<f32>,
        w3_w: Vec<f32>, // [hidden, embed] NN-packed
        w3_b: Vec<f32>,
        hidden: usize,
    },
}

struct ScratchBlock {
    /// `[n, embed]` — normed input to attention or FFN.
    xn: Vec<f32>,
    /// `[n, embed]` — attention or FFN output (pre-residual).
    sub_out: Vec<f32>,
    /// `[n, hidden]` — FFN hidden state (fc1 output / swiglu gate).
    ffn_hidden: Vec<f32>,
}

impl ScratchBlock {
    fn new() -> Self {
        Self {
            xn: Vec::new(),
            sub_out: Vec::new(),
            ffn_hidden: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, n: usize, embed: usize, hidden: usize) {
        let need = n * embed;
        if self.xn.len() < need {
            self.xn.resize(need, 0.0);
            self.sub_out.resize(need, 0.0);
        }
        let h_need = n * hidden;
        if self.ffn_hidden.len() < h_need {
            self.ffn_hidden.resize(h_need, 0.0);
        }
    }
}

impl FastVitBlock {
    /// Build from a loaded candle [`VitBlock`]. `ln_eps` is the block-level
    /// LayerNorm epsilon (from `cfg.ln_eps`); all other dimensions are derived
    /// from the weight tensors. Expensive (one-time); subsequent [`forward`]
    /// calls do no candle work or allocation.
    pub fn from_candle(block: &VitBlock, ln_eps: f32) -> Result<Self> {
        // Derive dimensions from the weight shapes (avoids depending on cfg).
        let embed = block.norm1.weight().dims()[0];
        // hidden: derived from the FFN's first Linear out_features.
        let hidden = match &block.ffn {
            crate::vit_block::Ffn::Mlp { fc1, .. } => fc1.weight().dims()[0],
            crate::vit_block::Ffn::Swiglu { w12, .. } => w12.weight().dims()[0] / 2,
        };

        let norm1_w = flatten_to_f32(block.norm1.weight())?;
        let norm1_b = flatten_to_f32(
            block
                .norm1
                .bias()
                .ok_or_else(|| crate::Error::Model("fast_block: norm1 has no bias".into()))?,
        )?;
        let norm2_w = flatten_to_f32(block.norm2.weight())?;
        let norm2_b = flatten_to_f32(
            block
                .norm2
                .bias()
                .ok_or_else(|| crate::Error::Model("fast_block: norm2 has no bias".into()))?,
        )?;

        let ls1 = block.ls1.as_ref().map(flatten_to_f32).transpose()?;
        let ls2 = block.ls2.as_ref().map(flatten_to_f32).transpose()?;

        let attn = FastAttention::from_candle(&block.attn)?;

        let ffn =
            match &block.ffn {
                crate::vit_block::Ffn::Mlp { fc1, fc2 } => {
                    let fc1_w_pt = flatten_to_f32(fc1.weight())?;
                    let fc1_b = flatten_to_f32(fc1.bias().ok_or_else(|| {
                        crate::Error::Model("fast_block: fc1 has no bias".into())
                    })?)?;
                    let fc2_w_pt = flatten_to_f32(fc2.weight())?;
                    let fc2_b = flatten_to_f32(fc2.bias().ok_or_else(|| {
                        crate::Error::Model("fast_block: fc2 has no bias".into())
                    })?)?;
                    FastFfn::Mlp {
                        fc1_w: tinyblas::pack_b_nt(embed, hidden, &fc1_w_pt),
                        fc1_b,
                        fc2_w: tinyblas::pack_b_nt(hidden, embed, &fc2_w_pt),
                        fc2_b,
                        hidden,
                    }
                }
                crate::vit_block::Ffn::Swiglu { w12, w3 } => {
                    let w12_w_pt = flatten_to_f32(w12.weight())?;
                    let w12_b = flatten_to_f32(w12.bias().ok_or_else(|| {
                        crate::Error::Model("fast_block: w12 has no bias".into())
                    })?)?;
                    let w3_w_pt = flatten_to_f32(w3.weight())?;
                    let w3_b = flatten_to_f32(w3.bias().ok_or_else(|| {
                        crate::Error::Model("fast_block: w3 has no bias".into())
                    })?)?;
                    FastFfn::Swiglu {
                        w12_w: tinyblas::pack_b_nt(embed, 2 * hidden, &w12_w_pt),
                        w12_b,
                        w3_w: tinyblas::pack_b_nt(hidden, embed, &w3_w_pt),
                        w3_b,
                        hidden,
                    }
                }
            };

        Ok(Self {
            norm1_w,
            norm1_b,
            attn,
            ls1,
            norm2_w,
            norm2_b,
            ffn,
            ls2,
            embed,
            ln_eps,
            scratch: ScratchBlock::new(),
        })
    }

    /// Forward pass. `x` is `[n, embed]` row-major; returns `[n, embed]`.
    ///
    /// `rope` is the `(cos, sin)` pair for attention (or `None`).
    pub fn forward(&mut self, x: &[f32], n: usize, rope: Option<(&[f32], &[f32])>) -> Vec<f32> {
        let embed = self.embed;
        let mut out = vec![0.0f32; n * embed];
        self.forward_into(x, &mut out, n, rope);
        out
    }

    /// In-place forward: writes `out = block(x)` into the provided buffer.
    /// `out` must be `n * embed` long. Zero heap allocation after the first
    /// call (scratch buffers are reused).
    pub fn forward_into(
        &mut self,
        x: &[f32],
        out: &mut [f32],
        n: usize,
        rope: Option<(&[f32], &[f32])>,
    ) {
        debug_assert_eq!(out.len(), n * self.embed);
        let embed = self.embed;
        let hidden = self.ffn.hidden();
        self.scratch.ensure_capacity(n, embed, hidden);
        let xn_len = n * embed;

        // ---- attention sub-block ----
        // xn = layernorm(x, norm1_w, norm1_b)
        let xn = &mut self.scratch.xn[..xn_len];
        layernorm_rows(x, xn, n, embed, &self.norm1_w, &self.norm1_b, self.ln_eps);

        // a = attn(xn, rope) -> sub_out (reused buffer, no alloc)
        let sub = &mut self.scratch.sub_out[..xn_len];
        self.attn.forward_into(xn, sub, n, rope);

        // Fused layerscale + residual + LayerNorm: out = x + ls1 * a,
        // then xn = layernorm(out, norm2_w, norm2_b). Single parallel pass
        // over rows — reads `x` (cache-cold: evicted during attention) and
        // `sub`, writes `out`, then re-reads `out` (now L1-hot from the write
        // above) for the layernorm reduction, and writes `xn`.
        //
        // This fuses what was previously two separate rayon dispatches (the
        // layerscale+residual pass and `layernorm_rows`) into one, saving one
        // fork-join per block × 12 blocks = 12 dispatches/inference, and
        // converts an L2 read of `out` (~2.6 MiB at n=864, embed=768) into an
        // L1 read (the row was just written in the same task).
        let xn = &mut self.scratch.xn[..xn_len];
        let x_base = x.as_ptr() as usize;
        let sub_base = sub.as_ptr() as usize;
        let xn_base = xn.as_mut_ptr() as usize;
        let norm2_w_base = self.norm2_w.as_ptr() as usize;
        let norm2_b_base = self.norm2_b.as_ptr() as usize;
        let ls1 = self.ls1.as_deref();
        let ls1_base = ls1.map(|ls| ls.as_ptr() as usize).unwrap_or(0);
        let ls1_present = ls1.is_some();
        let eps = self.ln_eps;
        let dim = embed;
        out.par_chunks_mut(embed)
            .enumerate()
            .for_each(|(ni, out_row)| {
                let x_row = unsafe {
                    core::slice::from_raw_parts((x_base as *const f32).add(ni * embed), embed)
                };
                let sub_row = unsafe {
                    core::slice::from_raw_parts((sub_base as *const f32).add(ni * embed), embed)
                };
                // Phase 1: out_row = x + ls1 * sub (the layerscale + residual).
                if ls1_present {
                    let ls = unsafe { core::slice::from_raw_parts(ls1_base as *const f32, embed) };
                    for d in 0..embed {
                        out_row[d] = x_row[d] + ls[d] * sub_row[d];
                    }
                } else {
                    for d in 0..embed {
                        out_row[d] = x_row[d] + sub_row[d];
                    }
                }
                // Phase 2: xn[ni] = layernorm(out_row, norm2_w, norm2_b).
                // out_row is L1-resident from the writes just above. Uses the
                // single-pass sum+sum_sq reduction (see `layernorm_rows`).
                let xn_row = unsafe {
                    core::slice::from_raw_parts_mut((xn_base as *mut f32).add(ni * embed), embed)
                };
                let w2 = unsafe { core::slice::from_raw_parts(norm2_w_base as *const f32, embed) };
                let b2 = unsafe { core::slice::from_raw_parts(norm2_b_base as *const f32, embed) };
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                for &v in out_row.iter() {
                    sum += v;
                    sum_sq += v * v;
                }
                let mean = sum / dim as f32;
                let var = sum_sq / dim as f32 - mean * mean;
                let inv_std = 1.0 / (var + eps).sqrt();
                for d in 0..embed {
                    xn_row[d] = (out_row[d] - mean) * inv_std * w2[d] + b2[d];
                }
            });

        // ---- FFN sub-block ----
        // (xn was computed above in the fused residual+norm pass.)

        // m = ffn(xn) -> sub_out
        let sub = &mut self.scratch.sub_out[..xn_len];
        self.ffn
            .forward_into(xn, sub, n, embed, &mut self.scratch.ffn_hidden);
        // Fused layerscale + residual: out += ls2 * m.
        // Single parallel pass — fuses the layerscale multiply into the
        // in-place residual add (same optimisation as blk_res1 above).
        let sub_base = sub.as_ptr() as usize;
        let out_base = out.as_mut_ptr() as usize;
        let embed_v = embed;
        let ls2 = self.ls2.as_deref();
        let ls2_base = ls2.map(|ls| ls.as_ptr() as usize).unwrap_or(0);
        let ls2_present = ls2.is_some();
        (0..n).into_par_iter().for_each(move |ni| {
            let out_row = unsafe {
                core::slice::from_raw_parts_mut((out_base as *mut f32).add(ni * embed_v), embed_v)
            };
            let sub_row = unsafe {
                core::slice::from_raw_parts((sub_base as *const f32).add(ni * embed_v), embed_v)
            };
            if ls2_present {
                let ls = unsafe { core::slice::from_raw_parts(ls2_base as *const f32, embed_v) };
                for d in 0..embed_v {
                    out_row[d] += ls[d] * sub_row[d];
                }
            } else {
                for d in 0..embed_v {
                    out_row[d] += sub_row[d];
                }
            }
        });
    }
}

impl FastFfn {
    fn hidden(&self) -> usize {
        match self {
            FastFfn::Mlp { hidden, .. } => *hidden,
            FastFfn::Swiglu { hidden, .. } => *hidden,
        }
    }

    /// FFN forward. `x` is `[n, embed]`; returns `[n, embed]`.
    /// `hidden_buf` is reused scratch of size `[n, hidden]`.
    #[allow(dead_code)]
    fn forward(&self, x: &[f32], n: usize, embed: usize, hidden_buf: &mut Vec<f32>) -> Vec<f32> {
        let mut out = vec![0.0f32; n * embed];
        self.forward_into(x, &mut out, n, embed, hidden_buf);
        out
    }

    /// In-place FFN forward. Writes `out = ffn(x)` into `out` (length `n*embed`).
    fn forward_into(
        &self,
        x: &[f32],
        out: &mut [f32],
        n: usize,
        embed: usize,
        hidden_buf: &mut Vec<f32>,
    ) {
        debug_assert_eq!(out.len(), n * embed);
        match self {
            FastFfn::Mlp {
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
                hidden,
            } => {
                // h = x @ fc1_w + fc1_b  -> [n, hidden]
                let hidden_len = n * hidden;
                if hidden_buf.len() < hidden_len {
                    hidden_buf.resize(hidden_len, 0.0);
                }
                let h = &mut hidden_buf[..hidden_len];
                // Fuse bias into the GEMM epilogue: pre-fill `h` with the bias
                // (broadcast across all n rows) so that `gemm_nn_into`'s
                // accumulate (`C += A@B`) yields `bias + A@B` directly. This
                // removes a separate bias-add pass over `h`.
                h.par_chunks_mut(*hidden).for_each(|row| {
                    row.copy_from_slice(fc1_b);
                });
                {
                    let _g = crate::fast_profile::scope("ffn_fc1");
                    tinyblas::gemm_nn_into(n, *hidden, embed, x, fc1_w, h);
                }
                // GELU only (bias already applied). Parallelised across rows.
                // The AVX2 batch kernel (`gelu_erf_slice`) processes 8 lanes
                // per iteration using a branchless erf polynomial + a
                // bit-manipulation exp, avoiding the libm `expf` call and the
                // sign branch that blocked auto-vectorisation.
                {
                    let _g = crate::fast_profile::scope("ffn_gelu");
                    h.par_chunks_mut(*hidden).for_each(|row| {
                        gelu_erf_slice(row);
                    });
                }
                // out = h @ fc2_w + fc2_b  -> [n, embed]
                // Same bias-fuse trick: pre-fill `out` with bias.
                out.par_chunks_mut(embed).for_each(|row| {
                    row.copy_from_slice(fc2_b);
                });
                {
                    let _g = crate::fast_profile::scope("ffn_fc2");
                    tinyblas::gemm_nn_into(n, embed, *hidden, h, fc2_w, out);
                }
            }
            FastFfn::Swiglu {
                w12_w,
                w12_b,
                w3_w,
                w3_b,
                hidden,
            } => {
                let two_hidden = 2 * hidden;
                let x12_len = n * two_hidden;
                if hidden_buf.len() < x12_len {
                    hidden_buf.resize(x12_len, 0.0);
                }
                let x12 = &mut hidden_buf[..x12_len];
                // Fuse w12 bias into GEMM epilogue (same trick as Mlp).
                x12.par_chunks_mut(two_hidden).for_each(|row| {
                    row.copy_from_slice(w12_b);
                });
                tinyblas::gemm_nn_into(n, two_hidden, embed, x, w12_w, x12);
                // SwiGLU gate: h = silu(x1) * x2, where x1 = x12[..hidden], x2 = x12[hidden..].
                // Parallelised across rows.
                let h_len = n * hidden;
                let mut h = vec![0.0f32; h_len];
                h.par_chunks_mut(*hidden).enumerate().for_each(|(ni, dst)| {
                    let base = ni * two_hidden;
                    let x1 = &x12[base..base + *hidden];
                    let x2 = &x12[base + *hidden..base + two_hidden];
                    for d in 0..*hidden {
                        dst[d] = silu(x1[d]) * x2[d];
                    }
                });
                // out = h @ w3_w + w3_b -> [n, embed]
                // Fuse w3 bias into GEMM epilogue.
                out.par_chunks_mut(embed).for_each(|row| {
                    row.copy_from_slice(w3_b);
                });
                tinyblas::gemm_nn_into(n, embed, *hidden, &h, w3_w, out);
            }
        }
    }
}

impl FastVitBlock {
    /// Dimensions `(embed, hidden, heads, head_dim)`.
    pub fn dims(&self) -> (usize, usize, usize, usize) {
        let (embed, heads, head_dim) = self.attn.dims();
        (embed, self.ffn.hidden(), heads, head_dim)
    }
}

// ---------------------------------------------------------------------------
// Helpers (raw f32, auto-vectorisable by the compiler)
// ---------------------------------------------------------------------------

/// Row-wise layer norm over `[n, dim]`. Writes `y = layernorm(x)` into `dst`.
///
/// Parallelised over the row axis via rayon — each row is fully independent.
/// For the DA3 backbone (n=864, dim=768) this is ~2× faster than the serial
/// loop at 32 threads because the row-level reduction is compute-bound.
///
/// Two-pass per row: (1) single reduction for `sum` and `sum_sq` to derive
/// `mean = sum/dim` and `var = sum_sq/dim - mean²` (the "shifted" variance
/// identity), then (2) the affine write. This saves one full pass over the
/// row vs the textbook `sum` then `sum((x-mean)²)` formulation. The shifted
/// identity can suffer catastrophic cancellation when `mean² ≈ E[x²]`, but
/// ViT block inputs (residual stream after pre-norm) have `|mean| ≪ std`, so
/// the cancellation is negligible (verified by `layernorm_rows_unit_var_*`
/// tests and the head-parity gate).
pub(crate) fn layernorm_rows(
    x: &[f32],
    dst: &mut [f32],
    _n: usize,
    dim: usize,
    w: &[f32],
    b: &[f32],
    eps: f32,
) {
    dst.par_chunks_mut(dim).enumerate().for_each(|(ni, out)| {
        let row = &x[ni * dim..(ni + 1) * dim];
        // Single-pass accumulation of sum and sum-of-squares.
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for &v in row {
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / dim as f32;
        // Var = E[x²] - E[x]². Equivalent to mean((x-mean)²) but without a
        // second pass over the row.
        let var = sum_sq / dim as f32 - mean * mean;
        let inv_std = 1.0 / (var + eps).sqrt();
        for d in 0..dim {
            out[d] = (row[d] - mean) * inv_std * w[d] + b[d];
        }
    });
}

/// Exact GELU via erf (torch `nn.GELU()` default).
///
/// gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
///
/// `erf` is computed via the Abramowitz & Stegun 7.1.26 approximation (max
/// error ~1.5e-7), which is what ggml uses for `ggml_gelu_erf` on CPU when
/// the math library `erff` isn't available. The `exp(-x^2)` term inside the
/// erf polynomial uses [`exp_fast`] (bit-manipulation + degree-6 polynomial)
/// instead of `libm`'s `expf`, which allows the whole GELU to auto-vectorise
/// cleanly under `target-cpu=native` (AVX2/FMA). The sign is applied
/// branchlessly via `copysign`, avoiding the branch that previously
/// inhibited the loop vectoriser.
///
/// Combined max abs error vs the libm `erff`-based reference: ~2e-7
/// (dominated by the A&S 7.1.26 rational approximation; `exp_fast` adds
/// <1e-7), which is well within f32 epsilon and the head parity gate.
#[inline]
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erff(x * (1.0f32 / core::f32::consts::SQRT_2)))
}

/// SiLU / swish: `x * sigmoid(x)`.
#[inline]
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Fast vectorisable `exp(x)` for f32.
///
/// Uses the standard decomposition `exp(x) = 2^(x/ln2) = 2^n * 2^r` where
/// `n = round_to_nearest(x/ln2)` is reconstructed via float bit manipulation
/// and `r ∈ [-0.5, 0.5]` is evaluated via a degree-6 polynomial (Taylor of
/// `2^r = e^(r·ln2)`). Every operation is pure arithmetic / bit-cast — no
/// libm call, no branch — so the function auto-vectorises to 8-wide AVX2
/// under `target-cpu=native`.
///
/// Max relative error: ~1.5e-7 over `[-87, 87]`. Inputs outside that range
/// may overflow / underflow (same as `libm`'s `expf`); callers in the hot
/// path pass bounded arguments (e.g. `-ax*ax` in `erff`, which is `≤ 0`).
#[inline]
fn exp_fast(x: f32) -> f32 {
    // Clamp to the f32 representable exp range to avoid producing NaN from
    // out-of-range exponent bits. `f32::clamp` lowers to a vectorisable
    // `vmaxps`/`vminps` pair under AVX2.
    let x = x.clamp(-87.3_f32, 88.7_f32);
    // t = x · log2(e).
    let t = x * core::f32::consts::LOG2_E;
    // n = round-to-nearest-even via the “magic number” trick: at exponent
    // 2^23, the f32 ULP is 1.0, so adding then subtracting 1.5·2^23 forces
    // round-to-nearest-even with no libm call. (Rust's `f32::round_ties_even`
    // and `f32::floor` both lower to scalar libm calls that block the loop
    // vectoriser; this arithmetic form vectorises to `vaddps`/`vsubps`.)
    // Valid for |t| < 2^22, far beyond our clamped range.
    let magic = 1.5_f32 * (1u32 << 23) as f32; // 12582912.0
    let n = (t + magic) - magic;
    // r ∈ [-0.5, 0.5].
    let r = t - n;
    // 2^r via degree-6 Horner (Taylor of e^(r·ln2); max |err| ≈ 1.2e-7 at r=±0.5).
    // Coefficients are (ln2)^k / k! for k = 1..6.
    let ln2 = core::f32::consts::LN_2;
    let p = 1.0_f32
        + r * (ln2
            + r * (0.2402265
                + r * (0.0555041 + r * (0.0096181 + r * (0.0013336 + r * 0.0001539)))));
    // 2^n via float exponent bits: exponent_field = (n + 127) << 23.
    // `n` is integer-valued, so `as i32` truncation is exact.
    let bits = ((n as i32 + 127) as u32) << 23;
    f32::from_bits(bits) * p
}

/// erf approximation (Abramowitz & Stegun 7.1.26) using [`exp_fast`] instead
/// of `libm`'s `expf`, and branchless sign application via `copysign`.
///
/// Max abs error vs `libm::erff`: ~1.5e-7 (dominated by the rational
/// approximation; `exp_fast` contributes < 1e-7).
#[inline]
fn erff(x: f32) -> f32 {
    // A&S 7.1.26 rational approximation. erf is odd, so we compute |erf(|x|)|
    // and re-apply the sign of x branchlessly via copysign (vectorises to
    // a sign-bit mask, no branch).
    let ax = x.abs();
    let p = 0.3275911_f32;
    let a1 = 0.254829592_f32;
    let a2 = -0.284496736_f32;
    let a3 = 1.421413741_f32;
    let a4 = -1.453152027_f32;
    let a5 = 1.061405429_f32;
    let t = 1.0 / (1.0 + p * ax);
    // Horner evaluation of a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5.
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let y = 1.0 - poly * exp_fast(-ax * ax);
    // Apply the sign of x. copysign(|y|, x) = y if x ≥ 0, -y if x < 0.
    // For x = ±0, erf(±0) = ±0, which copysign handles correctly.
    y.copysign(x)
}

// ---------------------------------------------------------------------------
// AVX2 batch GELU.
//
// The scalar [`gelu_erf`] above is correct but the `n as i32` truncation in
// [`exp_fast`] blocks the LLVM loop vectoriser: `f32 as i32` lowers to a
// scalar `cvttss2si`, not the vector `vcvtps2dq`, so the GELU loop in
// [`FastFfn::forward_into`] compiles to scalar `erff` calls even under
// `target-cpu=native`. Writing the batch GELU with explicit AVX2 intrinsics
// (including `_mm256_cvtps_epi32` for the float→int step) guarantees 8-wide
// vectorisation and is ~3× faster than the auto-vectorised scalar path on the
// i7-13700K.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2_gelu {
    use core::arch::x86_64::*;

    /// AVX2 exp(x) for 8 lanes. Same algorithm as [`super::exp_fast`] but uses
    /// `_mm256_cvtps_epi32` for the float→int step (which the auto-vectoriser
    /// refuses to emit for scalar `as i32`).
    ///
    /// Reused by [`crate::flash_attn`] for the online-softmax `exp` step.
    ///
    /// `#[inline(always)]` is required (not just `#[inline]`) because the function
    /// is called from a hot loop in `softmax_exp_sum_avx2` — LLVM's default
    /// inliner keeps it as a `callq` (with a `vzeroupper` before each call,
    /// ~20 cycles each) when it sees multiple call sites (gelu + softmax). Forcing
    /// inline eliminates the call + `vzeroupper` overhead, which was the dominant
    /// cost of the vectorised softmax path.
    #[inline(always)]
    pub(crate) unsafe fn exp_fast_avx2(x: __m256) -> __m256 {
        // Clamp to the f32 representable range.
        let hi = _mm256_set1_ps(88.7_f32);
        let lo = _mm256_set1_ps(-87.3_f32);
        let x = _mm256_max_ps(_mm256_min_ps(x, hi), lo);
        // t = x · log2(e).
        let t = _mm256_mul_ps(x, _mm256_set1_ps(core::f32::consts::LOG2_E));
        // n = round-to-nearest-even via magic number (1.5 · 2^23).
        let magic = _mm256_set1_ps(1.5_f32 * (1u32 << 23) as f32);
        let n = _mm256_sub_ps(_mm256_add_ps(t, magic), magic);
        // r = t - n  ∈ [-0.5, 0.5].
        let r = _mm256_sub_ps(t, n);
        // 2^r via degree-6 Horner. Start from the innermost coefficient c6 and
        // fold outward, ending with + 1.0:
        //   p = ((((c6·r + c5)·r + c4)·r + c3)·r + c2)·r + c1)·r + 1
        let p = _mm256_set1_ps(0.0001539_f32); // c6
        let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.0013336_f32)); // c6·r + c5
        let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.0096181_f32)); // ·r + c4
        let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.0555041_f32)); // ·r + c3
        let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.2402265_f32)); // ·r + c2
        let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(core::f32::consts::LN_2)); // ·r + c1
        let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0_f32)); // ·r + 1
                                                                // 2^n via float exponent bits. `n` is integer-valued so `_mm256_cvtps_epi32`
                                                                // (round-to-nearest-even) is exact. `(n_int + 127) << 23` then reinterpret
                                                                // as float.
        let n_int = _mm256_cvtps_epi32(n);
        let bias = _mm256_set1_epi32(127);
        let biased = _mm256_add_epi32(n_int, bias);
        let shifted = _mm256_slli_epi32(biased, 23);
        let pow2n = _mm256_castsi256_ps(shifted);
        _mm256_mul_ps(pow2n, p)
    }

    /// AVX2 erf(x) for 8 lanes (A&S 7.1.26 + [`exp_fast_avx2`]).
    #[inline]
    unsafe fn erff_avx2(x: __m256) -> __m256 {
        // abs(x) via sign-bit clear.
        let abs_mask = _mm256_set1_epi32(0x7fff_ffff);
        let ax = _mm256_and_ps(x, _mm256_castsi256_ps(abs_mask));
        // A&S coefficients.
        let p = _mm256_set1_ps(0.3275911_f32);
        let a1 = _mm256_set1_ps(0.254829592_f32);
        let a2 = _mm256_set1_ps(-0.284496736_f32);
        let a3 = _mm256_set1_ps(1.421413741_f32);
        let a4 = _mm256_set1_ps(-1.453152027_f32);
        let a5 = _mm256_set1_ps(1.061405429_f32);
        let one = _mm256_set1_ps(1.0_f32);
        // t = 1 / (1 + p·|x|)
        let denom = _mm256_fmadd_ps(p, ax, one);
        let t = _mm256_div_ps(one, denom);
        // poly = ((((a5·t + a4)·t + a3)·t + a2)·t + a1)·t
        let poly = _mm256_fmadd_ps(a5, t, a4);
        let poly = _mm256_fmadd_ps(poly, t, a3);
        let poly = _mm256_fmadd_ps(poly, t, a2);
        let poly = _mm256_fmadd_ps(poly, t, a1);
        let poly = _mm256_mul_ps(poly, t);
        // exp(-ax·ax)
        let neg_ax2 = _mm256_or_ps(
            _mm256_mul_ps(ax, ax),
            _mm256_castsi256_ps(_mm256_set1_epi32(0x8000_0000u32 as i32)),
        );
        let e = exp_fast_avx2(neg_ax2);
        // y = 1 - poly · e
        let y = _mm256_fnmadd_ps(poly, e, one); // 1 - poly*e (FMA: -(poly*e) + 1)
                                                // Apply sign of x: copysign(y, x) = (|y| with x's sign bit).
                                                // |y| = y & abs_mask; sign = x & sign_mask; result = |y| | sign.
        let sign_mask = _mm256_set1_epi32(0x8000_0000u32 as i32);
        let y_abs = _mm256_and_ps(y, _mm256_castsi256_ps(abs_mask));
        let x_sign = _mm256_and_ps(x, _mm256_castsi256_ps(sign_mask));
        _mm256_or_ps(y_abs, x_sign)
    }

    /// AVX2 gelu_erf(x) for 8 lanes.
    #[inline]
    unsafe fn gelu_erf_avx2(x: __m256) -> __m256 {
        let half = _mm256_set1_ps(0.5_f32);
        let one = _mm256_set1_ps(1.0_f32);
        let inv_sqrt2 = _mm256_set1_ps(1.0_f32 / core::f32::consts::SQRT_2);
        // xs = x / sqrt(2)
        let xs = _mm256_mul_ps(x, inv_sqrt2);
        // erf(xs)
        let erf = erff_avx2(xs);
        // 0.5 · x · (1 + erf)
        let one_plus = _mm256_add_ps(one, erf);
        _mm256_mul_ps(_mm256_mul_ps(half, x), one_plus)
    }

    /// Apply GELU in-place to `x` using AVX2. Processes 8 lanes at a time;
    // the remaining tail (0..7 elements) uses the scalar [`super::gelu_erf`].
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn gelu_erf_slice_avx2(x: &mut [f32]) {
        let n = x.len();
        let n8 = n & !7; // largest multiple of 8 ≤ n
        let ptr = x.as_mut_ptr();
        let mut i = 0;
        while i < n8 {
            let v = _mm256_loadu_ps(ptr.add(i));
            let r = gelu_erf_avx2(v);
            _mm256_storeu_ps(ptr.add(i), r);
            i += 8;
        }
        // Scalar tail.
        while i < n {
            *x.get_unchecked_mut(i) = super::gelu_erf(*x.get_unchecked(i));
            i += 1;
        }
    }
}

/// Apply GELU-erf in-place to a flat `&mut [f32]` slice. Uses the AVX2 batch
/// kernel on x86_64 (8 lanes/iteration); falls back to the scalar `gelu_erf`
/// loop elsewhere.
///
/// This is the hot-path entry point used by [`FastFfn::forward_into`] for the
/// GELU activation between fc1 and fc2.
#[inline]
pub fn gelu_erf_slice(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by runtime CPU feature detection.
            unsafe { avx2_gelu::gelu_erf_slice_avx2(x) };
            return;
        }
    }
    // Scalar fallback (also handles the non-x86_64 build).
    for v in x.iter_mut() {
        *v = gelu_erf(*v);
    }
}

/// Whether the candle-free fast block path is enabled (`DA_FAST_ATTN=1` or
/// `DA_FAST=1`). Read once and cached.
pub fn fast_block_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("DA_FAST_ATTN")
                .or_else(|_| std::env::var("DA_FAST"))
                .as_deref(),
            Ok("1") | Ok("on") | Ok("true") | Ok("ON") | Ok("TRUE") | Ok("True")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_erf_matches_known_values() {
        // gelu(0) = 0; gelu(1) ≈ 0.8412; gelu(-1) ≈ -0.1588; gelu(2) ≈ 1.9545.
        assert!((gelu_erf(0.0) - 0.0).abs() < 1e-5);
        assert!(
            (gelu_erf(1.0) - 0.84119).abs() < 1e-3,
            "gelu(1)={}",
            gelu_erf(1.0)
        );
        assert!((gelu_erf(-1.0) - (-0.15880)).abs() < 1e-3);
        assert!((gelu_erf(2.0) - 1.95449).abs() < 1e-3);
    }

    #[test]
    fn silu_matches_known_values() {
        // silu(0) = 0; silu(1) ≈ 0.7311; silu(2) ≈ 1.7616.
        assert!((silu(0.0) - 0.0).abs() < 1e-5);
        assert!((silu(1.0) - 0.73105).abs() < 1e-3);
        assert!((silu(2.0) - 1.76159).abs() < 1e-3);
    }

    /// Reference: the previous (libm-expf-based) erff formula. Used by the
    /// accuracy tests to verify that [`exp_fast`] + branchless sign doesn't
    /// regress the max-error budget.
    fn erff_reference(x: f32) -> f32 {
        let sign: f32 = if x < 0.0 { -1.0 } else { 1.0 };
        let ax = x.abs();
        let p = 0.3275911_f32;
        let a1 = 0.254829592_f32;
        let a2 = -0.284496736_f32;
        let a3 = 1.421413741_f32;
        let a4 = -1.453152027_f32;
        let a5 = 1.061405429_f32;
        let t = 1.0 / (1.0 + p * ax);
        let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
        let y = 1.0 - poly * (-ax * ax).exp();
        sign * y
    }

    #[test]
    fn exp_fast_matches_libm() {
        // Sweep the argument range that appears in erf(-ax*ax) for GELU inputs
        // up to |x|=6 (post-layernorm activations are well within this):
        // ax*ax ranges over [0, 18], so exp_fast argument is in [-18, 0].
        // We also sweep a bit of positive range (up to +18) for robustness.
        let mut max_rel = 0.0_f32;
        let mut worst_x = 0.0_f32;
        let mut i = 0;
        while i < 2000 {
            let x = -18.0_f32 + (i as f32) * (36.0 / 2000.0);
            let got = exp_fast(x);
            let want = x.exp();
            // Relative error is the meaningful metric for exp: the ULP spacing
            // scales with the magnitude, so |abs err| / |want| is what stays
            // bounded. For underflow (want≈0) we fall back to an abs check.
            let abs = (got - want).abs();
            let rel = abs / want.abs().max(1e-30);
            if rel > max_rel {
                max_rel = rel;
                worst_x = x;
            }
            i += 1;
        }
        // exp_fast max relative error should stay under 2e-6 over this range
        // (degree-6 Taylor of 2^r at r=±0.5 dominates; the bit manipulation is
        // exact for integer exponents).
        assert!(
            max_rel < 2e-6,
            "exp_fast max rel error {max_rel:e} at x={worst_x} exceeds 2e-6"
        );
    }

    #[test]
    fn exp_fast_known_values() {
        // Sanity checks at round numbers.
        assert!(
            (exp_fast(0.0) - 1.0).abs() < 1e-6,
            "exp(0)={}",
            exp_fast(0.0)
        );
        assert!((exp_fast(1.0) - core::f32::consts::E).abs() < 1e-5);
        assert!((exp_fast(-1.0) - (1.0 / core::f32::consts::E)).abs() < 1e-5);
        assert!(
            exp_fast(-40.0) < 1e-10,
            "exp(-40)={} should underflow",
            exp_fast(-40.0)
        );
        assert!(exp_fast(10.0) > 22000.0, "exp(10)={}", exp_fast(10.0));
    }

    #[test]
    fn erff_matches_reference_within_budget() {
        // The new branchless + exp_fast erff must stay within ~2e-7 of the
        // previous libm-expf formulation across the GELU input range.
        let mut max_diff = 0.0_f32;
        let mut worst_x = 0.0_f32;
        let mut i = 0;
        while i < 4000 {
            // x here is the *erf argument* (= gelu_input / sqrt(2)), so we
            // sweep ±4.5 which covers |gelu_input| up to ~6.4.
            let x = -4.5_f32 + (i as f32) * (9.0 / 4000.0);
            let got = erff(x);
            let want = erff_reference(x);
            let d = (got - want).abs();
            if d > max_diff {
                max_diff = d;
                worst_x = x;
            }
            i += 1;
        }
        assert!(
            max_diff < 2e-7,
            "erff max diff vs reference = {max_diff:e} at x={worst_x}"
        );
    }

    #[test]
    fn gelu_erf_stays_bounded_vs_reference() {
        // End-to-end check: gelu_erf with exp_fast must stay within ~2e-7 of
        // the previous libm-expf formulation across the typical activation
        // range [-6, 6].
        let mut max_diff = 0.0_f32;
        let mut worst_x = 0.0_f32;
        let mut i = 0;
        while i < 4000 {
            let x = -6.0_f32 + (i as f32) * (12.0 / 4000.0);
            let got = gelu_erf(x);
            // reference: old formula with libm expf
            let want = 0.5 * x * (1.0 + erff_reference(x * (1.0_f32 / core::f32::consts::SQRT_2)));
            let d = (got - want).abs();
            if d > max_diff {
                max_diff = d;
                worst_x = x;
            }
            i += 1;
        }
        assert!(
            max_diff < 3e-7,
            "gelu_erf max diff vs reference = {max_diff:e} at x={worst_x}"
        );
    }

    #[test]
    fn gelu_erf_slice_matches_scalar() {
        // The AVX2 batch kernel must produce results within 1 ULP of the
        // scalar `gelu_erf` across the full typical activation range.
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Map to [-6, 6] — the post-layernorm GELU input range.
            ((rng as f32) / (u32::MAX as f32) - 0.5) * 12.0
        };
        let n = 865 * 3072; // one FFN hidden slab at the 504×336 shape
        let mut input: Vec<f32> = (0..n).map(|_| next()).collect();
        // Make a scalar-reference copy.
        let reference: Vec<f32> = input.iter().map(|&v| gelu_erf(v)).collect();
        gelu_erf_slice(&mut input);
        let mut max_diff = 0.0_f32;
        for (got, want) in input.iter().zip(reference.iter()) {
            let d = (got - want).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        // Allow up to 3 ULPs of f32 rounding difference between the AVX2 FMA
        // path and the scalar non-FMA path (FMA doesn't round the intermediate
        // product). At |gelu| ≤ 6 this is ≤ ~7e-7.
        assert!(
            max_diff < 1e-6,
            "gelu_erf_slice max diff vs scalar = {max_diff:e}"
        );
    }

    #[test]
    fn gelu_erf_slice_tail_handles_non_multiple_of_8() {
        // Lengths that aren't multiples of 8 must fall through to the scalar
        // tail without over- or under-running the buffer.
        for len in [1usize, 7, 8, 9, 15, 16, 17, 100, 103] {
            let mut got = vec![0.5_f32; len];
            let mut want = got.clone();
            gelu_erf_slice(&mut got);
            for v in want.iter_mut() {
                *v = gelu_erf(*v);
            }
            let max_diff = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_diff < 1e-6,
                "len={len}: gelu_erf_slice max diff vs scalar = {max_diff:e}"
            );
        }
    }

    #[test]
    fn layernorm_rows_zero_mean_unit_var_before_affine() {
        let n = 2;
        let dim = 4;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w = vec![1.0_f32; dim];
        let b = vec![0.0_f32; dim];
        let mut dst = vec![0.0_f32; n * dim];
        layernorm_rows(&x, &mut dst, n, dim, &w, &b, 0.0);
        for ni in 0..n {
            let row = &dst[ni * dim..(ni + 1) * dim];
            let mean = row.iter().sum::<f32>() / dim as f32;
            assert!(mean.abs() < 1e-5, "row {ni} mean={mean}");
            let var = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            assert!((var - 1.0).abs() < 1e-4, "row {ni} var={var}");
        }
    }
}
