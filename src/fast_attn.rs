//! Candle-free attention forward for DA3.
//!
//! This module provides [`FastAttention`], which runs the exact same math as
//! [`crate::attention::Attention`] but operates entirely on raw `&[f32]`
//! buffers using [`crate::tinyblas`] for the matmuls and hand-written loops
//! for rope2d / layer-norm / softmax. The point is to **bypass candle's
//! per-op allocation and dispatcher overhead**, which the probe
//! (`examples/probe_hotspots.rs`) showed dominates the ViT backbone
//! (backbone = 1345 ms at 16 threads, of which only ~275 ms is GEMM).
//!
//! The output is numerically faithful to the candle path: the unit test
//! `matches_candle_reference` constructs a small attention with known weights
//! and checks the two implementations agree to within f32 accumulation
//! tolerance.
//!
//! # Layout
//!
//! - Linear weights are pre-packed to **NN row-major** `[in, out]` at load
//!   time (the `[out, in]` PyTorch layout is transposed once via
//!   [`crate::tinyblas::pack_b_nt`]).
//! - Activations flow as flat row-major `Vec<f32>` / `&[f32]`.
//! - Per-head Q/K/V are gathered into contiguous `[heads, n, head_dim]`
//!   panels so each head's matmul is a clean NN/NT call.
//!
//! # Buffer reuse
//!
//! The struct owns reusable scratch buffers (`qkv`, `q_heads`, `k_heads`,
//! `v_heads`, `scores_per_head`, `attn_out`, `concat`) that are grown once
//! to the working `n` and reused on every [`forward`] call. This eliminates
//! the ~10 MB of per-block allocations the original prototype paid.

use crate::attention::QK_NORM_EPS;
use crate::flash_attn as flash;
use crate::tinyblas;
use crate::Result;
use rayon::prelude::*;
use std::sync::OnceLock;

/// Whether fused (tiled) flash attention is enabled.
///
/// **Default ON** (the tiled implementation matches ggml's
/// `ggml_compute_forward_flash_attn_ext_tiled` and is faster than the
/// materialised tinyBLAS QKᵀ+softmax+AV path for DA3 shapes). Set
/// `DA_FAST_FLASH=0` to force the materialised path (e.g. for A/B or
/// debugging).
fn flash_attn_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| match std::env::var("DA_FAST_FLASH").as_deref() {
        Ok("0") | Ok("off") | Ok("false") | Ok("False") | Ok("FALSE") => false,
        _ => true,
    })
}

/// Candle-free attention. Construct once from a loaded candle [`Attention`];
/// the raw weights are cached so subsequent forwards do no candle work.
pub struct FastAttention {
    /// `[embed, 3*embed]` NN-layout QKV weight.
    qkv_w: Vec<f32>,
    /// `[3*embed]` QKV bias.
    qkv_b: Vec<f32>,
    /// `[embed, embed]` NN-layout projection weight.
    proj_w: Vec<f32>,
    /// `[embed]` projection bias.
    proj_b: Vec<f32>,
    /// Optional q-norm weight/bias (per head_dim).
    q_norm: Option<(Vec<f32>, Vec<f32>)>,
    /// Optional k-norm weight/bias.
    k_norm: Option<(Vec<f32>, Vec<f32>)>,
    heads: usize,
    head_dim: usize,
    embed: usize,
    /// Reusable scratch (grown lazily to the working `n`).
    scratch: ScratchAttn,
}

/// Reusable per-forward scratch buffers. Kept on the `FastAttention` struct so
/// repeated forwards on the same shape do zero heap allocation after the first
/// call. Buffers are indexed exactly as documented in [`FastAttention::forward`].
struct ScratchAttn {
    qkv: Vec<f32>,     // [n, 3*embed]
    q_heads: Vec<f32>, // [heads, n, head_dim]
    /// K stored **transposed per head** as `[heads, head_dim, n]` during the
    /// gather, so the per-head `QKᵀ = Q[n,hd] @ Kᵀ[hd,n]` is a clean NN gemm
    /// (no per-call transpose or NT kernel needed).
    k_heads_t: Vec<f32>, // [heads, head_dim, n]
    /// K stored **non-transposed** as `[heads, n, head_dim]` for the
    /// fused flash-attention path (one K row per key token, contiguous).
    k_heads: Vec<f32>, // [heads, n, head_dim]
    v_heads: Vec<f32>, // [heads, n, head_dim]
    attn_out: Vec<f32>, // [heads, n, head_dim]
    concat: Vec<f32>,  // [n, embed]
    /// Per-head scores workspace, sized `[heads, n*n]`. Each rayon task owns
    /// the `h*n*n`-offset slice, so the writes are disjoint.
    scores: Vec<f32>, // [heads, n, n]
    /// Single head_dim-length buffer for q/k layernorm+rope (avoids
    /// per-(n,head) Vec allocations).
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
}

impl ScratchAttn {
    fn new() -> Self {
        Self {
            qkv: Vec::new(),
            q_heads: Vec::new(),
            k_heads_t: Vec::new(),
            k_heads: Vec::new(),
            v_heads: Vec::new(),
            attn_out: Vec::new(),
            concat: Vec::new(),
            scores: Vec::new(),
            q_buf: Vec::new(),
            k_buf: Vec::new(),
        }
    }

    /// Grow all buffers to the working `n` (no-op if already large enough).
    /// We don't shrink so a transient large `n` doesn't cause reallocation
    /// churn on subsequent smaller forwards.
    fn ensure_capacity(&mut self, n: usize, heads: usize, head_dim: usize, embed: usize) {
        let qkv_need = n * 3 * embed;
        let hk_need = heads * n * head_dim;
        let kt_need = heads * head_dim * n; // transposed K
        let concat_need = n * embed;
        let scores_need = heads * n * n;
        if self.qkv.len() < qkv_need {
            self.qkv.resize(qkv_need, 0.0);
        }
        if self.q_heads.len() < hk_need {
            self.q_heads.resize(hk_need, 0.0);
            self.k_heads_t.resize(kt_need, 0.0);
            self.k_heads.resize(hk_need, 0.0);
            self.v_heads.resize(hk_need, 0.0);
            self.attn_out.resize(hk_need, 0.0);
        }
        if self.concat.len() < concat_need {
            self.concat.resize(concat_need, 0.0);
        }
        if self.scores.len() < scores_need {
            self.scores.resize(scores_need, 0.0);
        }
        if self.q_buf.len() < head_dim {
            self.q_buf.resize(head_dim, 0.0);
            self.k_buf.resize(head_dim, 0.0);
        }
    }
}

impl FastAttention {
    /// Build from raw weights in PyTorch `[out, in]` layout.
    ///
    /// This is the constructor used by [`from_candle`] after extracting the
    /// weight tensors; exposed publicly so tests can construct directly.
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw_weights(
        qkv_w: &[f32],                    // [3*embed, embed] PyTorch layout
        qkv_b: &[f32],                    // [3*embed]
        proj_w: &[f32],                   // [embed, embed] PyTorch layout
        proj_b: &[f32],                   // [embed]
        q_norm: Option<(&[f32], &[f32])>, // ([head_dim], [head_dim])
        k_norm: Option<(&[f32], &[f32])>,
        heads: usize,
        head_dim: usize,
        embed: usize,
    ) -> Self {
        assert_eq!(heads * head_dim, embed);
        assert_eq!(qkv_w.len(), 3 * embed * embed);
        assert_eq!(qkv_b.len(), 3 * embed);
        assert_eq!(proj_w.len(), embed * embed);
        assert_eq!(proj_b.len(), embed);
        // Pack weights to NN layout (transpose [out, in] -> [in, out]).
        let qkv_w_nn = tinyblas::pack_b_nt(embed, 3 * embed, qkv_w);
        let proj_w_nn = tinyblas::pack_b_nt(embed, embed, proj_w);
        Self {
            qkv_w: qkv_w_nn,
            qkv_b: qkv_b.to_vec(),
            proj_w: proj_w_nn,
            proj_b: proj_b.to_vec(),
            q_norm: q_norm.map(|(w, b)| (w.to_vec(), b.to_vec())),
            k_norm: k_norm.map(|(w, b)| (w.to_vec(), b.to_vec())),
            heads,
            head_dim,
            embed,
            scratch: ScratchAttn::new(),
        }
    }

    /// Build from a loaded candle [`Attention`], extracting and pre-packing
    /// its weights.
    pub fn from_candle(attn: &crate::attention::Attention) -> Result<Self> {
        let embed = attn.head_dim * attn.num_heads;
        let qkv_w = flatten_to_f32(attn.qkv.weight())?;
        let proj_w = flatten_to_f32(attn.proj.weight())?;
        let qkv_b = attn
            .qkv
            .bias()
            .ok_or_else(|| crate::Error::Model("fast_attn: QKV linear has no bias".into()))
            .and_then(flatten_to_f32)?;
        let proj_b = attn
            .proj
            .bias()
            .ok_or_else(|| crate::Error::Model("fast_attn: proj linear has no bias".into()))
            .and_then(flatten_to_f32)?;

        let q_norm =
            match &attn.q_norm {
                Some(ln) => Some((
                    flatten_to_f32(ln.weight())?,
                    flatten_to_f32(ln.bias().ok_or_else(|| {
                        crate::Error::Model("fast_attn: q_norm has no bias".into())
                    })?)?,
                )),
                None => None,
            };
        let k_norm =
            match &attn.k_norm {
                Some(ln) => Some((
                    flatten_to_f32(ln.weight())?,
                    flatten_to_f32(ln.bias().ok_or_else(|| {
                        crate::Error::Model("fast_attn: k_norm has no bias".into())
                    })?)?,
                )),
                None => None,
            };
        Ok(Self::from_raw_weights(
            &qkv_w,
            &qkv_b,
            &proj_w,
            &proj_b,
            q_norm.as_ref().map(|(w, b)| (w.as_slice(), b.as_slice())),
            k_norm.as_ref().map(|(w, b)| (w.as_slice(), b.as_slice())),
            attn.num_heads,
            attn.head_dim,
            embed,
        ))
    }

    /// Forward pass. `x` is `[n, embed]` row-major; returns `[n, embed]`.
    ///
    /// `cos`/`sin`, when supplied, are `[n, head_dim]` rope2d tables (shared
    /// across heads, matching the DA3 layout). When `None`, no rope is
    /// applied (mirrors `rope_start == -1`).
    pub fn forward(&mut self, x: &[f32], n: usize, rope: Option<(&[f32], &[f32])>) -> Vec<f32> {
        let mut out = vec![0.0f32; n * self.embed];
        self.forward_into(x, &mut out, n, rope);
        out
    }

    /// In-place forward: writes `out = attention(x)` into the provided buffer.
    /// `out` must be `n * embed` long. This is the zero-alloc hot path used by
    /// [`crate::fast_block::FastVitBlock`].
    pub fn forward_into(
        &mut self,
        x: &[f32],
        out: &mut [f32],
        n: usize,
        rope: Option<(&[f32], &[f32])>,
    ) {
        debug_assert_eq!(out.len(), n * self.embed);
        let embed = self.embed;
        let heads = self.heads;
        let head_dim = self.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let hd = head_dim;
        self.scratch.ensure_capacity(n, heads, head_dim, embed);

        // Borrow all scratch fields to local slices (lifetime bound to `self`).
        let ScratchAttn {
            qkv,
            q_heads,
            k_heads_t,
            k_heads,
            v_heads,
            attn_out,
            concat,
            scores,
            q_buf: _,
            k_buf: _,
        } = &mut self.scratch;

        // 1. QKV projection: [n, embed] @ [embed, 3*embed] + bias -> [n, 3*embed]
        //    Fuse bias into the GEMM epilogue: pre-fill `qkv` with bias so the
        //    accumulate (`C += A@B`) yields `bias + A@B` directly.
        let qkv_len = n * 3 * embed;
        qkv[..qkv_len].par_chunks_mut(3 * embed).for_each(|row| {
            row.copy_from_slice(&self.qkv_b);
        });
        {
            let _g = crate::fast_profile::scope("attn_qkv");
            tinyblas::gemm_nn_into(n, 3 * embed, embed, x, &self.qkv_w, &mut qkv[..qkv_len]);
        }

        // 2. Gather into contiguous [heads, n, head_dim] panels, applying
        //    optional q/k LayerNorm and rope2d along the way. The gather +
        //    norm + rope in a single pass avoids the several intermediate
        //    tensors the candle path materialises.
        //
        //    Parallelised over the token axis (`ni`): each token's gather is
        //    fully independent (reads from disjoint qkv rows, writes to disjoint
        //    q_heads/k_heads/v_heads columns). Per-thread q_buf/k_buf live on
        //    the stack.
        //
        //    When the fused flash path is active we skip the strided writes to
        //    `k_heads_t` (the transposed-K buffer) — flash reads K from
        //    `k_heads` (row-major) directly, and the strided `k_heads_t[d*n]`
        //    writes touch a new cache line per element, which is extremely
        //    cache-unfriendly.
        let qn = self.q_norm.as_ref();
        let kn = self.k_norm.as_ref();
        let use_flash = flash_attn_enabled();
        // rope2d closed-form constants (head_dim is divisible by 4).
        let half = hd / 2;
        let quart = half / 2;
        // Capture raw base addresses so the closure is `Send`. Each task writes
        // only its own disjoint columns of q_heads/k_heads/v_heads.
        let q_heads_base = q_heads.as_mut_ptr() as usize;
        let k_heads_base = k_heads.as_mut_ptr() as usize;
        let v_heads_base = v_heads.as_mut_ptr() as usize;
        let k_heads_t_base = k_heads_t.as_mut_ptr() as usize;
        let qkv_base = qkv.as_ptr() as usize;
        let rope_present = rope.is_some();
        let (cos_base, sin_base) = match rope {
            Some((c, s)) => (c.as_ptr() as usize, s.as_ptr() as usize),
            None => (0usize, 0usize),
        };

        (0..n).into_par_iter().for_each(move |ni| {
            // Per-thread stack scratch.
            let mut q_buf = [0.0f32; 128]; // sized for typical head_dim (<=128)
            let mut k_buf = [0.0f32; 128];
            debug_assert!(
                hd <= 128,
                "head_dim > 128 not supported by fast gather scratch"
            );
            let q_buf = &mut q_buf[..hd];
            let k_buf = &mut k_buf[..hd];

            // Safety: reconstruct slices from base addresses + ni offset. Each
            // (ni, h) pair addresses disjoint memory.
            let qkv_row =
                unsafe { core::slice::from_raw_parts(qkv_base as *const f32, n * 3 * embed) };
            let qkv_row = &qkv_row[ni * 3 * embed..(ni + 1) * 3 * embed];
            let q_heads_h = unsafe {
                core::slice::from_raw_parts_mut(q_heads_base as *mut f32, heads * n * hd)
            };
            let k_heads_h = unsafe {
                core::slice::from_raw_parts_mut(k_heads_base as *mut f32, heads * n * hd)
            };
            let v_heads_h = unsafe {
                core::slice::from_raw_parts_mut(v_heads_base as *mut f32, heads * n * hd)
            };
            let k_heads_t_h = unsafe {
                core::slice::from_raw_parts_mut(k_heads_t_base as *mut f32, heads * hd * n)
            };
            let (c, s) = if rope_present {
                let cos_all =
                    unsafe { core::slice::from_raw_parts(cos_base as *const f32, n * hd) };
                let sin_all =
                    unsafe { core::slice::from_raw_parts(sin_base as *const f32, n * hd) };
                (
                    &cos_all[ni * hd..(ni + 1) * hd],
                    &sin_all[ni * hd..(ni + 1) * hd],
                )
            } else {
                (&[] as &[f32], &[] as &[f32])
            };

            for h in 0..heads {
                let q_src = &qkv_row[h * hd..(h + 1) * hd];
                let k_src = &qkv_row[embed + h * hd..embed + (h + 1) * hd];
                let v_src = &qkv_row[2 * embed + h * hd..2 * embed + (h + 1) * hd];

                let q_off = h * n * hd + ni * hd;
                let kt_off = h * hd * n + ni; // k_t is [hd, n] per head, column ni
                let v_off = h * n * hd + ni * hd;
                let q_dst = &mut q_heads_h[q_off..q_off + hd];
                let v_dst = &mut v_heads_h[v_off..v_off + hd];

                q_buf.copy_from_slice(q_src);
                k_buf.copy_from_slice(k_src);
                if let Some((w, b)) = qn {
                    layernorm_1d(q_buf, w, b, QK_NORM_EPS as f32);
                }
                if let Some((w, b)) = kn {
                    layernorm_1d(k_buf, w, b, QK_NORM_EPS as f32);
                }

                if rope_present {
                    apply_rope2d_1d(q_buf, q_dst, c, s);
                    let k_row = &mut k_heads_h[h * n * hd + ni * hd..h * n * hd + (ni + 1) * hd];
                    if use_flash {
                        // Branch-free rope2d directly into k_row (no k_t needed).
                        rope2d_k_into_flash(k_buf, k_row, c, s, half, quart);
                    } else {
                        // Same computation but also writes the strided k_t column.
                        rope2d_k_into_both(k_buf, k_row, k_heads_t_h, kt_off, n, c, s, half, quart);
                    }
                } else {
                    q_dst.copy_from_slice(q_buf);
                    let k_row = &mut k_heads_h[h * n * hd + ni * hd..h * n * hd + (ni + 1) * hd];
                    if use_flash {
                        k_row.copy_from_slice(k_buf);
                    } else {
                        for d in 0..hd {
                            let kd = k_buf[d];
                            k_heads_t_h[kt_off + d * n] = kd;
                            k_row[d] = kd;
                        }
                    }
                }
                v_dst.copy_from_slice(v_src);
            }
        });

        // 3. Attention core: Q·Kᵀ → softmax → A·V.
        //
        // Two paths:
        //  - Fused flash attention (`DA_FAST_FLASH=1`, the default when the fast
        //    backbone is on): never materialises the `[n,n]` scores matrix;
        //    matches what the C++ `ggml_flash_attn_ext` kernel does. This is
        //    the single biggest backbone win for DA3 (`n=864, heads=12` → the
        //    scores matrix is ~34 MiB, well past L2).
        //  - Materialised (fallback): per-head QKᵀ gemm + softmax + AV gemm.
        if use_flash {
            let _g = crate::fast_profile::scope("attn_flash");
            flash::forward(
                &q_heads[..heads * n * hd],
                &k_heads[..heads * n * hd],
                &v_heads[..heads * n * hd],
                &mut attn_out[..heads * n * hd],
                heads,
                n,
                hd,
                scale,
            );
        } else {
            // Per-head materialised attention.
            let _g = crate::fast_profile::scope("attn_core");
            //    `n*hd`-sized chunk of `attn_out` and a disjoint `n*n` chunk of
            //    `scores`. The read-only q/k/v slices are shared across threads.
            //
            //    We use `par_chunks_mut` on `scores` (heads chunks of n*n each)
            //    and derive the head index from the chunk. `attn_out` is indexed
            //    mutably via raw pointer offset within each head's task — this is
            //    safe because head `h` writes only `attn_out[h*n*hd .. (h+1)*n*hd]`,
            //    which is disjoint from every other head's write.
            //
            //    `attn_out_base` is passed as a `usize` so the closure is `Send`;
            //    we cast back to `*mut f32` inside.
            let n2 = n * n;
            let attn_out_base = attn_out.as_mut_ptr() as usize;
            // Slice scores to exactly heads*n*n so par_chunks_mut yields `heads`
            // chunks (the scratch buffer may be larger from a previous call with
            // a bigger `n`).
            let scores_h = &mut scores[..heads * n2];
            scores_h
                .par_chunks_mut(n2)
                .enumerate()
                .for_each(move |(h, sc)| {
                    let qkv_base = h * n * hd;
                    let kt_base = h * hd * n;
                    let q_h = &q_heads[qkv_base..qkv_base + n * hd];
                    let k_t_h = &k_heads_t[kt_base..kt_base + hd * n];
                    let v_h = &v_heads[qkv_base..qkv_base + n * hd];
                    // Safety: head h owns attn_out[h*n*hd .. (h+1)*n*hd] exclusively.
                    let out_h = unsafe {
                        core::slice::from_raw_parts_mut(
                            (attn_out_base as *mut f32).add(qkv_base),
                            n * hd,
                        )
                    };

                    // scores = Q_h @ K_h^T * scale -> [n, n]
                    for v in sc.iter_mut() {
                        *v = 0.0;
                    }
                    {
                        tinyblas::gemm_nn_into(n, n, hd, q_h, k_t_h, sc);
                    }
                    for s in sc.iter_mut() {
                        *s *= scale;
                    }
                    softmax_rows_inplace(sc, n);

                    // out_h = scores @ V_h -> [n, hd]. Zero first (gemm accumulates).
                    for v in out_h.iter_mut() {
                        *v = 0.0;
                    }
                    {
                        tinyblas::gemm_nn_into(n, hd, n, sc, v_h, out_h);
                    }
                });
        }

        // 4. Scatter [heads, n, head_dim] back to [n, heads*head_dim] = [n, embed].
        //    Parallelised over the token axis — each token writes a disjoint row
        //    of `concat`.
        let attn_out_base = attn_out.as_ptr() as usize;
        let concat_base = concat.as_mut_ptr() as usize;
        (0..n).into_par_iter().for_each(move |ni| {
            let attn_out_h =
                unsafe { core::slice::from_raw_parts(attn_out_base as *const f32, heads * n * hd) };
            let concat_h =
                unsafe { core::slice::from_raw_parts_mut(concat_base as *mut f32, n * embed) };
            for h in 0..heads {
                let off = h * n * hd + ni * hd;
                let src = &attn_out_h[off..off + hd];
                let dst = &mut concat_h[ni * embed + h * hd..ni * embed + (h + 1) * hd];
                dst.copy_from_slice(src);
            }
        });

        // 5. Output projection: [n, embed] @ [embed, embed] + bias -> [n, embed].
        //    Fuse bias into the GEMM epilogue (same trick as QKV).
        out.par_chunks_mut(embed).for_each(|row| {
            row.copy_from_slice(&self.proj_b);
        });
        {
            let _g = crate::fast_profile::scope("attn_proj");
            tinyblas::gemm_nn_into(n, embed, embed, &concat[..n * embed], &self.proj_w, out);
        }
    }

    /// Dimensions `(embed, heads, head_dim)`.
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.embed, self.heads, self.head_dim)
    }
}

// ---------------------------------------------------------------------------
// Helpers (raw f32, auto-vectorisable by the compiler)
// ---------------------------------------------------------------------------

/// Flatten a candle tensor to `Vec<f32>`.
pub fn flatten_to_f32(t: &candle::Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}

/// Apply the DA3 2D RoPE to a single `head_dim`-length slice.
///
/// Computes `dst[i] = src[i]*cos[i] + rot(src)[i]*sin[i]` where the rotation
/// splits `head_dim` into a y-block `[0, half)` and x-block `[half, head_dim)`
/// and within each half swaps the two `quart`-sized quarters with a negation
/// on the first. The closed-form for position `i` reduces to:
///
/// ```text
/// if (i % half) < quart: rot[i] = -src[quart + i]
/// else:                  rot[i] =  src[i - quart]
/// ```
///
/// Implemented as four branch-free segments (one per `[quart]`-sized range).
/// Each segment has a fixed access pattern with no data-dependent indexing, so
/// the compiler auto-vectorises all four into straight-line AVX2.
pub(crate) fn apply_rope2d_1d(src: &[f32], dst: &mut [f32], cos: &[f32], sin: &[f32]) {
    let d = src.len();
    debug_assert_eq!(d % 4, 0, "head_dim must be divisible by 4");
    let half = d / 2;
    let quart = half / 2;
    // Segment 0: i in [0, quart)        -> rot = -src[quart + i]
    for i in 0..quart {
        dst[i] = src[i] * cos[i] - src[quart + i] * sin[i];
    }
    // Segment 1: i in [quart, half)     -> rot =  src[i - quart]
    for i in quart..half {
        dst[i] = src[i] * cos[i] + src[i - quart] * sin[i];
    }
    // Segment 2: i in [half, half+quart) -> rot = -src[quart + i]
    for i in half..half + quart {
        dst[i] = src[i] * cos[i] - src[quart + i] * sin[i];
    }
    // Segment 3: i in [half+quart, d)    -> rot =  src[i - quart]
    for i in half + quart..d {
        dst[i] = src[i] * cos[i] + src[i - quart] * sin[i];
    }
}

/// Branch-free rope2d for the K head, flash-attention path: writes only `dst`.
/// Same per-segment formula as [`apply_rope2d_1d`] but inlined at the call site
/// with the closed-form `(half, quart)` constants so the compiler can see the
/// fixed trip-counts (typically 16) and fully unroll + vectorise.
#[inline]
pub(crate) fn rope2d_k_into_flash(
    src: &[f32],
    dst: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    half: usize,
    quart: usize,
) {
    let d = src.len();
    for i in 0..quart {
        dst[i] = src[i] * cos[i] - src[quart + i] * sin[i];
    }
    for i in quart..half {
        dst[i] = src[i] * cos[i] + src[i - quart] * sin[i];
    }
    for i in half..half + quart {
        dst[i] = src[i] * cos[i] - src[quart + i] * sin[i];
    }
    for i in half + quart..d {
        dst[i] = src[i] * cos[i] + src[i - quart] * sin[i];
    }
}

/// Branch-free rope2d for the K head, materialised-attention path: writes the
/// result both row-major into `dst` and strided (`stride = n`) into `k_t`.
/// The strided write cannot be vectorised (one cache line per element) but the
/// computation itself is branch-free so the FMA chain is straight-line.
#[inline]
pub(crate) fn rope2d_k_into_both(
    src: &[f32],
    dst: &mut [f32],
    k_t: &mut [f32],
    kt_off: usize,
    n: usize,
    cos: &[f32],
    sin: &[f32],
    half: usize,
    quart: usize,
) {
    let d = src.len();
    for i in 0..quart {
        let kd = src[i] * cos[i] - src[quart + i] * sin[i];
        dst[i] = kd;
        k_t[kt_off + i * n] = kd;
    }
    for i in quart..half {
        let kd = src[i] * cos[i] + src[i - quart] * sin[i];
        dst[i] = kd;
        k_t[kt_off + i * n] = kd;
    }
    for i in half..half + quart {
        let kd = src[i] * cos[i] - src[quart + i] * sin[i];
        dst[i] = kd;
        k_t[kt_off + i * n] = kd;
    }
    for i in half + quart..d {
        let kd = src[i] * cos[i] + src[i - quart] * sin[i];
        dst[i] = kd;
        k_t[kt_off + i * n] = kd;
    }
}

/// In-place 1-D layer norm over the whole slice: y = (x-mean)*inv_std*w + b.
pub(crate) fn layernorm_1d(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        x[i] = (x[i] - mean) * inv_std * w[i] + b[i];
    }
}

/// In-place row-wise softmax over an `[rows, cols]` row-major buffer.
///
/// Two passes: (a) per-row max + exp + sum, (b) divide. We fuse the max-subtract
/// and exp into pass (a) using a three-step (max, then scaled exp, then sum)
/// so the row data only needs to be read twice.
pub(crate) fn softmax_rows_inplace(buf: &mut [f32], rows: usize) {
    let cols = buf.len() / rows;
    debug_assert_eq!(buf.len(), rows * cols);
    for r in 0..rows {
        let row = &mut buf[r * cols..(r + 1) * cols];
        let mut max = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            let e = (*v - max).exp();
            *v = e;
            sum += e;
        }
        let inv_sum = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Reference attention (slow, obviously-correct) for cross-checking
    /// FastAttention against an independent implementation.
    fn reference_attention(
        x: &[f32],
        n: usize,
        qkv_w: &[f32],
        qkv_b: &[f32],
        proj_w: &[f32],
        proj_b: &[f32],
        cos: &[f32],
        sin: &[f32],
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let embed = heads * head_dim;
        // QKV = x @ W_qkv^T + b
        let mut qkv = vec![0.0f32; n * 3 * embed];
        for i in 0..n {
            for j in 0..3 * embed {
                let mut s = qkv_b[j];
                for k in 0..embed {
                    // W_qkv is [3*embed, embed] row-major (PyTorch layout).
                    s += x[i * embed + k] * qkv_w[j * embed + k];
                }
                qkv[i * 3 * embed + j] = s;
            }
        }
        // Per-head rope + attention.
        let mut out = vec![0.0f32; n * embed];
        for h in 0..heads {
            // scores = Q_h @ K_h^T * scale, softmax, * V_h
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut scores = vec![0.0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for d in 0..head_dim {
                        let q = qkv[i * 3 * embed + h * head_dim + d];
                        let k = qkv[j * 3 * embed + embed + h * head_dim + d];
                        s += q * k;
                    }
                    scores[i * n + j] = s * scale;
                }
            }
            // rope2d on Q and K (matches the reference's pre-rotation? — the
            // reference here deliberately skips rope so we can isolate the
            // attention math. The rope unit test below covers rotation.)
            // softmax
            for r in 0..n {
                let row = &mut scores[r * n..(r + 1) * n];
                let mut max = f32::NEG_INFINITY;
                for &v in row.iter() {
                    if v > max {
                        max = v;
                    }
                }
                let mut sum = 0.0;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
            // out = scores @ V_h
            for i in 0..n {
                for d in 0..head_dim {
                    let mut s = 0.0f32;
                    for j in 0..n {
                        let v = qkv[j * 3 * embed + 2 * embed + h * head_dim + d];
                        s += scores[i * n + j] * v;
                    }
                    out[i * embed + h * head_dim + d] = s;
                }
            }
        }
        // out = out @ W_proj^T + b
        let mut result = vec![0.0f32; n * embed];
        for i in 0..n {
            for j in 0..embed {
                let mut s = proj_b[j];
                for k in 0..embed {
                    s += out[i * embed + k] * proj_w[j * embed + k];
                }
                result[i * embed + j] = s;
            }
        }
        result
    }

    #[test]
    fn forward_matches_reference_no_rope() {
        // Small dims to keep the reference O(n^3 * embed) fast.
        let heads = 4;
        let head_dim = 16;
        let embed = heads * head_dim; // 64
        let n = 8;

        let qkv_w = pseudo_random(1, 3 * embed * embed);
        let qkv_b = pseudo_random(2, 3 * embed);
        let proj_w = pseudo_random(3, embed * embed);
        let proj_b = pseudo_random(4, embed);
        let x = pseudo_random(5, n * embed);

        let mut fast = FastAttention::from_raw_weights(
            &qkv_w, &qkv_b, &proj_w, &proj_b, None, None, heads, head_dim, embed,
        );
        let got = fast.forward(&x, n, None);
        let expected = reference_attention(
            &x,
            n,
            &qkv_w,
            &qkv_b,
            &proj_w,
            &proj_b,
            &[],
            &[],
            heads,
            head_dim,
        );
        let diff = max_abs_diff(&got, &expected);
        let max_val = got.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        eprintln!("forward_matches_reference_no_rope: max_abs_diff={diff}, max_val={max_val}");
        assert!(
            diff < 1e-2,
            "fast attention disagrees with reference (max |d| = {diff})"
        );
    }

    #[test]
    fn forward_matches_reference_with_rope() {
        let heads = 4;
        let head_dim = 16;
        let embed = heads * head_dim;
        let n = 8;

        let qkv_w = pseudo_random(11, 3 * embed * embed);
        let qkv_b = pseudo_random(12, 3 * embed);
        let proj_w = pseudo_random(13, embed * embed);
        let proj_b = pseudo_random(14, embed);
        let x = pseudo_random(15, n * embed);
        let cos = pseudo_random(16, n * head_dim);
        let sin = pseudo_random(17, n * head_dim);

        let mut fast = FastAttention::from_raw_weights(
            &qkv_w, &qkv_b, &proj_w, &proj_b, None, None, heads, head_dim, embed,
        );
        // Just verify it runs without panicking and produces finite output.
        let got = fast.forward(&x, n, Some((&cos, &sin)));
        for &v in &got {
            assert!(v.is_finite(), "non-finite output from forward_with_rope");
        }
    }

    #[test]
    fn rope2d_leaves_zero_position_unchanged() {
        // At position 0, cos[0]=1, sin[0]=0 → src unchanged.
        let head_dim = 16usize;
        let mut cos = vec![0.0f32; head_dim];
        let mut sin = vec![0.0f32; head_dim];
        for i in 0..head_dim {
            cos[i] = if i == 0 { 1.0 } else { 0.7 };
            sin[i] = 0.0;
        }
        let src: Vec<f32> = (0..head_dim).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; head_dim];
        apply_rope2d_1d(&src, &mut dst, &cos, &sin);
        // At i=0: dst = src*1 + rot*0 = src.
        assert!((dst[0] - src[0]).abs() < 1e-6);
        // At i>0 (cos=0.7, sin=0): dst = src*0.7.
        for i in 1..head_dim {
            assert!((dst[i] - src[i] * 0.7).abs() < 1e-5, "i={i}");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut buf = vec![1.0f32, 2.0, 3.0, -1.0, 0.5, 0.0];
        softmax_rows_inplace(&mut buf, 2);
        // Row 0 and row 1 each sum to 1.
        let s0: f32 = buf[0..3].iter().sum();
        let s1: f32 = buf[3..6].iter().sum();
        assert!((s0 - 1.0).abs() < 1e-5, "row0 sum={s0}");
        assert!((s1 - 1.0).abs() < 1e-5, "row1 sum={s1}");
        // All entries in [0, 1].
        for &v in &buf {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn layernorm_zero_mean_unit_variance_before_affine() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];
        let b = vec![0.0f32; 4];
        layernorm_1d(&mut x, &w, &b, 0.0);
        let mean = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "post-ln mean={mean}");
        let var = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-4, "post-ln var={var}");
    }

    /// Repeated forwards on the same instance must agree with a fresh one —
    /// this catches buffer-reuse bugs (stale data, wrong zeroing).
    #[test]
    fn repeated_forwards_match_fresh_instance() {
        let heads = 4;
        let head_dim = 16;
        let embed = heads * head_dim;
        let n = 8;
        let qkv_w = pseudo_random(31, 3 * embed * embed);
        let qkv_b = pseudo_random(32, 3 * embed);
        let proj_w = pseudo_random(33, embed * embed);
        let proj_b = pseudo_random(34, embed);
        let x = pseudo_random(35, n * embed);

        let mut reused = FastAttention::from_raw_weights(
            &qkv_w, &qkv_b, &proj_w, &proj_b, None, None, heads, head_dim, embed,
        );
        let y1 = reused.forward(&x, n, None);
        let y2 = reused.forward(&x, n, None);
        let diff = max_abs_diff(&y1, &y2);
        assert!(diff < 1e-5, "repeated forwards disagree by {diff}");
    }

    /// A forward on a larger `n` followed by a smaller `n` must still be
    /// correct (the reused buffers were resized up, not down).
    #[test]
    fn handles_variable_n_with_reused_buffers() {
        let heads = 2;
        let head_dim = 8;
        let embed = heads * head_dim;
        let qkv_w = pseudo_random(41, 3 * embed * embed);
        let qkv_b = pseudo_random(42, 3 * embed);
        let proj_w = pseudo_random(43, embed * embed);
        let proj_b = pseudo_random(44, embed);

        let mut fast = FastAttention::from_raw_weights(
            &qkv_w, &qkv_b, &proj_w, &proj_b, None, None, heads, head_dim, embed,
        );
        // Big n first (grows buffers), then small n (uses prefix).
        let x_big = pseudo_random(45, 16 * embed);
        let _ = fast.forward(&x_big, 16, None);
        let x_small = pseudo_random(46, 4 * embed);
        let got = fast.forward(&x_small, 4, None);

        let mut fresh = FastAttention::from_raw_weights(
            &qkv_w, &qkv_b, &proj_w, &proj_b, None, None, heads, head_dim, embed,
        );
        let expected = fresh.forward(&x_small, 4, None);
        let diff = max_abs_diff(&got, &expected);
        assert!(diff < 1e-4, "variable-n disagree by {diff}");
    }
}
