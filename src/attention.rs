//! Multi-head attention with qk-norm and 2D RoPE, matching `src/attention.cpp`.
//!
//! Shape convention: token sequences are kept as `[1, N, embed]` throughout.
//! For attention we split into `[1, N, H, D]`, apply per-head LayerNorm and
//! 2D RoPE to Q and K, then run scaled dot-product attention as `[1, H, N, D]`.

use crate::config::Config;
use crate::weights::load_linear;
use crate::Result;
use candle::{DType, Device, Tensor};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

/// Epsilon for the q/k LayerNorms. **Not** the block's `ln_eps` (1e-6) — this
/// matches torch's default `nn.LayerNorm` eps, which DA3 uses for q/k norms.
pub const QK_NORM_EPS: f64 = 1e-5;

/// Per-block attention weights, loaded once.
///
/// `q_norm`/`k_norm` are `Option` because DA3 only stores them for blocks with
/// `index >= qknorm_start` (blocks before that have no q/k LayerNorm). This
/// mirrors `src/attention.cpp` where `AttnWeights.qn_w` is a nullable pointer
/// loaded via `ml.tensor(...)` (returns null when absent) and the forward path
/// guards `if (w.qn_w)`.
pub struct Attention {
    pub qkv: Linear,
    pub proj: Linear,
    pub q_norm: Option<LayerNorm>,
    pub k_norm: Option<LayerNorm>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    pub fn load(vb: &VarBuilder, cfg: &Config, _device: &Device) -> Result<Self> {
        let embed = cfg.embed_dim as usize;
        let h = cfg.num_heads as usize;
        let d = cfg.head_dim as usize;
        let qkv = load_linear(vb, "attn_qkv", 3 * embed, embed)?;
        let proj = load_linear(vb, "attn_proj", embed, embed)?;
        // Optional: present only when block index >= qknorm_start.
        let q_norm = layer_norm(d, QK_NORM_EPS, vb.pp("attn_qnorm")).ok();
        let k_norm = layer_norm(d, QK_NORM_EPS, vb.pp("attn_knorm")).ok();
        Ok(Self {
            qkv,
            proj,
            q_norm,
            k_norm,
            num_heads: h,
            head_dim: d,
        })
    }

    /// Forward pass.
    ///
    /// - `x`: `[1, N, embed]`.
    /// - `cos`, `sin`: optional RoPE tables as `[N, head_dim]` tensors. When
    ///   `None`, no RoPE is applied (mirrors `rope_start == -1`).
    pub fn forward(
        &self,
        x: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let h = self.num_heads;
        let d = self.head_dim;
        let embed = h * d;

        // Fused QKV projection: [1, N, embed] -> [1, N, 3*embed]
        let qkv = self.qkv.forward(x)?;
        // -> [1, N, 3, H, D], then split along the q/k/v axis.
        let qkv = qkv.reshape((b, n, 3, h, d))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?; // [1, N, H, D]
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;

        // Per-head qk LayerNorm over the head_dim (last axis). Only applied
        // when the norms exist (block index >= qknorm_start); matches the C++
        // `if (w.qn_w)` guard.
        let q = match &self.q_norm {
            Some(ln) => ln.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(ln) => ln.forward(&k)?,
            None => k,
        };

        // 2D RoPE on Q and K only.
        let q = if let (Some(cos), Some(sin)) = (cos, sin) {
            apply_rope2d(&q, cos, sin, d)?
        } else {
            q
        };
        let k = if let (Some(cos), Some(sin)) = (cos, sin) {
            apply_rope2d(&k, cos, sin, d)?
        } else {
            k
        };

        // -> [1, H, N, D] for scaled-dot-product attention.
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (d as f64).sqrt();
        // candle_nn::ops::sdpa has no CPU implementation in candle 0.8 (it is
        // CUDA/Metal-only), so implement scaled dot-product attention manually:
        //   scores = Q·K^T * scale  -> [1, H, N, N]
        //   attn   = softmax(scores) · V -> [1, H, N, D]
        // This mirrors the C++ "manual" attention path in src/attention.cpp
        // (the materialized-scores variant; bit-tight parity with flash on CPU).
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = scores.affine(scale, 0.0)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;

        // -> [1, N, H, D] -> [1, N, embed]
        let attn = attn.matmul(&v)?;
        let attn = attn.transpose(1, 2)?.contiguous()?.reshape((b, n, embed))?;
        let out = self.proj.forward(&attn)?;
        Ok(out)
    }
}

/// Apply the DA3 2D RoPE to `x` (`[..., head_dim]`), given precomputed
/// `cos`/`sin` tables of shape `[N, head_dim]`. Internally `cos`/`sin` are
/// broadcast to the head axis.
///
/// The rotation is the standard `x*cos + rotate_half_2d(x)*sin`, where
/// `rotate_half_2d` splits `head_dim` into two halves (y-block, x-block), each
/// of size `half = head_dim/2`, and within each half produces
/// `[-h[quart:], h[:quart]]` with `quart = half/2`. The two rotated halves are
/// concatenated back along the last axis.
fn apply_rope2d(x: &Tensor, cos: &Tensor, sin: &Tensor, head_dim: usize) -> Result<Tensor> {
    let half = head_dim / 2;
    let quart = half / 2;

    // Broadcast cos/sin from [N, head_dim] to x's shape.
    // x is [1, N, H, head_dim]; we want cos as [1, N, 1, head_dim].
    let x_dims = x.dims();
    let last_axis = x_dims.len() - 1; // index of the head_dim axis
    let broadcast_shape: Vec<usize> = x_dims.to_vec();
    let cos_b = cos
        .unsqueeze(0)? // [1, N, head_dim]
        .unsqueeze(2)? // [1, N, 1, head_dim]
        .broadcast_as(broadcast_shape.clone())?;
    let sin_b = sin
        .unsqueeze(0)?
        .unsqueeze(2)?
        .broadcast_as(broadcast_shape)?;

    let last = last_axis;
    // y-block = x[..., :half], x-block = x[..., half:head_dim].
    let ay = x.narrow(last, 0, half)?;
    let ax = x.narrow(last, half, half)?;

    let rotate_half = |h: &Tensor| -> candle::Result<Tensor> {
        let h0 = h.narrow(last, 0, quart)?;
        let h1 = h.narrow(last, quart, quart)?;
        Tensor::cat(&[&h1.neg()?, &h0], last)
    };

    let rot_y = rotate_half(&ay)?;
    let rot_x = rotate_half(&ax)?;
    let rot = Tensor::cat(&[&rot_y, &rot_x], last)?;

    // x*cos + rot*sin
    let out = (x.broadcast_mul(&cos_b)? + rot.broadcast_mul(&sin_b)?)?;
    let _ = DType::F32; // keep the import meaningful
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope2d::build_rope_tables;

    #[test]
    fn rope_leaves_cls_unchanged() {
        // Token 0 is at (0,0): cos=1, sin=0 everywhere → x unchanged.
        let device = Device::Cpu;
        let pos = vec![0.0, 0.0, 1.0, 1.0];
        let tbl = build_rope_tables(&pos, 8, 100.0);
        let cos = Tensor::from_slice(&tbl.cos, (2, 8), &device).unwrap();
        let sin = Tensor::from_slice(&tbl.sin, (2, 8), &device).unwrap();
        // x: [1, 2, 1, 8] (1 head). Token 0 row should be identical post-rope.
        let x = Tensor::arange(0.0f32, 16.0, &device)
            .unwrap()
            .reshape((1, 2, 1, 8))
            .unwrap();
        let y = apply_rope2d(&x, &cos, &sin, 8).unwrap();
        let x0 = x.narrow(1, 0, 1).unwrap().flatten_all().unwrap();
        let y0 = y.narrow(1, 0, 1).unwrap().flatten_all().unwrap();
        let xv: Vec<f32> = x0.to_vec1().unwrap();
        let yv: Vec<f32> = y0.to_vec1().unwrap();
        for i in 0..8 {
            assert!(
                (xv[i] - yv[i]).abs() < 1e-5,
                "CLS token modified by rope at {i}"
            );
        }
    }
}
