//! A single ViT transformer block (pre-norm, LayerScale), matching
//! `src/vit_block.cpp`.

use crate::attention::Attention;
use crate::config::{Config, FfnType};
use crate::weights::load_linear;
use crate::Result;
use candle::{Device, Tensor};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

/// One feed-forward variant of a block's FFN weights.
pub enum Ffn {
    /// GELU-erf MLP: `fc1 → gelu → fc2`.
    Mlp { fc1: Linear, fc2: Linear },
    /// SwiGLU: `silu(w12[:h]) * w12[h:] → w3`.
    Swiglu { w12: Linear, w3: Linear },
}

impl Ffn {
    pub fn load(vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let embed = cfg.embed_dim as usize;
        match cfg.ffn_type {
            FfnType::Mlp => {
                let hidden = cfg.mlp_hidden as usize;
                let fc1 = load_linear(vb, "mlp_fc1", hidden, embed)?;
                let fc2 = load_linear(vb, "mlp_fc2", embed, hidden)?;
                Ok(Ffn::Mlp { fc1, fc2 })
            }
            FfnType::Swiglu => {
                let hidden = cfg.mlp_hidden as usize;
                let w12 = load_linear(vb, "mlp_w12", 2 * hidden, embed)?;
                let w3 = load_linear(vb, "mlp_w3", embed, hidden)?;
                Ok(Ffn::Swiglu { w12, w3 })
            }
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Ffn::Mlp { fc1, fc2 } => {
                // Exact-erf GELU (torch nn.GELU default). candle's `gelu` is the
                // sigmoid approximation; `gelu_erf` is the exact erf variant.
                let h = fc1.forward(x)?.gelu_erf()?;
                fc2.forward(&h).map_err(Into::into)
            }
            Ffn::Swiglu { w12, w3 } => {
                let x12 = w12.forward(x)?; // [..., 2*hidden]
                let two_hidden = x12.dim(x12.dims().len() - 1)?;
                let hidden = two_hidden / 2;
                let last = x12.dims().len() - 1;
                let x1 = x12.narrow(last, 0, hidden)?;
                let x2 = x12.narrow(last, hidden, hidden)?;
                let h = x1.silu()?.broadcast_mul(&x2)?;
                w3.forward(&h).map_err(Into::into)
            }
        }
    }
}

/// One ViT block: pre-norm attention + pre-norm FFN, each with a residual and
/// a LayerScale (`ls1`/`ls2`).
pub struct VitBlock {
    pub norm1: LayerNorm,
    pub attn: Attention,
    pub ls1: Option<Tensor>,
    pub norm2: LayerNorm,
    pub ffn: Ffn,
    pub ls2: Option<Tensor>,
    /// Block-level LayerNorm epsilon (cached for the fast path).
    ln_eps: f32,
    /// Candle-free fast path (built lazily on first `forward` when
    /// `DA_FAST_ATTN=1`). `Err` means construction failed; falls back to candle.
    fast: std::sync::Mutex<Option<crate::fast_block::FastVitBlock>>,
}

impl VitBlock {
    /// Load block `i` from `vit.blk.{i}.*`.
    pub fn load(vb: &VarBuilder, cfg: &Config, i: usize, device: &Device) -> Result<Self> {
        let block_vb = vb.pp("vit").pp("blk").pp(i);
        let embed = cfg.embed_dim as usize;
        let norm1 = layer_norm(embed, cfg.ln_eps as f64, block_vb.pp("norm1"))?;
        let norm2 = layer_norm(embed, cfg.ln_eps as f64, block_vb.pp("norm2"))?;
        let attn = Attention::load(&block_vb, cfg, device)?;
        let ffn = Ffn::load(&block_vb, cfg)?;

        // LayerScale tensors are optional in the engine; present for DA3.
        let ls1 = block_vb.get((embed,), "ls1").ok();
        let ls2 = block_vb.get((embed,), "ls2").ok();

        Ok(Self {
            norm1,
            attn,
            ls1,
            norm2,
            ffn,
            ls2,
            ln_eps: cfg.ln_eps as f32,
            fast: std::sync::Mutex::new(None),
        })
    }

    /// Forward pass. `cos`/`sin` are the RoPE tables for this block (None if
    /// the block doesn't use RoPE).
    pub fn forward(
        &self,
        x: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
    ) -> Result<Tensor> {
        if fast_attn_enabled() {
            return self.forward_fast(x, cos, sin);
        }
        // Attention sub-block (candle path).
        let xn = self.norm1.forward(x)?;
        let a = self.attn.forward(&xn, cos, sin)?;
        let a = scale_layer(&a, &self.ls1)?;
        let x = (x + a)?;

        // FFN sub-block.
        let xn = self.norm2.forward(&x)?;
        let m = self.ffn.forward(&xn)?;
        let m = scale_layer(&m, &self.ls2)?;
        let x = (x + m)?;
        Ok(x)
    }

    /// Candle-free fast path: the entire block (norm1 + attn + ls1 + residual
    /// + norm2 + ffn + ls2 + residual) runs on raw `&[f32]` via
    /// [`crate::fast_block::FastVitBlock`], bypassing candle's per-op overhead.
    /// Falls back to [`forward`] if batch != 1 or construction fails.
    fn forward_fast(
        &self,
        x: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, n, _e) = x.dims3()?;
        if b != 1 {
            // Multi-batch isn't supported by the raw-f32 path; fall back.
            return self.forward_candle(x, cos, sin);
        }
        let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
        let (c_v, s_v) = match (cos, sin) {
            (Some(c), Some(s)) => (
                Some(c.flatten_all()?.to_vec1::<f32>()?),
                Some(s.flatten_all()?.to_vec1::<f32>()?),
            ),
            _ => (None, None),
        };
        let rope = match (&c_v, &s_v) {
            (Some(c), Some(s)) => Some((c.as_slice(), s.as_slice())),
            _ => None,
        };

        let embed = x.dim(2)? as usize;
        let mut guard = self.fast.lock().expect("fast_block mutex poisoned");
        if guard.is_none() {
            *guard = crate::fast_block::FastVitBlock::from_candle(self, self.ln_eps).ok();
        }
        let Some(fast) = guard.as_mut() else {
            return self.forward_candle(x, cos, sin);
        };
        // Reuse a single output buffer across calls (stored in the Mutex guard
        // alongside the FastVitBlock — but since forward_into needs &mut self
        // and we hold the guard, we allocate here and let the caller deal).
        // For simplicity we allocate once per call; the block's internal scratch
        // (the expensive part) is reused.
        let mut out_buf = vec![0.0f32; n * embed];
        fast.forward_into(&x_flat, &mut out_buf, n, rope);
        Tensor::from_vec(out_buf, (1, n, embed), x.device()).map_err(Into::into)
    }

    /// The candle path (used as fallback or when fast path is disabled).
    fn forward_candle(
        &self,
        x: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
    ) -> Result<Tensor> {
        let xn = self.norm1.forward(x)?;
        let a = self.attn.forward(&xn, cos, sin)?;
        let a = scale_layer(&a, &self.ls1)?;
        let x = (x + a)?;
        let xn = self.norm2.forward(&x)?;
        let m = self.ffn.forward(&xn)?;
        let m = scale_layer(&m, &self.ls2)?;
        let x = (x + m)?;
        Ok(x)
    }
}

/// Read `DA_FAST_ATTN` once and cache. When set to "1" (or "on"/"true"), the
/// ViT block routes attention through [`Attention::forward_fast`] (the
/// candle-free tinyBLAS path). Any other value (or unset) uses the candle path.
fn fast_attn_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("DA_FAST_ATTN").as_deref(),
            Ok("1") | Ok("on") | Ok("true") | Ok("ON") | Ok("TRUE") | Ok("True")
        )
    })
}

fn scale_layer(x: &Tensor, gamma: &Option<Tensor>) -> Result<Tensor> {
    match gamma {
        Some(g) => {
            // g: [embed] -> broadcast over the leading token dims.
            let mut dims: Vec<usize> = x.dims().to_vec();
            for d in dims.iter_mut() {
                *d = 1;
            }
            *dims.last_mut().unwrap() = g.dim(0)?;
            x.broadcast_mul(&g.reshape(dims)?).map_err(Into::into)
        }
        None => Ok(x.clone()),
    }
}
