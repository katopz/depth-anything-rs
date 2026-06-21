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
        // Attention sub-block.
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
