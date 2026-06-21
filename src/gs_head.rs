//! The 3D-Gaussian DPT head (GSDPT), matching `src/gs_head.cpp` and the
//! Python `depth_anything_3.model.gs_dpt.GSDPT._forward_impl`.
//!
//! Subclasses the single-head DPT fusion pyramid but with:
//! - `norm_type="idt"` — NO LayerNorm at the input of each stage.
//! - GSDPT fixed config: `features=256`, `out_channels=[256,512,1024,1024]`,
//!   `output_dim=38` (= 37 Gaussian channels + 1 confidence).
//! - An `images_merger` that injects the input image (3→32→64→128, 3×3 pad1 +
//!   exact-erf GELU after each conv including the last) before `output_conv2`.
//! - `output_conv2` splits into `out2a` (3×3 pad1, 128→32, ReLU) and `out2b`
//!   (1×1, 32→38).
//!
//! All weights live under `gs.*` (NOT `head.*`); the head is detected in
//! [`crate::engine::Engine::load`] by the presence of `gs.proj.0.weight`.

use crate::config::Config;
use crate::gguf::GgufFile;
use crate::uv_embed::uv_embed_chw_cached;
use crate::Result;
use candle::{DType, Device, Tensor};
use candle_nn::conv::{Conv2d, ConvTranspose2d};
use candle_nn::{Conv2dConfig, ConvTranspose2dConfig, Module, VarBuilder};

/// Scale on the UV positional embedding added to feature maps (matches C++).
const UV_RATIO: f32 = 0.1;

/// The GSDPT output is `38` channels per pixel = 37 Gaussian channels + 1
/// confidence channel. (Kept as a `const` for clarity; the head's `out2b`
/// shape is read dynamically from the GGUF, but the activate step assumes 38.)
pub const GS_OUT_CHANNELS: usize = 38;

/// The raw per-pixel GSDPT output, already split into the 37 Gaussian channels
/// and the 1 confidence channel.
#[derive(Debug, Clone)]
pub struct RawGaussians {
    /// `[H*W*37]`, channels-last: `(h*W+w)*37 + c`.
    pub raw_gs: Vec<f32>,
    /// `[H*W]`, row-major.
    pub gs_conf: Vec<f32>,
    /// Image height (`H`).
    pub h: usize,
    /// Image width (`W`).
    pub w: usize,
}

/// One GSDPT head, loaded from `gs.*`. The fusion pyramid uses `feat_half = 128`
/// (GSDPT's `features/2`) and produces 38 output channels.
pub struct GsHead {
    proj: [(Tensor, Tensor); 4],
    resize0: (Tensor, Tensor),
    resize1: (Tensor, Tensor),
    resize3: (Tensor, Tensor),
    layer_rn: [Tensor; 4],
    rn1: Fusion,
    rn2: Fusion,
    rn3: Fusion,
    rn4: Fusion,
    out1: (Tensor, Tensor),
    out2a: (Tensor, Tensor),
    out2b: (Tensor, Tensor),
    /// `gs.merger.{0..3}` — 3 sequential 3×3 pad1 convs, GELU-erf after each.
    merger: [(Tensor, Tensor); 3],
}

/// One `refinenet{N}` fusion stage (identical layout to `DptHead`'s `Fusion`).
struct Fusion {
    /// Lateral residual conv unit. `None` for refinenet4.
    rc1: Option<((Tensor, Tensor), (Tensor, Tensor))>,
    /// Top residual conv unit (always present).
    rc2: ((Tensor, Tensor), (Tensor, Tensor)),
    /// 1×1 out conv (with bias), `feat_half → feat_half`.
    outc: (Tensor, Tensor),
}

impl GsHead {
    /// Load the GSDPT head from a GGUF. Returns `Err` if any required tensor is
    /// missing — callers usually check `has_gs_head(file)` first.
    pub fn load(vb: &VarBuilder, cfg: &Config, _file: &GgufFile, _device: &Device) -> Result<Self> {
        let embed = cfg.embed_dim as usize;
        let in_channels = if cfg.cat_token { 2 * embed } else { embed };
        // GSDPT fixed config.
        let features = 256usize; // gs.features
        let feat_half = 128usize; // gs.features/2
        let oc = [256usize, 512, 1024, 1024];
        let gs_vb = vb.pp("gs");

        // ---- proj: 1x1 conv, in_channels -> oc[i] (NO LayerNorm at input). ----
        let proj = [
            conv_weights_1x1(&gs_vb, "proj.0", in_channels, oc[0])?,
            conv_weights_1x1(&gs_vb, "proj.1", in_channels, oc[1])?,
            conv_weights_1x1(&gs_vb, "proj.2", in_channels, oc[2])?,
            conv_weights_1x1(&gs_vb, "proj.3", in_channels, oc[3])?,
        ];

        let resize0 = conv_transpose_weights(&gs_vb, "resize.0", oc[0], oc[0], 4)?;
        let resize1 = conv_transpose_weights(&gs_vb, "resize.1", oc[1], oc[1], 2)?;
        let resize3 = conv_weights(&gs_vb, "resize.3", oc[3], oc[3], 3)?;

        let scratch = gs_vb.pp("scratch");
        // layer{i}_rn: 3x3 (no bias), out_channels[i] -> features (256).
        let layer_rn = [
            conv_no_bias(&scratch, "layer1_rn", oc[0], features, 3)?,
            conv_no_bias(&scratch, "layer2_rn", oc[1], features, 3)?,
            conv_no_bias(&scratch, "layer3_rn", oc[2], features, 3)?,
            conv_no_bias(&scratch, "layer4_rn", oc[3], features, 3)?,
        ];

        // Fusion (refinenet) pyramid operates entirely at `features` (256).
        let rn1 = load_fusion(&gs_vb, "rn1", true, features)?;
        let rn2 = load_fusion(&gs_vb, "rn2", true, features)?;
        let rn3 = load_fusion(&gs_vb, "rn3", true, features)?;
        let rn4 = load_fusion(&gs_vb, "rn4", false, features)?;

        // output_conv1: 3x3, features(256) -> feat_half(128).
        let out1 = conv_weights(&gs_vb, "scratch.out1", feat_half, features, 3)?;
        // output_conv2a: 3x3, feat_half(128) -> 32.
        let out2a = conv_weights(&gs_vb, "scratch.out2a", 32, feat_half, 3)?;
        // output_conv2b: 1x1, 32 -> 38 (37 raw + 1 conf).
        let out2b = conv_weights_1x1(&gs_vb, "scratch.out2b", 32, GS_OUT_CHANNELS)?;

        // images_merger: 3 convs (3->32->64->128), 3x3 pad1 + GELU-erf after each.
        let merger = [
            conv_weights(&gs_vb, "merger.0", 32, 3, 3)?,
            conv_weights(&gs_vb, "merger.1", 64, 32, 3)?,
            conv_weights(&gs_vb, "merger.2", feat_half, 64, 3)?,
        ];

        Ok(Self {
            proj,
            resize0,
            resize1,
            resize3,
            layer_rn,
            rn1,
            rn2,
            rn3,
            rn4,
            out1,
            out2a,
            out2b,
            merger,
        })
    }

    /// Forward the GSDPT head and produce the per-pixel raw-Gaussian output.
    ///
    /// * `feats` — the 4 backbone out-layer features, each `[1, n_patch, in_channels]`.
    /// * `image` — the ImageNet-normalized input image as `[1, 3, H, W]`
    ///   (the SAME tensor the backbone consumed — NOT a `[0,1]` image).
    /// * `gh`, `gw` — patch-grid dims (`H/patch`, `W/patch`).
    /// * `h`, `w` — full processed-image dims.
    ///
    /// Returns the channels-last `[H*W*37]` raw Gaussian channels + `[H*W]` conf.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        feats: &[Tensor],
        image: &Tensor,
        gh: usize,
        gw: usize,
        h: usize,
        w: usize,
        device: &Device,
    ) -> Result<RawGaussians> {
        let pw = gw;
        let ph = gh;
        let aspect = gw as f32 / gh as f32;

        let pad0 = Conv2dConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        };
        let pad1 = Conv2dConfig {
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        };
        let down = Conv2dConfig {
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
        };

        // ---- per-stage: project + UV + resize. NO LayerNorm (norm_type="idt"). ----
        let mut l = Vec::with_capacity(4);
        for (s, feat) in feats.iter().enumerate().take(4) {
            let x = feat.clone();
            // NO LayerNorm: GSDPT norm_type="idt" -> identity.
            // [1, n_patch, C] -> [1, C, ph, pw]
            let x = x
                .contiguous()?
                .reshape((1, ph, pw, x.dim(2)?))?
                .permute((0, 3, 1, 2))?;
            let mut x = conv_fwd(&self.proj[s].0, &self.proj[s].1, pad0, &x)?;
            // + 0.1 * UV at [pw, ph, oc[s]].
            x = add_uv(&x, pw, ph, x.dim(1)?, aspect, device)?;
            x = match s {
                0 => conv_t_fwd(
                    &self.resize0.0,
                    &self.resize0.1,
                    ConvTranspose2dConfig {
                        stride: 4,
                        padding: 0,
                        dilation: 1,
                        output_padding: 0,
                    },
                    &x,
                )?,
                1 => conv_t_fwd(
                    &self.resize1.0,
                    &self.resize1.1,
                    ConvTranspose2dConfig {
                        stride: 2,
                        padding: 0,
                        dilation: 1,
                        output_padding: 0,
                    },
                    &x,
                )?,
                2 => x,
                3 => conv_fwd(&self.resize3.0, &self.resize3.1, down, &x)?,
                _ => unreachable!(),
            };
            l.push(x);
        }

        // layer{i}_rn: 3x3 no bias -> features (256).
        let features = 256usize;
        let zero = zero_bias(device, features);
        let lat: Vec<Tensor> = (0..4)
            .map(|s| conv_fwd(&self.layer_rn[s], &zero, pad1, &l[s]))
            .collect::<Result<Vec<_>>>()?;

        // Top-down fusion.
        let out4 = fusion_forward(&self.rn4, &lat[3], None, pw, ph)?;
        let out3 = fusion_forward(&self.rn3, &out4, Some(&lat[2]), 2 * pw, 2 * ph)?;
        let out2 = fusion_forward(&self.rn2, &out3, Some(&lat[1]), 4 * pw, 4 * ph)?;
        let mut out = fusion_forward(&self.rn1, &out2, Some(&lat[0]), 0, 0)?;

        // output_conv1: 3x3 256->128.
        out = conv_fwd(&self.out1.0, &self.out1.1, pad1, &out)?;

        // Upsample to the full processed image resolution (H, W).
        if (out.dim(2)?, out.dim(3)?) != (h, w) {
            out = upsample_bilinear_ac(&out, h, w)?;
        }

        // images_merger: 3 convs (3->32->64->128), 3x3 pad1 + GELU-erf after EACH
        // (including the last). Image is the ImageNet-normalized input.
        let mut m = conv_fwd(&self.merger[0].0, &self.merger[0].1, pad1, image)?;
        m = m.gelu_erf()?;
        m = conv_fwd(&self.merger[1].0, &self.merger[1].1, pad1, &m)?;
        m = m.gelu_erf()?;
        m = conv_fwd(&self.merger[2].0, &self.merger[2].1, pad1, &m)?;
        m = m.gelu_erf()?;
        out = out.broadcast_add(&m)?;

        // + 0.1 * UV(128) at full resolution.
        out = add_uv(&out, w, h, out.dim(1)?, aspect, device)?;

        // output_conv2: 3x3 pad1 (128->32) -> ReLU -> 1x1 (32->38).
        out = conv_fwd(&self.out2a.0, &self.out2a.1, pad1, &out)?;
        out = out.relu()?;
        let logits = conv_fwd(&self.out2b.0, &self.out2b.1, pad0, &out)?;

        // logits is [1, 38, H, W]. Read back to host and reorder to channels-last.
        let (b, c, dh, dw) = logits.dims4()?;
        if (b, c) != (1, GS_OUT_CHANNELS) {
            return Err(crate::Error::Model(format!(
                "gs_head: expected logits [1, {GS_OUT_CHANNELS}, H, W], got [{b}, {c}, {dh}, {dw}]"
            )));
        }
        if (dh, dw) != (h, w) {
            return Err(crate::Error::Model(format!(
                "gs_head: expected logits at {h}x{w}, got {dh}x{dw}"
            )));
        }
        let host = logits
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // candle layout is channel-major [c, H, W] (c slowest). The reference
        // raw_gs layout is (h*W + w)*37 + c (c fastest).
        let hw = h * w;
        let mut raw_gs = vec![0.0f32; hw * 37];
        let mut gs_conf = vec![0.0f32; hw];
        for i in 0..hw {
            for c in 0..37 {
                raw_gs[i * 37 + c] = host[c * hw + i];
            }
            // gs_conf = sigmoid(logit[37]).
            gs_conf[i] = 1.0 / (1.0 + (-host[37 * hw + i]).exp());
        }
        Ok(RawGaussians {
            raw_gs,
            gs_conf,
            h,
            w,
        })
    }
}

// ---- helpers (mirrors `src/dpt_head.rs`; kept local to avoid leaking
// `dpt_head`'s private helpers into the public API). -----------------------

fn conv_weights(
    vb: &VarBuilder,
    name: &str,
    out_c: usize,
    in_c: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    Ok((
        vb.pp(name).get((out_c, in_c, k, k), "weight")?,
        vb.pp(name).get((out_c,), "bias")?,
    ))
}

fn conv_weights_1x1(
    vb: &VarBuilder,
    name: &str,
    in_c: usize,
    out_c: usize,
) -> Result<(Tensor, Tensor)> {
    conv_weights(vb, name, out_c, in_c, 1)
}

fn conv_no_bias(
    vb: &VarBuilder,
    name: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
) -> Result<Tensor> {
    Ok(vb.pp(name).get((out_c, in_c, k, k), "weight")?)
}

fn conv_transpose_weights(
    vb: &VarBuilder,
    name: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    // PyTorch ConvTranspose2d weight is [in_c, out_c, kH, kW].
    Ok((
        vb.pp(name).get((in_c, out_c, k, k), "weight")?,
        vb.pp(name).get((in_c,), "bias")?,
    ))
}

fn zero_bias(device: &Device, n: usize) -> Tensor {
    Tensor::zeros((n,), DType::F32, device).unwrap()
}

/// Add the cached UV embedding to a `[1, C, H, W]` feature map.
fn add_uv(
    x: &Tensor,
    w: usize,
    h: usize,
    c: usize,
    aspect: f32,
    device: &Device,
) -> Result<Tensor> {
    let buf = uv_embed_chw_cached(w, h, c, aspect, UV_RATIO);
    let pe = Tensor::from_vec(buf, (1, c, h, w), device)?.to_dtype(DType::F32)?;
    Ok(x.broadcast_add(&pe)?)
}

fn conv_fwd(w: &Tensor, b: &Tensor, cfg: Conv2dConfig, x: &Tensor) -> Result<Tensor> {
    Conv2d::new(w.clone(), Some(b.clone()), cfg)
        .forward(x)
        .map_err(Into::into)
}

fn conv_t_fwd(w: &Tensor, b: &Tensor, cfg: ConvTranspose2dConfig, x: &Tensor) -> Result<Tensor> {
    ConvTranspose2d::new(w.clone(), Some(b.clone()), cfg)
        .forward(x)
        .map_err(Into::into)
}

fn load_fusion(vb: &VarBuilder, name: &str, has_rc1: bool, features: usize) -> Result<Fusion> {
    // `vb` is already the `gs.*` VarBuilder; do NOT re-prepend "gs".
    let rn = vb.pp("scratch").pp(name);
    let rc1 = if has_rc1 {
        Some((
            conv_weights(&rn, "rc1.c1", features, features, 3)?,
            conv_weights(&rn, "rc1.c2", features, features, 3)?,
        ))
    } else {
        None
    };
    let rc2 = (
        conv_weights(&rn, "rc2.c1", features, features, 3)?,
        conv_weights(&rn, "rc2.c2", features, features, 3)?,
    );
    // 1x1 out conv: features -> features (256 -> 256).
    let outc = conv_weights_1x1(&rn, "out", features, features)?;
    Ok(Fusion { rc1, rc2, outc })
}

/// Residual conv unit: relu → 3x3 pad1 → relu → 3x3 pad1 → +x.
fn residual_conv_unit(x: &Tensor, c1: &(Tensor, Tensor), c2: &(Tensor, Tensor)) -> Result<Tensor> {
    let h = x.relu()?;
    let h = conv_fwd(
        &c1.0,
        &c1.1,
        Conv2dConfig {
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        },
        &h,
    )?;
    let h = h.relu()?;
    let h = conv_fwd(
        &c2.0,
        &c2.1,
        Conv2dConfig {
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        },
        &h,
    )?;
    Ok((&h + x)?)
}

/// Feature fusion stage, matching `dpt_blocks.cpp::feature_fusion`.
fn fusion_forward(
    f: &Fusion,
    top: &Tensor,
    lateral: Option<&Tensor>,
    out_w: usize,
    out_h: usize,
) -> Result<Tensor> {
    let mut y = top.clone();
    if let (Some(lat), Some(rc1)) = (lateral, &f.rc1) {
        let res = residual_conv_unit(lat, &rc1.0, &rc1.1)?;
        y = (&y + &res)?;
    }
    y = residual_conv_unit(&y, &f.rc2.0, &f.rc2.1)?;
    let (oh, ow) = if out_w == 0 && out_h == 0 {
        let dims = y.dims();
        (dims[2] * 2, dims[3] * 2)
    } else {
        (out_h, out_w)
    };
    y = upsample_bilinear_ac(&y, oh, ow)?;
    let out = conv_fwd(
        &f.outc.0,
        &f.outc.1,
        Conv2dConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        },
        &y,
    )?;
    Ok(out)
}

/// Bilinear upsample with `align_corners=true` (matches `interp_bilinear_ac`).
///
/// Identical to `dpt_head::upsample_bilinear_ac`; kept local so the GS head is
/// self-contained (no `pub(crate)` dependency on the DPT head's private helper).
fn upsample_bilinear_ac(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    if out_h == h && out_w == w {
        return Ok(x.clone());
    }
    let x = x.contiguous()?;
    let data = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0.0f32; b * c * out_h * out_w];
    let scale_y = if out_h > 1 {
        (h - 1) as f32 / (out_h - 1) as f32
    } else {
        0.0
    };
    let scale_x = if out_w > 1 {
        (w - 1) as f32 / (out_w - 1) as f32
    } else {
        0.0
    };
    for bi in 0..b {
        for ci in 0..c {
            let in_off = (bi * c + ci) * h * w;
            let out_off = (bi * c + ci) * out_h * out_w;
            for oy in 0..out_h {
                let fy = oy as f32 * scale_y;
                let y0 = fy.floor() as usize;
                let y1 = (y0 + 1).min(h - 1);
                let wy = fy - y0 as f32;
                for ox in 0..out_w {
                    let fx = ox as f32 * scale_x;
                    let x0 = fx.floor() as usize;
                    let x1 = (x0 + 1).min(w - 1);
                    let wx = fx - x0 as f32;
                    let p00 = data[in_off + y0 * w + x0];
                    let p01 = data[in_off + y0 * w + x1];
                    let p10 = data[in_off + y1 * w + x0];
                    let p11 = data[in_off + y1 * w + x1];
                    let top = p00 * (1.0 - wx) + p01 * wx;
                    let bot = p10 * (1.0 - wx) + p11 * wx;
                    out[out_off + oy * out_w + ox] = top * (1.0 - wy) + bot * wy;
                }
            }
        }
    }
    Ok(Tensor::from_vec(out, (b, c, out_h, out_w), x.device())?.to_dtype(DType::F32)?)
}

/// Whether a GGUF contains the GSDPT head (`gs.proj.0.weight`).
///
/// Mirrors the C++ engine's `bool Engine::has_gs_head()` (presence of the
/// `gs.*` tensor namespace). Use this to gate [`Engine::reconstruct_image`].
pub fn has_gs_head(file: &GgufFile) -> bool {
    file.tensor_info("gs.proj.0.weight").is_some()
}
