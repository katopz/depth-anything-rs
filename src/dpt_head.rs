//! The DPT depth decoder head, matching `src/dpt_head.cpp` + `src/dpt_blocks.cpp`.
//!
//! Four backbone features at out-layers `[5,7,9,11]` are projected to the
//! pyramid channel widths `[oc0..oc3]`, resized to four pyramid levels, fed
//! through `layer{1..4}_rn` lateral 3×3 convs (→ `feat_half` channels), then
//! fused top-down by `feature_fusion` (residual conv units with bilinear upsample
//! and a 1×1 out conv). A final 3×3 conv upsamples to `(W,H)`; after adding the
//! UV embed, three more convs (3×3, ReLU, 1×1) produce the depth/confidence/sky
//! logits.

use crate::config::Config;
use crate::gguf::GgufFile;
use crate::uv_embed::uv_embed_chw_cached;
use crate::Result;
use candle::{DType, Device, Tensor};
use candle_nn::conv::{Conv2d, ConvTranspose2d};
use candle_nn::{Conv2dConfig, ConvTranspose2dConfig, Module, VarBuilder};

const UV_RATIO: f32 = 0.1;
/// Epsilon for the optional `head.norm` input LayerNorm (matches torch default).
const HEAD_NORM_EPS: f32 = 1e-5;

/// One DPT head, loaded from `head.*`.
pub struct DptHead {
    cfg: Config,
    in_norm: Option<(Tensor, Tensor)>,
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
    sky: Option<((Tensor, Tensor), (Tensor, Tensor))>,
    /// Optional DualDPT auxiliary ray head (`head.scratch.*_aux`). Present only
    /// in `--with-aux` GGUFs. `None` => this model has no ray head.
    aux: Option<AuxRayHead>,
}

/// One `refinenet{N}` fusion stage.
struct Fusion {
    /// Lateral residual conv unit. `None` for refinenet4.
    rc1: Option<((Tensor, Tensor), (Tensor, Tensor))>,
    /// Top residual conv unit (always present).
    rc2: ((Tensor, Tensor), (Tensor, Tensor)),
    /// 1×1 out conv (with bias), `feat_half → 128`.
    outc: (Tensor, Tensor),
}

/// The DualDPT auxiliary ray head — a fully independent fusion pyramid sharing
/// only the main `layer{i}_rn` laterals, followed by a 5-conv output block, a
/// channels-last LayerNorm, and a final 1×1 conv producing 7 channels
/// (6 ray + 1 confidence). Matches `build_depth_graph`'s aux branch and
/// `DptHead::rays` in `src/dpt_head.cpp`.
struct AuxRayHead {
    rn1: Fusion,
    rn2: Fusion,
    rn3: Fusion,
    rn4: Fusion,
    /// `out1_aux.{0..4}`: 5 sequential 3×3 pad1 convs, no activations between.
    out1: Vec<(Tensor, Tensor)>,
    /// `out2a_aux`: 3×3 pad1 conv (`64 → 32`).
    out2a: (Tensor, Tensor),
    /// `out2_aux_ln`: channels-last LayerNorm over 32 channels.
    ln: (Tensor, Tensor),
    /// `out2b_aux`: 1×1 conv (`32 → 7`).
    out2b: (Tensor, Tensor),
}

impl DptHead {
    pub fn load(vb: &VarBuilder, cfg: &Config, file: &GgufFile, device: &Device) -> Result<Self> {
        let _ = device;
        let embed = cfg.embed_dim as usize;
        let in_channels = if cfg.cat_token { 2 * embed } else { embed };
        // `features` is the DPT fusion-pyramid channel width = head_features
        // (128 for BASE). `feat_half` (= head_features/2 = 64) is the narrower
        // width used only by output_conv1's output / output_conv2a's input /
        // the UV embed / sky head input. This split mirrors `src/dpt_head.cpp`,
        // where `layer{i}_rn` -> head_features but `out1` projects 128 -> 64.
        let features = if cfg.head_features > 0 {
            cfg.head_features as usize
        } else {
            128
        };
        let feat_half = cfg.feat_half() as usize;
        let oc = cfg.head_out_channels_arr();
        let head_vb = vb.pp("head");

        let in_norm = match head_vb.get((in_channels,), "norm.weight") {
            Ok(w) => Some((w, head_vb.get((in_channels,), "norm.bias")?)),
            Err(_) => None,
        };

        let proj = [
            conv_weights_1x1(&head_vb, "proj.0", in_channels, oc[0] as usize)?,
            conv_weights_1x1(&head_vb, "proj.1", in_channels, oc[1] as usize)?,
            conv_weights_1x1(&head_vb, "proj.2", in_channels, oc[2] as usize)?,
            conv_weights_1x1(&head_vb, "proj.3", in_channels, oc[3] as usize)?,
        ];

        let resize0 =
            conv_transpose_weights(&head_vb, "resize.0", oc[0] as usize, oc[0] as usize, 4)?;
        let resize1 =
            conv_transpose_weights(&head_vb, "resize.1", oc[1] as usize, oc[1] as usize, 2)?;
        let resize3 = conv_weights(&head_vb, "resize.3", oc[3] as usize, oc[3] as usize, 3)?;

        let scratch = head_vb.pp("scratch");
        // layer{i}_rn: 3x3 (no bias), out_channels[i] -> features (128).
        let layer_rn = [
            conv_no_bias(&scratch, "layer1_rn", oc[0] as usize, features, 3)?,
            conv_no_bias(&scratch, "layer2_rn", oc[1] as usize, features, 3)?,
            conv_no_bias(&scratch, "layer3_rn", oc[2] as usize, features, 3)?,
            conv_no_bias(&scratch, "layer4_rn", oc[3] as usize, features, 3)?,
        ];

        // Fusion (refinenet) pyramid operates entirely at `features` (128).
        let rn1 = load_fusion(&head_vb, "rn1", true, features)?;
        let rn2 = load_fusion(&head_vb, "rn2", true, features)?;
        let rn3 = load_fusion(&head_vb, "rn3", true, features)?;
        let rn4 = load_fusion(&head_vb, "rn4", false, features)?;

        // output_conv1: 3x3, features(128) -> feat_half(64).
        let out1 = conv_weights(&head_vb, "scratch.out1", feat_half, features, 3)?;
        // output_conv2a: 3x3, feat_half(64) -> 32.
        let out2a = conv_weights(&head_vb, "scratch.out2a", 32, feat_half, 3)?;

        let output_dim = file
            .tensor_info("head.scratch.out2b.weight")
            .and_then(|i| i.candle_shape().first().copied())
            .ok_or_else(|| crate::Error::Model("missing head.scratch.out2b.weight".into()))?;
        let out2b = (
            head_vb.get((output_dim, 32, 1, 1), "scratch.out2b.weight")?,
            head_vb.get((output_dim,), "scratch.out2b.bias")?,
        );

        let sky = match head_vb.get((32, feat_half, 3, 3), "scratch.sky_out2a.weight") {
            Ok(w) => Some((
                (w, head_vb.get((32,), "scratch.sky_out2a.bias")?),
                (
                    head_vb.get((1, 32, 1, 1), "scratch.sky_out2b.weight")?,
                    head_vb.get((1,), "scratch.sky_out2b.bias")?,
                ),
            )),
            Err(_) => None,
        };

        // Optional DualDPT auxiliary ray head (only present in --with-aux GGUFs).
        // Detected by `head.scratch.rn1_aux.out.weight`. Conv shapes are read
        // dynamically from the GGUF so this is robust to the exact channel
        // widths (the reference alternates 128/64 across the 5 `out1_aux` convs).
        let aux = if file
            .tensor_info("head.scratch.rn1_aux.out.weight")
            .is_some()
        {
            Some(load_aux_ray_head(&head_vb, features, file)?)
        } else {
            None
        };

        Ok(Self {
            cfg: cfg.clone(),
            in_norm,
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
            sky,
            aux,
        })
    }

    /// Forward the decoder. Returns `(depth_logits, sky_logits)`.
    ///
    /// `gh`, `gw` are the patch-grid dims (`H/patch`, `W/patch`); `h`, `w` are
    /// the full processed image dims, used for the final bilinear upsample to
    /// full resolution (mirrors `interp_bilinear_ac(ctx, out, W, H)` in the C++).
    pub fn forward(
        &self,
        feats: &[Tensor],
        gh: usize,
        gw: usize,
        h: usize,
        w: usize,
        device: &Device,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let pw = gw;
        let ph = gh;
        let aspect = gw as f32 / gh as f32;

        let pad1 = Conv2dConfig {
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        };
        let pad0 = Conv2dConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        };

        let lat = self.compute_laterals(feats, pw, ph, aspect, device)?;

        // Top-down fusion.
        let out4 = fusion_forward(&self.rn4, &lat[3], None, pw, ph)?;
        let out3 = fusion_forward(&self.rn3, &out4, Some(&lat[2]), 2 * pw, 2 * ph)?;
        let out2 = fusion_forward(&self.rn2, &out3, Some(&lat[1]), 4 * pw, 4 * ph)?;
        let out1 = fusion_forward(&self.rn1, &out2, Some(&lat[0]), 0, 0)?;

        // output_conv1: 3x3 128→64.
        let mut out = conv_fwd(&self.out1.0, &self.out1.1, pad1, &out1)?;
        // Upsample to the full processed image resolution (H, W). This matches
        // `interp_bilinear_ac(ctx, out, W, H)` in src/dpt_head.cpp: the DPT
        // pyramid ends at ph*8 (= H * 8/patch) which is NOT the full image when
        // patch != 8, so an explicit bilinear upsample to (H, W) is required.
        let target_h = h;
        let target_w = w;
        if (out.dim(2)?, out.dim(3)?) != (target_h, target_w) {
            out = upsample_bilinear_ac(&out, target_h, target_w)?;
        }
        if self.cfg.head_pos_embed {
            let oc = out.dim(1)?;
            out = add_uv(&out, target_w, target_h, oc, aspect, device)?;
        }

        // Optional sky head.
        let sky = if let Some(((sa_w, sa_b), (sb_w, sb_b))) = &self.sky {
            let sk = conv_fwd(sa_w, sa_b, pad1, &out)?;
            let sk = sk.relu()?;
            let sk = conv_fwd(sb_w, sb_b, pad0, &sk)?;
            Some(sk)
        } else {
            None
        };

        // output_conv2: 3x3 64→32, ReLU, 1x1 32→output_dim.
        let out = conv_fwd(&self.out2a.0, &self.out2a.1, pad1, &out)?;
        let out = out.relu()?;
        let out = conv_fwd(&self.out2b.0, &self.out2b.1, pad0, &out)?;

        Ok((out, sky))
    }

    /// Project + resize each backbone stage and apply its `layer{i}_rn` lateral
    /// conv, producing the four `[1, features, h_i, w_i]` lateral feature maps
    /// shared by the main depth pyramid and the aux ray pyramid.
    ///
    /// Mirrors the first half of `build_depth_graph` (up through the
    /// `layer{i}_rn` convs). Returning these separately lets [`Self::forward`]
    /// and [`Self::forward_rays`] each build their own fusion pyramid on top.
    fn compute_laterals(
        &self,
        feats: &[Tensor],
        pw: usize,
        ph: usize,
        aspect: f32,
        device: &Device,
    ) -> Result<Vec<Tensor>> {
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

        // Project each stage and resize to its pyramid level.
        let mut l = Vec::with_capacity(4);
        for (s, feat) in feats.iter().enumerate().take(4) {
            let mut x = feat.clone();
            if let Some((w, b)) = &self.in_norm {
                x = candle_nn::ops::layer_norm(&x, w, b, HEAD_NORM_EPS)?;
            }
            // [1, n_patch, C] -> [1, C, ph, pw]
            let x = x
                .contiguous()?
                .reshape((1, ph, pw, x.dim(2)?))?
                .permute((0, 3, 1, 2))?;
            let mut x = conv_fwd(&self.proj[s].0, &self.proj[s].1, pad0, &x)?;
            if self.cfg.head_pos_embed {
                let oc = x.dim(1)?;
                x = add_uv(&x, pw, ph, oc, aspect, device)?;
            }
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

        // Laterals: 3x3, no bias (zero bias). layer_rn outputs `features` (128).
        let features = if self.cfg.head_features > 0 {
            self.cfg.head_features as usize
        } else {
            128
        };
        let zero = zero_bias(device, features);
        let lat: Vec<Tensor> = (0..4)
            .map(|s| conv_fwd(&self.layer_rn[s], &zero, pad1, &l[s]))
            .collect::<Result<Vec<_>>>()?;
        Ok(lat)
    }

    pub fn output_dim(&self) -> usize {
        self.out2b.0.dim(0).unwrap()
    }

    pub fn has_sky(&self) -> bool {
        self.sky.is_some()
    }

    /// Whether this head has the DualDPT auxiliary ray head (only `--with-aux`
    /// GGUFs do).
    pub fn has_aux(&self) -> bool {
        self.aux.is_some()
    }

    /// Forward the auxiliary ray head. Returns `(ray, ray_conf, ray_h, ray_w)`
    /// where:
    /// - `ray` is `6 * ray_h * ray_w` row-major `(h, w, c)` (`c = 0..6`), with
    ///   `identity` activation (matches `DptHead::rays` in the C++).
    /// - `ray_conf` is `ray_h * ray_w`, with `exp(x) + 1` activation.
    /// - `ray_h == 8 * ph`, `ray_w == 8 * pw` (the finest aux level).
    ///
    /// Returns [`crate::Error::Unimplemented`] if this head has no aux head.
    pub fn forward_rays(
        &self,
        feats: &[Tensor],
        gh: usize,
        gw: usize,
        device: &Device,
    ) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
        let aux = self.aux.as_ref().ok_or_else(|| {
            crate::Error::Unimplemented("aux ray head (rebuild GGUF with --with-aux)")
        })?;
        let pw = gw;
        let ph = gh;
        let aspect = gw as f32 / gh as f32;

        let pad1 = Conv2dConfig {
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        };
        let pad0 = Conv2dConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        };

        // Reuse the main pyramid's laterals (aux shares only `layer{i}_rn`).
        let lat = self.compute_laterals(feats, pw, ph, aspect, device)?;

        // Aux fusion pyramid (mirrors the main one, weights from `rn{i}_aux`).
        let a4 = fusion_forward(&aux.rn4, &lat[3], None, pw, ph)?;
        let a3 = fusion_forward(&aux.rn3, &a4, Some(&lat[2]), 2 * pw, 2 * ph)?;
        let a2 = fusion_forward(&aux.rn2, &a3, Some(&lat[1]), 4 * pw, 4 * ph)?;
        let mut a = fusion_forward(&aux.rn1, &a2, Some(&lat[0]), 0, 0)?;

        // output_conv1_aux: 5 sequential 3x3 pad1 convs, no activations between.
        for (w, b) in &aux.out1 {
            a = conv_fwd(w, b, pad1, &a)?;
        }
        // Optional pos_embed on the finest aux feature, at its own resolution.
        if self.cfg.head_pos_embed {
            let (_, ac, ah, aw) = a.dims4()?;
            a = add_uv(&a, aw, ah, ac, aspect, device)?;
        }
        // output_conv2_aux: 3x3 pad1 -> channels-last LayerNorm(32) -> ReLU -> 1x1.
        a = conv_fwd(&aux.out2a.0, &aux.out2a.1, pad1, &a)?;
        a = layernorm_channels_last(&a, &aux.ln.0, &aux.ln.1, HEAD_NORM_EPS)?;
        a = a.relu()?;
        a = conv_fwd(&aux.out2b.0, &aux.out2b.1, pad0, &a)?;

        // `a` is `[1, 7, ray_h, ray_w]`. Read back to host and reorder to the
        // reference's (h, w, c) layout (identity on the 6 ray channels,
        // exp+1 on the confidence channel).
        let ray_w = 8 * pw;
        let ray_h = 8 * ph;
        let (b0, c7, dh, dw) = a.dims4()?;
        debug_assert_eq!((b0, c7), (1, 7));
        debug_assert_eq!((dh, dw), (ray_h, ray_w));
        let host = a
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let hwa = ray_h * ray_w;
        // candle layout is channel-major [c, ray_h, ray_w] (c slowest). The
        // reference ray layout is (h*ray_w + w)*6 + c (c fastest).
        let mut ray = vec![0.0f32; 6 * hwa];
        let mut ray_conf = vec![0.0f32; hwa];
        for p in 0..hwa {
            for c in 0..6 {
                ray[p * 6 + c] = host[c * hwa + p];
            }
            ray_conf[p] = host[6 * hwa + p].exp() + 1.0;
        }
        Ok((ray, ray_conf, ray_h, ray_w))
    }
}

// ---- helpers ----------------------------------------------------------

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

/// Functional conv2d: build a throwaway `Conv2d` module from loaded weights
/// and run one forward. (candle has no stateless functional conv2d op.)
fn conv_fwd(w: &Tensor, b: &Tensor, cfg: Conv2dConfig, x: &Tensor) -> Result<Tensor> {
    Conv2d::new(w.clone(), Some(b.clone()), cfg)
        .forward(x)
        .map_err(Into::into)
}

/// Functional conv-transpose2d.
fn conv_t_fwd(w: &Tensor, b: &Tensor, cfg: ConvTranspose2dConfig, x: &Tensor) -> Result<Tensor> {
    ConvTranspose2d::new(w.clone(), Some(b.clone()), cfg)
        .forward(x)
        .map_err(Into::into)
}

fn load_fusion(vb: &VarBuilder, name: &str, has_rc1: bool, features: usize) -> Result<Fusion> {
    // `vb` is already the `head.*` VarBuilder; do NOT re-prepend "head".
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
    // 1x1 out conv: features -> features (e.g. 128 -> 128). Matches the actual
    // `rn{i}.out.weight` shape [features, features, 1, 1].
    let outc = conv_weights_1x1(&rn, "out", features, features)?;
    Ok(Fusion { rc1, rc2, outc })
}

/// Load a conv `(weight, bias)` whose shape is read dynamically from the GGUF
/// (rather than hardcoded). Used by the aux head, whose 5 `out1_aux` convs
/// alternate channel widths the port does not want to bake in.
///
/// `weight_tensor` is the full dotted name `head.scratch.<name>.weight`.
fn conv_dyn_shape(
    file: &GgufFile,
    vb: &VarBuilder,
    name: &str,
) -> Result<Option<(Tensor, Tensor)>> {
    let full_w = format!("{name}.weight");
    let info = match file.tensor_info(&full_w) {
        Some(i) => i,
        None => return Ok(None),
    };
    let shape = info.candle_shape();
    if shape.len() != 4 {
        return Err(crate::Error::Model(format!(
            "conv weight {full_w}: expected 4 dims, got {shape:?}"
        )));
    }
    let w = vb.pp(name).get(shape.as_slice(), "weight")?;
    let bias_shape = shape[0]; // out_channels
    let b = vb.pp(name).get((bias_shape,), "bias")?;
    Ok(Some((w, b)))
}

/// Load the DualDPT auxiliary ray head (`head.scratch.*_aux`). `features` is the
/// main-pyramid width (128 for BASE) shared by the aux fusion stages.
///
/// See [`AuxRayHead`] and `src/dpt_head.cpp::build_depth_graph`'s aux branch.
fn load_aux_ray_head(head_vb: &VarBuilder, features: usize, file: &GgufFile) -> Result<AuxRayHead> {
    let rn1 = load_fusion(head_vb, "rn1_aux", true, features)?;
    let rn2 = load_fusion(head_vb, "rn2_aux", true, features)?;
    let rn3 = load_fusion(head_vb, "rn3_aux", true, features)?;
    let rn4 = load_fusion(head_vb, "rn4_aux", false, features)?;

    // out1_aux.{0..4}: 5 sequential 3x3 pad1 convs. Read shapes dynamically —
    // the reference alternates 128/64 out-channels across them.
    let scratch = head_vb.pp("scratch");
    let mut out1 = Vec::with_capacity(5);
    for i in 0..5 {
        let name = format!("out1_aux.{i}");
        let conv = conv_dyn_shape(file, &scratch, &name)?
            .ok_or_else(|| crate::Error::Model(format!("aux head: missing {name}.weight")))?;
        out1.push(conv);
    }
    // out2a_aux: 3x3 pad1 (final aux feature -> 32). out2_aux_ln: LayerNorm(32).
    // out2b_aux: 1x1 (32 -> 7 = 6 rays + 1 conf).
    let out2a = conv_dyn_shape(file, &scratch, "out2a_aux")?
        .ok_or_else(|| crate::Error::Model("aux head: missing out2a_aux.weight".into()))?;
    let ln = (
        scratch.get((32,), "out2_aux_ln.weight")?,
        scratch.get((32,), "out2_aux_ln.bias")?,
    );
    let out2b = conv_dyn_shape(file, &scratch, "out2b_aux")?
        .ok_or_else(|| crate::Error::Model("aux head: missing out2b_aux.weight".into()))?;

    Ok(AuxRayHead {
        rn1,
        rn2,
        rn3,
        rn4,
        out1,
        out2a,
        ln,
        out2b,
    })
}

/// Channels-last LayerNorm matching `layernorm_channels_last` in
/// `src/dpt_head.cpp` / `dpt_blocks.cpp`: permute `[N,C,H,W] -> [N,H,W,C]`,
/// normalize over C, apply affine, permute back. `w`, `b` are `[C]`.
///
/// This is equivalent to a single `nn.LayerNorm(C)` applied to the channels of
/// a `[N,C,H,W]` feature map — NOT a per-pixel normalization. Implemented with
/// candle's `layer_norm` over the last dim after a permute, to match the
/// reference's `ggml_norm`-over-C exactly.
fn layernorm_channels_last(x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
    // [N,C,H,W] -> [N,H,W,C]
    let (n, c, h, wd) = x.dims4()?;
    let x = x.permute((0, 2, 3, 1))?.contiguous()?;
    let x = x.reshape((n * h * wd, c))?;
    let x = candle_nn::ops::layer_norm(&x, w, b, eps)?;
    let x = x.reshape((n, h, wd, c))?;
    // [N,H,W,C] -> [N,C,H,W]
    Ok(x.permute((0, 3, 1, 2))?.contiguous()?)
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// `layernorm_channels_last` must equal a single `nn.LayerNorm(C)` applied
    /// across the channel dim of a `[N,C,H,W]` map (NOT a per-pixel norm).
    /// Construct a known input where each spatial location has a distinct
    /// channel distribution, then check against a hand-computed reference.
    #[test]
    fn layernorm_channels_last_matches_per_channel_norm() {
        let dev = Device::Cpu;
        // [N=1, C=2, H=1, W=2] — two pixels, each with a 2-vector.
        // channel-major layout (c slowest): [c0p0, c0p1, c1p0, c1p1]
        let x = Tensor::from_vec(
            vec![0.0f32, 4.0, 2.0, 6.0], // p0=(0,2), p1=(4,6)
            (1, 2, 1, 2),
            &dev,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
        // weight=1, bias=0, eps tiny so it doesn't perturb the analytic values.
        let w = Tensor::ones((2,), DType::F32, &dev).unwrap();
        let b = Tensor::zeros((2,), DType::F32, &dev).unwrap();
        let y = layernorm_channels_last(&x, &w, &b, 1e-12).unwrap();
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Per pixel: mean of the 2-vector, then (x-mean)/std where
        // std = sqrt(mean((x-mean)^2)) — torch LayerNorm uses biased var over C.
        // p0=(0,2): mean=1, var=((0-1)^2+(2-1)^2)/2 = 1, std=1 -> (-1, 1)
        // p1=(4,6): mean=5, var=1, std=1 -> (-1, 1)
        // Output layout (channel-major [c0p0, c0p1, c1p0, c1p1]):
        // c0 = (-1, -1), c1 = (1, 1)
        let expected = [-1.0f32, -1.0, 1.0, 1.0];
        for (i, (got, exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "idx {i}: {got} != {exp}");
        }
    }

    /// Channels-last LN must be invariant to the spatial dimensions (it normalizes
    /// over C only), and must broadcast the affine correctly across H, W.
    #[test]
    fn layernorm_channels_last_affine_broadcast() {
        let dev = Device::Cpu;
        // [1, 2, 2, 1] — H=2, W=1, C=2. Channel-major layout
        // [c0(h0,w0), c0(h1,w0), c1(h0,w0), c1(h1,w0)].
        // Pixel (h0,w0) = (c0=0, c1=2); pixel (h1,w0) = (c0=4, c1=6).
        let x = Tensor::from_vec(vec![0.0f32, 4.0, 2.0, 6.0], (1, 2, 2, 1), &dev)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        // Scale channel 0 by 3, channel 1 by -1; bias (10, 20).
        let w = Tensor::from_vec(vec![3.0f32, -1.0], (2,), &dev).unwrap();
        let b = Tensor::from_vec(vec![10.0f32, 20.0], (2,), &dev).unwrap();
        let y = layernorm_channels_last(&x, &w, &b, 1e-12).unwrap();
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Each pixel (0,2) and (4,6): mean=1 resp 5, var=1, std=1 -> (-1, 1).
        // affine: c0 -> -1*3 + 10 = 7 ; c1 -> 1*(-1) + 20 = 19.
        // Output layout [c0(h0), c0(h1), c1(h0), c1(h1)] = [7, 7, 19, 19]
        let expected = [7.0f32, 7.0, 19.0, 19.0];
        for (i, (got, exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-5, "idx {i}: {got} != {exp}");
        }
    }
}
