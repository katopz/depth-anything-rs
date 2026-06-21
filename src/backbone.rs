//! The DINOv2-style ViT backbone, matching `src/dino_backbone.cpp`.
//!
//! Pipeline: patch-embed conv → CLS prepend → bicubic pos-embed add →
//! transformer block loop (with camera-token injection at `alt_start` and a
//! local/nodiff RoPE-table switch for global blocks) → per-out-layer feature
//! assembly as `cat[local_x, vit.norm(x)]` over the channel dim (when
//! `cat_token=true`).

use crate::config::Config;
use crate::pos_embed::interp_pos_embed_cached;
use crate::rope2d::{build_backbone_tables, build_rope_tables};
use crate::vit_block::VitBlock;
use crate::Result;
use candle::{DType, Device, Tensor};
use candle_nn::conv::Conv2d;
use candle_nn::{layer_norm, Conv2dConfig, LayerNorm, Module, VarBuilder};

/// The loaded backbone: patch-embed conv, blocks, final norm, the CLS and
/// camera tokens, and the stored (pre-interp) positional embedding.
pub struct Backbone {
    patch_weight: Tensor,
    patch_bias: Tensor,
    cls_token: Tensor,
    camera_token: Tensor,
    pos_embed_raw: Vec<f32>,
    blocks: Vec<VitBlock>,
    final_norm: LayerNorm,
    cfg: Config,
}

/// The assembled per-out-layer features, each `[1, n_patch, channels]`.
#[derive(Debug, Clone)]
pub struct BackboneFeatures {
    /// One tensor per out-layer (length 4 for DA3: layers 5,7,9,11).
    pub feats: Vec<Tensor>,
    /// The camera token (`[1, 1, channels]`) used as input to the pose head.
    pub cam_token: Tensor,
}

/// The multi-view analogue of [`BackboneFeatures`]: per-view per-out-layer
/// features and cam tokens, plus the selected reference view index.
///
/// `feats[l][s]` is `[1, n_patch, channels]` and `cam_tokens[l][s]` is
/// `[1, 1, channels]` for out-layer `l` and view `s`.
#[derive(Debug)]
pub struct BackboneFeaturesMv {
    /// `[out_layer][view]` -> `[1, n_patch, channels]` feature tensor.
    pub feats: Vec<Vec<Tensor>>,
    /// `[out_layer][view]` -> `[1, 1, channels]` camera token.
    pub cam_tokens: Vec<Vec<Tensor>>,
    /// The selected reference view index (0 when no selection was applied,
    /// e.g. S < 3 or `alt_start < 0`).
    pub ref_view: usize,
}

/// Internal return type of `forward_mv_ordered`: `(feats, cam_tokens)` each
/// indexed `[out_layer][view]`.
type MvFeatures = (Vec<Vec<Tensor>>, Vec<Vec<Tensor>>);

impl Backbone {
    pub fn load(
        vb: &VarBuilder,
        cfg: &Config,
        file: &crate::gguf::GgufFile,
        device: &Device,
    ) -> Result<Self> {
        let embed = cfg.embed_dim as usize;

        // patch_embed.{weight,bias}
        let patch_weight = vb.pp("vit").pp("patch_embed").get(
            (embed, 3, cfg.patch_size as usize, cfg.patch_size as usize),
            "weight",
        )?;
        let patch_bias = vb.pp("vit").pp("patch_embed").get((embed,), "bias")?;

        // cls_token: ggml ne=[embed,1] → candle [1, embed]. Reshape to [1,1,embed].
        let cls_token = vb
            .pp("vit")
            .get((embed,), "cls_token")
            .or_else(|_| {
                // Some GGUFs store it as [1, embed].
                vb.pp("vit").get((1, embed), "cls_token")
            })?
            .reshape((1, 1, embed))?;

        // camera_token: ne=[embed,2] → candle [2, embed]. We use row 0.
        let camera_token = vb
            .pp("vit")
            .get((2, embed), "camera_token")
            .or_else(|_| vb.pp("vit").get((embed,), "camera_token"))?;

        // pos_embed: ne=[embed, 1+M*M] → candle [1+M*M, embed]. We read it as a
        // raw f32 vec for the cached bicubic interpolation.
        let pos_embed_raw = file.tensor_f32("vit.pos_embed")?;
        let expected = (1 + cfg.pos_embed_grid as usize * cfg.pos_embed_grid as usize) * embed;
        if pos_embed_raw.len() != expected {
            return Err(crate::Error::Model(format!(
                "vit.pos_embed: expected {expected} elements, got {}",
                pos_embed_raw.len()
            )));
        }

        let blocks = (0..cfg.depth as usize)
            .map(|i| VitBlock::load(vb, cfg, i, device))
            .collect::<Result<Vec<_>>>()?;

        let final_norm = layer_norm(embed, cfg.ln_eps as f64, vb.pp("vit").pp("norm"))?;

        Ok(Self {
            patch_weight,
            patch_bias,
            cls_token,
            camera_token,
            pos_embed_raw,
            blocks,
            final_norm,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Run the full backbone forward pass on a CHW-float image `[1, 3, H, W]`
    /// (H, W already multiples of `patch_size`), producing the per-out-layer
    /// features and the camera token.
    pub fn forward(&self, image: &Tensor, gh: usize, gw: usize) -> Result<BackboneFeatures> {
        let device = image.device();
        let embed = self.cfg.embed_dim as usize;
        let patch = self.cfg.patch_size as usize;

        // ---- patch embed: conv2d k=s=patch, pad=0 ----
        let conv_cfg = Conv2dConfig {
            stride: patch,
            padding: 0,
            dilation: 1,
            groups: 1,
        };
        let conv = Conv2d::new(
            self.patch_weight.clone(),
            Some(self.patch_bias.clone()),
            conv_cfg,
        );
        let conv_out = conv.forward(image)?;
        // conv: [1, embed, gh, gw]
        let (b, _, h, w) = conv_out.dims4()?;
        debug_assert_eq!((h, w), (gh, gw));
        let n_patch = gh * gw;

        // -> [1, n_patch, embed]
        let mut x = conv_out
            .reshape((b, embed, n_patch))?
            .transpose(1, 2)?
            .contiguous()?;

        // ---- CLS prepend -> [1, 1+n_patch, embed] ----
        x = Tensor::cat(&[&self.cls_token, &x], 1)?;
        let n_tok = 1 + n_patch;

        // ---- positional embedding (cached bicubic) ----
        let pe = interp_pos_embed_cached(
            &self.pos_embed_raw,
            embed,
            self.cfg.pos_embed_grid as usize,
            gh,
            gw,
            self.cfg.interp_offset,
        );
        let pe_t = Tensor::from_vec(pe, (1, n_tok, embed), device)?.to_dtype(DType::F32)?;
        x = x.broadcast_add(&pe_t)?;

        // ---- camera token for injection at alt_start ----
        // camera_token: [2, embed] or [embed]; take row 0 -> [1, 1, embed].
        let cam_row0 = self.camera_token.narrow(0, 0, 1)?.reshape((1, 1, embed))?;

        // ---- RoPE tables (if rope_start >= 0) ----
        let rope_tables = if self.cfg.rope_start >= 0 {
            let (local, nodiff) =
                build_backbone_tables(n_tok, gw, self.cfg.head_dim as usize, self.cfg.rope_freq);
            let cos_l = Tensor::from_vec(local.cos, (n_tok, self.cfg.head_dim as usize), device)?;
            let sin_l = Tensor::from_vec(local.sin, (n_tok, self.cfg.head_dim as usize), device)?;
            let cos_n = Tensor::from_vec(nodiff.cos, (n_tok, self.cfg.head_dim as usize), device)?;
            let sin_n = Tensor::from_vec(nodiff.sin, (n_tok, self.cfg.head_dim as usize), device)?;
            Some(((cos_l, sin_l), (cos_n, sin_n)))
        } else {
            None
        };

        // ---- block loop ----
        let mut local_x = x.clone();
        let mut captured: Vec<Option<(Tensor /*local_x*/, Tensor /*x*/)>> =
            vec![None; self.blocks.len()];

        for (i, block) in self.blocks.iter().enumerate() {
            // Camera-token injection at alt_start: overwrite x[:,0] (the CLS slot).
            if self.cfg.alt_start >= 0 && i as i32 == self.cfg.alt_start {
                let rest = x.narrow(1, 1, n_tok - 1)?.contiguous()?;
                x = Tensor::cat(&[&cam_row0, &rest], 1)?;
            }

            let is_global =
                self.cfg.alt_start >= 0 && i as i32 >= self.cfg.alt_start && (i % 2 == 1);
            let use_rope = self.cfg.rope_start >= 0 && i as i32 >= self.cfg.rope_start;
            let (cos, sin) = if use_rope {
                if let Some(((cl, sl), (cn, sn))) = &rope_tables {
                    if is_global {
                        (Some(cn.clone() as Tensor), Some(sn.clone() as Tensor))
                    } else {
                        (Some(cl.clone() as Tensor), Some(sl.clone() as Tensor))
                    }
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            x = block.forward(&x, cos.as_ref(), sin.as_ref())?;

            if !is_global {
                local_x = x.clone();
            }

            if self.cfg.out_layers.contains(&(i as i32)) {
                captured[i] = Some((local_x.clone(), x.clone()));
            }
        }

        // ---- feature assembly ----
        let mut feats = Vec::with_capacity(self.cfg.out_layers.len());
        let mut cam_feature: Option<Tensor> = None;
        for &layer in &self.cfg.out_layers {
            let (lx, xx) = captured
                .get(layer as usize)
                .cloned()
                .flatten()
                .ok_or_else(|| crate::Error::Model(format!("out_layer {layer} not captured")))?;
            if self.cfg.cat_token {
                // feat[t=1..] = cat[local_x[:,t], vit.norm(x)[:,t]] over channels.
                let normed = self.final_norm.forward(&xx)?;
                let lx_patch = lx.narrow(1, 1, n_patch)?.contiguous()?;
                let nx_patch = normed.narrow(1, 1, n_patch)?.contiguous()?;
                let feat = Tensor::cat(&[&lx_patch, &nx_patch], 2)?; // [1, n_patch, 2*embed]
                feats.push(feat);
                // cam token: cat[local_x[:,0], x[:,0]] (no norm on the 2nd half).
                let lx0 = lx.narrow(1, 0, 1)?;
                let xx0 = xx.narrow(1, 0, 1)?;
                cam_feature = Some(Tensor::cat(&[&lx0, &xx0], 2)?);
            } else {
                // feat[t=1..] = vit.norm(x)[:,1:].
                let normed = self.final_norm.forward(&xx)?;
                let feat = normed.narrow(1, 1, n_patch)?.contiguous()?;
                feats.push(feat);
                cam_feature = Some(xx.narrow(1, 0, 1)?.contiguous()?);
            }
        }
        let cam_token = cam_feature
            .ok_or_else(|| crate::Error::Model("no out_layers captured for camera token".into()))?;

        Ok(BackboneFeatures { feats, cam_token })
    }

    // ===================================================================
    // Multi-view batched forward (port of `DinoBackbone::forward_mv`).
    // ===================================================================
    //
    // A single backbone pass over S views with cross-view global attention at
    // odd-indexed blocks >= `alt_start`. Two internal paths:
    //
    // 1. `forward_mv_ordered` — the actual batched forward: patch-embed all S
    //    views into `[S, Ntok, embed]`, run the block loop with per-view LOCAL
    //    attention for most blocks and a single cross-view GLOBAL attention at
    //    odd blocks >= alt_start (token sequence flattened to `[1, S·Ntok,
    //    embed]`), then assemble per-view per-out-layer feats + cam tokens.
    //
    // 2. reference-view selection (S >= 3 and `alt_start >= 0` only): a light
    //    pass-A runs blocks `[0, alt_start-1)` per view, captures each view's
    //    CLS feature, and `select_reference_view_saddle` picks the view whose
    //    {avg cosine-sim, L2 norm, normalized variance} jointly sit closest to
    //    the per-metric median. Views are reordered ref-first for pass-B and
    //    restored to input order before the result is returned.
    //
    // The cam-token injection at `alt_start` writes the REFERENCE cam into
    // view-0's CLS slot and the SOURCE cam into every other view's. This is
    // what makes the global attention layers compare against the right anchor.

    /// Run the multi-view batched backbone over `images` `[S, 3, H, W]`, with
    /// `gh = H / patch_size`, `gw = W / patch_size`. Returns per-view
    /// per-out-layer features and cam tokens, plus the selected reference view
    /// index (0 when no selection is applied, e.g. S < 3).
    pub fn forward_mv(&self, images: &Tensor, gh: usize, gw: usize) -> Result<BackboneFeaturesMv> {
        let s = images.dim(0)?;
        if s == 0 {
            return Err(crate::Error::Model("forward_mv: empty view batch".into()));
        }

        // Reference-view selection: only for S >= 3 and alt_start >= 0.
        const THRESH_FOR_REF_SELECTION: usize = 3;
        if s < THRESH_FOR_REF_SELECTION || self.cfg.alt_start < 0 {
            let (feats, cam_tokens) = self.forward_mv_ordered(images, gh, gw)?;
            return Ok(BackboneFeaturesMv {
                feats,
                cam_tokens,
                ref_view: 0,
            });
        }

        // Pass A: capture CLS at the input of block (alt_start-1) = output of
        // block (alt_start-2). Run per-view LOCAL blocks [0, alt_start-1).
        let upto = (self.cfg.alt_start - 1) as usize;
        let cls = self.capture_local_cls(images, upto, gh, gw)?;
        let cls_host: Vec<Vec<f64>> = cls
            .iter()
            .map(|t| {
                let v: Vec<f32> = t
                    .to_dtype(DType::F32)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1()?;
                Ok::<Vec<f64>, candle::Error>(v.iter().map(|&x| x as f64).collect())
            })
            .collect::<candle::Result<Vec<Vec<f64>>>>()?;
        let ref_view = select_reference_view_saddle(&cls_host);

        // reorder_by_reference: position 0 <- b_idx; positions 1..b_idx <-
        // 0..b_idx-1; positions >b_idx unchanged.
        let mut order = vec![0usize; s];
        order[0] = ref_view;
        for (p, slot) in order.iter_mut().enumerate().skip(1) {
            *slot = if p <= ref_view { p - 1 } else { p };
        }
        let order_i64: Vec<i64> = order.iter().map(|&x| x as i64).collect();
        let order_t = Tensor::from_vec(order_i64, (s,), images.device())?;
        let ordered = images.index_select(&order_t, 0)?;

        // Pass B: full forward on reordered views (ref at position 0).
        let (feats_ordered, cams_ordered) = self.forward_mv_ordered(&ordered, gh, gw)?;

        // restore_original_order: target t<b_idx <- current t+1; target b_idx <-
        // current 0; target t>b_idx <- current t.
        let mut restore = vec![0usize; s];
        for (t, slot) in restore.iter_mut().enumerate() {
            *slot = if t < ref_view { t + 1 } else { t };
        }
        restore[ref_view] = 0;

        let nl = feats_ordered.len();
        let mut feats = vec![Vec::with_capacity(s); nl];
        let mut cam_tokens = vec![Vec::with_capacity(s); nl];
        for o in 0..nl {
            for t in 0..s {
                feats[o].push(feats_ordered[o][restore[t]].clone());
                cam_tokens[o].push(cams_ordered[o][restore[t]].clone());
            }
        }
        Ok(BackboneFeaturesMv {
            feats,
            cam_tokens,
            ref_view,
        })
    }

    /// The batched forward over views already in processing order. `images`
    /// is `[S, 3, H, W]`; returns `(feats, cam_tokens)` each indexed
    /// `[out_layer][view]`.
    fn forward_mv_ordered(&self, images: &Tensor, gh: usize, gw: usize) -> Result<MvFeatures> {
        let device = images.device();
        let embed = self.cfg.embed_dim as usize;
        let patch = self.cfg.patch_size as usize;
        let n_patch = gh * gw;
        let n_tok = 1 + n_patch;
        let s = images.dim(0)?;
        let head_dim = self.cfg.head_dim as usize;

        // ---- patch embed: conv2d k=s=patch on [S,3,H,W] -> [S,embed,gh,gw] ----
        let conv_cfg = Conv2dConfig {
            stride: patch,
            padding: 0,
            dilation: 1,
            groups: 1,
        };
        let conv = Conv2d::new(
            self.patch_weight.clone(),
            Some(self.patch_bias.clone()),
            conv_cfg,
        );
        let conv_out = conv.forward(images)?; // [S, embed, gh, gw]

        // -> [S, n_patch, embed] -> CLS prepend -> [S, Ntok, embed].
        let mut x = conv_out
            .reshape((s, embed, n_patch))?
            .transpose(1, 2)?
            .contiguous()?; // [S, n_patch, embed]
        let cls_b = self.cls_token.broadcast_as((s, 1, embed))?;
        x = Tensor::cat(&[&cls_b, &x], 1)?; // [S, Ntok, embed]

        // ---- positional embedding (broadcast over S) ----
        let pe = interp_pos_embed_cached(
            &self.pos_embed_raw,
            embed,
            self.cfg.pos_embed_grid as usize,
            gh,
            gw,
            self.cfg.interp_offset,
        );
        let pe_t = Tensor::from_vec(pe, (1, n_tok, embed), device)?.to_dtype(DType::F32)?;
        x = x.broadcast_add(&pe_t)?;

        // ---- per-view camera token slots: ref = row 0, src = row 1 ----
        // At block alt_start, view 0's CLS slot is overwritten with cam_ref and
        // every other view's with cam_src. Build [S, 1, embed].
        let cam_ref = self.camera_token.narrow(0, 0, 1)?.reshape((1, 1, embed))?;
        let cam_src = self.camera_token.narrow(0, 1, 1)?.reshape((1, 1, embed))?;
        let cam_per_view = {
            let mut cams: Vec<Tensor> = Vec::with_capacity(s);
            cams.push(cam_ref.clone());
            for _ in 1..s {
                cams.push(cam_src.clone());
            }
            let cam_refs: Vec<&Tensor> = cams.iter().collect();
            Tensor::cat(&cam_refs, 0)? // [S, 1, embed]
        };

        // ---- RoPE tables ----
        // local: [Ntok, head_dim] (identical per view; broadcasts over S).
        // global nodiff: pos_nodiff tiled S times -> [S·Ntok, head_dim].
        let rope_tables = if self.cfg.rope_start >= 0 {
            let (rt_local, _) = build_backbone_tables(n_tok, gw, head_dim, self.cfg.rope_freq);
            let cos_l = Tensor::from_vec(rt_local.cos, (n_tok, head_dim), device)?;
            let sin_l = Tensor::from_vec(rt_local.sin, (n_tok, head_dim), device)?;
            // Tile the nodiff positions S times (view-major) and build the
            // global table: token 0 = (0,0), tokens 1+ = (1,1).
            let mut pos_nodiff_g = vec![0.0f32; 2 * n_tok * s];
            for ss in 0..s {
                for t in 0..n_tok {
                    let (py, px) = if t == 0 { (0.0, 0.0) } else { (1.0, 1.0) };
                    pos_nodiff_g[2 * (ss * n_tok + t)] = py;
                    pos_nodiff_g[2 * (ss * n_tok + t) + 1] = px;
                }
            }
            let rt_g = build_rope_tables(&pos_nodiff_g, head_dim, self.cfg.rope_freq);
            let cos_g = Tensor::from_vec(rt_g.cos, (s * n_tok, head_dim), device)?;
            let sin_g = Tensor::from_vec(rt_g.sin, (s * n_tok, head_dim), device)?;
            Some(((cos_l, sin_l), (cos_g, sin_g)))
        } else {
            None
        };

        // ---- block loop ----
        let mut local_x = x.clone();
        let mut captured: Vec<Option<(Tensor /*local_x*/, Tensor /*x*/)>> =
            vec![None; self.blocks.len()];

        for (i, block) in self.blocks.iter().enumerate() {
            // Camera-token overwrite BEFORE block alt_start: view 0 <- cam_ref,
            // views >= 1 <- cam_src.
            if self.cfg.alt_start >= 0 && i as i32 == self.cfg.alt_start {
                let rest = x.narrow(1, 1, n_tok - 1)?.contiguous()?; // [S, Ntok-1, embed]
                x = Tensor::cat(&[&cam_per_view, &rest], 1)?; // [S, Ntok, embed]
            }

            let is_global =
                self.cfg.alt_start >= 0 && i as i32 >= self.cfg.alt_start && (i % 2 == 1);
            let use_rope = self.cfg.rope_start >= 0 && i as i32 >= self.cfg.rope_start;

            if is_global {
                // Cross-view: flatten [S, Ntok, embed] -> [1, S·Ntok, embed].
                let (cos, sin) = if use_rope {
                    rope_tables
                        .as_ref()
                        .map(|(_, g)| (Some(&g.0), Some(&g.1)))
                        .unwrap_or((None, None))
                } else {
                    (None, None)
                };
                let xf = x.contiguous()?.reshape((1, s * n_tok, embed))?;
                let xf = block.forward(&xf, cos, sin)?;
                // Materialize as a real tensor (mirrors the C++ double-ggml_cont).
                x = xf.contiguous()?.reshape((s, n_tok, embed))?.contiguous()?;
            } else {
                // Local: per-view independent attention (batch dim = S).
                let (cos, sin) = if use_rope {
                    rope_tables
                        .as_ref()
                        .map(|(l, _)| (Some(&l.0), Some(&l.1)))
                        .unwrap_or((None, None))
                } else {
                    (None, None)
                };
                x = block.forward(&x, cos, sin)?;
                local_x = x.clone();
            }

            if self.cfg.out_layers.contains(&(i as i32)) {
                captured[i] = Some((local_x.clone(), x.clone()));
            }
        }

        // ---- per-view feature assembly ----
        // For each out-layer: feat[s] = cat[local_x[s,1:], norm(x[s,1:])],
        // cam[s] = cat[local_x[s,0], x[s,0]]. Identical to the single-view
        // assembly, sliced per view.
        let mut feats: Vec<Vec<Tensor>> = Vec::with_capacity(self.cfg.out_layers.len());
        let mut cam_tokens: Vec<Vec<Tensor>> = Vec::with_capacity(self.cfg.out_layers.len());
        for &layer in &self.cfg.out_layers {
            let (lx, xx) = captured
                .get(layer as usize)
                .cloned()
                .flatten()
                .ok_or_else(|| crate::Error::Model(format!("out_layer {layer} not captured")))?;

            let mut layer_feats = Vec::with_capacity(s);
            let mut layer_cams = Vec::with_capacity(s);
            for v in 0..s {
                let lx_s = lx.narrow(0, v, 1)?; // [1, Ntok, embed]
                let xx_s = xx.narrow(0, v, 1)?;
                if self.cfg.cat_token {
                    let normed = self.final_norm.forward(&xx_s)?;
                    let lx_patch = lx_s.narrow(1, 1, n_patch)?.contiguous()?;
                    let nx_patch = normed.narrow(1, 1, n_patch)?.contiguous()?;
                    let feat = Tensor::cat(&[&lx_patch, &nx_patch], 2)?; // [1, n_patch, 2·embed]
                    layer_feats.push(feat);
                    let lx0 = lx_s.narrow(1, 0, 1)?;
                    let xx0 = xx_s.narrow(1, 0, 1)?;
                    layer_cams.push(Tensor::cat(&[&lx0, &xx0], 2)?); // [1, 1, 2·embed]
                } else {
                    let normed = self.final_norm.forward(&xx_s)?;
                    let feat = normed.narrow(1, 1, n_patch)?.contiguous()?;
                    layer_feats.push(feat);
                    layer_cams.push(xx_s.narrow(1, 0, 1)?.contiguous()?);
                }
            }
            feats.push(layer_feats);
            cam_tokens.push(layer_cams);
        }

        Ok((feats, cam_tokens))
    }

    /// Pass-A helper: run the per-view LOCAL blocks `[0, upto)` on every view
    /// and return each view's token-0 (CLS) feature `[embed]` (un-normalized).
    /// Used by [`Self::forward_mv`] for reference-view selection.
    fn capture_local_cls(
        &self,
        images: &Tensor,
        upto: usize,
        gh: usize,
        gw: usize,
    ) -> Result<Vec<Tensor>> {
        let device = images.device();
        let embed = self.cfg.embed_dim as usize;
        let patch = self.cfg.patch_size as usize;
        let n_patch = gh * gw;
        let n_tok = 1 + n_patch;
        let s = images.dim(0)?;
        let head_dim = self.cfg.head_dim as usize;

        let conv_cfg = Conv2dConfig {
            stride: patch,
            padding: 0,
            dilation: 1,
            groups: 1,
        };
        let conv = Conv2d::new(
            self.patch_weight.clone(),
            Some(self.patch_bias.clone()),
            conv_cfg,
        );
        let conv_out = conv.forward(images)?; // [S, embed, gh, gw]
        let mut x = conv_out
            .reshape((s, embed, n_patch))?
            .transpose(1, 2)?
            .contiguous()?; // [S, n_patch, embed]
        let cls_b = self.cls_token.broadcast_as((s, 1, embed))?;
        x = Tensor::cat(&[&cls_b, &x], 1)?; // [S, Ntok, embed]

        let pe = interp_pos_embed_cached(
            &self.pos_embed_raw,
            embed,
            self.cfg.pos_embed_grid as usize,
            gh,
            gw,
            self.cfg.interp_offset,
        );
        let pe_t = Tensor::from_vec(pe, (1, n_tok, embed), device)?.to_dtype(DType::F32)?;
        x = x.broadcast_add(&pe_t)?;

        // Only register local RoPE if a layer in [0, upto) actually uses it.
        let rope = if self.cfg.rope_start >= 0 && self.cfg.rope_start < upto as i32 {
            let (rt_local, _) = build_backbone_tables(n_tok, gw, head_dim, self.cfg.rope_freq);
            let cos_l = Tensor::from_vec(rt_local.cos, (n_tok, head_dim), device)?;
            let sin_l = Tensor::from_vec(rt_local.sin, (n_tok, head_dim), device)?;
            Some((cos_l, sin_l))
        } else {
            None
        };

        for (i, block) in self.blocks.iter().take(upto).enumerate() {
            let use_rope = self.cfg.rope_start >= 0 && i as i32 >= self.cfg.rope_start;
            let (cos, sin) = if use_rope {
                rope.as_ref()
                    .map(|(c, sn)| (Some(c.clone() as Tensor), Some(sn.clone() as Tensor)))
                    .unwrap_or((None, None))
            } else {
                (None, None)
            };
            x = block.forward(&x, cos.as_ref(), sin.as_ref())?;
        }

        // Extract token-0 per view -> [embed].
        let mut cls = Vec::with_capacity(s);
        for v in 0..s {
            cls.push(x.narrow(0, v, 1)?.narrow(1, 0, 1)?.reshape((embed,))?);
        }
        Ok(cls)
    }
}

/// `saddle_balanced` reference-view selection from per-view CLS features.
///
/// Computes three per-view metrics, normalizes each to `[0, 1]`, and picks the
/// view whose normalized metrics are jointly closest to the per-metric median
/// (0.5): `argmin_v |sim - 0.5| + |norm - 0.5| + |var - 0.5|`.
///
/// - `sim_score[v]` = mean over `w != v` of `cos(cls[v], cls[w])`.
/// - `feat_norm[v]` = L2 norm of `cls[v]`.
/// - `feat_var[v]`  = unbiased variance `(1/(C-1)) · Σ(cn[v] - mean)²` of the
///   unit-normalized cls.
///
/// `cls[s]` is the raw `[embed]` CLS vector (length `embed` for every view).
/// Returns 0 for `S <= 1`.
pub fn select_reference_view_saddle(cls: &[Vec<f64>]) -> usize {
    let s = cls.len();
    if s <= 1 {
        return 0;
    }
    let embed = cls[0].len();
    if embed == 0 {
        return 0;
    }

    // L2 norms + unit-normalized cls.
    let mut norm = vec![0.0f64; s];
    let mut cn = vec![vec![0.0f64; embed]; s];
    for v in 0..s {
        let mut n = 0.0f64;
        for &val in &cls[v] {
            n += val * val;
        }
        n = n.sqrt();
        norm[v] = n;
        let inv = if n > 0.0 { 1.0 / n } else { 0.0 };
        for (cn_v, &cls_v) in cn[v].iter_mut().zip(&cls[v]) {
            *cn_v = cls_v * inv;
        }
    }

    // sim_score, feat_norm, feat_var per view.
    let mut sim_score = vec![0.0f64; s];
    let mut feat_norm = vec![0.0f64; s];
    let mut feat_var = vec![0.0f64; s];
    for v in 0..s {
        let mut sum_sim = 0.0f64;
        for w in 0..s {
            if w == v {
                continue;
            }
            let dot: f64 = cn[v].iter().zip(&cn[w]).map(|(a, b)| a * b).sum();
            sum_sim += dot;
        }
        sim_score[v] = sum_sim / (s as f64 - 1.0);
        feat_norm[v] = norm[v];
        // Unbiased variance of the normalized cls over channels.
        let mean: f64 = cn[v].iter().sum::<f64>() / embed as f64;
        let var: f64 = cn[v].iter().map(|&x| (x - mean) * (x - mean)).sum();
        feat_var[v] = var / (embed as f64 - 1.0);
    }

    // min-max normalize each metric to [0, 1] (matches C++ norm01).
    let norm01 = |m: &mut [f64]| {
        let mut mn = m[0];
        let mut mx = m[0];
        for &v in m.iter() {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        let denom = mx - mn + 1e-8;
        for v in m.iter_mut() {
            *v = (*v - mn) / denom;
        }
    };
    norm01(&mut sim_score);
    norm01(&mut feat_norm);
    norm01(&mut feat_var);

    // Pick the view closest to (0.5, 0.5, 0.5).
    let mut best = 0usize;
    let mut best_bal = f64::INFINITY;
    for v in 0..s {
        let bal =
            (sim_score[v] - 0.5).abs() + (feat_norm[v] - 0.5).abs() + (feat_var[v] - 0.5).abs();
        if bal < best_bal {
            best_bal = bal;
            best = v;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- select_reference_view_saddle ----------------------------------

    #[test]
    fn saddle_empty_returns_zero() {
        assert_eq!(select_reference_view_saddle(&[]), 0);
    }

    #[test]
    fn saddle_single_returns_zero() {
        assert_eq!(select_reference_view_saddle(&[vec![1.0, 2.0, 3.0]]), 0);
    }

    #[test]
    fn saddle_picks_central_view() {
        // Three views with different norms: [1,0,0] (norm 1), [1,1,0] (norm √2),
        // [1,1,1] (norm √3). After norm01 normalization, the median-norm view
        // is the one whose normalized norm is closest to 0.5. The middle view
        // [1,1,0] should win.
        let cls = vec![
            vec![1.0f64, 0.0, 0.0],
            vec![1.0f64, 1.0, 0.0],
            vec![1.0f64, 1.0, 1.0],
        ];
        let best = select_reference_view_saddle(&cls);
        // The middle view (index 1) has the median norm.
        assert_eq!(best, 1);
    }

    #[test]
    fn saddle_all_identical_returns_zero() {
        // When all views are identical, all metrics are identical, so the
        // first view (index 0) is picked (tie goes to the first).
        let cls = vec![
            vec![1.0f64, 2.0, 3.0],
            vec![1.0f64, 2.0, 3.0],
            vec![1.0f64, 2.0, 3.0],
        ];
        assert_eq!(select_reference_view_saddle(&cls), 0);
    }

    #[test]
    fn saddle_orthogonal_views() {
        // Three orthogonal unit vectors: sim_score all 0 (after norm01 -> 0.5
        // for all), norms all 1 (norm01 -> 0.5 for all), vars all equal.
        // All metrics tie at 0.5 -> first view (0) wins.
        let cls = vec![
            vec![1.0f64, 0.0, 0.0],
            vec![0.0f64, 1.0, 0.0],
            vec![0.0f64, 0.0, 1.0],
        ];
        assert_eq!(select_reference_view_saddle(&cls), 0);
    }

    // ---- reorder / restore permutation ---------------------------------

    /// Reconstruct the C++ reorder and restore permutations for a given ref_view
    /// and view count S, and verify they're inverses.
    fn reorder_restore_roundtrip(s: usize, ref_view: usize) {
        // reorder: position 0 <- ref_view; 1..ref_view <- 0..ref_view-1;
        // >ref_view unchanged.
        let mut order = vec![0usize; s];
        order[0] = ref_view;
        for (p, slot) in order.iter_mut().enumerate().take(s).skip(1) {
            *slot = if p <= ref_view { p - 1 } else { p };
        }
        // restore: t < ref_view <- t+1; t == ref_view <- 0; t > ref_view <- t.
        let mut restore = vec![0usize; s];
        for (t, slot) in restore.iter_mut().enumerate() {
            *slot = if t < ref_view { t + 1 } else { t };
        }
        restore[ref_view] = 0;

        // restore[order[p]] == p for all p (restore is the inverse of order).
        for p in 0..s {
            assert_eq!(
                restore[order[p]], p,
                "restore∘order not identity at p={p}, s={s}, ref={ref_view}"
            );
        }
        // order is a valid permutation of 0..s.
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            (0..s).collect::<Vec<_>>(),
            "order not a permutation: s={s}, ref={ref_view}"
        );
    }

    #[test]
    fn reorder_restore_roundtrip_all_configs() {
        for s in 1..=6 {
            for ref_view in 0..s {
                reorder_restore_roundtrip(s, ref_view);
            }
        }
    }

    #[test]
    fn reorder_example_ref1_s4() {
        // S=4, ref_view=1: order = [1, 0, 2, 3], restore = [1, 0, 2, 3].
        let s = 4;
        let ref_view = 1;
        let mut order = vec![0usize; s];
        order[0] = ref_view;
        for (p, slot) in order.iter_mut().enumerate().take(s).skip(1) {
            *slot = if p <= ref_view { p - 1 } else { p };
        }
        assert_eq!(order, vec![1, 0, 2, 3]);
    }

    #[test]
    fn reorder_example_ref0_s3() {
        // S=3, ref_view=0: order = [0, 1, 2] (no change).
        let s = 3;
        let ref_view = 0;
        let mut order = vec![0usize; s];
        order[0] = ref_view;
        for (p, slot) in order.iter_mut().enumerate().take(s).skip(1) {
            *slot = if p <= ref_view { p - 1 } else { p };
        }
        assert_eq!(order, vec![0, 1, 2]);
    }
}
