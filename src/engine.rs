//! The top-level inference engine: load a GGUF, preprocess, run the backbone,
//! DPT head, and camera-pose head, and apply the depth/conf/sky activations.

use std::sync::Arc;
use std::time::Instant;

use crate::backbone::{Backbone, BackboneFeatures};
use crate::cam_pose::{decode_pose, CamPose};
use crate::config::Config;
use crate::depth_export::activate;
use crate::dpt_head::DptHead;
use crate::gguf::GgufFile;
use crate::gs_adapter::{Gaussians, GsAdapter};
use crate::gs_head::{has_gs_head, GsHead};
use crate::nested::{AnyviewOut, MetricOut, NestedAligner, NestedOut};
use crate::preprocess::{preprocess_real, Image, Preprocessed};
use crate::ray_pose::{solve_ray_pose, RayPoseIndices, RayPoseOut, RayPoseParams};
use crate::weights::{has_tensor, var_builder, var_builder_metric};
use crate::Result;
use candle::{DType, Device, Tensor};

/// Output of a depth inference call.
#[derive(Debug, Clone)]
pub struct DepthOutput {
    /// Row-major `[H*W]` depth (already activated — `exp`, `σ·m`, or `relu`).
    pub depth: Vec<f32>,
    /// Row-major `[H*W]` confidence (empty if the model has no conf channel).
    pub conf: Vec<f32>,
    /// Row-major `[H*W]` sky mask (empty if the model has no sky head).
    pub sky: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

/// Output of a camera-pose inference call.
#[derive(Debug, Clone)]
pub struct PoseOutput {
    /// Raw 9-vector `pose_enc`.
    pub pose_enc: Vec<f32>,
    /// Extrinsics, 3×4 row-major (12 floats).
    pub ext: [f32; 12],
    /// Intrinsics, 3×3 row-major (9 floats).
    pub intr: [f32; 9],
}

/// Per-view result of the multi-view pipeline ([`Engine::depth_pose_multi_image`]).
///
/// Mirrors `da::ViewResult` in `src/engine.hpp`. `depth`/`conf` are `[H*W]`
/// row-major at the processed resolution (identical H,W for every view).
#[derive(Debug, Clone)]
pub struct ViewResult {
    /// Row-major `[H*W]` depth (already activated).
    pub depth: Vec<f32>,
    /// Row-major `[H*W]` confidence (empty if the model has no conf channel).
    pub conf: Vec<f32>,
    /// Extrinsics, 3×4 row-major (12 floats).
    pub ext: [f32; 12],
    /// Intrinsics, 3×3 row-major (9 floats).
    pub intr: [f32; 9],
}

/// Output of [`Engine::depth_pose_multi_image`]: per-view depth + pose results,
/// plus the common processing resolution and the selected reference view.
#[derive(Debug)]
pub struct MultiViewOutput {
    /// One [`ViewResult`] per input view, in input order.
    pub views: Vec<ViewResult>,
    /// Common processing height (every view preprocesses to the same H).
    pub h: usize,
    /// Common processing width.
    pub w: usize,
    /// Selected reference view index (0 when S < 3 or `alt_start < 0`).
    pub ref_view: usize,
}

/// Output of [`Engine::reconstruct_image`]: the world-space 3D Gaussians plus
/// the common processing resolution.
#[derive(Debug)]
pub struct ReconstructOutput {
    /// The reconstructed Gaussians (`N = H * W`).
    pub gaussians: Gaussians,
    /// Processing height.
    pub h: usize,
    /// Processing width.
    pub w: usize,
}

/// Output of the ray-based pose path ([`Engine::depth_pose_rays_image`]).
///
/// The aux ray head's per-pixel ray field is turned into camera extrinsics +
/// intrinsics by the RANSAC homography + QL solver in [`crate::ray_pose`].
/// Unlike [`PoseOutput`], there is no `pose_enc` (the aux head emits a ray field,
/// not an MLP encoding).
#[derive(Debug, Clone, Default)]
pub struct RayPoseOutput {
    /// Extrinsics, 3×4 camera-to-world row-major (12 floats).
    pub ext: [f32; 12],
    /// Intrinsics, 3×3 row-major (9 floats).
    pub intr: [f32; 9],
    /// Diagnostics from the solver (best-hypothesis inlier count, focal, pp, …).
    pub diag: RayPoseDiag,
}

/// Solver diagnostics surfaced by [`Engine::depth_pose_rays_image`].
#[derive(Debug, Clone, Default)]
pub struct RayPoseDiag {
    /// Best-hypothesis inlier count (pre-subsample).
    pub n_inlier_best: usize,
    /// Returned focal = 1/f (per axis).
    pub focal: [f64; 2],
    /// Returned principal point = pp_raw + 1 (per axis).
    pub pp: [f64; 2],
}

/// Timings (milliseconds) for the last forward pass. Populated only when the
/// engine was constructed with `timings: true`.
#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub load_ms: f64,
    pub preprocess_ms: f64,
    pub backbone_ms: f64,
    pub head_ms: f64,
    pub pose_ms: f64,
    pub activate_ms: f64,
    pub total_ms: f64,
}

/// The inference engine.
pub struct Engine {
    cfg: Config,
    backbone: Backbone,
    head: DptHead,
    pose: Option<CamPose>,
    /// Optional GSDPT head (`gs.*`). Present only in DA3-Giant GGUFs that
    /// include the 3D-Gaussian reconstruction head. Required for
    /// [`Self::reconstruct_image`].
    gs_head: Option<GsHead>,
    device: Device,
    _file: Arc<GgufFile>,
    enable_timings: bool,
    last: std::sync::Mutex<Timings>,
    /// Nested-metric branch (loaded via [`Self::load_nested`]). When present,
    /// [`Self::depth_metric_image`] runs both branches and aligns them.
    metric: Option<MetricBranch>,
}

/// The metric (ViT-L + DPT/sky) branch of a nested-metric engine. Holds its
/// own backbone + head + config + GGUF file handle, independent of the
/// anyview branch's. Mirrors the C++ engine's `metric_ml_` + `metric_be_`.
struct MetricBranch {
    cfg: Config,
    backbone: Backbone,
    head: DptHead,
    _file: Arc<GgufFile>,
}

impl Engine {
    /// Load a GGUF model file onto `device` (or the default device).
    ///
    /// `enable_timings` records per-stage wall-clock timings, retrievable via
    /// [`Self::last_timings`]. Useful for the benchmark harness.
    pub fn load(path: &str, device: Option<Device>) -> Result<Self> {
        Self::load_with(path, device, false)
    }

    /// Like [`Self::load`] but with per-stage timing enabled.
    pub fn load_with_timings(path: &str, device: Option<Device>) -> Result<Self> {
        Self::load_with(path, device, true)
    }

    fn load_with(path: &str, device: Option<Device>, enable_timings: bool) -> Result<Self> {
        let t0 = Instant::now();
        let device = match device {
            Some(d) => d,
            None => crate::default_device()?,
        };
        let file = Arc::new(GgufFile::open(path)?);
        let cfg = Config::from_gguf(&file)?;
        let vb = var_builder(file.clone(), device.clone());

        let backbone = Backbone::load(&vb, &cfg, &file, &device)?;
        let head = DptHead::load(&vb, &cfg, &file, &device)?;
        let pose = if has_tensor(&file, "cam.bb0.weight") {
            Some(CamPose::load(&vb, &cfg, &file, &device)?)
        } else {
            None
        };
        let gs_head = if has_gs_head(&file) {
            Some(GsHead::load(&vb, &cfg, &file, &device)?)
        } else {
            None
        };
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let timings = Timings {
            load_ms,
            ..Default::default()
        };

        Ok(Self {
            cfg,
            backbone,
            head,
            pose,
            gs_head,
            device,
            _file: file,
            enable_timings,
            last: std::sync::Mutex::new(timings),
            metric: None,
        })
    }

    /// Load a nested-metric engine: the anyview (GIANT) GGUF plus a separate
    /// metric (ViT-L + DPT/sky) GGUF. After this, [`Self::depth_metric_image`]
    /// runs both branches + the alignment.
    ///
    /// Mirrors `Engine::load_nested` in `src/engine.cpp`. The metric GGUF's
    /// tensors live under `m_vit.*` / `m_head.*`; the engine's `VarBuilder`
    /// rewrites lookups so the same `Backbone`/`DptHead` loaders work unchanged
    /// (mirrors the alias map inserted by `ModelLoader::load`).
    pub fn load_nested(
        anyview_gguf: &str,
        metric_gguf: &str,
        device: Option<Device>,
    ) -> Result<Self> {
        let mut eng = Self::load(anyview_gguf, device)?;
        let m_file = Arc::new(GgufFile::open(metric_gguf)?);
        let m_cfg = Config::from_gguf_metric(&m_file)?;
        let m_vb = var_builder_metric(m_file.clone(), eng.device.clone());
        let m_backbone = Backbone::load(&m_vb, &m_cfg, &m_file, &eng.device)
            .map_err(|e| crate::Error::Model(format!("nested metric backbone load failed: {e}")))?;
        let m_head = DptHead::load(&m_vb, &m_cfg, &m_file, &eng.device)
            .map_err(|e| crate::Error::Model(format!("nested metric head load failed: {e}")))?;
        eng.metric = Some(MetricBranch {
            cfg: m_cfg,
            backbone: m_backbone,
            head: m_head,
            _file: m_file,
        });
        Ok(eng)
    }

    /// True iff this engine was created via [`Self::load_nested`] (anyview +
    /// metric branches both loaded). [`Self::depth_metric_image`] is then the
    /// valid inference path.
    pub fn is_nested(&self) -> bool {
        self.metric.is_some()
    }

    /// Config of the metric branch, if loaded.
    pub fn metric_config(&self) -> Option<&Config> {
        self.metric.as_ref().map(|m| &m.cfg)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn has_pose_head(&self) -> bool {
        self.pose.is_some()
    }

    /// Whether this model has the DualDPT auxiliary ray head (only `--with-aux`
    /// GGUFs do). Required for [`Self::depth_pose_rays_image`].
    pub fn has_aux_head(&self) -> bool {
        self.head.has_aux()
    }

    /// Timings recorded for the last forward (all zeros until a forward runs,
    /// and always zero if timing was disabled at construction).
    pub fn last_timings(&self) -> Timings {
        self.last.lock().unwrap().clone()
    }

    fn record(&self, t: &mut Timings) {
        if self.enable_timings {
            t.total_ms = t.preprocess_ms + t.backbone_ms + t.head_ms + t.pose_ms + t.activate_ms;
            *self.last.lock().unwrap() = t.clone();
        }
    }

    /// Run depth inference on an already-loaded [`Image`].
    pub fn depth_image(&self, img: &Image) -> Result<DepthOutput> {
        let mut t = Timings::default();
        let t0 = Instant::now();
        let pre = preprocess_real(img, &self.cfg)?;
        t.preprocess_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.depth_from_preprocessed(&pre, &mut t)
    }

    /// Run depth inference on an image file path.
    pub fn depth_path(&self, path: &str) -> Result<DepthOutput> {
        let img = Image::load(path)?;
        self.depth_image(&img)
    }

    fn depth_from_preprocessed(&self, pre: &Preprocessed, t: &mut Timings) -> Result<DepthOutput> {
        let (h, w) = (pre.h, pre.w);
        let gh = h / self.cfg.patch_size as usize;
        let gw = w / self.cfg.patch_size as usize;

        // Upload the CHW image as a candle tensor [1, 3, H, W].
        let img_t =
            Tensor::from_vec(pre.chw.clone(), (1, 3, h, w), &self.device)?.to_dtype(DType::F32)?;

        // Backbone.
        let t0 = Instant::now();
        let feats: BackboneFeatures = self.backbone.forward(&img_t, gh, gw)?;
        t.backbone_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // DPT head.
        let t0 = Instant::now();
        let (logits, sky) = self
            .head
            .forward(&feats.feats, gh, gw, h, w, &self.device)?;
        t.head_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Read back logits to host and apply activations.
        let t0 = Instant::now();
        let output_dim = self.head.output_dim();
        let hw = h * w;
        // logits is [1, output_dim, H, W]. Flatten WITHOUT permuting so the host
        // buffer is channel-major [output_dim, H*W] — i.e. logits[0..HW] = all
        // depth (ch0), logits[HW..2*HW] = all conf (ch1). This matches the C++
        // host post-process in `Engine::depth_native_fused` / `DptHead::run`, and
        // is what `activate` below expects (depth = exp(logits[i]) for i in 0..HW).
        let logits_host = logits
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let sky_host = match sky {
            Some(s) => {
                // sky is [1, 1, H, W]; single channel, plain flatten is fine.
                let v = s
                    .to_dtype(DType::F32)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                Some(v)
            }
            None => None,
        };
        let maps = activate(
            &logits_host,
            sky_host.as_deref(),
            hw,
            output_dim,
            self.cfg.head_max_depth,
        );
        t.activate_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.record(t);

        Ok(DepthOutput {
            depth: maps.depth,
            conf: maps.conf,
            sky: maps.sky,
            h,
            w,
        })
    }

    /// Run depth + camera pose on an already-loaded [`Image`].
    pub fn depth_pose_image(&self, img: &Image) -> Result<(DepthOutput, PoseOutput)> {
        let mut t = Timings::default();
        let t0 = Instant::now();
        let pre = preprocess_real(img, &self.cfg)?;
        t.preprocess_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let (h, w) = (pre.h, pre.w);
        let gh = h / self.cfg.patch_size as usize;
        let gw = w / self.cfg.patch_size as usize;

        let img_t =
            Tensor::from_vec(pre.chw.clone(), (1, 3, h, w), &self.device)?.to_dtype(DType::F32)?;

        let t0 = Instant::now();
        let feats = self.backbone.forward(&img_t, gh, gw)?;
        t.backbone_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let (logits, sky) = self
            .head
            .forward(&feats.feats, gh, gw, h, w, &self.device)?;
        t.head_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Pose head (optional).
        let t0 = Instant::now();
        let pose = if let Some(p) = &self.pose {
            let enc = p.forward_enc(&feats.cam_token)?;
            let enc_host = enc.flatten_all()?.to_vec1::<f32>()?;
            let (ext, intr) = decode_pose(&enc_host, w, h)?;
            Some(PoseOutput {
                pose_enc: enc_host,
                ext,
                intr,
            })
        } else {
            None
        };
        t.pose_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Activations.
        let t0 = Instant::now();
        let output_dim = self.head.output_dim();
        let hw = h * w;
        // Channel-major flatten [1, output_dim, H, W] -> [output_dim, H*W] (see
        // depth_from_preprocessed for why no permute).
        let logits_host = logits
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let sky_host = match sky {
            Some(s) => {
                let v = s
                    .to_dtype(DType::F32)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                Some(v)
            }
            None => None,
        };
        let maps = activate(
            &logits_host,
            sky_host.as_deref(),
            hw,
            output_dim,
            self.cfg.head_max_depth,
        );
        t.activate_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.record(&mut t);

        let depth = DepthOutput {
            depth: maps.depth,
            conf: maps.conf,
            sky: maps.sky,
            h,
            w,
        };
        let pose = pose.ok_or_else(|| {
            crate::Error::Unimplemented(
                "this model has no camera-pose head (cam.*); use depth_path instead",
            )
        })?;
        Ok((depth, pose))
    }

    /// Run depth + camera pose on an image file path.
    pub fn depth_pose_path(&self, path: &str) -> Result<(DepthOutput, PoseOutput)> {
        let img = Image::load(path)?;
        self.depth_pose_image(&img)
    }

    /// Run the **multi-view** pipeline on a batch of already-loaded [`Image`]s:
    /// a single backbone pass over all S views (with cross-view global
    /// attention at odd blocks >= `alt_start`), then per-view DPT depth head +
    /// camera-pose head. Mirrors `Engine::depth_pose_multi` in `src/engine.cpp`.
    ///
    /// All images must preprocess to the same `(H, W)` (else returns an error).
    /// The returned [`MultiViewOutput`] has one [`ViewResult`] per input view, in
    /// input order, plus the common `(h, w)` and the selected reference view.
    ///
    /// Requires a camera-pose head (`cam.*`); use [`Self::depth_image`] per view
    /// otherwise.
    pub fn depth_pose_multi_image(&self, imgs: &[Image]) -> Result<MultiViewOutput> {
        if imgs.is_empty() {
            return Err(crate::Error::Model("depth_pose_multi: no images".into()));
        }
        let pose = self.pose.as_ref().ok_or_else(|| {
            crate::Error::Unimplemented(
                "depth_pose_multi: this model has no camera-pose head (cam.*); \
                 use depth_image per view instead",
            )
        })?;

        let mut t = Timings::default();

        // Preprocess every image; all must yield identical (H, W).
        let t0 = Instant::now();
        let mut pres: Vec<Preprocessed> = Vec::with_capacity(imgs.len());
        let mut h = 0usize;
        let mut w = 0usize;
        for (i, img) in imgs.iter().enumerate() {
            let pre = preprocess_real(img, &self.cfg)?;
            if i == 0 {
                h = pre.h;
                w = pre.w;
            } else if (pre.h, pre.w) != (h, w) {
                return Err(crate::Error::Model(format!(
                    "depth_pose_multi: view {i} preprocesses to {}x{}, expected {h}x{w}",
                    pre.h, pre.w
                )));
            }
            pres.push(pre);
        }
        t.preprocess_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let s = pres.len();
        let gh = h / self.cfg.patch_size as usize;
        let gw = w / self.cfg.patch_size as usize;

        // Stack all views into a single [S, 3, H, W] tensor.
        let view_tensors: Vec<Tensor> = pres
            .iter()
            .map(|pre| {
                Tensor::from_vec(pre.chw.clone(), (1, 3, h, w), &self.device)
                    .and_then(|t| t.to_dtype(DType::F32))
            })
            .collect::<candle::Result<Vec<_>>>()?;
        let view_refs: Vec<&Tensor> = view_tensors.iter().collect();
        let img_t = Tensor::cat(&view_refs, 0)?; // [S, 3, H, W]

        // Backbone (one pass over all views).
        let t0 = Instant::now();
        let feats = self.backbone.forward_mv(&img_t, gh, gw)?;
        t.backbone_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Per-view: DPT depth head + camera pose head.
        let hw = h * w;
        let output_dim = self.head.output_dim();
        let mut views = Vec::with_capacity(s);
        for v in 0..s {
            let t0 = Instant::now();
            let feats_v: Vec<Tensor> = feats.feats.iter().map(|l| l[v].clone()).collect();
            let (logits, sky) = self.head.forward(&feats_v, gh, gw, h, w, &self.device)?;
            // Pose from the last out-layer's cam token.
            let cam_v = &feats.cam_tokens[feats.cam_tokens.len() - 1][v];
            let enc = pose.forward_enc(cam_v)?;
            let enc_host = enc.flatten_all()?.to_vec1::<f32>()?;
            let (ext, intr) = decode_pose(&enc_host, w, h)?;
            t.head_ms += t0.elapsed().as_secs_f64() * 1000.0;

            // Activations.
            let t0 = Instant::now();
            let logits_host = logits
                .to_dtype(DType::F32)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let sky_host = match sky {
                Some(sk) => Some(
                    sk.to_dtype(DType::F32)?
                        .contiguous()?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                ),
                None => None,
            };
            let maps = activate(
                &logits_host,
                sky_host.as_deref(),
                hw,
                output_dim,
                self.cfg.head_max_depth,
            );
            t.activate_ms += t0.elapsed().as_secs_f64() * 1000.0;

            views.push(ViewResult {
                depth: maps.depth,
                conf: maps.conf,
                ext,
                intr,
            });
        }
        // Backbone/head times are for the whole batch; normalize per-view.
        let sf = s as f64;
        t.backbone_ms /= sf;
        t.head_ms /= sf;
        t.activate_ms /= sf;
        self.record(&mut t);

        Ok(MultiViewOutput {
            views,
            h,
            w,
            ref_view: feats.ref_view,
        })
    }

    /// Like [`Self::depth_pose_multi_image`] but takes file paths.
    pub fn depth_pose_multi_paths(&self, paths: &[&str]) -> Result<MultiViewOutput> {
        let imgs: Vec<Image> = paths
            .iter()
            .map(|p| Image::load(p))
            .collect::<Result<Vec<_>>>()?;
        self.depth_pose_multi_image(&imgs)
    }

    /// Run the **nested metric** pipeline on an already-loaded [`Image`]:
    /// anyview GIANT (depth + conf + pose) and metric ViT-L (depth + sky)
    /// branches are run on the same preprocessed input, then fused by
    /// [`NestedAligner::align`] into a metric-scale depth map + rescaled pose.
    ///
    /// Requires an engine created via [`Self::load_nested`]. Mirrors
    /// `Engine::depth_metric` in `src/engine.cpp`. The returned `NestedOut`
    /// carries the metric-scale depth `[H*W]`, the scaled extrinsics
    /// (translation only), the unchanged intrinsics, and the fitted
    /// `scale_factor`.
    pub fn depth_metric_image(&self, img: &Image) -> Result<NestedOut> {
        let m = self.metric.as_ref().ok_or_else(|| {
            crate::Error::Unimplemented(
                "nested metric: engine not loaded via load_nested; no metric branch",
            )
        })?;
        // Both branches consume the SAME preprocessed input (da3.py
        // NestedDepthAnything3Net). Preprocessing uses the anyview config
        // (the outer engine's), matching the C++ which passes `ml_.config()`.
        let pre = preprocess_real(img, &self.cfg)?;
        let (h, w) = (pre.h, pre.w);
        let gh = h / self.cfg.patch_size as usize;
        let gw = w / self.cfg.patch_size as usize;

        let img_t =
            Tensor::from_vec(pre.chw.clone(), (1, 3, h, w), &self.device)?.to_dtype(DType::F32)?;

        // --- anyview (GIANT): backbone once -> depth + conf + cam pose ---
        let any_feats = self.backbone.forward(&img_t, gh, gw)?;
        let (any_logits, _) = self
            .head
            .forward(&any_feats.feats, gh, gw, h, w, &self.device)?;
        let pose = self.pose.as_ref().ok_or_else(|| {
            crate::Error::Unimplemented(
                "nested metric: anyview model has no camera-pose head (cam.*); load a DA3-GIANT anyview GGUF",
            )
        })?;
        let pe = pose.forward_enc(&any_feats.cam_token)?;
        let pe_host = pe.flatten_all()?.to_vec1::<f32>()?;
        let (extrinsics, intrinsics) = decode_pose(&pe_host, w, h)?;

        // Activations for the anyview depth + conf (output_dim==2 -> exp/exp+1).
        let hw = h * w;
        let output_dim = self.head.output_dim();
        let any_logits_host = any_logits
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let any_maps = activate(
            &any_logits_host,
            None,
            hw,
            output_dim,
            self.cfg.head_max_depth,
        );

        // --- metric (ViT-L + DPT/sky): backbone + depth_sky head ---
        let m_patch = m.cfg.patch_size as usize;
        if (h, w) != (h / m_patch * m_patch, w / m_patch * m_patch) {
            return Err(crate::Error::Model(format!(
                "nested metric: processed dims {h}x{w} not divisible by metric patch_size {m_patch}"
            )));
        }
        let mgh = h / m_patch;
        let mgw = w / m_patch;
        let m_feats = m.backbone.forward(&img_t, mgh, mgw)?;
        let (m_logits, m_sky) = m
            .head
            .forward(&m_feats.feats, mgh, mgw, h, w, &self.device)?;
        let m_logits_host = m_logits
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let m_sky_host = m_sky
            .map(|s| {
                s.to_dtype(DType::F32)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<f32>()
            })
            .transpose()?;
        let m_maps = activate(
            &m_logits_host,
            m_sky_host.as_deref(),
            hw,
            m.head.output_dim(),
            m.cfg.head_max_depth,
        );
        // Metric branch must have a sky map for the alignment; surface a clear
        // error otherwise (load_nested should have caught this at load time via
        // the GGUF not carrying head.scratch.sky_out2b, but double-check here).
        if m_maps.sky.is_empty() {
            return Err(crate::Error::Model(
                "nested metric: metric branch produced no sky map (expected head.scratch.sky_out2b)"
                    .into(),
            ));
        }

        let any = AnyviewOut {
            depth: any_maps.depth,
            depth_conf: any_maps.conf,
            extrinsics,
            intrinsics,
        };
        // process_mono_sky is the metric branch's own sky-fill applied inside
        // da3_metric(x) *before* alignment (NOT capped at 200 here).
        let mut metric_depth = m_maps.depth;
        let metric_sky = m_maps.sky;
        NestedAligner::process_mono_sky(&mut metric_depth, &metric_sky);
        let metric = MetricOut {
            depth: metric_depth,
            sky: metric_sky,
        };
        Ok(NestedAligner::align(&any, &metric, h, w))
    }

    /// Like [`Self::depth_metric_image`] but takes a file path.
    pub fn depth_metric_path(&self, path: &str) -> Result<NestedOut> {
        let img = Image::load(path)?;
        self.depth_metric_image(&img)
    }

    /// Whether this model has the GSDPT head (only DA3-Giant GGUFs do).
    /// Required for [`Self::reconstruct_image`].
    pub fn has_gs_head(&self) -> bool {
        self.gs_head.is_some()
    }

    /// Reconstruct world-space 3D Gaussians for `img` and write them to `path`
    /// as an INRIA-gaussian-splatting-compatible binary `.ply`.
    ///
    /// Requires the GSDPT head (`gs.*`); use [`Self::has_gs_head`] to gate.
    /// Mirrors `Engine::reconstruct` in `src/engine.cpp`. The pipeline is:
    ///
    /// 1. Preprocess once (anyview config).
    /// 2. Backbone forward → 4 out-layer features + last-layer cam token.
    /// 3. DPT depth head → depth (+ conf).
    /// 4. Camera-pose head → extrinsics + intrinsics.
    /// 5. GSDPT head → raw 37-ch Gaussian field + conf.
    /// 6. [`GsAdapter`] → world-space Gaussians (means, scales, rotations, SH,
    ///    opacities).
    /// 7. [`write_gaussian_ply`] → binary `.ply`.
    pub fn reconstruct_image(&self, img: &Image) -> Result<ReconstructOutput> {
        let gs = self.gs_head.as_ref().ok_or_else(|| {
            crate::Error::Unimplemented(
                "reconstruct: this model has no GSDPT head (gs.*); load a DA3-Giant GGUF",
            )
        })?;
        let pose = self.pose.as_ref().ok_or_else(|| {
            crate::Error::Unimplemented(
                "reconstruct: this model has no camera-pose head (cam.*); required for world-space Gaussians",
            )
        })?;

        let mut t = Timings::default();
        let t0 = Instant::now();
        let pre = preprocess_real(img, &self.cfg)?;
        t.preprocess_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (h, w) = (pre.h, pre.w);
        let gh = h / self.cfg.patch_size as usize;
        let gw = w / self.cfg.patch_size as usize;

        // Upload the CHW image as a candle tensor [1, 3, H, W].
        let img_t =
            Tensor::from_vec(pre.chw.clone(), (1, 3, h, w), &self.device)?.to_dtype(DType::F32)?;

        // Backbone (one pass; features feed depth, pose, AND the gs_head).
        let t0 = Instant::now();
        let feats: BackboneFeatures = self.backbone.forward(&img_t, gh, gw)?;
        t.backbone_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // DPT depth head + GSDPT head (independent; share only backbone feats).
        let t0 = Instant::now();
        let (logits, sky) = self
            .head
            .forward(&feats.feats, gh, gw, h, w, &self.device)?;
        let raw = gs.forward(&feats.feats, &img_t, gh, gw, h, w, &self.device)?;
        t.head_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Pose from the last out-layer's cam token.
        let t0 = Instant::now();
        let pe = pose.forward_enc(&feats.cam_token)?;
        let pe_host = pe.flatten_all()?.to_vec1::<f32>()?;
        let (ext, intr) = decode_pose(&pe_host, w, h)?;
        t.pose_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Activations on the depth head (giant: output_dim==2 -> exp/exp+1).
        let t0 = Instant::now();
        let output_dim = self.head.output_dim();
        let hw = h * w;
        let logits_host = logits
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let sky_host = match sky {
            Some(s) => Some(
                s.to_dtype(DType::F32)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            ),
            None => None,
        };
        let maps = activate(
            &logits_host,
            sky_host.as_deref(),
            hw,
            output_dim,
            self.cfg.head_max_depth,
        );
        t.activate_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Adapter: convert raw Gaussians + depth + pose into world-space.
        let adapter = GsAdapter::default();
        let gaussians = adapter
            .build(&raw.raw_gs, &maps.depth, &raw.gs_conf, &ext, &intr, h, w)
            .map_err(crate::Error::from)?;
        self.record(&mut t);

        Ok(ReconstructOutput { gaussians, h, w })
    }

    /// Like [`Self::reconstruct_image`] but takes a file path.
    pub fn reconstruct_path(&self, path: &str) -> Result<ReconstructOutput> {
        let img = Image::load(path)?;
        self.reconstruct_image(&img)
    }

    /// Run depth + ray-based pose on an already-loaded [`Image`].
    ///
    /// Requires the aux ray head (rebuild the GGUF with `--with-aux`). The ray
    /// field produced by the aux head is solved by [`crate::ray_pose`] with the
    /// **production** sampling path (seeded internal RANSAC — see the module's
    /// parity note). The depth output is identical to [`Self::depth_image`].
    pub fn depth_pose_rays_image(&self, img: &Image) -> Result<(DepthOutput, RayPoseOutput)> {
        self.depth_pose_rays_image_with(img, &RayPoseParams::default(), &RayPoseIndices::default())
    }

    /// Like [`Self::depth_pose_rays_image`] but with explicit solver tunables
    /// and (optionally) injected RANSAC indices for bit-exact parity with the
    /// C++ reference (the gated path).
    pub fn depth_pose_rays_image_with(
        &self,
        img: &Image,
        params: &RayPoseParams,
        indices: &RayPoseIndices<'_>,
    ) -> Result<(DepthOutput, RayPoseOutput)> {
        if !self.head.has_aux() {
            return Err(crate::Error::Unimplemented(
                "ray-pose: this GGUF has no aux ray head (rebuild with --with-aux)",
            ));
        }
        let mut t = Timings::default();
        let t0 = Instant::now();
        let pre = preprocess_real(img, &self.cfg)?;
        t.preprocess_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let (h, w) = (pre.h, pre.w);
        let gh = h / self.cfg.patch_size as usize;
        let gw = w / self.cfg.patch_size as usize;

        let img_t =
            Tensor::from_vec(pre.chw.clone(), (1, 3, h, w), &self.device)?.to_dtype(DType::F32)?;

        let t0 = Instant::now();
        let feats = self.backbone.forward(&img_t, gh, gw)?;
        t.backbone_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let (logits, sky) = self
            .head
            .forward(&feats.feats, gh, gw, h, w, &self.device)?;
        // Aux ray head runs on the SAME backbone features.
        let (ray, ray_conf, ray_h, ray_w) =
            self.head.forward_rays(&feats.feats, gh, gw, &self.device)?;
        t.head_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Ray -> pose solver (host-side double precision).
        let t0 = Instant::now();
        let RayPoseOut {
            extrinsics,
            intrinsics,
            focal,
            pp,
            n_inlier_best,
            ..
        } = solve_ray_pose(&ray, &ray_conf, ray_h, ray_w, h, w, indices, params)
            .map_err(|e| crate::Error::Model(format!("ray-pose solver: {e}")))?;
        t.pose_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Activations (depth head).
        let t0 = Instant::now();
        let output_dim = self.head.output_dim();
        let hw = h * w;
        let logits_host = logits
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let sky_host = match sky {
            Some(s) => {
                let v = s
                    .to_dtype(DType::F32)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                Some(v)
            }
            None => None,
        };
        let maps = activate(
            &logits_host,
            sky_host.as_deref(),
            hw,
            output_dim,
            self.cfg.head_max_depth,
        );
        t.activate_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.record(&mut t);

        let depth = DepthOutput {
            depth: maps.depth,
            conf: maps.conf,
            sky: maps.sky,
            h,
            w,
        };
        let pose = RayPoseOutput {
            ext: extrinsics,
            intr: intrinsics,
            diag: RayPoseDiag {
                n_inlier_best,
                focal,
                pp,
            },
        };
        Ok((depth, pose))
    }

    /// Run depth + ray-based pose on an image file path.
    pub fn depth_pose_rays_path(&self, path: &str) -> Result<(DepthOutput, RayPoseOutput)> {
        let img = Image::load(path)?;
        self.depth_pose_rays_image(&img)
    }

    /// Run depth + camera pose on `img` and write the requested 3D exports
    /// (glb / colmap). Mirrors the C++ `cmd_depth_export` single-view path:
    /// preprocess → backbone → depth head + camera pose → exporter.
    ///
    /// The camera-pose head is required (use a DA3 model with `cam.*` tensors);
    /// otherwise this returns [`crate::Error::Unimplemented`]. Only the exports
    /// named in `spec` are written; pass `None` for a field to skip it.
    pub fn depth_pose_export_image(&self, img: &Image, spec: &ExportSpec) -> Result<ExportResult> {
        // Reuse the full depth+pose pipeline so the exporters see exactly the
        // same depth/conf/ext/intr the inference path produces.
        let pre = preprocess_real(img, &self.cfg)?;
        let (depth, pose) = self.depth_pose_from_preprocessed(&pre)?;
        self.write_exports(&pre, &depth, &pose, spec)
    }

    /// Like [`Self::depth_pose_export_image`] but takes a file path.
    pub fn depth_pose_export_path(&self, path: &str, spec: &ExportSpec) -> Result<ExportResult> {
        let img = Image::load(path)?;
        self.depth_pose_export_image(&img, spec)
    }

    fn depth_pose_from_preprocessed(
        &self,
        pre: &Preprocessed,
    ) -> Result<(DepthOutput, PoseOutput)> {
        let (h, w) = (pre.h, pre.w);
        let gh = h / self.cfg.patch_size as usize;
        let gw = w / self.cfg.patch_size as usize;
        let img_t =
            Tensor::from_vec(pre.chw.clone(), (1, 3, h, w), &self.device)?.to_dtype(DType::F32)?;
        let feats = self.backbone.forward(&img_t, gh, gw)?;
        let (logits, sky) = self
            .head
            .forward(&feats.feats, gh, gw, h, w, &self.device)?;

        let pose = if let Some(p) = &self.pose {
            let enc = p.forward_enc(&feats.cam_token)?;
            let enc_host = enc.flatten_all()?.to_vec1::<f32>()?;
            let (ext, intr) = decode_pose(&enc_host, w, h)?;
            PoseOutput {
                pose_enc: enc_host,
                ext,
                intr,
            }
        } else {
            return Err(crate::Error::Unimplemented(
                "export needs a camera-pose head (cam.*); this model has none",
            ));
        };

        let output_dim = self.head.output_dim();
        let hw = h * w;
        let logits_host = logits
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let sky_host = match sky {
            Some(s) => {
                let v = s
                    .to_dtype(DType::F32)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                Some(v)
            }
            None => None,
        };
        let maps = activate(
            &logits_host,
            sky_host.as_deref(),
            hw,
            output_dim,
            self.cfg.head_max_depth,
        );
        let depth = DepthOutput {
            depth: maps.depth,
            conf: maps.conf,
            sky: maps.sky,
            h,
            w,
        };
        Ok((depth, pose))
    }

    fn write_exports(
        &self,
        pre: &Preprocessed,
        depth: &DepthOutput,
        pose: &PoseOutput,
        spec: &ExportSpec,
    ) -> Result<ExportResult> {
        let h = depth.h;
        let w = depth.w;
        let ext4 = pose_ext_to_4x4(&pose.ext);
        let k = pose.intr;
        let k_arr = [k];
        let ext_arr = [ext4];
        let img_slice: &[u8] = pre.rgb_u8.as_slice();
        let images: Vec<&[u8]> = vec![img_slice];
        let depth_slice: &[f32] = depth.depth.as_slice();
        let conf_slice: &[f32] = if depth.conf.is_empty() {
            &[]
        } else {
            depth.conf.as_slice()
        };
        let orig_wh: [(i64, i64); 1] = [(pre.orig_w as i64, pre.orig_h as i64)];

        let mut out = ExportResult::default();
        if let Some(glb_path) = spec.glb_path.as_deref() {
            crate::glb_export::write_glb(
                glb_path,
                depth_slice,
                conf_slice,
                &k_arr,
                &ext_arr,
                &images,
                h,
                w,
                1,
                spec.glb_options(),
            )?;
            out.glb_written = true;
        }
        if let Some(colmap_dir) = spec.colmap_dir.as_deref() {
            // Derive a stable image name from the input (caller can rename).
            let name = spec
                .image_name
                .clone()
                .unwrap_or_else(|| "image0001.png".to_string());
            let names = vec![name];
            crate::colmap_export::write_colmap(
                colmap_dir,
                depth_slice,
                conf_slice,
                &k_arr,
                &ext_arr,
                &images,
                &names,
                &orig_wh,
                h,
                w,
                1,
                spec.colmap_binary,
            )?;
            out.colmap_written = true;
        }
        Ok(out)
    }
}

/// Pad a 3×4 camera extrinsic (row-major, 12 floats) to a 4×4 row-major matrix
/// by appending the row `[0, 0, 0, 1]`. Required because the exporters take 4×4
/// world-to-camera matrices (the camera-pose head emits only the 3×4 part).
pub fn pose_ext_to_4x4(ext: &[f32; 12]) -> [f32; 16] {
    [
        ext[0], ext[1], ext[2], ext[3], //
        ext[4], ext[5], ext[6], ext[7], //
        ext[8], ext[9], ext[10], ext[11], //
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Which 3D exports to write from a single-view depth+pose result, and the
/// options for each. Mirrors the C++ CLI `--glb` / `--colmap` / `--ply` flags
/// (PLY is reserved for the not-yet-ported Gaussian path).
#[derive(Debug, Clone, Default)]
pub struct ExportSpec {
    /// Output `.glb` path; `None` to skip the glb export.
    pub glb_path: Option<String>,
    /// Output COLMAP directory; `None` to skip. Created if missing.
    pub colmap_dir: Option<String>,
    /// COLMAP binary (`.bin`) vs text (`.txt`) format. Defaults to binary.
    pub colmap_binary: bool,
    /// Override the COLMAP image basename (defaults to `"image0001.png"`).
    pub image_name: Option<String>,
    /// GLB options; `None` uses [`crate::glb_export::GlbOptions::default`].
    pub glb_opts: Option<crate::glb_export::GlbOptions>,
}

impl ExportSpec {
    fn glb_options(&self) -> &crate::glb_export::GlbOptions {
        self.glb_opts.as_ref().unwrap_or(&DEFAULT_GLB_OPTIONS)
    }
}

// A single shared default so the borrow in `write_exports` can use `&`.
static DEFAULT_GLB_OPTIONS: crate::glb_export::GlbOptions = crate::glb_export::GlbOptions {
    num_max_points: 1_000_000,
    show_cameras: true,
    camera_size: 0.03,
    conf_thresh: 1.05,
    conf_thresh_percentile: 40.0,
    ensure_thresh_percentile: 90.0,
};

/// What was actually written by [`Engine::depth_pose_export_image`].
#[derive(Debug, Clone, Default)]
pub struct ExportResult {
    /// Whether a `.glb` file was written.
    pub glb_written: bool,
    /// Whether a COLMAP model directory was written.
    pub colmap_written: bool,
}
