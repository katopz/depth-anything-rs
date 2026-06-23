//! Candle-free DPT depth head.
//!
//! [`FastDptHead`] runs the entire DPT decoder (input layer-norm, projection
//! convs, resize conv-transpose/convs, fusion pyramid with residual conv units,
//! bilinear upsamples, output convs, UV positional embed, optional sky head)
//! on raw `&[f32]` buffers, eliminating all of candle's per-op allocation and
//! dispatcher overhead for non-conv ops.
//!
//! Convolutions are delegated to [`crate::fast_conv`] (Winograd 3×3 + 1×1 GEMM).
//! Conv-transpose (k=stride, pad=0) and strided 3×3 conv are implemented inline.
//!
//! Activations flow as flat row-major `Vec<f32>` in NCHW `[1, C, H, W]` layout,
//! wrapped in a small [`Feat`] struct that tracks `(c, h, w)`.
//!
//! # Env var
//!
//! Enabled by `DA_FAST_HEAD=1` (same flag that enables [`crate::fast_conv`]);
//! see [`fast_dpt_enabled`].

use crate::fast_attn::flatten_to_f32;
use crate::fast_conv;
use crate::uv_embed::uv_embed_chw_cached;
use crate::Result;

use candle::Tensor;

/// Epsilon for the optional `head.norm` input LayerNorm (matches torch default).
const HEAD_NORM_EPS: f32 = 1e-5;
/// UV embed scale (matches C++ and [`crate::dpt_head`]).
const UV_RATIO: f32 = 0.1;

// ---------------------------------------------------------------------------
// Feature buffer: Vec<f32> + spatial dims
// ---------------------------------------------------------------------------

/// A 4D NCHW feature map with N=1.
#[derive(Clone)]
pub struct Feat {
    pub data: Vec<f32>,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl Feat {
    #[inline]
    pub fn new(data: Vec<f32>, c: usize, h: usize, w: usize) -> Self {
        debug_assert_eq!(data.len(), c * h * w, "Feat size mismatch");
        Self { data, c, h, w }
    }

    #[inline]
    pub fn hw(&self) -> usize {
        self.h * self.w
    }
}

// ---------------------------------------------------------------------------
// Weight structs (pre-extracted to flat f32 buffers)
// ---------------------------------------------------------------------------

/// 3×3 conv weights + bias.
struct Conv3x3W {
    weight: Vec<f32>, // [OC, IC, 3, 3]
    bias: Vec<f32>,   // [OC]
    ic: usize,
    oc: usize,
}

/// 1×1 conv weights + bias.
struct Conv1x1W {
    weight: Vec<f32>, // [OC, IC]
    bias: Vec<f32>,   // [OC]
    ic: usize,
    oc: usize,
}

/// Conv-transpose weights.
///
/// Weights are stored as `[in_c, ky, kx, out_c]` (out_c innermost) so that
/// the inner loop can use a contiguous `&[out_c]` slice per (ic, ky, kx)
/// triple. This is a transpose of PyTorch's `[in_c, out_c, ky, kx]` layout,
/// done once at extraction time.
struct ConvTransposeW {
    weight: Vec<f32>, // [in_c, ky, kx, out_c]
    bias: Vec<f32>,
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
}

struct ResConvUnitW {
    c1: Conv3x3W,
    c2: Conv3x3W,
}

struct FusionW {
    rc1: Option<ResConvUnitW>,
    rc2: ResConvUnitW,
    outc: Conv1x1W,
}

// ---------------------------------------------------------------------------
// FastDptHead
// ---------------------------------------------------------------------------

/// Candle-free DPT head. Holds pre-extracted weights.
///
/// Intermediate buffers are allocated per-`forward` call as local `Vec<f32>`s;
/// the global allocator (mimalloc/jemalloc on most platforms) recycles these
/// efficiently, and the cost is dwarfed by the conv kernels. The win over
/// candle comes from avoiding its per-op dispatcher overhead (type checks,
/// device dispatch, layout validation, tensor construction), not from buffer
/// reuse.
pub struct FastDptHead {
    // Optional input LayerNorm (`head.norm`).
    in_norm: Option<(Vec<f32>, Vec<f32>)>,
    in_channels: usize,
    // Projection convs (1×1) for each of 4 stages.
    proj: [Conv1x1W; 4],
    // Resize conv-transpose / strided conv per stage.
    resize0: ConvTransposeW,
    resize1: ConvTransposeW,
    resize3: Conv3x3W,
    // `layer{i}_rn` 3×3 no-bias lateral convs.
    layer_rn: [Conv3x3W; 4],
    // Fusion pyramid.
    rn1: FusionW,
    rn2: FusionW,
    rn3: FusionW,
    rn4: FusionW,
    // Output convs.
    out1: Conv3x3W,
    out2a: Conv3x3W,
    out2b: Conv1x1W,
    // Optional sky head.
    sky: Option<(Conv3x3W, Conv1x1W)>,
    // Config.
    features: usize,
    head_pos_embed: bool,
}

/// Whether the candle-free DPT-head path is enabled (`DA_FAST_HEAD=1`).
pub fn fast_dpt_enabled() -> bool {
    fast_conv::fast_head_enabled()
}

impl FastDptHead {
    /// Build from a loaded candle [`crate::dpt_head::DptHead`]. Expensive
    /// (one-time weight copy); subsequent [`Self::forward`] calls do no candle
    /// work or allocation.
    pub fn from_candle(head: &crate::dpt_head::DptHead) -> Result<Self> {
        use crate::dpt_head::DptHead;
        let DptHead {
            cfg,
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
            ..
        } = head;

        let in_norm = match in_norm {
            Some((w, b)) => Some((flatten_to_f32(w)?, flatten_to_f32(b)?)),
            None => None,
        };

        let proj = [
            conv1x1_from(&proj[0])?,
            conv1x1_from(&proj[1])?,
            conv1x1_from(&proj[2])?,
            conv1x1_from(&proj[3])?,
        ];

        let resize0 = conv_t_from(resize0, 4)?;
        let resize1 = conv_t_from(resize1, 2)?;
        let resize3 = conv3x3_from(resize3)?;

        let layer_rn = [
            conv3x3_nobias_from(&layer_rn[0])?,
            conv3x3_nobias_from(&layer_rn[1])?,
            conv3x3_nobias_from(&layer_rn[2])?,
            conv3x3_nobias_from(&layer_rn[3])?,
        ];

        let rn1 = fusion_from(rn1)?;
        let rn2 = fusion_from(rn2)?;
        let rn3 = fusion_from(rn3)?;
        let rn4 = fusion_from(rn4)?;

        let out1 = conv3x3_from(out1)?;
        let out2a = conv3x3_from(out2a)?;
        let out2b = conv1x1_from(out2b)?;

        let sky = match sky {
            Some(((sa_w, sa_b), (sb_w, sb_b))) => {
                let sa = conv3x3_from(&(sa_w.clone(), sa_b.clone()))?;
                let sb = conv1x1_from(&(sb_w.clone(), sb_b.clone()))?;
                Some((sa, sb))
            }
            None => None,
        };

        let features = if cfg.head_features > 0 {
            cfg.head_features as usize
        } else {
            128
        };

        Ok(Self {
            in_norm,
            in_channels: if cfg.cat_token {
                2 * cfg.embed_dim as usize
            } else {
                cfg.embed_dim as usize
            },
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
            features,
            head_pos_embed: cfg.head_pos_embed,
        })
    }

    /// Forward the decoder on raw f32 buffers. Returns
    /// `(depth_logits, sky_logits)` as NCHW `[1, C, H, W]` flat buffers.
    ///
    /// `feats` is one `[1, n_patch, C]` row-major buffer per stage (4 stages),
    /// matching the candle `Tensor` layout `[1, n_patch, C]` flattened.
    pub fn forward(
        &self,
        feats: &[Vec<f32>],
        gh: usize,
        gw: usize,
        h: usize,
        w: usize,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        use std::time::Instant;
        let prof = Self::profile_enabled();
        let pw = gw;
        let ph = gh;
        let aspect = gw as f32 / gh as f32;

        let t0 = Instant::now();
        let lat = self.compute_laterals(feats, pw, ph, aspect)?;
        let t_laterals = if prof {
            t0.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };

        let t0 = Instant::now();
        // Top-down fusion pyramid. Spatial dims track the candle path:
        //   rn4: top=lat[3]            → up to (pw, ph)
        //   rn3: top=out4 + lat[2]     → up to (2pw, 2ph)
        //   rn2: top=out3 + lat[1]     → up to (4pw, 4ph)
        //   rn1: top=out2 + lat[0]     → up to (8pw, 8ph)  [doubling]
        //
        // Overlap optimisation: RCU(lat[0]) (8.4ms, needed by rn1) is
        // independent of the rn4→rn3→rn2 chain. We spawn it as a background
        // rayon task that runs concurrently with the early fusion levels.
        // Rayon's work-stealing lets the RCU's sub-tasks (conv tile-blocks)
        // share the thread pool with the fusion chain. By the time rn1 needs
        // the RCU result, it's already computed.
        //
        // We use rayon::scope with a channel to pass the RCU result back.
        // The fusion chain runs in the scope's main thread; the RCU runs in a
        // spawned task. The channel recv() at rn1 blocks until the RCU is done
        // (which should already be the case by then).
        let (out, t_fusion) = if let Some(rc1) = &self.rn1.rc1 {
            // Overlap RCU(lat[0]) with the rn4→rn3→rn2 chain via rayon::join.
            // Both branches share the thread pool via work-stealing.
            let (rcu0, out2_res) = rayon::join(
                || residual_conv_unit(&lat[0], rc1).expect("RCU(lat[0]) failed"),
                || -> Result<Feat> {
                    let out4 = self.fusion_forward(&self.rn4, &lat[3], None, pw, ph)?;
                    let out3 =
                        self.fusion_forward(&self.rn3, &out4, Some(&lat[2]), 2 * pw, 2 * ph)?;
                    let out2 =
                        self.fusion_forward(&self.rn2, &out3, Some(&lat[1]), 4 * pw, 4 * ph)?;
                    Ok(out2)
                },
            );
            let out2 = out2_res?;
            let out =
                self.fusion_forward_with_lateral_rcu(&self.rn1, &out2, &rcu0, 8 * pw, 8 * ph)?;
            let t_fusion = if prof {
                t0.elapsed().as_secs_f64() * 1000.0
            } else {
                0.0
            };
            (out, t_fusion)
        } else {
            let out4 = self.fusion_forward(&self.rn4, &lat[3], None, pw, ph)?;
            let out3 = self.fusion_forward(&self.rn3, &out4, Some(&lat[2]), 2 * pw, 2 * ph)?;
            let out2 = self.fusion_forward(&self.rn2, &out3, Some(&lat[1]), 4 * pw, 4 * ph)?;
            let out = self.fusion_forward(&self.rn1, &out2, Some(&lat[0]), 8 * pw, 8 * ph)?;
            let t_fusion = if prof {
                t0.elapsed().as_secs_f64() * 1000.0
            } else {
                0.0
            };
            (out, t_fusion)
        };

        let t0 = Instant::now();
        // output_conv1: 3×3 features(128) → feat_half(64).
        let t_out1 = Instant::now();
        let out1_out = fast_conv::conv3x3_pad1(
            &out.data,
            &self.out1.weight,
            &self.out1.bias,
            1,
            self.out1.ic,
            out.h,
            out.w,
            self.out1.oc,
        );
        let t_out1_ms = if prof {
            t_out1.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let out_c = self.out1.oc;
        let out_h_pre = out.h;
        let out_w_pre = out.w;

        // Output stage: upsample (if needed) + out2a (3×3 64→32, ReLU).
        //
        // Common case (no sky head, upsample needed): fuse the bilinear
        // upsample directly into out2a's Winograd input transform. This
        // skips materialising the ~43 MiB upsampled tensor and reading it
        // back, saving one full DRAM round-trip. The un-upsampling input
        // (~14 MiB) fits entirely in L2.
        //
        // Fallback (sky head present OR no upsample): materialise the
        // upsample (with UV-PE add fused in), then run out2a from the
        // materialised tensor. The sky head also needs the upsampled tensor,
        // so fusion can't skip the materialisation in that case.
        let uv_pe = if self.head_pos_embed {
            Some(uv_embed_chw_cached(w, h, out_c, aspect, UV_RATIO))
        } else {
            None
        };
        let uv_pe_slice: Option<&[f32]> = uv_pe.as_deref().map(|v| v.as_slice());

        let need_upsample = out_h_pre != h || out_w_pre != w;
        // Allow A/B testing: DA_FAST_FUSE_UPSAMPLE=0 forces the materialised
        // path even when fusion is possible.
        let fuse_upsample = need_upsample && self.sky.is_none() && Self::fuse_upsample_enabled();

        let (a, sky, t_up_ms, t_out2a_ms) = if fuse_upsample {
            // Fused path: out2a reads un-upsampled out1 output, computing
            // each upsampled pixel via bilinear interpolation on the fly.
            // No separate upsample pass → t_up_ms is 0.
            let t_out2a = Instant::now();
            let a = fast_conv::conv3x3_pad1_relu_out_upsample(
                &out1_out,
                &self.out2a.weight,
                &self.out2a.bias,
                1,
                self.out2a.ic,
                out_h_pre,
                out_w_pre,
                self.out2a.oc,
                h,
                w,
                uv_pe_slice,
            );
            let t_out2a_ms = if prof {
                t_out2a.elapsed().as_secs_f64() * 1000.0
            } else {
                0.0
            };
            (a, None, 0.0, t_out2a_ms)
        } else {
            // Fallback path: materialise the upsampled tensor (with UV-PE
            // add fused in), then run sky head and out2a from it.
            let t_up_inner = Instant::now();
            let mut out_data = out1_out;
            let mut out_h = out_h_pre;
            let mut out_w = out_w_pre;
            if need_upsample {
                let in_feat = Feat::new(out_data, out_c, out_h, out_w);
                let mut up = vec![0.0f32; out_c * h * w];
                upsample_bilinear_ac(&in_feat, h, w, &mut up, uv_pe_slice);
                out_data = up;
                out_h = h;
                out_w = w;
            } else if let Some(pe) = &uv_pe {
                // No upsample needed, but UV embed still applies. Add directly.
                let hw = out_h * out_w;
                out_data
                    .par_chunks_mut(hw)
                    .zip(pe.par_chunks(hw))
                    .for_each(|(a_ch, b_ch)| {
                        for (a, b) in a_ch.iter_mut().zip(b_ch.iter()) {
                            *a += *b;
                        }
                    });
            }
            let t_up_ms = if prof {
                t_up_inner.elapsed().as_secs_f64() * 1000.0
            } else {
                0.0
            };
            let out = Feat::new(out_data, out_c, out_h, out_w);

            // Optional sky head (only present when the fallback path is taken).
            let sky = if let Some((sa, sb)) = &self.sky {
                let sa_out = fast_conv::conv3x3_pad1_relu_out(
                    &out.data, &sa.weight, &sa.bias, 1, sa.ic, out.h, out.w, sa.oc,
                );
                let sa_feat = Feat::new(sa_out, sa.oc, out.h, out.w);
                let sb_out = fast_conv::conv1x1(
                    &sa_feat.data,
                    &sb.weight,
                    &sb.bias,
                    1,
                    sb.ic,
                    sa_feat.h,
                    sa_feat.w,
                    sb.oc,
                );
                Some(sb_out)
            } else {
                None
            };

            // output_conv2a: 3×3 64→32, ReLU fused into output scatter.
            let t_out2a = Instant::now();
            let a = fast_conv::conv3x3_pad1_relu_out(
                &out.data,
                &self.out2a.weight,
                &self.out2a.bias,
                1,
                self.out2a.ic,
                out.h,
                out.w,
                self.out2a.oc,
            );
            let t_out2a_ms = if prof {
                t_out2a.elapsed().as_secs_f64() * 1000.0
            } else {
                0.0
            };
            (a, sky, t_up_ms, t_out2a_ms)
        };

        let a_feat = Feat::new(a, self.out2a.oc, h, w);
        let data = fast_conv::conv1x1(
            &a_feat.data,
            &self.out2b.weight,
            &self.out2b.bias,
            1,
            self.out2b.ic,
            a_feat.h,
            a_feat.w,
            self.out2b.oc,
        );
        let t_output = if prof {
            t0.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };

        if prof {
            eprintln!(
                "[fast_dpt] laterals={:.2}ms fusion={:.2}ms output={:.2}ms total={:.2}ms",
                t_laterals,
                t_fusion,
                t_output,
                t_laterals + t_fusion + t_output
            );
            eprintln!(
                "    [output] out1={:.2}ms upsample={:.2}ms out2a={:.2}ms total={:.2}ms",
                t_out1_ms, t_up_ms, t_out2a_ms, t_output
            );
        }

        Ok((data, sky))
    }

    /// Whether to print per-stage timing for the fast head (`DA_FAST_HEAD_PROFILE=1`).
    /// Off by default; useful for finding bottlenecks.
    fn profile_enabled() -> bool {
        use std::sync::OnceLock;
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| {
            matches!(
                std::env::var("DA_FAST_HEAD_PROFILE").as_deref(),
                Ok("1") | Ok("on") | Ok("true") | Ok("ON") | Ok("TRUE") | Ok("True")
            )
        })
    }

    /// Whether the upsample+out2a fusion is enabled. Default ON; set
    /// `DA_FAST_FUSE_UPSAMPLE=0` to force the materialised upsample path
    /// (for A/B testing).
    fn fuse_upsample_enabled() -> bool {
        use std::sync::OnceLock;
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| !matches!(std::env::var("DA_FAST_FUSE_UPSAMPLE").as_deref(), Ok("0")))
    }

    /// Whether the fusion-stage upsample+conv1x1 fusion is enabled.
    ///
    /// **Default OFF** because the upsampled activation tensor for DA3-BASE
    /// (`[128, 192, 288]` = 28 MiB at rn1; smaller at rn2/rn3) fits within the
    /// 30 MiB L3 cache, so the unfused path (parallel upsample → GEMM reading
    /// B from L3) beats the fused path (per-panel B-strip materialisation
    /// inside the GEMM). Microbench and end-to-end both show the fused path is
    /// ~5 ms slower for DA3-BASE fusion shapes.
    ///
    /// The fusion becomes a win when the upsampled B exceeds L3 capacity
    /// (e.g. a hypothetical `features=256` model → 56 MiB B at rn1). Set
    /// `DA_FAST_FUSE_FUSION_UPSAMPLE=1` to opt in for experimentation.
    fn fuse_fusion_upsample_enabled() -> bool {
        use std::sync::OnceLock;
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| {
            matches!(
                std::env::var("DA_FAST_FUSE_FUSION_UPSAMPLE").as_deref(),
                Ok("1") | Ok("on") | Ok("true") | Ok("ON") | Ok("TRUE") | Ok("True")
            )
        })
    }

    /// Run the tail of a fusion stage: optional bilinear upsample `(y_h, y_w) →
    /// (out_h, out_w)` followed by `conv1x1` projecting to `f.outc.oc` channels.
    ///
    /// When the upsample is non-trivial AND the fusion flag is enabled, the
    /// upsample and conv1x1 are fused into a single
    /// [`fast_conv::conv1x1_upsample`] call that materialises the upsampled
    /// activation panel-by-panel inside the GEMM (saving the DRAM round-trip of
    /// the full upsampled tensor). Otherwise the two passes run separately.
    ///
    /// Returns `(out_data, t_up_ms, t_conv_ms)`.
    fn fusion_upsample_conv1x1(
        f: &FusionW,
        y: Vec<f32>,
        y_c: usize,
        y_h: usize,
        y_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> (Vec<f32>, f64, f64) {
        use std::time::Instant;
        let need_up = y_h != out_h || y_w != out_w;

        if need_up && Self::fuse_fusion_upsample_enabled() {
            // Fused path: conv1x1 reads the upsampled activation on-the-fly
            // from `y` (L3-resident). No separate upsample pass is timed; the
            // total fused cost is reported as `t_conv` with `t_up = 0`.
            let t = Instant::now();
            let out_data = fast_conv::conv1x1_upsample(
                &y,
                &f.outc.weight,
                &f.outc.bias,
                1,
                f.outc.ic,
                y_h,
                y_w,
                f.outc.oc,
                out_h,
                out_w,
            );
            let t_fused = t.elapsed().as_secs_f64() * 1000.0;
            (out_data, 0.0, t_fused)
        } else {
            // Unfused path: materialise upsample then run conv1x1.
            let mut y = y;
            let t_up;
            if need_up {
                let start = Instant::now();
                let in_feat = Feat::new(y, y_c, y_h, y_w);
                let mut up = vec![0.0f32; y_c * out_h * out_w];
                upsample_bilinear_ac(&in_feat, out_h, out_w, &mut up, None);
                y = up;
                t_up = start.elapsed().as_secs_f64() * 1000.0;
            } else {
                t_up = 0.0;
            }
            let start = Instant::now();
            let out_data = fast_conv::conv1x1(
                &y,
                &f.outc.weight,
                &f.outc.bias,
                1,
                f.outc.ic,
                out_h,
                out_w,
                f.outc.oc,
            );
            let t_conv = start.elapsed().as_secs_f64() * 1000.0;
            (out_data, t_up, t_conv)
        }
    }

    /// Project + resize each backbone stage and apply its `layer{i}_rn` lateral
    /// conv. Returns the four `[features, h_i, w_i]` lateral feature maps.
    fn compute_laterals(
        &self,
        feats: &[Vec<f32>],
        pw: usize,
        ph: usize,
        aspect: f32,
    ) -> Result<[Feat; 4]> {
        use std::time::Instant;
        let prof = Self::profile_enabled();
        let n_patch = ph * pw;

        // The 4 laterals are independent (each reads feats[s], writes lats[s]).
        // Run them concurrently via rayon so their inner rayon-parallel conv
        // work can share the thread pool via work-stealing. The total work is
        // unchanged, but dispatch-overhead-bound operations (small GEMMs in
        // proj/resize/lrn for the smaller laterals) can fill idle threads
        // while the larger laterals (lat3) are compute-bound.
        //
        // All-4-parallel beats two-phase (lat3-then-rest) because the lighter
        // laterals' dispatch-bound work fills idle threads during lat3's
        // compute-bound GEMMs, giving better overall utilisation.
        let mut collected: Vec<(usize, Feat)> = (0..4usize)
            .into_par_iter()
            .map(|s| {
                let t = Instant::now();
                let lat = self.compute_one_lateral(s, &feats[s], pw, ph, aspect, n_patch);
                if prof {
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("    [lateral {s}] took {ms:.2}ms");
                }
                (s, lat)
            })
            .collect();
        collected.sort_by_key(|(s, _)| *s);

        let lat0 = collected.remove(0).1;
        let lat1 = collected.remove(0).1;
        let lat2 = collected.remove(0).1;
        let lat3 = collected.remove(0).1;
        Ok([lat0, lat1, lat2, lat3])
    }

    /// Compute one lateral (stage `s`). Extracted from `compute_laterals` so
    /// each lateral can run in its own rayon task.
    fn compute_one_lateral(
        &self,
        s: usize,
        feat: &[f32],
        pw: usize,
        ph: usize,
        aspect: f32,
        n_patch: usize,
    ) -> Feat {
        // 1. Optional input LayerNorm over channels ([n_patch, in_channels]).
        let feat_src: Vec<f32> = if let Some((nw, nb)) = &self.in_norm {
            let mut dst = vec![0.0f32; n_patch * self.in_channels];
            layernorm_rows(
                feat,
                &mut dst,
                n_patch,
                self.in_channels,
                nw,
                nb,
                HEAD_NORM_EPS,
            );
            dst
        } else {
            feat.to_vec()
        };

        // 2. Transpose [n_patch, C] → [C, ph, pw] NCHW.
        let mut chw = vec![0.0f32; self.in_channels * n_patch];
        transpose_npatch_chw(&feat_src, n_patch, self.in_channels, ph, pw, &mut chw);

        // 3. Project 1×1: in_channels → oc[s].
        let oc_s = self.proj[s].oc;
        let mut x = fast_conv::conv1x1(
            &chw,
            &self.proj[s].weight,
            &self.proj[s].bias,
            1,
            self.in_channels,
            ph,
            pw,
            oc_s,
        );

        // 4. UV pos embed at (ph, pw) before resize.
        if self.head_pos_embed {
            let pe = uv_embed_chw_cached(pw, ph, oc_s, aspect, UV_RATIO);
            let hw = ph * pw;
            x.par_chunks_mut(hw)
                .zip(pe.par_chunks(hw))
                .for_each(|(a_ch, b_ch)| {
                    for (a, b) in a_ch.iter_mut().zip(b_ch.iter()) {
                        *a += *b;
                    }
                });
        }

        // 5. Resize to pyramid level.
        let resized: Vec<f32> = match s {
            0 => {
                let mut out = Vec::new();
                conv_transpose_k_stride(&self.resize0, &x, ph, pw, &mut out);
                out
            }
            1 => {
                let mut out = Vec::new();
                conv_transpose_k_stride(&self.resize1, &x, ph, pw, &mut out);
                out
            }
            2 => x,
            3 => {
                let mut out = Vec::new();
                conv3x3_stride2_pad1(&self.resize3, &x, ph, pw, &mut out);
                out
            }
            _ => unreachable!(),
        };

        // 6. layer_rn 3×3 (no bias): oc[s] → features (128).
        let (rh, rw) = match s {
            0 => (ph * 4, pw * 4),
            1 => (ph * 2, pw * 2),
            2 => (ph, pw),
            3 => (ph / 2, pw / 2),
            _ => unreachable!(),
        };
        let lrn = &self.layer_rn[s];
        let lat =
            fast_conv::conv3x3_pad1(&resized, &lrn.weight, &lrn.bias, 1, lrn.ic, rh, rw, lrn.oc);
        Feat::new(lat, lrn.oc, rh, rw)
    }

    /// One fusion stage.
    fn fusion_forward(
        &self,
        f: &FusionW,
        top: &Feat,
        lateral: Option<&Feat>,
        out_w: usize,
        out_h: usize,
    ) -> Result<Feat> {
        use std::time::Instant;
        let prof = Self::profile_enabled();
        let t0 = Instant::now();

        // y = top
        // if lateral && rc1: y += residual_conv_unit(lateral, rc1)
        // y = residual_conv_unit(y, rc2)
        // y = upsample(y, out_h, out_w)
        // y = conv1x1(outc, y)

        let y_c = top.c;
        let y_h = top.h;
        let y_w = top.w;

        // Lateral add. When a lateral is present, compute y = top + rc1(lat)
        // directly into a fresh buffer (saves one read+write pass vs cloning
        // top and adding in-place). When no lateral, y is just top cloned
        // (only happens at the smallest fusion level — cheap).
        let mut y: Vec<f32>;
        let t_lat = if let (Some(lat), Some(rc1)) = (lateral, &f.rc1) {
            let t = Instant::now();
            let res = residual_conv_unit(lat, rc1)?;
            // y = top + res, written directly (no clone of top).
            let hw = y_h * y_w;
            y = vec![0.0f32; y_c * hw];
            y.par_chunks_mut(hw)
                .zip(top.data.par_chunks(hw))
                .zip(res.data.par_chunks(hw))
                .for_each(|((y_ch, t_ch), r_ch)| {
                    for ((y_v, t_v), r_v) in y_ch.iter_mut().zip(t_ch.iter()).zip(r_ch.iter()) {
                        *y_v = *t_v + *r_v;
                    }
                });
            t.elapsed().as_secs_f64() * 1000.0
        } else {
            y = top.data.clone();
            0.0
        };

        let t_rc2 = {
            let t = Instant::now();
            // residual_conv_unit clones x internally, so we can pass &y directly.
            let y_feat = Feat::new(y, y_c, y_h, y_w);
            let rc2_out = residual_conv_unit(&y_feat, &f.rc2)?;
            y = rc2_out.data;
            t.elapsed().as_secs_f64() * 1000.0
        };

        let (t_up, t_conv, out_data, y_h, y_w) = {
            let (out_data, t_up, t_conv) =
                Self::fusion_upsample_conv1x1(f, y, y_c, y_h, y_w, out_h, out_w);
            (t_up, t_conv, out_data, out_h, out_w)
        };

        if prof {
            eprintln!(
                "    [fusion] y_h={} y_w={} lat={:.2}ms rc2={:.2}ms up={:.2}ms conv={:.2}ms total={:.2}ms",
                y_h, y_w, t_lat, t_rc2, t_up, t_conv,
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(Feat::new(out_data, f.outc.oc, y_h, y_w))
    }

    /// Like [`fusion_forward`] but takes a pre-computed lateral RCU result
    /// (`lateral_rcu`) instead of computing `RCU(lateral)` inline. This allows
    /// the caller to overlap the lateral RCU with prior fusion work.
    ///
    /// `lateral_rcu` must be the output of `residual_conv_unit(lateral, f.rc1)`.
    fn fusion_forward_with_lateral_rcu(
        &self,
        f: &FusionW,
        top: &Feat,
        lateral_rcu: &Feat,
        out_w: usize,
        out_h: usize,
    ) -> Result<Feat> {
        use std::time::Instant;
        let prof = Self::profile_enabled();
        let t0 = Instant::now();

        let y_c = top.c;
        let y_h = top.h;
        let y_w = top.w;

        // y = top + lateral_rcu (pre-computed RCU(lat)), written directly.
        let hw = y_h * y_w;
        let mut y = vec![0.0f32; y_c * hw];
        y.par_chunks_mut(hw)
            .zip(top.data.par_chunks(hw))
            .zip(lateral_rcu.data.par_chunks(hw))
            .for_each(|((y_ch, t_ch), r_ch)| {
                for ((y_v, t_v), r_v) in y_ch.iter_mut().zip(t_ch.iter()).zip(r_ch.iter()) {
                    *y_v = *t_v + *r_v;
                }
            });

        let t_rc2 = {
            let t = Instant::now();
            let y_feat = Feat::new(y, y_c, y_h, y_w);
            let rc2_out = residual_conv_unit(&y_feat, &f.rc2)?;
            y = rc2_out.data;
            t.elapsed().as_secs_f64() * 1000.0
        };

        let (t_up, t_conv, out_data, y_h, y_w) = {
            let (out_data, t_up, t_conv) =
                Self::fusion_upsample_conv1x1(f, y, y_c, y_h, y_w, out_h, out_w);
            (t_up, t_conv, out_data, out_h, out_w)
        };

        if prof {
            eprintln!(
                "    [fusion-pre-rcu] y_h={} y_w={} rc2={:.2}ms up={:.2}ms conv={:.2}ms total={:.2}ms",
                y_h, y_w, t_rc2, t_up, t_conv,
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(Feat::new(out_data, f.outc.oc, y_h, y_w))
    }

    // (No scratch buffer helpers — see struct doc.)
}

// ---------------------------------------------------------------------------
// Elementwise / normalization helpers
// ---------------------------------------------------------------------------

#[inline]
#[allow(dead_code)] // currently all callers use fused relu variants
fn relu_inplace(buf: &mut [f32]) {
    buf.par_iter_mut().for_each(|v| {
        if *v < 0.0 {
            *v = 0.0;
        }
    });
}

/// Row-wise layer norm over `[n, dim]`. Writes into `dst`.
///
/// Parallelised over the row axis via rayon — each row is fully independent.
/// For the DPT input LayerNorm (n=ph*pw patches, dim=embed_dim) this lets us
/// use the same 32-thread oversubscription that makes the backbone fast.
fn layernorm_rows(
    src: &[f32],
    dst: &mut [f32],
    n: usize,
    dim: usize,
    w: &[f32],
    b: &[f32],
    eps: f32,
) {
    debug_assert_eq!(src.len(), n * dim);
    debug_assert_eq!(dst.len(), n * dim);
    dst.par_chunks_mut(dim).enumerate().for_each(|(ni, out)| {
        let row = &src[ni * dim..(ni + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for d in 0..dim {
            out[d] = (row[d] - mean) * inv_std * w[d] + b[d];
        }
    });
}

/// Transpose `[1, n_patch, C]` (token-major) → `[1, C, ph, pw]` (NCHW).
///
/// Parallelised over the channel axis `c`. Each channel writes a contiguous
/// `[ph, pw]` plane (ph*pw floats) into `dst`, so writes from different
/// channels never alias. For DA3-base (c=768, ph*pw=864+) this is plenty of
/// parallel work for 32 threads.
fn transpose_npatch_chw(
    src: &[f32],
    n_patch: usize,
    c: usize,
    ph: usize,
    pw: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), n_patch * c);
    debug_assert_eq!(dst.len(), c * ph * pw);
    // Each output channel plane is contiguous: dst[ci*ph*pw .. (ci+1)*ph*pw].
    dst.par_chunks_mut(ph * pw)
        .enumerate()
        .for_each(|(ci, plane)| {
            for py in 0..ph {
                let src_row = py * pw;
                for px in 0..pw {
                    // token index = py*pw + px, channel offset = ci
                    plane[src_row + px] = src[(src_row + px) * c + ci];
                }
            }
        });
}

// ---------------------------------------------------------------------------
// Convolution helpers
// ---------------------------------------------------------------------------

// (Convs are called directly via `fast_conv::conv3x3_pad1` / `fast_conv::conv1x1`;
// no wrappers needed.)

use rayon::prelude::*;

use crate::tinyblas;

/// Conv-transpose with k=stride, pad=0.
///
/// Decomposed into k×k independent GEMMs (one per (ky, kx) offset):
/// For each (ky, kx), the output at strided positions
/// `[oc, iy*stride+ky, ix*stride+kx]` is `W[:,:,ky,kx] · X[:,iy,ix]`, i.e.
/// a GEMM of shape `[out_c, in_h*in_w] = [out_c, in_c] · [in_c, in_h*in_w]`.
/// This is far faster than the naive nested loop (each GEMM hits tinyblas's
/// AVX2/FMA microkernel).
#[allow(clippy::uninit_vec)] // every element is bias-filled before any read
fn conv_transpose_k_stride(
    w: &ConvTransposeW,
    x: &[f32],
    in_h: usize,
    in_w: usize,
    out: &mut Vec<f32>,
) {
    let k = w.k;
    let stride = w.stride;
    debug_assert_eq!(k, stride);
    let out_h = (in_h - 1) * stride + k;
    let out_w = (in_w - 1) * stride + k;
    let in_c = w.in_c;
    let out_c = w.out_c;
    debug_assert_eq!(x.len(), in_c * in_h * in_w);
    let total = out_c * out_h * out_w;
    // Allocate without zero-fill — every element is written by the bias
    // pre-fill below before any reader sees it (the scatter loop then
    // accumulates onto the bias values).
    out.clear();
    out.reserve(total);
    // SAFETY: `capacity >= total` (just reserved). Every element in
    // `0..total` is written by the bias-fill `par_chunks_mut` below before
    // any subsequent read (the GEMM/scatter only reads after bias fill).
    unsafe { out.set_len(total) };
    // Pre-fill with bias so the scatter accumulate below yields `bias + conv`.
    out.par_chunks_mut(out_h * out_w)
        .enumerate()
        .for_each(|(oc, out_ch)| {
            let bv = w.bias[oc];
            for v in out_ch.iter_mut() {
                *v = bv;
            }
        });

    // x is [in_c, in_h, in_w] NCHW. Treat it as [in_c, in_h*in_w] for the GEMM.
    let hw = in_h * in_w;
    // Pre-transpose all k×k weight slices into [out_c, in_c] (NN) layout, once.
    // The source weight is [in_c, ky, kx, out_c] with out_c innermost, so each
    // (ky, kx) slice is a contiguous [in_c, out_c] block that we transpose to
    // [out_c, in_c]. Cached by weight pointer so we only pay this once per model.
    let w_t = get_or_build_conv_transpose_w(w);

    let mut tmp = vec![0.0f32; out_c * hw];
    for ky in 0..k {
        for kx in 0..k {
            let slice = (ky * k + kx) * out_c * in_c;
            let w_kxk = &w_t[slice..slice + out_c * in_c];
            // tmp = w_kxk @ x   shape [out_c, hw] = [out_c, in_c] @ [in_c, hw]
            for v in tmp.iter_mut() {
                *v = 0.0;
            }
            tinyblas::gemm_nn_into(out_c, hw, in_c, w_kxk, x, &mut tmp);
            // Scatter tmp[oc, iy, ix] into out[oc, iy*stride+ky, ix*stride+kx].
            out.par_chunks_mut(out_h * out_w)
                .enumerate()
                .for_each(|(oc, out_ch)| {
                    let tmp_off = oc * hw;
                    for iy in 0..in_h {
                        let oy = iy * stride + ky;
                        for ix in 0..in_w {
                            let ox = ix * stride + kx;
                            out_ch[oy * out_w + ox] += tmp[tmp_off + iy * in_w + ix];
                        }
                    }
                });
        }
    }

    // Bias was fused into the pre-fill above.
}

// Cache of pre-transposed conv-transpose weights. Keyed by weight pointer so
// the transpose runs once per model load (weights are immutable after load).
type ConvTCache = std::collections::HashMap<usize, std::sync::Arc<Vec<f32>>>;
static CONVT_CACHE: std::sync::Mutex<Option<ConvTCache>> = std::sync::Mutex::new(None);

/// Get the pre-transposed [k*k, out_c, in_c] weight slices for a conv-transpose.
/// Transposes from [in_c, ky, kx, out_c] to [ky*kx, out_c, in_c] (NN layout for
/// tinyblas GEMM). Cached by source weight pointer.
fn get_or_build_conv_transpose_w(w: &ConvTransposeW) -> std::sync::Arc<Vec<f32>> {
    let key = w.weight.as_ptr() as usize;
    let mut guard = CONVT_CACHE.lock().expect("CONVT_CACHE poisoned");
    if let Some(cache) = guard.as_mut() {
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }
    }
    // Build: transpose each (ky, kx) slice from [in_c, out_c] to [out_c, in_c].
    let k = w.k;
    let in_c = w.in_c;
    let out_c = w.out_c;
    let mut w_t = vec![0.0f32; k * k * out_c * in_c];
    for ky in 0..k {
        for kx in 0..k {
            let slice = (ky * k + kx) * out_c * in_c;
            for ic in 0..in_c {
                let src = ((ic * k + ky) * k + kx) * out_c;
                for oc in 0..out_c {
                    w_t[slice + oc * in_c + ic] = w.weight[src + oc];
                }
            }
        }
    }
    let arc = std::sync::Arc::new(w_t);
    if guard.is_none() {
        *guard = Some(std::collections::HashMap::new());
    }
    guard.as_mut().unwrap().insert(key, arc.clone());
    arc
}

/// 3×3 stride-2 pad-1 conv. Output spatial = ((in+1)/2, (in+1)/2).
///
/// Uses im2col + GEMM: gather each output's 3×3 (ic) input patch into a
/// column of a `[9*ic, out_h*out_w]` matrix, then GEMM with the
/// `[oc, 9*ic]` weight matrix (reshaped from `[oc, ic, 3, 3]`).
#[allow(clippy::uninit_vec)] // every element is bias-filled before any read
fn conv3x3_stride2_pad1(w: &Conv3x3W, x: &[f32], in_h: usize, in_w: usize, out: &mut Vec<f32>) {
    let out_h = (in_h + 1) / 2;
    let out_w = (in_w + 1) / 2;
    let ic = w.ic;
    let oc = w.oc;
    debug_assert_eq!(x.len(), ic * in_h * in_w);
    let hw_out = out_h * out_w;
    let total = oc * hw_out;
    // Allocate without zero-fill — every element is written by the bias
    // pre-fill below before the GEMM reads it.
    out.clear();
    out.reserve(total);
    // SAFETY: `capacity >= total` (just reserved). Every element in `0..total`
    // is written by the bias-fill below before the GEMM (which reads C as the
    // accumulator base) touches it.
    unsafe { out.set_len(total) };
    // Pre-fill with bias so the GEMM accumulate below yields `bias + conv`.
    // For tiny outputs (hw_out small), serial is faster than rayon dispatch.
    if hw_out >= 1024 {
        out.par_chunks_mut(hw_out)
            .enumerate()
            .for_each(|(oc_i, out_ch)| {
                let bv = w.bias[oc_i];
                for v in out_ch.iter_mut() {
                    *v = bv;
                }
            });
    } else {
        for oc_i in 0..oc {
            let bv = w.bias[oc_i];
            let base = oc_i * hw_out;
            for i in 0..hw_out {
                out[base + i] = bv;
            }
        }
    }
    let ksq = 9; // 3*3

    // im2col: build a `[ic*9, out_h*out_w]` matrix where each column is the
    // flattened (ic, ky, kx) patch for one output position. This matches
    // w.weight's [oc, ic, 3, 3] = [oc, ic*9] layout, so no permutation is needed.
    let mut col = vec![0.0f32; ic * ksq * hw_out];
    // Parallelise over channels (ic tasks, each handling 9 (ky,kx) patches)
    // rather than ic*ksq rows, to avoid rayon dispatch overhead dominating for
    // tiny outputs (e.g. lateral 3 resize with hw_out=54, where 768*9=6912
    // fine-grained tasks spent more time on dispatch than work). The serial
    // fallback handles cases where ic is too small to benefit from rayon.
    if ic >= 32 && hw_out >= 64 {
        col.par_chunks_mut(ksq * hw_out)
            .enumerate()
            .for_each(|(ic_i, col_ic)| {
                // col_ic is [9, hw_out] for this channel.
                for ky in 0..3 {
                    for kx in 0..3 {
                        let col_ch =
                            &mut col_ic[(ky * 3 + kx) * hw_out..(ky * 3 + kx + 1) * hw_out];
                        for oy in 0..out_h {
                            let iy_raw = 2 * oy + ky;
                            for ox in 0..out_w {
                                let ix_raw = 2 * ox + kx;
                                let v = if iy_raw == 0 || ix_raw == 0 {
                                    0.0
                                } else {
                                    let iy = iy_raw - 1;
                                    let ix = ix_raw - 1;
                                    if iy >= in_h || ix >= in_w {
                                        0.0
                                    } else {
                                        x[(ic_i * in_h + iy) * in_w + ix]
                                    }
                                };
                                col_ch[oy * out_w + ox] = v;
                            }
                        }
                    }
                }
            });
    } else {
        // Serial: for tiny outputs the rayon dispatch overhead dominates.
        for ic_i in 0..ic {
            for ky in 0..3 {
                for kx in 0..3 {
                    let kic = (ic_i * 3 + ky) * 3 + kx;
                    let col_ch = &mut col[kic * hw_out..(kic + 1) * hw_out];
                    for oy in 0..out_h {
                        let iy_raw = 2 * oy + ky;
                        for ox in 0..out_w {
                            let ix_raw = 2 * ox + kx;
                            let v = if iy_raw == 0 || ix_raw == 0 {
                                0.0
                            } else {
                                let iy = iy_raw - 1;
                                let ix = ix_raw - 1;
                                if iy >= in_h || ix >= in_w {
                                    0.0
                                } else {
                                    x[(ic_i * in_h + iy) * in_w + ix]
                                }
                            };
                            col_ch[oy * out_w + ox] = v;
                        }
                    }
                }
            }
        }
    }

    // GEMM: out [oc, hw_out] = w.weight [oc, ic*9] @ col [ic*9, hw_out].
    // No permutation needed — both have ic*9 in (ic, ky, kx) order.
    // Accumulates onto the bias pre-filled above.
    tinyblas::gemm_nn_into(oc, hw_out, ic * ksq, &w.weight, &col, out);
}

/// Bilinear upsample with `align_corners=true`. Port of `interp_bilinear_ac`.
///
/// `x` is NCHW `[1, C, in_h, in_w]`. Output is `[1, C, out_h, out_w]`.
///
/// If `add` is `Some`, its `[C, out_h, out_w]` values are added to the upsampled
/// output in the same pass (`out = upsample(x) + add`), avoiding a separate
/// read+write pass over the output tensor.
fn upsample_bilinear_ac(
    x: &Feat,
    out_h: usize,
    out_w: usize,
    out: &mut Vec<f32>,
    add: Option<&[f32]>,
) {
    let (c, h, w) = (x.c, x.h, x.w);
    out.resize(c * out_h * out_w, 0.0);
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
    // Precompute x-direction indices and weights (shared across channels).
    let xw: Vec<(usize, usize, f32)> = (0..out_w)
        .map(|ox| {
            let fx = ox as f32 * scale_x;
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let wx = fx - x0 as f32;
            (x0, x1, wx)
        })
        .collect();
    let yw: Vec<(usize, usize, f32)> = (0..out_h)
        .map(|oy| {
            let fy = oy as f32 * scale_y;
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let wy = fy - y0 as f32;
            (y0, y1, wy)
        })
        .collect();
    out.par_chunks_mut(out_h * out_w)
        .enumerate()
        .for_each(|(ci, out_ch)| {
            let in_off = ci * h * w;
            let add_off = ci * out_h * out_w;
            for oy in 0..out_h {
                let (y0, y1, wy) = yw[oy];
                for ox in 0..out_w {
                    let (x0, x1, wx) = xw[ox];
                    let p00 = x.data[in_off + y0 * w + x0];
                    let p01 = x.data[in_off + y0 * w + x1];
                    let p10 = x.data[in_off + y1 * w + x0];
                    let p11 = x.data[in_off + y1 * w + x1];
                    let top = p00 * (1.0 - wx) + p01 * wx;
                    let bot = p10 * (1.0 - wx) + p11 * wx;
                    let v = top * (1.0 - wy) + bot * wy;
                    out_ch[oy * out_w + ox] = if let Some(add) = add {
                        v + add[add_off + oy * out_w + ox]
                    } else {
                        v
                    };
                }
            }
        });
}

/// Residual conv unit: relu → 3×3 pad1 → relu → 3×3 pad1 → +x.
fn residual_conv_unit(x: &Feat, rcu: &ResConvUnitW) -> Result<Feat> {
    // h = relu(x); h = conv1(h); h = relu(h); h = conv2(h); h += x.
    //
    // The two ReLUs and the residual add are all fused into the conv kernels:
    //   - conv1 reads x with on-the-fly relu (conv3x3_pad1_relu_in)
    //   - conv2 reads conv1's output with on-the-fly relu AND adds x to its
    //     output scatter (conv3x3_pad1_relu_in_res_out)
    // This eliminates three full read/write passes over the (up to 28 MiB at
    // the largest fusion level) intermediate tensors.
    //
    // conv1: relu(x) fused into input read.
    let h = fast_conv::conv3x3_pad1_relu_in(
        &x.data,
        &rcu.c1.weight,
        &rcu.c1.bias,
        1,
        rcu.c1.ic,
        x.h,
        x.w,
        rcu.c1.oc,
    );
    // conv2: relu(h) fused into input read; +x fused into output scatter.
    let h = fast_conv::conv3x3_pad1_relu_in_res_out(
        &h,
        &rcu.c2.weight,
        &rcu.c2.bias,
        1,
        rcu.c2.ic,
        x.h,
        x.w,
        rcu.c2.oc,
        &x.data,
    );
    Ok(Feat::new(h, rcu.c2.oc, x.h, x.w))
}

// ---------------------------------------------------------------------------
// Weight extraction helpers
// ---------------------------------------------------------------------------

fn conv1x1_from(t: &(Tensor, Tensor)) -> Result<Conv1x1W> {
    let weight = flatten_to_f32(&t.0)?;
    let bias = flatten_to_f32(&t.1)?;
    let dims = t.0.dims();
    let oc = dims[0];
    let ic = dims[1];
    Ok(Conv1x1W {
        weight,
        bias,
        ic,
        oc,
    })
}

fn conv3x3_from(t: &(Tensor, Tensor)) -> Result<Conv3x3W> {
    let weight = flatten_to_f32(&t.0)?;
    let bias = flatten_to_f32(&t.1)?;
    let dims = t.0.dims();
    let oc = dims[0];
    let ic = dims[1];
    Ok(Conv3x3W {
        weight,
        bias,
        ic,
        oc,
    })
}

fn conv3x3_nobias_from(t: &Tensor) -> Result<Conv3x3W> {
    let weight = flatten_to_f32(t)?;
    let dims = t.dims();
    let oc = dims[0];
    let ic = dims[1];
    let bias = vec![0.0f32; oc];
    Ok(Conv3x3W {
        weight,
        bias,
        ic,
        oc,
    })
}

fn conv_t_from(t: &(Tensor, Tensor), stride: usize) -> Result<ConvTransposeW> {
    let raw = flatten_to_f32(&t.0)?;
    let bias = flatten_to_f32(&t.1)?;
    let dims = t.0.dims();
    let in_c = dims[0];
    let out_c = dims[1];
    let k = dims[2];
    // Transpose PyTorch [in_c, out_c, k, k] → [in_c, k, k, out_c] for
    // contiguous OC access in the inner loop of `conv_transpose_k_stride`.
    let mut weight = vec![0.0f32; in_c * k * k * out_c];
    for ic in 0..in_c {
        for ky in 0..k {
            for kx in 0..k {
                for oc in 0..out_c {
                    let src = ((ic * out_c + oc) * k + ky) * k + kx;
                    let dst = ((ic * k + ky) * k + kx) * out_c + oc;
                    weight[dst] = raw[src];
                }
            }
        }
    }
    Ok(ConvTransposeW {
        weight,
        bias,
        in_c,
        out_c,
        k,
        stride,
    })
}

fn fusion_from(f: &crate::dpt_head::Fusion) -> Result<FusionW> {
    use crate::dpt_head::Fusion as F;
    let F { rc1, rc2, outc } = f;
    let rc1 = if let Some(rc1) = rc1 {
        Some(ResConvUnitW {
            c1: conv3x3_from(&rc1.0)?,
            c2: conv3x3_from(&rc1.1)?,
        })
    } else {
        None
    };
    let rc2 = ResConvUnitW {
        c1: conv3x3_from(&rc2.0)?,
        c2: conv3x3_from(&rc2.1)?,
    };
    let outc = conv1x1_from(outc)?;
    Ok(FusionW { rc1, rc2, outc })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layernorm_rows_matches_naive() {
        let n = 3;
        let dim = 5;
        let src: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let w: Vec<f32> = (0..dim).map(|i| 0.5 + 0.1 * (i as f32)).collect();
        let b: Vec<f32> = (0..dim).map(|i| 0.01 * (i as f32)).collect();
        let mut dst = vec![0.0f32; n * dim];
        layernorm_rows(&src, &mut dst, n, dim, &w, &b, 1e-5);

        // Naive reference.
        for ni in 0..n {
            let row = &src[ni * dim..(ni + 1) * dim];
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + 1e-5).sqrt();
            for d in 0..dim {
                let expected = (row[d] - mean) * inv_std * w[d] + b[d];
                assert!(
                    (dst[ni * dim + d] - expected).abs() < 1e-5,
                    "row {ni} dim {d}: got {} expected {}",
                    dst[ni * dim + d],
                    expected
                );
            }
        }
    }

    #[test]
    fn conv_transpose_k_stride_matches_naive() {
        // Small: in_c=2, out_c=3, k=stride=2, in 2×2 → out 4×4.
        let in_c = 2;
        let out_c = 3;
        let k = 2;
        let stride = 2;
        let in_h = 2;
        let in_w = 2;
        // PyTorch-style [in_c, out_c, k, k].
        let raw_weight: Vec<f32> = (0..in_c * out_c * k * k)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let bias: Vec<f32> = vec![0.1; out_c];
        let x: Vec<f32> = (0..in_c * in_h * in_w).map(|i| (i as f32) * 0.5).collect();

        // Transpose to [in_c, k, k, out_c] (matches conv_t_from).
        let mut weight = vec![0.0f32; in_c * k * k * out_c];
        for ic in 0..in_c {
            for ky in 0..k {
                for kx in 0..k {
                    for oc in 0..out_c {
                        let src = ((ic * out_c + oc) * k + ky) * k + kx;
                        let dst = ((ic * k + ky) * k + kx) * out_c + oc;
                        weight[dst] = raw_weight[src];
                    }
                }
            }
        }

        let w = ConvTransposeW {
            weight,
            bias: bias.clone(),
            in_c,
            out_c,
            k,
            stride,
        };

        let mut out = Vec::new();
        conv_transpose_k_stride(&w, &x, in_h, in_w, &mut out);

        // Naive reference using the original PyTorch [in_c, out_c, k, k] layout.
        let out_h = (in_h - 1) * stride + k;
        let out_w = (in_w - 1) * stride + k;
        let mut naive = vec![0.0f32; out_c * out_h * out_w];
        for ic in 0..in_c {
            for iy in 0..in_h {
                for ix in 0..in_w {
                    let xv = x[(ic * in_h + iy) * in_w + ix];
                    for ky in 0..k {
                        for kx in 0..k {
                            let oy = iy * stride + ky;
                            let ox = ix * stride + kx;
                            for oc in 0..out_c {
                                let wv = raw_weight[((ic * out_c + oc) * k + ky) * k + kx];
                                naive[(oc * out_h + oy) * out_w + ox] += wv * xv;
                            }
                        }
                    }
                }
            }
        }
        for oc in 0..out_c {
            let bv = bias[oc];
            for i in 0..out_h * out_w {
                naive[oc * out_h * out_w + i] += bv;
            }
        }

        for i in 0..naive.len() {
            assert!(
                (out[i] - naive[i]).abs() < 1e-5,
                "idx {i}: got {} expected {}",
                out[i],
                naive[i]
            );
        }
    }

    #[test]
    fn conv3x3_stride2_pad1_matches_naive() {
        // Small: ic=2, oc=3, in 5×5 → out 3×3.
        let ic = 2;
        let oc = 3;
        let in_h = 5;
        let in_w = 5;
        let out_h = (in_h + 1) / 2;
        let out_w = (in_w + 1) / 2;
        let weight: Vec<f32> = (0..oc * ic * 9).map(|i| (i as f32) * 0.01).collect();
        let bias: Vec<f32> = vec![0.1; oc];
        let x: Vec<f32> = (0..ic * in_h * in_w).map(|i| (i as f32) * 0.05).collect();
        let w = Conv3x3W {
            weight: weight.clone(),
            bias: bias.clone(),
            ic,
            oc,
        };

        let mut out = Vec::new();
        conv3x3_stride2_pad1(&w, &x, in_h, in_w, &mut out);

        // Naive reference: Y[oc, oy, ox] = bias[oc] + sum_{ic, ky, kx}
        // W[oc, ic, ky, kx] * X[ic, 2*oy+ky-1, 2*ox+kx-1] with zero padding.
        let mut naive = vec![0.0f32; oc * out_h * out_w];
        for oc_i in 0..oc {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = bias[oc_i];
                    for ic_i in 0..ic {
                        for ky in 0..3 {
                            let iy = (2 * oy + ky).wrapping_sub(1);
                            if iy >= in_h {
                                continue;
                            }
                            for kx in 0..3 {
                                let ix = (2 * ox + kx).wrapping_sub(1);
                                if ix >= in_w {
                                    continue;
                                }
                                let wv = weight[((oc_i * ic + ic_i) * 3 + ky) * 3 + kx];
                                acc += wv * x[(ic_i * in_h + iy) * in_w + ix];
                            }
                        }
                    }
                    naive[(oc_i * out_h + oy) * out_w + ox] = acc;
                }
            }
        }

        for i in 0..naive.len() {
            assert!(
                (out[i] - naive[i]).abs() < 1e-4,
                "idx {i}: got {} expected {}",
                out[i],
                naive[i]
            );
        }
    }

    #[test]
    fn upsample_bilinear_ac_matches_naive() {
        // Small: c=2, in 2×2 → out 4×4.
        let c = 2;
        let in_h = 2;
        let in_w = 2;
        let out_h = 4;
        let out_w = 4;
        let data: Vec<f32> = (0..c * in_h * in_w).map(|i| (i as f32) * 0.5).collect();
        let x = Feat::new(data, c, in_h, in_w);
        let mut out = Vec::new();
        upsample_bilinear_ac(&x, out_h, out_w, &mut out, None);

        // Naive: align_corners=true → scale = (in-1)/(out-1).
        let scale_y = (in_h - 1) as f32 / (out_h - 1) as f32;
        let scale_x = (in_w - 1) as f32 / (out_w - 1) as f32;
        let mut naive = vec![0.0f32; c * out_h * out_w];
        for ci in 0..c {
            for oy in 0..out_h {
                let fy = oy as f32 * scale_y;
                let y0 = fy.floor() as usize;
                let y1 = (y0 + 1).min(in_h - 1);
                let wy = fy - y0 as f32;
                for ox in 0..out_w {
                    let fx = ox as f32 * scale_x;
                    let x0 = fx.floor() as usize;
                    let x1 = (x0 + 1).min(in_w - 1);
                    let wx = fx - x0 as f32;
                    let p00 = x.data[ci * in_h * in_w + y0 * in_w + x0];
                    let p01 = x.data[ci * in_h * in_w + y0 * in_w + x1];
                    let p10 = x.data[ci * in_h * in_w + y1 * in_w + x0];
                    let p11 = x.data[ci * in_h * in_w + y1 * in_w + x1];
                    let top = p00 * (1.0 - wx) + p01 * wx;
                    let bot = p10 * (1.0 - wx) + p11 * wx;
                    naive[ci * out_h * out_w + oy * out_w + ox] = top * (1.0 - wy) + bot * wy;
                }
            }
        }
        for i in 0..naive.len() {
            assert!(
                (out[i] - naive[i]).abs() < 1e-5,
                "idx {i}: got {} expected {}",
                out[i],
                naive[i]
            );
        }
    }
}
