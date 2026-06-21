//! Host-only nested-metric alignment, ported from `src/nested.cpp` /
//! `nested.hpp`.
//!
//! `NestedDepthAnything3Net` (da3.py) runs two branches on the same input:
//!
//! 1. **anyview (GIANT)** — DualDPT depth + conf + camera-pose head. Produces
//!    relative-scale depth `{depth, depth_conf, extrinsics, intrinsics}`.
//! 2. **metric (ViT-L + DPT/sky)** — single-channel DPT with a parallel sky
//!    head. Produces metric-scale depth `{depth, sky}`.
//!
//! The two are fused by a least-squares scalar fit (`scale_factor =
//! Σ(a·b) / Σ(b·b)`) over high-confidence, non-sky pixels, then the anyview
//! depth + translation is rescaled into metric units. Sky pixels are then
//! filled with the 99th percentile of the non-sky depth (capped at 200).
//!
//! All math is host-side `Vec<f32>` (no tensors); see `Engine::depth_metric_*`
//! for the branch forwards that feed it.

// The index-based `for i in 0..n` loops below mirror the C++
// `for (size_t i = 0; i < N; ++i)` line-for-line so the two can be diffed.
// Clippy's iterator suggestions would obscure that correspondence.
#![allow(clippy::needless_range_loop)]

use crate::depth_export::write_pfm;
use crate::Result;
use std::path::Path;

/// Anyview (GIANT) branch outputs for the nested metric alignment.
///
/// All buffers are `[H*W]` row-major `(h, w)`; `extrinsics`/`intrinsics` are
/// row-major 3×4 / 3×3.
#[derive(Debug, Clone)]
pub struct AnyviewOut {
    pub depth: Vec<f32>,
    pub depth_conf: Vec<f32>,
    pub extrinsics: [f32; 12],
    pub intrinsics: [f32; 9],
}

/// Metric (ViT-L + DPT/sky) branch outputs for the nested metric alignment.
///
/// `depth` is `depth_metric_raw` (post `exp`, **pre** the focal/300 scaling
/// applied inside [`NestedAligner::align`]). `sky` is the relu-activated sky
/// map.
#[derive(Debug, Clone)]
pub struct MetricOut {
    pub depth: Vec<f32>,
    pub sky: Vec<f32>,
}

/// Final metric-scale depth + rescaled pose produced by
/// [`NestedAligner::align`].
#[derive(Debug, Clone)]
pub struct NestedOut {
    /// `[H*W]` row-major metric-scale depth (sky-filled).
    pub depth: Vec<f32>,
    /// 3×4 row-major extrinsics (translation scaled by `scale_factor`).
    pub extrinsics: [f32; 12],
    /// 3×3 row-major intrinsics (unchanged by alignment).
    pub intrinsics: [f32; 9],
    pub scale_factor: f32,
    /// Processed-image height (rows) — matches `depth.len() / w`.
    pub h: usize,
    /// Processed-image width (columns).
    pub w: usize,
}

impl NestedOut {
    /// Write the depth map to a little-endian PFM file (matches
    /// `da::write_pfm`).
    pub fn write_pfm<P: AsRef<Path>>(&self, path: P, w: usize, h: usize) -> Result<()> {
        write_pfm(path, &self.depth, w, h)
    }
}

/// Host (`Vec<f32>`) implementation of `NestedDepthAnything3Net`'s metric
/// alignment (da3.py + utils/alignment.py).
pub struct NestedAligner;

impl NestedAligner {
    /// Run the two-branch alignment. See the module docs for the algorithm.
    ///
    /// Mirrors `NestedAligner::align` in `src/nested.cpp`.
    pub fn align(any: &AnyviewOut, metric: &MetricOut, h: usize, w: usize) -> NestedOut {
        let n = h * w;
        let mut out = NestedOut {
            depth: any.depth.clone(),
            extrinsics: any.extrinsics,
            intrinsics: any.intrinsics,
            scale_factor: 0.0,
            h,
            w,
        };
        if n == 0 || any.depth.len() != n || metric.depth.len() != n || metric.sky.len() != n {
            // Mirrors the C++ early-return: leave out.depth = any.depth,
            // scale_factor = 0. Caller can detect the empty case via h*w.
            return out;
        }

        // --- 1) apply_metric_scaling: scale metric depth by anyview focal/300.
        // focal = (intr[0,0] + intr[1,1]) / 2 ; scale_factor const = 300.
        let focal = (any.intrinsics[0] + any.intrinsics[4]) * 0.5;
        let metric_scale = focal / 300.0;
        let mut metric_depth = vec![0.0f32; n];
        for i in 0..n {
            metric_depth[i] = metric.depth[i] * metric_scale;
        }

        // --- 2) _apply_depth_alignment ---
        // non_sky_mask = sky < 0.3
        let mut non_sky = vec![false; n];
        let mut n_nonsky = 0usize;
        for i in 0..n {
            non_sky[i] = metric.sky[i] < 0.3;
            if non_sky[i] {
                n_nonsky += 1;
            }
        }

        // median_conf = quantile(depth_conf[non_sky], 0.5). 224^2 < 100_000 -> no sampling.
        let mut conf_ns = Vec::with_capacity(n_nonsky);
        for i in 0..n {
            if non_sky[i] {
                conf_ns.push(any.depth_conf[i]);
            }
        }
        let median_conf = quantile(&conf_ns, 0.5) as f32;

        // align_mask = (conf >= median_conf) & non_sky & (metric_depth > 1e-2) & (depth > 1e-3)
        // least_squares_scale_scalar(a=metric, b=depth) = sum(a*b) / sum(b*b)
        let min_depth_thresh = 1e-3f32;
        let min_metric_thresh = 1e-2f32;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..n {
            if non_sky[i]
                && any.depth_conf[i] >= median_conf
                && metric_depth[i] > min_metric_thresh
                && any.depth[i] > min_depth_thresh
            {
                let a = metric_depth[i] as f64;
                let b = any.depth[i] as f64;
                num += a * b;
                den += b * b;
            }
        }
        if den < 1e-12 {
            den = 1e-12;
        }
        let scale_factor = (num / den) as f32;
        out.scale_factor = scale_factor;

        // output.depth *= scale_factor ; output.extrinsics[:,:,:3,3] *= scale_factor
        for v in out.depth.iter_mut() {
            *v *= scale_factor;
        }
        out.extrinsics[3] *= scale_factor; // row0 col3 (translation x)
        out.extrinsics[7] *= scale_factor; // row1 col3 (translation y)
        out.extrinsics[11] *= scale_factor; // row2 col3 (translation z)

        // --- 3) _handle_sky_regions ---
        // non_sky_max = min(quantile(depth[non_sky], 0.99), 200). Then sky pixels -> non_sky_max.
        let mut depth_ns = Vec::with_capacity(n_nonsky);
        for i in 0..n {
            if non_sky[i] {
                depth_ns.push(out.depth[i]);
            }
        }
        let mut non_sky_max = quantile(&depth_ns, 0.99) as f32;
        if non_sky_max > 200.0 {
            non_sky_max = 200.0;
        }
        for i in 0..n {
            if !non_sky[i] {
                out.depth[i] = non_sky_max;
            }
        }

        out
    }

    /// `DepthAnything3Net._process_mono_sky_estimation`: the metric branch's
    /// own sky-fill, applied inside `da3_metric(x)` *before* the nested
    /// alignment. Sets sky pixels (`sky >= 0.3`) to
    /// `quantile(non_sky_depth, 0.99)` (NOT capped at 200 here — that cap is
    /// applied later inside [`Self::align`] for the merged depth).
    ///
    /// No-op unless there are `> 10` non-sky AND `> 10` sky pixels (matches the
    /// reference early-returns). Mutates `depth` in place.
    ///
    /// Mirrors `NestedAligner::process_mono_sky` in `src/nested.cpp`.
    pub fn process_mono_sky(depth: &mut [f32], sky: &[f32]) {
        let n = depth.len();
        if n == 0 || sky.len() != n {
            return;
        }
        let mut n_nonsky = 0usize;
        let mut n_sky = 0usize;
        for &s in sky {
            if s < 0.3 {
                n_nonsky += 1;
            } else {
                n_sky += 1;
            }
        }
        if n_nonsky <= 10 || n_sky <= 10 {
            return;
        }
        let mut depth_ns = Vec::with_capacity(n_nonsky);
        for i in 0..n {
            if sky[i] < 0.3 {
                depth_ns.push(depth[i]);
            }
        }
        // NOT capped here (the 200 cap is applied later in `align`).
        let non_sky_max = quantile(&depth_ns, 0.99) as f32;
        for i in 0..n {
            if sky[i] >= 0.3 {
                depth[i] = non_sky_max;
            }
        }
    }
}

/// `torch.quantile` (linear interpolation): sort ascending, `pos = q*(n-1)`,
/// interpolate between `floor(pos)` and `ceil(pos)`. The input is copied and
/// sorted internally (matches the C++ `std::vector<float> v` by-value
/// semantics), so the caller's slice is left untouched.
///
/// Mirrors `NestedAligner::quantile` in `src/nested.cpp`.
pub fn quantile(v: &[f32], q: f32) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = v.to_vec();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n == 1 {
        return sorted[0] as f64;
    }
    let pos = (q as f64) * ((n - 1) as f64);
    let lo_d = pos.floor();
    let hi_d = pos.ceil();
    let mut lo = lo_d as usize;
    let mut hi = hi_d as usize;
    if hi >= n {
        hi = n - 1;
    }
    if lo >= n {
        lo = n - 1;
    }
    let frac = pos - lo_d;
    (sorted[lo] as f64) + frac * ((sorted[hi] as f64) - (sorted[lo] as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- quantile -------------------------------------------------------

    #[test]
    fn quantile_empty_returns_zero() {
        assert_eq!(quantile(&[], 0.5), 0.0);
    }

    #[test]
    fn quantile_single_element() {
        assert_eq!(quantile(&[7.5], 0.0), 7.5);
        assert_eq!(quantile(&[7.5], 1.0), 7.5);
        assert_eq!(quantile(&[7.5], 0.5), 7.5);
    }

    #[test]
    fn quantile_endpoints() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&v, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile(&v, 1.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn quantile_median_odd_n() {
        // n=5 -> pos = 0.5 * 4 = 2 -> v[2] = 3.0 (no interpolation).
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&v, 0.5) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn quantile_median_even_n_interpolates() {
        // n=4 -> pos = 0.5 * 3 = 1.5 -> v[1] + 0.5*(v[2]-v[1]) = 2 + 0.5*(4-2) = 3.0.
        let v = vec![1.0f32, 2.0, 4.0, 8.0];
        assert!((quantile(&v, 0.5) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn quantile_p25_interpolates_down() {
        // n=5 -> pos = 0.25 * 4 = 1 -> exactly v[1] = 2.0.
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&v, 0.25) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn quantile_p99_clamps_to_last() {
        // q=1.0 -> pos = n-1 -> hi clamped to n-1.
        let v = vec![10.0f32, 20.0, 30.0];
        assert!((quantile(&v, 1.0) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn quantile_unsorted_input() {
        // Input order shouldn't matter; the function sorts internally.
        let v = vec![5.0f32, 1.0, 4.0, 2.0, 3.0];
        assert!((quantile(&v, 0.5) - 3.0).abs() < 1e-12);
    }

    // ---- process_mono_sky ------------------------------------------------

    #[test]
    fn process_mono_sky_noop_when_too_few_sky() {
        // 11 non-sky, 1 sky -> n_sky <= 10 -> no-op.
        let mut depth = vec![1.0f32; 12];
        let sky = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let before = depth.clone();
        NestedAligner::process_mono_sky(&mut depth, &sky);
        assert_eq!(depth, before);
    }

    #[test]
    fn process_mono_sky_noop_when_too_few_nonsky() {
        // 1 non-sky, 11 sky -> n_nonsky <= 10 -> no-op.
        let mut depth = vec![1.0f32; 12];
        let sky = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let before = depth.clone();
        NestedAligner::process_mono_sky(&mut depth, &sky);
        assert_eq!(depth, before);
    }

    #[test]
    fn process_mono_sky_noop_when_empty() {
        let mut depth: Vec<f32> = Vec::new();
        let sky: Vec<f32> = Vec::new();
        // Should not panic.
        NestedAligner::process_mono_sky(&mut depth, &sky);
        assert!(depth.is_empty());
    }

    #[test]
    fn process_mono_sky_noop_on_size_mismatch() {
        let mut depth = vec![1.0f32; 12];
        let sky = vec![0.0f32; 5]; // mismatched length
        let before = depth.clone();
        NestedAligner::process_mono_sky(&mut depth, &sky);
        assert_eq!(depth, before);
    }

    #[test]
    fn process_mono_sky_fills_sky_with_p99_no_cap() {
        // 11 non-sky (depth 1..=11) + 11 sky (depth 999). p99 of [1..=11]:
        //   pos = 0.99 * 10 = 9.9 -> v[9] + 0.9*(v[10]-v[9]) = 10 + 0.9*(11-10) = 10.9.
        //   No 200 cap here (that's applied in `align`, not here).
        let mut depth: Vec<f32> = (1..=11).map(|x| x as f32).collect();
        depth.extend(std::iter::repeat(999.0).take(11));
        let mut sky = vec![0.0f32; 11];
        sky.extend(std::iter::repeat(1.0).take(11));
        NestedAligner::process_mono_sky(&mut depth, &sky);
        // Non-sky pixels unchanged.
        for i in 0..11 {
            assert_eq!(depth[i], (i + 1) as f32);
        }
        // Sky pixels == p99 (10.9), NOT capped.
        for i in 11..22 {
            assert!((depth[i] - 10.9).abs() < 1e-5, "depth[{i}] = {}", depth[i]);
        }
    }

    // ---- align -----------------------------------------------------------

    /// Build a minimal `AnyviewOut`/`MetricOut` pair where the math is
    /// hand-computable. Used by several `align` tests below.
    fn toy_pair() -> (AnyviewOut, MetricOut) {
        // 2x2 grid. anyview depth = [[1, 2], [3, 4]] (row-major), conf = 1.0
        // everywhere. metric depth = [[10, 20], [30, 40]], sky = all 0.0
        // (non-sky). intrinsics diagonal so focal = (fx+fy)/2 = (300+300)/2 =
        // 300 -> metric_scale = 1.0 (so metric_depth is unchanged by scaling).
        let any = AnyviewOut {
            depth: vec![1.0, 2.0, 3.0, 4.0],
            depth_conf: vec![1.0; 4],
            extrinsics: [
                1.0, 0.0, 0.0, 10.0, // row 0: tx=10
                0.0, 1.0, 0.0, 20.0, // row 1: ty=20
                0.0, 0.0, 1.0, 30.0, // row 2: tz=30
            ],
            intrinsics: [
                300.0, 0.0, 0.0, // fx=300
                0.0, 300.0, 0.0, // fy=300
                0.0, 0.0, 1.0,
            ],
        };
        let metric = MetricOut {
            depth: vec![10.0, 20.0, 30.0, 40.0],
            sky: vec![0.0; 4],
        };
        (any, metric)
    }

    #[test]
    fn align_empty_returns_zero_scale() {
        // n=0 -> early return: out.depth == any.depth (empty), scale_factor = 0.
        let any = AnyviewOut {
            depth: Vec::new(),
            depth_conf: Vec::new(),
            extrinsics: [0.0; 12],
            intrinsics: [0.0; 9],
        };
        let metric = MetricOut {
            depth: Vec::new(),
            sky: Vec::new(),
        };
        let out = NestedAligner::align(&any, &metric, 0, 0);
        assert!(out.depth.is_empty());
        assert_eq!(out.scale_factor, 0.0);
    }

    #[test]
    fn align_least_squares_scale_factor() {
        // metric_scale = focal/300 = 1.0 (focal=300). With all confs=1.0 and
        // median_conf=1.0, every pixel is in the align mask. metric_depth =
        // [10,20,30,40], any.depth = [1,2,3,4] -> scale_factor = sum(a*b)/sum(b*b)
        //   = (10*1 + 20*2 + 30*3 + 40*4) / (1*1 + 2*2 + 3*3 + 4*4)
        //   = (10 + 40 + 90 + 160) / (1 + 4 + 9 + 16)
        //   = 300 / 30 = 10.0.
        let (any, metric) = toy_pair();
        let out = NestedAligner::align(&any, &metric, 2, 2);
        assert!(
            (out.scale_factor - 10.0).abs() < 1e-5,
            "got {}",
            out.scale_factor
        );
    }

    #[test]
    fn align_rescales_depth_and_translation() {
        // scale_factor = 10 (from the test above). out.depth = any.depth * 10.
        // Translation column (ext[3], ext[7], ext[11]) = [10,20,30] * 10.
        let (any, metric) = toy_pair();
        let out = NestedAligner::align(&any, &metric, 2, 2);
        assert_eq!(out.depth, vec![10.0, 20.0, 30.0, 40.0]);
        assert!((out.extrinsics[3] - 100.0).abs() < 1e-5);
        assert!((out.extrinsics[7] - 200.0).abs() < 1e-5);
        assert!((out.extrinsics[11] - 300.0).abs() < 1e-5);
        // Non-translation entries unchanged.
        assert_eq!(out.extrinsics[0], 1.0);
        assert_eq!(out.extrinsics[5], 1.0);
    }

    #[test]
    fn align_preserves_intrinsics() {
        let (any, metric) = toy_pair();
        let out = NestedAligner::align(&any, &metric, 2, 2);
        assert_eq!(out.intrinsics, any.intrinsics);
    }

    #[test]
    fn align_metric_scale_uses_focal() {
        // Same as toy_pair but intrinsics = 600/600 -> focal=600, metric_scale=2.
        // metric_depth = [10,20,30,40] * 2 = [20,40,60,80].
        // scale_factor = sum(a*b)/sum(b*b) with a=metric_depth, b=any.depth:
        //   = (20*1 + 40*2 + 60*3 + 80*4) / (1+4+9+16) = 600/30 = 20.0.
        let (mut any, metric) = toy_pair();
        any.intrinsics[0] = 600.0;
        any.intrinsics[4] = 600.0;
        let out = NestedAligner::align(&any, &metric, 2, 2);
        assert!(
            (out.scale_factor - 20.0).abs() < 1e-5,
            "got {}",
            out.scale_factor
        );
    }

    #[test]
    fn align_fills_sky_pixels_with_p99_capped_at_200() {
        // 6 pixels: any.depth = [1,1,1,1,1,1000] (row-major), sky[5]=1.0 (sky),
        // rest non-sky. metric depth irrelevant once scale is known (all confs=1,
        // median_conf=1, so non-sky mask only matters for the final fill).
        // We use uniform metric.depth = any.depth so scale_factor = 1.0 exactly
        // (sum(a*b) = sum(b*b)).
        let any = AnyviewOut {
            depth: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1000.0],
            depth_conf: vec![1.0; 6],
            extrinsics: [0.0; 12],
            intrinsics: [300.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 1.0],
        };
        let metric = MetricOut {
            // metric.depth * metric_scale (=1.0) = same as any.depth.
            depth: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1000.0],
            sky: vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        };
        let out = NestedAligner::align(&any, &metric, 3, 2);
        // scale_factor should be 1.0 (a==b -> num=den).
        assert!((out.scale_factor - 1.0).abs() < 1e-5);
        // Non-sky depth unchanged (== 1.0).
        for i in 0..5 {
            assert!((out.depth[i] - 1.0).abs() < 1e-5);
        }
        // Sky pixel: p99 of non-sky depth = p99([1,1,1,1,1]) = 1.0. Capped at 200.
        // So out.depth[5] = min(1.0, 200.0) = 1.0.
        assert!((out.depth[5] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn align_sky_fill_caps_at_200() {
        // Same as above but with non-sky depth large enough that p99 > 200.
        let any = AnyviewOut {
            depth: vec![1000.0; 5]
                .into_iter()
                .chain(std::iter::once(0.0))
                .collect(),
            depth_conf: vec![1.0; 6],
            extrinsics: [0.0; 12],
            intrinsics: [300.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 1.0],
        };
        let metric = MetricOut {
            // a==b -> scale=1.
            depth: any.depth.clone(),
            sky: vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        };
        let out = NestedAligner::align(&any, &metric, 3, 2);
        // After scale (=1.0), non-sky depth = 1000. p99([1000]*5) = 1000 -> capped 200.
        assert!((out.depth[5] - 200.0).abs() < 1e-5, "got {}", out.depth[5]);
        // Non-sky depth stays at 1000 (cap only applies to sky fill).
        for i in 0..5 {
            assert!((out.depth[i] - 1000.0).abs() < 1e-5);
        }
    }

    #[test]
    fn align_low_confidence_excluded_from_fit() {
        // Pixel 0 has conf=0 (below median 1.0) -> excluded from the fit.
        // Remaining pixels: a=[20,30,40], b=[2,3,4].
        // scale = (40+90+160)/(4+9+16) = 290/29 = 10.0 (approx).
        // Then pixel 0 (any.depth=1) -> 1 * 10 = 10.
        // median_conf of [1,1,1] (non-sky) = 1.0. Pixel 0's conf (0) < 1.0 -> excluded.
        let any = AnyviewOut {
            depth: vec![1.0, 2.0, 3.0, 4.0],
            depth_conf: vec![0.0, 1.0, 1.0, 1.0],
            extrinsics: [0.0; 12],
            intrinsics: [300.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 1.0],
        };
        let metric = MetricOut {
            depth: vec![10.0, 20.0, 30.0, 40.0],
            sky: vec![0.0; 4],
        };
        let out = NestedAligner::align(&any, &metric, 2, 2);
        let expected = 290.0 / 29.0;
        assert!(
            (out.scale_factor - expected as f32).abs() < 1e-4,
            "got {} expected {}",
            out.scale_factor,
            expected
        );
    }

    #[test]
    fn align_size_mismatch_returns_any_depth_unchanged() {
        // any.depth.len() != H*W -> early return (scale_factor stays 0).
        let any = AnyviewOut {
            depth: vec![1.0, 2.0, 3.0], // len 3 but H*W=4
            depth_conf: vec![1.0; 3],
            extrinsics: [0.0; 12],
            intrinsics: [300.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 1.0],
        };
        let metric = MetricOut {
            depth: vec![10.0, 20.0, 30.0, 40.0],
            sky: vec![0.0; 4],
        };
        let out = NestedAligner::align(&any, &metric, 2, 2);
        assert_eq!(out.depth, vec![1.0, 2.0, 3.0]);
        assert_eq!(out.scale_factor, 0.0);
    }

    #[test]
    fn align_den_zero_clamped() {
        // All any.depth = 0 -> sum(b*b) = 0 -> den clamped to 1e-12 -> scale ~ 0.
        let any = AnyviewOut {
            depth: vec![0.0; 4],
            depth_conf: vec![1.0; 4],
            extrinsics: [0.0; 12],
            intrinsics: [300.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 1.0],
        };
        let metric = MetricOut {
            depth: vec![10.0; 4],
            sky: vec![0.0; 4],
        };
        let out = NestedAligner::align(&any, &metric, 2, 2);
        assert!(out.scale_factor.abs() < 1e-3, "got {}", out.scale_factor);
    }
}
