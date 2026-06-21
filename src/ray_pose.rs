//! Ray → camera-pose SOLVER (the `use_ray_pose` path).
//!
//! Faithful Rust port of `src/ray_pose.cpp`. Turns the aux ray head's per-pixel
//! ray field into camera extrinsics (3×4 c2w) + intrinsics (3×3), mirroring
//! `ray_utils.get_extrinsic_from_camray` + `da3._process_ray_pose_estimation`.
//!
//! Pure host-side double-precision arithmetic (no tensors) — depends only on
//! [`crate::linalg`].
//!
//! # Sampling parity caveat
//!
//! The production path generates RANSAC sample indices with a seeded [`Mt19937`]
//! (bit-identical to `std::mt19937` for a given seed, which is fully specified by
//! the C++ standard). The mapping from generator output to a bounded integer
//! (`std::uniform_int_distribution`) and the consensus reshuffle (`std::shuffle`)
//! are **not** standard-specified — libstdc++, libc++ and MSVC differ. This port
//! follows the libstdc++ algorithm. On a Windows/MSVC C++ build the exact sample
//! draws may therefore differ, so the *production* pose (with all index
//! arguments `None`) is statistically equivalent but not bit-identical to one
//! C++ run.
//!
//! For bit-exact parity, use the **gated path**: pass `rand_idx`, `sorted_idx`
//! and `refit_idx` produced by the reference (see `examples/ray_pose_parity.rs`).
//! The deterministic solver math (cloud build, homography fit/score, QL
//! decomposition, translation, intrinsics) matches the C++ exactly.

// Index loops and the 8-arg signature deliberately mirror the C source for
// side-by-side verification (see `src/ray_pose.cpp`).
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::linalg;

/// Tunables for [`solve_ray_pose`]. Defaults match the C++ `RayPoseParams`.
#[derive(Debug, Clone)]
pub struct RayPoseParams {
    /// RANSAC iterations.
    pub n_iter: usize,
    /// Points sampled per iteration (`num_sample_for_ransac`).
    pub num_sample: usize,
    /// `n_sample = max(num_sample, int(N*sample_ratio))`.
    pub sample_ratio: f64,
    /// Inlier reprojection threshold (in the normalized image plane).
    pub reproj_threshold: f64,
    /// Cap on the consensus-set size before the final refit.
    pub max_inlier_num: usize,
    /// Ray-direction z-component gate (below this, the point is zero-weighted).
    pub z_threshold: f64,
    /// Seed for the internal (production-path) RANSAC sampler.
    pub seed: u32,
}

impl Default for RayPoseParams {
    fn default() -> Self {
        Self {
            n_iter: 100,
            num_sample: 8,
            sample_ratio: 0.3,
            reproj_threshold: 0.2,
            max_inlier_num: 8000,
            z_threshold: 1e-4,
            seed: 1234,
        }
    }
}

/// Diagnostics + final pose produced by [`solve_ray_pose`].
#[derive(Debug, Clone)]
pub struct RayPoseOut {
    /// 3×4 camera-to-world extrinsics, row-major (12 floats).
    pub extrinsics: [f32; 12],
    /// 3×3 intrinsics, row-major (9 floats).
    pub intrinsics: [f32; 9],
    /// Rotation (w2c) from the QL decomposition.
    pub r: [f64; 9],
    /// Translation (w2c).
    pub t: [f64; 3],
    /// Returned focal = 1/f.
    pub focal: [f64; 2],
    /// Returned principal point = pp_raw + 1.
    pub pp: [f64; 2],
    /// Homography after det-normalization.
    pub a: [f64; 9],
    /// Best-hypothesis inlier count (pre-subsample).
    pub n_inlier_best: usize,
}

impl Default for RayPoseOut {
    fn default() -> Self {
        Self {
            extrinsics: [0.0; 12],
            intrinsics: [0.0; 9],
            r: [0.0; 9],
            t: [0.0; 3],
            focal: [0.0; 2],
            pp: [0.0; 2],
            a: [0.0; 9],
            n_inlier_best: 0,
        }
    }
}

/// Inputs to [`solve_ray_pose`] for the deterministic "gated" path.
///
/// When all fields are `None`, the solver runs the production path: it computes
/// `sorted_idx` (argsort of weights, descending), generates RANSAC samples with
/// a seeded [`Mt19937`], and selects its own consensus refit set. When provided,
/// these exact indices are used instead — enabling bit-exact parity with the C++
/// reference (see [`crate::ray_pose`] module docs).
#[derive(Default)]
pub struct RayPoseIndices<'a> {
    /// `n_iter * num_sample` sample indices into `[0, n_sample)`. `None` =>
    /// generated internally with [`RayPoseParams::seed`].
    pub rand_idx: Option<&'a [usize]>,
    /// `N` ints: argsort of weights descending. `None` => computed internally
    /// (stable sort).
    pub sorted_idx: Option<&'a [usize]>,
    /// Consensus refit indices into `[0, N)`. `None` => solver selects its own
    /// best-hypothesis inliers + deterministic subsample.
    pub refit_idx: Option<&'a [usize]>,
}

/// The z-normalized 2D point cloud the homography is fit on.
///
/// - `src[n] = (x_grid - 1, y_grid - 1)` — identity-cam-plane unprojected origin.
/// - `dst[n] = (dx/dz, dy/dz)` — ray-direction half, z-normalized.
/// - `w[n] = ray_conf`, zeroed where `|dz| <= z_threshold`.
/// - `origin_half[n] = ray channels [3..6]` (used for the translation `T`).
/// - `conf_raw[n] = ray_conf` (unmodified; used for `T`'s weighted mean).
#[derive(Debug, Clone, Default)]
pub struct RayCloud {
    /// Length `2*Hy*Wx`, `(x, y)` interleaved.
    pub src: Vec<f64>,
    /// Length `2*Hy*Wx`, `(x, y)` interleaved.
    pub dst: Vec<f64>,
    /// Length `Hy*Wx`.
    pub w: Vec<f64>,
    /// Length `3*Hy*Wx`, `(x, y, z)` interleaved.
    pub origin_half: Vec<f64>,
    /// Length `Hy*Wx`.
    pub conf_raw: Vec<f64>,
}

/// Build the z-normalized 2D point cloud the homography is fit on, exactly as
/// `compute_optimal_rotation_intrinsics_batch` does.
///
/// - `ray`: `Hy*Wx*6` row-major `(h, w, c)` — channels `0..3` are the direction
///   half, `3..6` are the origin half.
/// - `ray_conf`: `Hy*Wx`.
/// - `z_threshold`: see [`RayPoseParams::z_threshold`].
pub fn build_ray_cloud(
    ray: &[f32],
    ray_conf: &[f32],
    hy: usize,
    wx: usize,
    z_threshold: f64,
) -> RayCloud {
    let n = hy * wx;
    let mut cloud = RayCloud {
        src: vec![0.0; 2 * n],
        dst: vec![0.0; 2 * n],
        w: vec![0.0; n],
        origin_half: vec![0.0; 3 * n],
        conf_raw: vec![0.0; n],
    };

    // Identity-cam-plane grid (unproject_depth, ixt_normalized): for patch index j,
    // x_grid = linspace(dx, 2-dx, Wx)[j]; origin x = x_grid - 1 (I_K^{-1}
    // subtracts cx=1). The reference builds the grid in float32 in torch; the C++
    // uses double here and absorbs the ~1e-7 difference in downstream tolerance.
    let dx = 1.0 / wx as f64;
    let dy = 1.0 / hy as f64;
    let gx: Vec<f64> = (0..wx)
        .map(|j| {
            if wx == 1 {
                dx
            } else {
                dx + j as f64 * ((2.0 - dx) - dx) / (wx - 1) as f64
            }
        })
        .collect();
    let gy: Vec<f64> = (0..hy)
        .map(|i| {
            if hy == 1 {
                dy
            } else {
                dy + i as f64 * ((2.0 - dy) - dy) / (hy - 1) as f64
            }
        })
        .collect();

    for i in 0..hy {
        for j in 0..wx {
            let nn = i * wx + j;
            let r = &ray[(nn * 6)..(nn * 6 + 6)];
            let dirx = r[0] as f64;
            let diry = r[1] as f64;
            let dirz = r[2] as f64;
            let ox = r[3] as f64;
            let oy = r[4] as f64;
            let oz = r[5] as f64;
            let ox_plane = gx[j] - 1.0; // identity-plane origin x
            let oy_plane = gy[i] - 1.0; // identity-plane origin y
                                        // z_mask: |dir_z| > thr AND |origin_plane_z=1| > thr (always true).
            let zmask = dirz.abs() > z_threshold;
            let mut tx = dirx;
            let mut ty = diry;
            if zmask {
                tx = dirx / dirz;
                ty = diry / dirz;
            }
            // origin (plane) z = 1, so its x, y unchanged by z-norm.
            cloud.src[2 * nn] = ox_plane;
            cloud.src[2 * nn + 1] = oy_plane;
            cloud.dst[2 * nn] = tx;
            cloud.dst[2 * nn + 1] = ty;
            let c = ray_conf[nn] as f64;
            cloud.conf_raw[nn] = c;
            cloud.w[nn] = if zmask { c } else { 0.0 };
            cloud.origin_half[3 * nn] = ox;
            cloud.origin_half[3 * nn + 1] = oy;
            cloud.origin_half[3 * nn + 2] = oz;
        }
    }
    cloud
}

/// Fit a weighted homography on a set of point indices into the cloud.
fn fit_homography_idx(cloud: &RayCloud, idx: &[usize]) -> Option<[f64; 9]> {
    let m = idx.len();
    let mut s = vec![0.0; 2 * m];
    let mut d = vec![0.0; 2 * m];
    let mut ww = vec![0.0; m];
    for (k, &n) in idx.iter().enumerate() {
        s[2 * k] = cloud.src[2 * n];
        s[2 * k + 1] = cloud.src[2 * n + 1];
        d[2 * k] = cloud.dst[2 * n];
        d[2 * k + 1] = cloud.dst[2 * n + 1];
        ww[k] = cloud.w[n];
    }
    linalg::homography_weighted(&s, &d, &ww, m)
}

/// Weighted inlier score of homography `h` over the full cloud + fill inlier mask.
/// Returns the summed weight of the inliers.
fn score_homography(cloud: &RayCloud, h: &[f64; 9], reproj_thr: f64, inlier: &mut [u8]) -> f64 {
    let n = cloud.w.len();
    let mut total = 0.0;
    let thr2 = reproj_thr * reproj_thr;
    for nn in 0..n {
        let x = cloud.src[2 * nn];
        let y = cloud.src[2 * nn + 1];
        let px = h[0] * x + h[1] * y + h[2];
        let py = h[3] * x + h[4] * y + h[5];
        let pz = h[6] * x + h[7] * y + h[8];
        let ex = px / pz - cloud.dst[2 * nn];
        let ey = py / pz - cloud.dst[2 * nn + 1];
        let is_in = (ex * ex + ey * ey) < thr2; // error < reproj_thr (squared compare)
        inlier[nn] = if is_in { 1 } else { 0 };
        if is_in {
            total += cloud.w[nn];
        }
    }
    total
}

/// Full ray → pose solver.
///
/// Returns `Ok(out)` on success, or `Err` describing the degeneracy (rank-
/// deficient consensus, zero confidence, non-finite pose, etc.). On `Err`, `out`
/// is left in its default (zeroed) state and the caller should fall back to the
/// MLP pose head.
pub fn solve_ray_pose(
    ray: &[f32],
    ray_conf: &[f32],
    hy: usize,
    wx: usize,
    img_h: usize,
    img_w: usize,
    indices: &RayPoseIndices<'_>,
    p: &RayPoseParams,
) -> Result<RayPoseOut, &'static str> {
    let n = hy * wx;
    let cloud = build_ray_cloud(ray, ray_conf, hy, wx, p.z_threshold);

    // n_sample: candidate pool size; clamp to N so a tiny ray field
    // (N < num_sample) can never index sorted_idx out of range.
    let n_sample = n.min(p.num_sample.max((n as f64 * p.sample_ratio) as usize));

    // Candidate ordering: argsort(w, descending). Use the fed order if provided
    // (matches the reference exactly on the gated path); else compute one.
    let owned_sorted;
    let sorted_idx: &[usize] = if let Some(s) = indices.sorted_idx {
        s
    } else {
        let mut perm: Vec<usize> = (0..n).collect();
        // stable_sort by w desc (matches std::stable_sort predicate w[a] > w[b]).
        perm.sort_by(|&a, &b| {
            cloud.w[b]
                .partial_cmp(&cloud.w[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        owned_sorted = perm;
        &owned_sorted
    };

    // Sampling indices into [0, n_sample): fed (gated) or generated (production).
    let owned_rand;
    let ridx: &[usize] = if let Some(r) = indices.rand_idx {
        r
    } else {
        let mut rng = Mt19937::new(p.seed);
        // perm is initialized ONCE and carried across iterations (partial
        // Fisher-Yates: elements in [0, num_sample) keep getting overwritten, but
        // the tail [num_sample, n_sample) drifts, exactly as in the C++).
        let mut perm = (0..n_sample).collect::<Vec<_>>();
        let mut gen = vec![0usize; p.n_iter * p.num_sample];
        for it in 0..p.n_iter {
            for k in 0..p.num_sample {
                let j = rng.uniform_int(k, n_sample - 1);
                perm.swap(k, j);
                gen[it * p.num_sample + k] = perm[k];
            }
        }
        owned_rand = gen;
        &owned_rand
    };

    // RANSAC: fit per iter, score, keep best inlier mask.
    let mut best_mask = vec![0u8; n];
    let mut tmp_mask = vec![0u8; n];
    let mut best_score = -1.0f64;
    let mut sample = vec![0usize; p.num_sample];
    for it in 0..p.n_iter {
        for k in 0..p.num_sample {
            sample[k] = sorted_idx[ridx[it * p.num_sample + k]];
        }
        let h = match fit_homography_idx(&cloud, &sample) {
            Some(h) => h,
            None => continue,
        };
        let sc = score_homography(&cloud, &h, p.reproj_threshold, &mut tmp_mask);
        if sc > best_score {
            best_score = sc;
            best_mask.copy_from_slice(&tmp_mask);
        }
    }

    // Collect best-hypothesis inliers.
    let mut inliers: Vec<usize> = (0..n).filter(|&i| best_mask[i] != 0).collect();

    let n_inlier_best = inliers.len();
    let mut out = RayPoseOut {
        n_inlier_best,
        ..Default::default()
    };

    // Final consensus point set for the refit.
    let owned_refit;
    let refit: &[usize] = if let Some(r) = indices.refit_idx {
        r
    } else {
        // Production: sort inliers by weight desc; subsample if > max_inlier_num.
        inliers.sort_by(|&a, &b| {
            cloud.w[b]
                .partial_cmp(&cloud.w[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if inliers.len() > p.max_inlier_num {
            let keep_len = ((inliers.len() as f64) * 0.95).max(p.max_inlier_num as f64) as usize;
            let mut head: Vec<usize> = inliers[..keep_len].to_vec();
            let mut rng2 = Mt19937::new(p.seed ^ 0x9e3779b9u32);
            // std::shuffle equivalent (Fisher-Yates over the whole range).
            shuffle(&mut head, &mut rng2);
            head.truncate(p.max_inlier_num);
            owned_refit = head;
        } else {
            owned_refit = inliers.clone();
        }
        &owned_refit
    };
    if refit.len() < 4 {
        return Err("ray-pose: fewer than 4 consensus points");
    }

    // Refit homography on the consensus set, normalize det > 0.
    let mut a = match fit_homography_idx(&cloud, refit) {
        Some(a) => a,
        None => return Err("ray-pose: consensus homography degenerate"),
    };
    if linalg::det3(&a) < 0.0 {
        for v in a.iter_mut() {
            *v = -*v;
        }
    }
    out.a = a;

    // QL decomposition -> R (=Q), L.
    let (r_mat, mut l_mat) = linalg::ql_decomposition(&a);
    // L = L / L[2][2]. A rank-deficient consensus set gives l22~0 (and/or focal
    // f0/f1~0 below) -> Inf/NaN pose; bail rather than report garbage.
    let l22 = l_mat[8];
    if l22.abs() < 1e-12 {
        return Err("ray-pose: L[2][2] ~ 0 (rank-deficient consensus)");
    }
    for v in l_mat.iter_mut() {
        *v /= l22;
    }
    let f0 = l_mat[0];
    let f1 = l_mat[4];
    let pp0 = l_mat[6];
    let pp1 = l_mat[7];
    if f0.abs() < 1e-12 || f1.abs() < 1e-12 {
        return Err("ray-pose: focal ~ 0");
    }

    // Translation T = sum(origin_half * conf) / sum(conf).
    let mut tn = [0.0f64; 3];
    let mut csum = 0.0f64;
    for nn in 0..n {
        let c = cloud.conf_raw[nn];
        tn[0] += cloud.origin_half[3 * nn] * c;
        tn[1] += cloud.origin_half[3 * nn + 1] * c;
        tn[2] += cloud.origin_half[3 * nn + 2] * c;
        csum += c;
    }
    if csum.abs() < 1e-12 {
        return Err("ray-pose: all-zero confidence");
    }
    for v in tn.iter_mut() {
        *v /= csum;
    }

    out.r = r_mat;
    out.t = tn;
    // Returned focal = 1/f, pp = pp_raw + 1.
    out.focal[0] = 1.0 / f0;
    out.focal[1] = 1.0 / f1;
    out.pp[0] = pp0 + 1.0;
    out.pp[1] = pp1 + 1.0;

    // Assemble w2c = [[R, T],[0,0,0,1]] then affine_inverse ->
    // c2w = [[R^T, -R^T T],[..]].
    let rt = linalg::mat3_transpose(&r_mat);
    let mut c2w_t = [0.0f64; 3];
    for i in 0..3 {
        c2w_t[i] = -(rt[i * 3] * tn[0] + rt[i * 3 + 1] * tn[1] + rt[i * 3 + 2] * tn[2]);
    }
    // Extrinsics 3x4 row-major.
    for i in 0..3 {
        out.extrinsics[i * 4] = rt[i * 3] as f32;
        out.extrinsics[i * 4 + 1] = rt[i * 3 + 1] as f32;
        out.extrinsics[i * 4 + 2] = rt[i * 3 + 2] as f32;
        out.extrinsics[i * 4 + 3] = c2w_t[i] as f32;
    }
    // Intrinsics: K00 = fr0/2*W, K11 = fr1/2*H, K02 = pp0*W*0.5, K12 = pp1*H*0.5.
    out.intrinsics[0] = (out.focal[0] / 2.0 * img_w as f64) as f32;
    out.intrinsics[4] = (out.focal[1] / 2.0 * img_h as f64) as f32;
    out.intrinsics[2] = (out.pp[0] * img_w as f64 * 0.5) as f32;
    out.intrinsics[5] = (out.pp[1] * img_h as f64 * 0.5) as f32;
    out.intrinsics[8] = 1.0;

    // Final safety net: never report success with a non-finite pose.
    for &v in out.extrinsics.iter() {
        if !v.is_finite() {
            return Err("ray-pose: non-finite extrinsics");
        }
    }
    for &v in out.intrinsics.iter() {
        if !v.is_finite() {
            return Err("ray-pose: non-finite intrinsics");
        }
    }
    Ok(out)
}

/// `std::shuffle` equivalent: Fisher-Yates using a bounded-uniform draw.
///
/// Matches the standard's specified algorithm (`for i in 1..len: swap(i,
/// uniform(0, i))`). The integer mapping inside [`Mt19937::uniform_int`] follows
/// libstdc++; see the module-level parity note.
fn shuffle(slice: &mut [usize], rng: &mut Mt19937) {
    let len = slice.len();
    if len < 2 {
        return;
    }
    for i in (1..len).rev() {
        let j = rng.uniform_int(0, i);
        slice.swap(i, j);
    }
}

// =====================================================================
// mt19937 — bit-exact port of std::mt19937 (C++ standard, [rand.eng.mers]).
// =====================================================================

/// The 32-bit Mersenne Twister. Bit-exact with `std::mt19937` for a given seed:
/// the state-transition and output tempering are fully specified by the C++
/// standard, so this matches libstdc++, libc++ and MSVC identically.
pub struct Mt19937 {
    state: [u32; Mt19937::N],
    index: usize,
}

impl Mt19937 {
    const N: usize = 624;
    const M: usize = 397;
    const MATRIX_A: u32 = 0x9908_b0df;
    const UPPER_MASK: u32 = 0x8000_0000;
    const LOWER_MASK: u32 = 0x7fff_ffff;

    /// Construct from a seed (matches `std::mt19937(seed)`).
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; Self::N];
        state[0] = seed;
        for i in 1..Self::N {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            state,
            index: Self::N,
        }
    }

    /// Generate the next raw 32-bit word (matches `operator()`).
    fn next(&mut self) -> u32 {
        if self.index >= Self::N {
            self.twist();
        }
        let y = self.state[self.index];
        self.index += 1;
        // Tempering (exact bit operations from the standard).
        let y = y ^ (y >> 11);
        let y = y ^ ((y << 7) & 0x9d2c_5680);
        let y = y ^ ((y << 15) & 0xefc6_0000);
        y ^ (y >> 18)
    }

    fn twist(&mut self) {
        for k in 0..Self::N {
            let y = (self.state[k] & Self::UPPER_MASK)
                | (self.state[(k + 1) % Self::N] & Self::LOWER_MASK);
            let mut next = self.state[(k + Self::M) % Self::N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= Self::MATRIX_A;
            }
            self.state[k] = next;
        }
        self.index = 0;
    }

    /// Draw from the integer interval `[lo, hi]` (inclusive).
    ///
    /// Implements the libstdc++ rejection-sampling algorithm for
    /// `std::uniform_int_distribution<int>`. The output therefore matches a
    /// libstdc++ C++ build; it may differ from an MSVC or libc++ build (see the
    /// module-level parity note).
    pub fn uniform_int(&mut self, lo: usize, hi: usize) -> usize {
        let range = (hi - lo) as u64 + 1; // inclusive span
        if range == 0 {
            return lo;
        }
        // 32-bit engine; __int64 widening as in libstdc++.
        let scale = (u64::from(u32::MAX) + 1) / range; // u32::MAX+1 = 2^32
        let max_accepted = scale * range - 1;
        loop {
            let r = u64::from(self.next());
            if r <= max_accepted {
                return lo + (r / scale) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known first outputs of `std::mt19937` — verified against the canonical
    /// test vector (default seed 5489 -> 3499211612, which is reproducible across
    /// libstdc++ / libc++ / MSVC since the engine is fully standard-specified).
    #[test]
    fn mt19937_default_seed_known_output() {
        // std::mt19937 default seed (5489) first draw = 3499211612.
        let mut rng = Mt19937::new(5489);
        assert_eq!(mt_next_for_test(&mut rng), 3499211612);
        // A few more (also canonical).
        assert_eq!(mt_next_for_test(&mut rng), 581869302);
        assert_eq!(mt_next_for_test(&mut rng), 3890346734);
        assert_eq!(mt_next_for_test(&mut rng), 3586334585);
        assert_eq!(mt_next_for_test(&mut rng), 545404204);
    }

    // Test-only accessor for the private `next()`.
    fn mt_next_for_test(rng: &mut Mt19937) -> u32 {
        rng.next()
    }

    #[test]
    fn uniform_int_in_range() {
        let mut rng = Mt19937::new(42);
        for _ in 0..1000 {
            let v = rng.uniform_int(5, 9);
            assert!((5..=9).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn uniform_int_degenerate_single() {
        let mut rng = Mt19937::new(7);
        for _ in 0..10 {
            assert_eq!(rng.uniform_int(3, 3), 3);
        }
    }

    #[test]
    fn build_ray_cloud_shapes() {
        let hy = 4;
        let wx = 6;
        let n = hy * wx;
        let ray = vec![0.0f32; n * 6];
        let conf = vec![1.0f32; n];
        let cloud = build_ray_cloud(&ray, &conf, hy, wx, 1e-4);
        assert_eq!(cloud.src.len(), 2 * n);
        assert_eq!(cloud.dst.len(), 2 * n);
        assert_eq!(cloud.w.len(), n);
        assert_eq!(cloud.origin_half.len(), 3 * n);
        assert_eq!(cloud.conf_raw.len(), n);
    }

    #[test]
    fn build_ray_cloud_zmask_zero_weights() {
        // All ray directions flat (dir_z = 0) -> every weight zeroed.
        let hy = 2;
        let wx = 2;
        let n = hy * wx;
        // dir = (0,0,0), origin = (0,0,0): dirz=0 -> zmask false -> w=0.
        let ray = vec![0.0f32; n * 6];
        let conf = vec![1.0f32; n];
        let cloud = build_ray_cloud(&ray, &conf, hy, wx, 1e-4);
        for w in cloud.w.iter() {
            assert!((w - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn build_ray_cloud_identity_grid() {
        // For a 1x1 grid the grid point is dx=dy=1, so origin_plane = (0,0).
        let ray = vec![0.0f32, 0.0f32, 1.0, 0.0, 0.0, 0.0]; // dir=(0,0,1)
        let conf = vec![1.0f32];
        let cloud = build_ray_cloud(&ray, &conf, 1, 1, 1e-4);
        // src = (gx-1, gy-1) = (1-1, 1-1) = (0,0)
        assert!((cloud.src[0] - 0.0).abs() < 1e-12);
        assert!((cloud.src[1] - 0.0).abs() < 1e-12);
        // dst = (dirx/dirz, diry/dirz) = (0/1, 0/1) = (0,0)
        assert!((cloud.dst[0] - 0.0).abs() < 1e-12);
        assert!((cloud.dst[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn solve_ray_pose_degenerate_returns_err() {
        // All-zero rays -> all-zero confidence -> solver bails.
        let hy = 4;
        let wx = 4;
        let n = hy * wx;
        let ray = vec![0.0f32; n * 6];
        let conf = vec![0.0f32; n];
        let p = RayPoseParams::default();
        let res = solve_ray_pose(&ray, &conf, hy, wx, 32, 32, &RayPoseIndices::default(), &p);
        assert!(res.is_err());
    }
}
