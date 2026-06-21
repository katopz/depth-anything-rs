//! The host-side Gaussian adapter — pure math, NO learned weights. Matches
//! `src/gs_adapter.cpp` and the Python `depth_anything_3.model.gs_adapter.forward`
//! (plus `utils/geometry.py`, `model/utils/transform.py`, `utils/sh_helpers.py`).
//!
//! Converts the raw 37-channel GSDPT output + the giant depth map + a camera
//! pose into world-space 3D Gaussians (means, scales, rotations, SH harmonics,
//! opacities). All internal math is `f64`; outputs are narrowed to `f32`.
//!
//! # Raw channel layout (37, channels-last)
//!
//! ```text
//! ch 0,1     : offset_xy        (stripped from the FRONT, second)
//! ch 2,3,4   : scales (logits)
//! ch 5..8    : quaternion xyzw
//! ch 9..35   : sh (3 colors × 9 coeff; color-major: ch 9 + color*9 + coeff)
//! ch 36      : offset_depth     (stripped from the END, first)
//! ```

#![allow(clippy::needless_range_loop)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
#![allow(clippy::too_many_arguments)]

/// World-space 3D Gaussians produced by the adapter. All arrays are row-major;
/// `N = H * W` (single view, `B = V = 1`).
///
/// Mirrors `da::Gaussians` in `src/gs_adapter.hpp`.
#[derive(Debug, Clone, Default)]
pub struct Gaussians {
    /// Number of Gaussians (`H * W`).
    pub n: usize,
    /// `[N*3]` (pixel*3 + xyz).
    pub means: Vec<f32>,
    /// `[N*3]` (pixel*3 + xyz).
    pub scales: Vec<f32>,
    /// `[N*4]` wxyz (pixel*4 + c).
    pub rotations: Vec<f32>,
    /// `[N*3*9]` ((pixel*3 + color)*9 + coeff).
    pub harmonics: Vec<f32>,
    /// `[N]`.
    pub opacities: Vec<f32>,
}

/// The adapter's configuration knobs. Defaults match the reference
/// (`sh_degree=2`, `pred_offset_depth=true`, `pred_offset_xy=true`,
/// `pred_color=false`, `scale_min=1e-5`, `scale_max=30`).
#[derive(Debug, Clone)]
pub struct GsAdapter {
    pub sh_degree: usize,
    /// `(sh_degree + 1)^2`. Cached; equals 9 for `sh_degree = 2`.
    pub d_sh: usize,
    pub pred_offset_depth: bool,
    pub pred_offset_xy: bool,
    pub pred_color: bool,
    pub scale_min: f64,
    pub scale_max: f64,
}

impl Default for GsAdapter {
    fn default() -> Self {
        Self {
            sh_degree: 2,
            d_sh: 9,
            pred_offset_depth: true,
            pred_offset_xy: true,
            pred_color: false,
            scale_min: 1e-5,
            scale_max: 30.0,
        }
    }
}

impl GsAdapter {
    /// Build the world-space Gaussians. Returns `Err` on malformed input.
    ///
    /// * `raw_gs` — `[H*W*37]`, channels-last.
    /// * `depth`  — `[H*W]`, row-major.
    /// * `gs_conf` — `[H*W]`, row-major.
    /// * `ext` — 3×4 row-major world→camera extrinsics (w2c).
    /// * `intr` — 3×3 row-major camera intrinsics `K` (pixel units, `H`/`W`).
    pub fn build(
        &self,
        raw_gs: &[f32],
        depth: &[f32],
        gs_conf: &[f32],
        ext: &[f32; 12],
        intr: &[f32; 9],
        h: usize,
        w: usize,
    ) -> Result<Gaussians, BuildError> {
        let hw = h * w;
        if raw_gs.len() != hw * 37 {
            return Err(BuildError::BadInputLen {
                name: "raw_gs",
                expected: hw * 37,
                got: raw_gs.len(),
            });
        }
        if depth.len() != hw {
            return Err(BuildError::BadInputLen {
                name: "depth",
                expected: hw,
                got: depth.len(),
            });
        }
        if gs_conf.len() != hw {
            return Err(BuildError::BadInputLen {
                name: "gs_conf",
                expected: hw,
                got: gs_conf.len(),
            });
        }
        let dsh = self.d_sh;
        let n = hw;

        // --- extrinsics (w2c, 3x4) -> homogeneous 4x4 -> cam2world = affine_inverse.
        // c2w rotation Rc2w = R^T, translation Tc2w = -R^T t  (R,t are the w2c block).
        let mut r = [0.0f64; 9];
        let mut tt = [0.0f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] = ext[i * 4 + j] as f64;
            }
            tt[i] = ext[i * 4 + 3] as f64;
        }
        let mut rc2w = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                rc2w[i * 3 + j] = r[j * 3 + i]; // R^T
            }
        }
        let mut tc2w = [0.0f64; 3];
        for i in 0..3 {
            let mut s = 0.0;
            for j in 0..3 {
                s += rc2w[i * 3 + j] * tt[j];
            }
            tc2w[i] = -s;
        }

        // --- intr_normed = K with row0/=W, row1/=H ; inv used for unproject + scale mult.
        let mut kn = [0.0f64; 9];
        for j in 0..3 {
            kn[0 * 3 + j] = intr[0 * 3 + j] as f64 / w as f64;
            kn[1 * 3 + j] = intr[1 * 3 + j] as f64 / h as f64;
            kn[2 * 3 + j] = intr[2 * 3 + j] as f64;
        }
        let kninv = mat3_inverse(&kn);
        // get_scale_multiplier: 0.1 * sum( inv(Kn[:2,:2]) @ (1/W, 1/H) )
        let k2 = [kn[0], kn[1], kn[3], kn[4]];
        let det2 = k2[0] * k2[3] - k2[1] * k2[2];
        let id2 = if det2 != 0.0 { 1.0 / det2 } else { 0.0 };
        let k2inv = [k2[3] * id2, -k2[1] * id2, -k2[2] * id2, k2[0] * id2];
        let ps = [1.0 / w as f64, 1.0 / h as f64];
        let mx0 = k2inv[0] * ps[0] + k2inv[1] * ps[1];
        let mx1 = k2inv[2] * ps[0] + k2inv[3] * ps[1];
        let multiplier = 0.1 * (mx0 + mx1);

        // --- SH rotation matrices (same cam2world rotation for all pixels) --------
        // rotate_sh: permute axes yzx->xyz (P^-1 R P), -> e3nn angles -> wigner.
        // P = [[0,0,1],[1,0,0],[0,1,0]]; P^-1 = P^T.
        let (d1, d2) = {
            let p = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            let mut pinv = [0.0f64; 9];
            for i in 0..3 {
                for j in 0..3 {
                    pinv[i * 3 + j] = p[j * 3 + i];
                }
            }
            let t1 = matmul(&pinv, &rc2w, 3);
            let perm_r = matmul(&t1, &p, 3);
            // project_to_so3_strict is ~identity for a proper rotation; skip.
            let perm_r_arr: [f64; 9] = [
                perm_r[0], perm_r[1], perm_r[2], perm_r[3], perm_r[4], perm_r[5], perm_r[6],
                perm_r[7], perm_r[8],
            ];
            let (a, b, g) = matrix_to_angles(&perm_r_arr);
            let mut d1 = [0.0f64; 9];
            let mut d2 = [0.0f64; 25];
            wigner_d(1, a, -b, g, &mut d1);
            wigner_d(2, a, -b, g, &mut d2);
            (d1, d2)
        };

        // sh_mask[deg^2..(deg+1)^2] = 0.1*0.25^deg ; index 0 -> 1.
        let mut sh_mask = [0.0f64; 9];
        sh_mask[0] = 1.0;
        for deg in 1..=self.sh_degree {
            let v = 0.1 * 0.25f64.powi(deg as i32);
            for k in (deg * deg)..((deg + 1) * (deg + 1)) {
                sh_mask[k] = v;
            }
        }

        let mut out = Gaussians {
            n,
            means: vec![0.0; n * 3],
            scales: vec![0.0; n * 3],
            rotations: vec![0.0; n * 4],
            harmonics: vec![0.0; n * 3 * dsh],
            opacities: vec![0.0; n],
        };

        for hh in 0..h {
            for ww in 0..w {
                let pix = hh * w + ww;
                let rg = &raw_gs[pix * 37..pix * 37 + 37];

                // depth offset (ch 36) and depth.
                let mut gs_depth = depth[pix] as f64 + rg[36] as f64;
                if !self.pred_offset_depth {
                    gs_depth = depth[pix] as f64;
                }

                // xy offset (ch 0,1). sample_image_grid: x=(w+0.5)/W, y=(h+0.5)/H.
                let (ox, oy) = if self.pred_offset_xy {
                    (rg[0] as f64, rg[1] as f64)
                } else {
                    (0.0, 0.0)
                };
                let xr = (ww as f64 + 0.5) / w as f64 + ox / w as f64;
                let yr = (hh as f64 + 0.5) / h as f64 + oy / h as f64;

                // unproject -> camera dir = normalize(Kn^-1 @ (xr,yr,1)).
                let mut dx = kninv[0] * xr + kninv[1] * yr + kninv[2];
                let mut dy = kninv[3] * xr + kninv[4] * yr + kninv[5];
                let mut dz = kninv[6] * xr + kninv[7] * yr + kninv[8];
                let dn = (dx * dx + dy * dy + dz * dz).sqrt();
                if dn > 0.0 {
                    dx /= dn;
                    dy /= dn;
                    dz /= dn;
                }
                // world dir = Rc2w @ dir (homogeneous vector, w=0).
                let wdx = rc2w[0] * dx + rc2w[1] * dy + rc2w[2] * dz;
                let wdy = rc2w[3] * dx + rc2w[4] * dy + rc2w[5] * dz;
                let wdz = rc2w[6] * dx + rc2w[7] * dy + rc2w[8] * dz;
                out.means[pix * 3 + 0] = (tc2w[0] + wdx * gs_depth) as f32;
                out.means[pix * 3 + 1] = (tc2w[1] + wdy * gs_depth) as f32;
                out.means[pix * 3 + 2] = (tc2w[2] + wdz * gs_depth) as f32;

                // scales (ch 2,3,4) sigmoid -> [min,max] * depth * multiplier.
                for d in 0..3 {
                    let sg = 1.0 / (1.0 + (-(rg[2 + d] as f64)).exp());
                    let sc = self.scale_min + (self.scale_max - self.scale_min) * sg;
                    out.scales[pix * 3 + d] = (sc * gs_depth * multiplier) as f32;
                }

                // quaternion (ch 5,6,7,8) xyzw normalized -> world wxyz.
                let mut q = [rg[5] as f64, rg[6] as f64, rg[7] as f64, rg[8] as f64];
                let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt() + 1e-8;
                for v in q.iter_mut() {
                    *v /= qn;
                }
                let mut world_wxyz = [0.0f64; 4];
                cam_quat_to_world_wxyz(&q, &rc2w, &mut world_wxyz);
                for d in 0..4 {
                    out.rotations[pix * 4 + d] = world_wxyz[d] as f32;
                }

                // SH (ch 9.. ) : color-major (color*9 + coeff). mask then rotate.
                for color in 0..3 {
                    let mut sh = [0.0f64; 9];
                    for k in 0..dsh {
                        sh[k] = rg[9 + color * dsh + k] as f64 * sh_mask[k];
                    }
                    let mut rot = [0.0f64; 9];
                    rot[0] = sh[0]; // degree 0 unchanged
                                    // degree 1: D1 (3x3) on sh[1..3]
                    for i in 0..3 {
                        let mut s = 0.0;
                        for j in 0..3 {
                            s += d1[i * 3 + j] * sh[1 + j];
                        }
                        rot[1 + i] = s;
                    }
                    // degree 2: D2 (5x5) on sh[4..8]
                    for i in 0..5 {
                        let mut s = 0.0;
                        for j in 0..5 {
                            s += d2[i * 5 + j] * sh[4 + j];
                        }
                        rot[4 + i] = s;
                    }
                    let hp =
                        &mut out.harmonics[(pix * 3 + color) * dsh..(pix * 3 + color) * dsh + dsh];
                    for k in 0..dsh {
                        hp[k] = rot[k] as f32;
                    }
                }

                // opacity = map_pdf_to_opacity(conf) with global_step=0 -> identity.
                let pdf = gs_conf[pix] as f64;
                out.opacities[pix] = (0.5 * (1.0 - (1.0 - pdf) + pdf)) as f32; // == pdf
            }
        }
        Ok(out)
    }
}

/// Error returned by [`GsAdapter::build`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// An input array had the wrong length.
    BadInputLen {
        name: &'static str,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::BadInputLen {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "gs_adapter: {name} expected length {expected}, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for BuildError {}

impl From<BuildError> for crate::Error {
    fn from(e: BuildError) -> Self {
        crate::Error::Model(e.to_string())
    }
}

// ---- small dense linear algebra (row-major) --------------------------------

/// 3×3 inverse via the adjugate/cofactor formula. Returns the zero matrix on
/// singular input (matches `mat3_inverse` in the C++).
fn mat3_inverse(m: &[f64; 9]) -> [f64; 9] {
    let (a, b, c, d, e, f, g, h, i) = (m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
    let aa = e * i - f * h;
    let bb = -(d * i - f * g);
    let cc = d * h - e * g;
    let det = a * aa + b * bb + c * cc;
    let id = if det != 0.0 { 1.0 / det } else { 0.0 };
    [
        aa * id,
        (c * h - b * i) * id,
        (b * f - c * e) * id,
        bb * id,
        (a * i - c * g) * id,
        (c * d - a * f) * id,
        cc * id,
        (b * g - a * h) * id,
        (a * e - b * d) * id,
    ]
}

/// `A[n*n] @ B[n*n]` → out (`n <= 5` to keep stack-allocated buffers).
fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// Generic matrix exponential (scaling-and-squaring + Taylor), `n <= 5`.
fn mat_exp(a: &[f64], n: usize) -> Vec<f64> {
    let mut nrm = 0.0f64;
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j].abs();
        }
        if s > nrm {
            nrm = s;
        }
    }
    let mut k = 0;
    let mut scale = 1.0;
    while nrm * scale > 0.25 {
        scale *= 0.5;
        k += 1;
    }
    let mut b = vec![0.0; n * n];
    for i in 0..n * n {
        b[i] = a[i] * scale;
    }
    let mut term = vec![0.0; n * n];
    let mut acc = vec![0.0; n * n];
    for i in 0..n {
        term[i * n + i] = 1.0;
        acc[i * n + i] = 1.0;
    }
    for m in 1..=18 {
        let tmp = matmul(&term, &b, n);
        for i in 0..n * n {
            term[i] = tmp[i] / m as f64;
        }
        for i in 0..n * n {
            acc[i] += term[i];
        }
    }
    for _ in 0..k {
        let tmp = matmul(&acc, &acc, n);
        acc = tmp;
    }
    acc
}

// ---- quaternion <-> matrix (pytorch3d convention, scalar-LAST xyzw) ---------

/// Input `q` is **xyzw** (i, j, k, r). Matches `quat_xyzw_to_mat` in the C++.
fn quat_xyzw_to_mat(q: &[f64; 4]) -> [f64; 9] {
    let (i, j, k, r) = (q[0], q[1], q[2], q[3]);
    let two_s = 2.0 / (i * i + j * j + k * k + r * r);
    [
        1.0 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1.0 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1.0 - two_s * (i * i + j * j),
    ]
}

/// Returns xyzw (ijkr), standardized so the real part is `>= 0`. Matches
/// `mat_to_quat_xyzw` in the C++ (this is the historical-quirk path: the
/// stored "rotations" buffer keeps mat_to_quat's xyzw output verbatim).
fn mat_to_quat_xyzw(m: &[f64; 9]) -> [f64; 4] {
    let (m00, m01, m02, m10, m11, m12, m20, m21, m22) =
        (m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
    let mut qa = [
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ];
    for v in qa.iter_mut() {
        *v = if *v > 0.0 { v.sqrt() } else { 0.0 };
    }
    // candidate rows (rijk order); pick the best-conditioned (largest qa).
    let cand = [
        [qa[0] * qa[0], m21 - m12, m02 - m20, m10 - m01],
        [m21 - m12, qa[1] * qa[1], m10 + m01, m02 + m20],
        [m02 - m20, m10 + m01, qa[2] * qa[2], m12 + m21],
        [m10 - m01, m20 + m02, m21 + m12, qa[3] * qa[3]],
    ];
    let mut best = 0;
    for t in 1..4 {
        if qa[t] > qa[best] {
            best = t;
        }
    }
    let flr = qa[best].max(0.1);
    let denom = 2.0 * flr;
    let mut rijk = [0.0f64; 4];
    for t in 0..4 {
        rijk[t] = cand[best][t] / denom;
    }
    // rijk -> ijkr (out indices [1,2,3,0])
    let (mut x, mut y, mut z, mut w) = (rijk[1], rijk[2], rijk[3], rijk[0]);
    if w < 0.0 {
        // standardize: real part >= 0
        x = -x;
        y = -y;
        z = -z;
        w = -w;
    }
    [x, y, z, w]
}

/// Rotate the camera-space quaternion `cam_xyzw` into world space using the
/// cam2world rotation, writing the result (xyzw-labeled-as-wxyz to match the
/// reference) into `out_wxyz`.
///
/// See the long note in `gs_adapter.cpp::cam_quat_to_world_wxyz`: the python
/// re-labels `(w,x,y,z)` as the `(i,j,k,r)` input to `quat_to_mat`, i.e. it
/// feeds `[w,x,y,z]` positionally as xyzw. The result of `mat_to_quat` is then
/// stored verbatim — there is NO reordering.
fn cam_quat_to_world_wxyz(cam_xyzw: &[f64; 4], rc2w: &[f64; 9], out_wxyz: &mut [f64; 4]) {
    // xyzw -> wxyz, then build rotation matrix via quat_to_mat (which takes xyzw).
    let q_relabel = [cam_xyzw[3], cam_xyzw[0], cam_xyzw[1], cam_xyzw[2]]; // (w,x,y,z)
    let rcam = quat_xyzw_to_mat(&q_relabel);
    let rworld_mat = matmul(rc2w, &rcam, 3);
    let rworld = [
        rworld_mat[0],
        rworld_mat[1],
        rworld_mat[2],
        rworld_mat[3],
        rworld_mat[4],
        rworld_mat[5],
        rworld_mat[6],
        rworld_mat[7],
        rworld_mat[8],
    ];
    let q = mat_to_quat_xyzw(&rworld);
    out_wxyz[0] = q[0];
    out_wxyz[1] = q[1];
    out_wxyz[2] = q[2];
    out_wxyz[3] = q[3];
}

// ---- e3nn angle / wigner machinery -----------------------------------------

fn matrix_y(a: f64) -> [f64; 9] {
    let (c, s) = (a.cos(), a.sin());
    [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]
}
fn matrix_x(a: f64) -> [f64; 9] {
    let (c, s) = (a.cos(), a.sin());
    [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]
}

/// `angles_to_matrix = Ry(a) @ Rx(b) @ Ry(g)`.
fn angles_to_matrix(a: f64, b: f64, g: f64) -> [f64; 9] {
    let ya = matrix_y(a);
    let xb = matrix_x(b);
    let yg = matrix_y(g);
    let t = matmul(&ya, &xb, 3);
    let t_arr = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]];
    let r = matmul(&t_arr, &yg, 3);
    [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]]
}

/// e3nn `matrix_to_angles`: returns `(alpha, beta, gamma)`.
fn matrix_to_angles(r: &[f64; 9]) -> (f64, f64, f64) {
    // x = R @ [0,1,0]  -> second column of R.
    let (mut x0, mut x1, x2) = (r[1], r[4], r[7]);
    let nrm = (x0 * x0 + x1 * x1 + x2 * x2).sqrt();
    if nrm > 0.0 {
        x0 /= nrm;
        x1 /= nrm;
    }
    x1 = x1.clamp(-1.0, 1.0);
    let beta = x1.acos();
    let alpha = x0.atan2(x2);
    // R2 = angles_to_matrix(a,b,0)^T @ R ; gamma = atan2(R2[0,2], R2[0,0])
    let a = angles_to_matrix(alpha, beta, 0.0);
    let mut at = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            at[i * 3 + j] = a[j * 3 + i];
        }
    }
    let r2 = matmul(&at, r, 3);
    let gamma = r2[2].atan2(r2[0]);
    (alpha, beta, gamma)
}

/// `_z_rot_mat(angle, l)`: `(2l+1) × (2l+1)`.
fn z_rot_mat(angle: f64, l: usize) -> Vec<f64> {
    let d = 2 * l + 1;
    let mut m = vec![0.0; d * d];
    for idx in 0..d {
        let rev = d - 1 - idx;
        let freq = (l as f64) - idx as f64;
        m[idx * d + rev] = (freq * angle).sin();
        m[idx * d + idx] = (freq * angle).cos();
    }
    m
}

/// so(3) x-generator `X0` for degree `l` (exact entries) → dense `(2l+1)^2`.
fn x_gen(l: usize) -> Vec<f64> {
    let d = 2 * l + 1;
    let mut x0 = vec![0.0; d * d];
    if l == 1 {
        // [[0,0,0],[0,0,-1],[0,1,0]]
        x0[1 * 3 + 2] = -1.0;
        x0[2 * 3 + 1] = 1.0;
    } else {
        // l == 2
        let s3 = 3.0f64.sqrt();
        // [[0,1,0,0,0],[-1,0,0,0,0],[0,0,0,-s3,0],[0,0,s3,0,-1],[0,0,0,1,0]]
        x0[0 * 5 + 1] = 1.0;
        x0[1 * 5 + 0] = -1.0;
        x0[2 * 5 + 3] = -s3;
        x0[3 * 5 + 2] = s3;
        x0[3 * 5 + 4] = -1.0;
        x0[4 * 5 + 3] = 1.0;
    }
    x0
}

/// `wigner_D(l, a, b, g) = z_rot(a) @ expm(b*X0) @ z_rot(g)`.
///
/// Matches e3nn `wigner_D` via so3 generators: `matrix_exp(a*Xz) @
/// matrix_exp(b*Xx) @ matrix_exp(g*Xz)`, with `Xz` in closed form = `z_rot_mat`.
fn wigner_d(l: usize, a: f64, b: f64, g: f64, out: &mut [f64]) {
    let d = 2 * l + 1;
    let za = z_rot_mat(a, l);
    let zg = z_rot_mat(g, l);
    let x0 = x_gen(l);
    let mut bx0 = vec![0.0; d * d];
    for i in 0..d * d {
        bx0[i] = b * x0[i];
    }
    let eb = mat_exp(&bx0, d);
    let t = matmul(&za, &eb, d);
    let r = matmul(&t, &zg, d);
    out[..r.len()].copy_from_slice(&r);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    // ---- mat3_inverse ----------------------------------------------------

    #[test]
    fn mat3_inverse_identity() {
        let inv = mat3_inverse(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        for v in inv.iter() {
            assert!(approx(*v, 0.0, 1e-12) || approx(*v, 1.0, 1e-12));
        }
    }

    #[test]
    fn mat3_inverse_diagonal() {
        let inv = mat3_inverse(&[2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0]);
        assert!(approx(inv[0], 0.5, 1e-12));
        assert!(approx(inv[4], 0.25, 1e-12));
        assert!(approx(inv[8], 0.125, 1e-12));
    }

    #[test]
    fn mat3_inverse_singular_returns_zero() {
        let inv = mat3_inverse(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        // Rank-2 -> singular -> zero matrix (matches C++ behavior).
        for v in inv.iter() {
            assert!(approx(*v, 0.0, 1e-12));
        }
    }

    #[test]
    fn mat3_inverse_times_original_is_identity() {
        let m = [1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0];
        let inv = mat3_inverse(&m);
        let prod = matmul(&m, &inv, 3);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx(prod[i * 3 + j], want, 1e-9),
                    "[{i},{j}]={}",
                    prod[i * 3 + j]
                );
            }
        }
    }

    // ---- quaternion / rotation matrices ----------------------------------

    #[test]
    fn quat_xyzw_identity_to_mat() {
        // xyzw = (0, 0, 0, 1) is the identity quaternion.
        let r = quat_xyzw_to_mat(&[0.0, 0.0, 0.0, 1.0]);
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for i in 0..9 {
            assert!(approx(r[i], eye[i], 1e-12), "r[{i}]={}", r[i]);
        }
    }

    #[test]
    fn quat_xyzw_180_about_z_to_mat() {
        // xyzw = (0, 0, 1, 0) is 180° about z: diag(-1, -1, 1).
        let r = quat_xyzw_to_mat(&[0.0, 0.0, 1.0, 0.0]);
        assert!(approx(r[0], -1.0, 1e-12));
        assert!(approx(r[4], -1.0, 1e-12));
        assert!(approx(r[8], 1.0, 1e-12));
    }

    #[test]
    fn mat_to_quat_identity_round_trip() {
        // Identity matrix -> wxyz = (1, 0, 0, 0) (after standardization).
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let q = mat_to_quat_xyzw(&eye); // returns xyzw-labeled-as-wxyz quirk
                                        // r=1 -> ijkr=[0,0,0,1] -> (x,y,z,w)=(0,0,0,1)
        assert!(approx(q[3], 1.0, 1e-9), "w={}", q[3]);
        assert!(approx(q[0], 0.0, 1e-9));
        assert!(approx(q[1], 0.0, 1e-9));
        assert!(approx(q[2], 0.0, 1e-9));
    }

    #[test]
    fn mat_to_quat_180_about_y_round_trip() {
        // 180° about y: diag(-1, 1, -1). wxyz = (0, 0, 1, 0).
        let m = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        let q = mat_to_quat_xyzw(&m);
        // qa = [0, 0, 4, 0] -> best=2, qa[2]=sqrt(4)=2, denom=2*2=4.
        // candidate[2] = [m02-m20, m10+m01, qa[2]^2, m12+m21] = [0, 0, 4, 0].
        // rijk = candidate[best]/denom = [0, 0, 1, 0].
        // output (x,y,z,w) = (rijk[1], rijk[2], rijk[3], rijk[0]) = (0, 1, 0, 0).
        assert!(approx(q[1], 1.0, 1e-9), "y={}", q[1]);
        assert!(approx(q[0], 0.0, 1e-9));
        assert!(approx(q[2], 0.0, 1e-9));
        assert!(approx(q[3], 0.0, 1e-9));
    }

    // ---- wigner_D --------------------------------------------------------

    #[test]
    fn wigner_d_degree_1_identity() {
        // angles (0,0,0) -> D = identity for any l.
        let mut d1 = [0.0f64; 9];
        wigner_d(1, 0.0, 0.0, 0.0, &mut d1);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx(d1[i * 3 + j], want, 1e-12),
                    "D1[{i},{j}]={}",
                    d1[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn wigner_d_degree_2_identity() {
        let mut d2 = [0.0f64; 25];
        wigner_d(2, 0.0, 0.0, 0.0, &mut d2);
        for i in 0..5 {
            for j in 0..5 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx(d2[i * 5 + j], want, 1e-12),
                    "D2[{i},{j}]={}",
                    d2[i * 5 + j]
                );
            }
        }
    }

    #[test]
    fn wigner_d_degree_1_alpha_rotation() {
        // wigner_D(1, a, 0, 0) = z_rot(a, 1) @ I @ I = z_rot(a, 1).
        // z_rot_mat(angle, l=1) = [[c,0,s],[0,1,0],[-s,0,c]] with freq (1,0,-1).
        let a = 0.7;
        let mut d1 = [0.0f64; 9];
        wigner_d(1, a, 0.0, 0.0, &mut d1);
        // Expected: freq=+1 (idx=0) -> cos, sin; freq=-1 (idx=2) -> cos, -sin.
        // m[0,0]=cos(a), m[0,2]=sin(a), m[2,0]=-sin(a), m[2,2]=cos(a), m[1,1]=1.
        assert!(approx(d1[0], a.cos(), 1e-9));
        assert!(approx(d1[2], a.sin(), 1e-9));
        assert!(approx(d1[6], -a.sin(), 1e-9));
        assert!(approx(d1[8], a.cos(), 1e-9));
        assert!(approx(d1[4], 1.0, 1e-9));
    }

    // ---- GsAdapter::build end-to-end on a tiny hand-rolled input -----------

    /// A compact hand-rolled input bundle for `GsAdapter::build` tests.
    type TrivialIn = (Vec<f32>, Vec<f32>, Vec<f32>, [f32; 12], [f32; 9]);

    fn trivial_input(h: usize, w: usize) -> TrivialIn {
        let hw = h * w;
        let mut raw_gs = vec![0.0f32; hw * 37];
        // Zero raw_gs means: offset_xy=0, scales sigmoid(0)=0.5, quaternion=(0,0,0,0)
        // (degenerate), sh=0, offset_depth=0. We'll set a known quaternion below.
        for pix in 0..hw {
            // Identity quaternion xyzw -> (0,0,0,1) in raw channels 5..8.
            raw_gs[pix * 37 + 5] = 0.0;
            raw_gs[pix * 37 + 6] = 0.0;
            raw_gs[pix * 37 + 7] = 0.0;
            raw_gs[pix * 37 + 8] = 1.0;
            // Scale logits 0 -> sigmoid=0.5 (mid of [scale_min, scale_max]).
            raw_gs[pix * 37 + 2] = 0.0;
            raw_gs[pix * 37 + 3] = 0.0;
            raw_gs[pix * 37 + 4] = 0.0;
            // DC SH (coeff 0 of each color) set to a known value to verify masking+rotate.
            raw_gs[pix * 37 + 9] = 0.5; // r DC
            raw_gs[pix * 37 + 18] = 0.6; // g DC
            raw_gs[pix * 37 + 27] = 0.7; // b DC
        }
        let depth = vec![10.0; hw];
        let gs_conf = vec![0.5; hw];
        // Identity extrinsics: w2c = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        let ext = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        // Intrinsics: a focal of fx=fy=W (so normalized focal=1), principal point at center.
        let intr = [
            w as f32,
            0.0,
            w as f32 / 2.0,
            0.0,
            w as f32,
            w as f32 / 2.0,
            0.0,
            0.0,
            1.0,
        ];
        (raw_gs, depth, gs_conf, ext, intr)
    }

    #[test]
    fn build_rejects_bad_lengths() {
        let (raw_gs, depth, gs_conf, ext, intr) = trivial_input(2, 2);
        let ad = GsAdapter::default();
        // raw_gs too short
        let err = ad
            .build(&raw_gs[..10], &depth, &gs_conf, &ext, &intr, 2, 2)
            .unwrap_err();
        assert!(matches!(
            err,
            BuildError::BadInputLen { name: "raw_gs", .. }
        ));
        // depth too short
        let err = ad
            .build(&raw_gs, &depth[..2], &gs_conf, &ext, &intr, 2, 2)
            .unwrap_err();
        assert!(matches!(err, BuildError::BadInputLen { name: "depth", .. }));
        // gs_conf too short
        let err = ad
            .build(&raw_gs, &depth, &gs_conf[..2], &ext, &intr, 2, 2)
            .unwrap_err();
        assert!(matches!(
            err,
            BuildError::BadInputLen {
                name: "gs_conf",
                ..
            }
        ));
    }

    #[test]
    fn build_opacity_is_identity_at_global_step_zero() {
        // map_pdf_to_opacity(conf, global_step=0) == conf (the C++ identity form).
        let (raw_gs, depth, gs_conf, ext, intr) = trivial_input(2, 2);
        let ad = GsAdapter::default();
        let g = ad
            .build(&raw_gs, &depth, &gs_conf, &ext, &intr, 2, 2)
            .unwrap();
        assert_eq!(g.opacities.len(), 4);
        for v in &g.opacities {
            assert!(approx(*v as f64, 0.5, 1e-6));
        }
    }

    #[test]
    fn build_means_unproject_with_identity_extrinsics() {
        // Identity w2c => Rc2w=I, Tc2w=0. Kn normalized: fx/W=1, fy/H=(W/H).
        // For a 2x2 image with W=H=2: Kn = [[1,0,0.5],[0,1,0.5],[0,0,1]].
        // Kninv = [[1,0,-0.5],[0,1,-0.5],[0,0,1]] (the principal point offset
        // is subtracted when unprojecting).
        // dir at (w=0,h=0): xr = (0+0.5)/2 = 0.25, yr = 0.25.
        // dir = normalize(Kninv @ (0.25, 0.25, 1)) = normalize(-0.25, -0.25, 1).
        // means = dir * depth(=10).
        let (raw_gs, depth, gs_conf, ext, intr) = trivial_input(2, 2);
        let ad = GsAdapter::default();
        let g = ad
            .build(&raw_gs, &depth, &gs_conf, &ext, &intr, 2, 2)
            .unwrap();
        let dx = -0.25f64;
        let dy = -0.25f64;
        let dz = 1.0f64;
        let dn = (dx * dx + dy * dy + dz * dz).sqrt();
        let mx = dx / dn * 10.0;
        let my = dy / dn * 10.0;
        let mz = dz / dn * 10.0;
        assert!(
            approx(g.means[0] as f64, mx, 1e-5),
            "got {} want {}",
            g.means[0],
            mx
        );
        assert!(
            approx(g.means[1] as f64, my, 1e-5),
            "got {} want {}",
            g.means[1],
            my
        );
        assert!(
            approx(g.means[2] as f64, mz, 1e-5),
            "got {} want {}",
            g.means[2],
            mz
        );
    }

    #[test]
    fn build_scales_use_sigmoid_depth_multiplier() {
        let (raw_gs, depth, gs_conf, ext, intr) = trivial_input(1, 1);
        let ad = GsAdapter::default();
        let g = ad
            .build(&raw_gs, &depth, &gs_conf, &ext, &intr, 1, 1)
            .unwrap();
        // For trivial 1x1 input: depth=10, sigmoid(0)=0.5, scale=[1e-5, 30] midpoint.
        // multiplier = 0.1 * sum(K2inv @ (1/W, 1/H)).
        // W=H=1: Kn=[[1,0,0.5],[0,1,0.5],[0,0,1]], K2=[[1,0],[0,1]], K2inv=I.
        // ps=(1,1) -> (mx0,mx1)=(1,1) -> multiplier=0.1*2=0.2.
        // scale_min + (scale_max-scale_min)*0.5 = 1e-5 + (30-1e-5)*0.5 ≈ 15.0
        let expected_scale = (1e-5 + (30.0 - 1e-5) * 0.5) * 10.0 * 0.2;
        for d in 0..3 {
            assert!(
                approx(g.scales[d] as f64, expected_scale, 1e-3),
                "scales[{d}]={} expected={expected_scale}",
                g.scales[d]
            );
        }
    }

    #[test]
    fn build_sh_dc_passthrough_when_rotation_is_identity() {
        // Identity extrinsics -> perm_R = I -> angles (0,0,0) -> wigner_D = I.
        // sh_mask[0]=1, sh_mask[1..]=0.1*0.25^deg. So SH coeff 0 passes through
        // unchanged; coeffs 1..8 are scaled by 0.1*0.25 or 0.1*0.0625.
        let (raw_gs, depth, gs_conf, ext, intr) = trivial_input(1, 1);
        let ad = GsAdapter::default();
        let g = ad
            .build(&raw_gs, &depth, &gs_conf, &ext, &intr, 1, 1)
            .unwrap();
        // Color 0 DC = 0.5 (set in trivial_input). sh_mask[0]=1, no rotation -> 0.5.
        assert!(
            approx(g.harmonics[0] as f64, 0.5, 1e-6),
            "harmonics[0]={}",
            g.harmonics[0]
        );
        assert!(
            approx(g.harmonics[9] as f64, 0.6, 1e-6),
            "harmonics[9]={}",
            g.harmonics[9]
        );
        assert!(
            approx(g.harmonics[18] as f64, 0.7, 1e-6),
            "harmonics[18]={}",
            g.harmonics[18]
        );
    }

    #[test]
    fn build_rotation_buffer_matches_cam_quat_world_with_identity_extrinsics() {
        // Identity Rc2w: Rworld = matmul(I, Rcam) = Rcam. The raw quaternion
        // is (0,0,0,1) xyzw (identity in pytorch3d convention).
        // cam_quat_to_world_wxyz: relabels to (w,x,y,z)=(1,0,0,0) and feeds
        // positionally as xyzw=(1,0,0,0).
        // quat_xyzw_to_mat(1,0,0,0): i=1, j=k=r=0, two_s=2/(1)=2.
        // m00=1-2*(j^2+k^2)=1, m11=1-2*(i^2+k^2)=-1, m22=1-2*(i^2+j^2)=-1.
        // Off-diags all 0. Rcam = diag(1,-1,-1) (a 180° flip about x — this is
        // the pytorch3d convention's interpretation of (w=1)).
        // mat_to_quat_xyzw(diag(1,-1,-1)): qa=[0,4,0,0], best=1, qa[1]=2,
        // denom=4. candidate[1]=[0,4,0,0]/4 = [0,1,0,0] = rijk.
        // output (x,y,z,w) = (rijk[1], rijk[2], rijk[3], rijk[0]) = (1,0,0,0).
        let (raw_gs, depth, gs_conf, ext, intr) = trivial_input(1, 1);
        let ad = GsAdapter::default();
        let g = ad
            .build(&raw_gs, &depth, &gs_conf, &ext, &intr, 1, 1)
            .unwrap();
        // rotations is the stored "wxyz"-labeled buffer; the historical quirk
        // stores mat_to_quat's xyzw output verbatim. So x=1, y=z=w=0.
        assert!(
            approx(g.rotations[0] as f64, 1.0, 1e-9),
            "x={}",
            g.rotations[0]
        );
        assert!(
            approx(g.rotations[1] as f64, 0.0, 1e-9),
            "y={}",
            g.rotations[1]
        );
        assert!(
            approx(g.rotations[2] as f64, 0.0, 1e-9),
            "z={}",
            g.rotations[2]
        );
        assert!(
            approx(g.rotations[3] as f64, 0.0, 1e-9),
            "w={}",
            g.rotations[3]
        );
    }

    #[test]
    fn build_pred_offset_depth_false_ignores_ch_36() {
        let (mut raw_gs, depth, gs_conf, ext, intr) = trivial_input(1, 1);
        // Set ch 36 to a large value; with pred_offset_depth=false it should be ignored.
        raw_gs[36] = 100.0;
        let ad = GsAdapter {
            pred_offset_depth: false,
            ..GsAdapter::default()
        };
        let g = ad
            .build(&raw_gs, &depth, &gs_conf, &ext, &intr, 1, 1)
            .unwrap();
        // With identity extrinsics and a 1x1 image: xr = yr = 0.5,
        // Kn = [[1,0,0.5],[0,1,0.5],[0,0,1]], Kninv = [[1,0,-0.5],[0,1,-0.5],[0,0,1]].
        // dir = normalize(Kninv @ (0.5, 0.5, 1)) = normalize(0, 0, 1). mz = 1 * 10.
        // With pred_offset_depth=false, the depth offset is ignored so mz stays
        // at depth(=10) regardless of ch 36.
        assert!(
            approx(g.means[2] as f64, 10.0, 1e-5),
            "got {} want 10",
            g.means[2]
        );
    }

    #[test]
    fn build_pred_offset_xy_false_ignores_ch_0_1() {
        let (mut raw_gs, depth, gs_conf, ext, intr) = trivial_input(1, 1);
        raw_gs[0] = 100.0; // ignored when pred_offset_xy=false
        raw_gs[1] = 100.0;
        let ad = GsAdapter {
            pred_offset_xy: false,
            ..GsAdapter::default()
        };
        let g = ad
            .build(&raw_gs, &depth, &gs_conf, &ext, &intr, 1, 1)
            .unwrap();
        // Same as above: dir = (0, 0, 1) with mz = 10. The xy offset would have
        // shifted xr,yr if it had been applied (giving a different mz), so
        // passing this assertion proves pred_offset_xy was honored.
        assert!(
            approx(g.means[2] as f64, 10.0, 1e-5),
            "got {} want 10",
            g.means[2]
        );
    }
}
