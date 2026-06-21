//! Geometry helpers for 3D reconstruction, matching `src/reconstruct.cpp`.
//!
//! These are the shared primitives consumed by the glb/COLMAP exporters
//! ([`crate::glb_export`], [`crate::colmap_export`]). They back-project per-frame
//! depth maps into a shared world frame and provide the small linear-algebra
//! helpers (`inv3`, `inv4`, `percentile_linear`, `rotmat2qvec`) the exporters
//! need. All matrix math is done in `f64` internally (mirroring the C++) and
//! narrowed to `f32` only at the boundaries, because the reference pipeline
//! (`trimesh.transform_points`, `np.linalg.inv`) runs in float64.

#![allow(clippy::needless_range_loop)]

/// World points produced by back-projecting per-frame depth into a shared world
/// frame, mirroring the reference `_depths_to_world_points_with_colors`.
///
/// All parallel arrays are indexed by point (`0..num_points-1`) in the order the
/// points were appended: frames outer, then row-major pixel order (v outer,
/// u inner) within each frame, keeping only valid pixels.
#[derive(Debug, Clone, Default)]
pub struct WorldPoints {
    /// `3*num_points` (`x,y,z` per point).
    pub xyz: Vec<f32>,
    /// `3*num_points` (`r,g,b` per point).
    pub rgb: Vec<u8>,
    /// `num_points`: source frame index.
    pub frame: Vec<i32>,
    /// `num_points`: source pixel column.
    pub u: Vec<i32>,
    /// `num_points`: source pixel row.
    pub v: Vec<i32>,
}

impl WorldPoints {
    /// Number of points held.
    pub fn len(&self) -> usize {
        self.frame.len()
    }

    /// Whether there are no points.
    pub fn is_empty(&self) -> bool {
        self.frame.is_empty()
    }
}

/// 3×3 inverse (row-major). Returns `None` if singular.
///
/// Uses the adjugate/cofactor formula in `f64`. Matches `reconstruct.cpp::inv3`.
pub fn inv3(m: &[f32; 9]) -> Option<[f32; 9]> {
    let a = m[0] as f64;
    let b = m[1] as f64;
    let c = m[2] as f64;
    let d = m[3] as f64;
    let e = m[4] as f64;
    let f = m[5] as f64;
    let g = m[6] as f64;
    let h = m[7] as f64;
    let i = m[8] as f64;
    let aa = e * i - f * h;
    let bb = -(d * i - f * g);
    let cc = d * h - e * g;
    let det = a * aa + b * bb + c * cc;
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        (aa * inv) as f32,
        (-(b * i - c * h) * inv) as f32,
        ((b * f - c * e) * inv) as f32,
        (bb * inv) as f32,
        ((a * i - c * g) * inv) as f32,
        (-(a * f - c * d) * inv) as f32,
        (cc * inv) as f32,
        (-(a * h - b * g) * inv) as f32,
        ((a * e - b * d) * inv) as f32,
    ])
}

/// 4×4 inverse (row-major) via Gauss-Jordan with partial pivoting, in `f64`.
/// Returns `None` if singular. Matches `reconstruct.cpp::inv4`.
pub fn inv4(m: &[f32; 16]) -> Option<[f32; 16]> {
    // Augmented [m | I] in f64.
    let mut a = [[0.0f64; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            a[r][c] = m[r * 4 + c] as f64;
        }
        for c in 0..4 {
            a[r][4 + c] = if r == c { 1.0 } else { 0.0 };
        }
    }
    for col in 0..4 {
        // Find the pivot (largest |a[r][col]| at or below `col`).
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..4 {
            let v = a[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best == 0.0 || !best.is_finite() {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
        }
        let d = a[col][col];
        for c in 0..8 {
            a[col][c] /= d;
        }
        for r in 0..4 {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..8 {
                a[r][c] -= factor * a[col][c];
            }
        }
    }
    let mut out = [0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = a[r][4 + c] as f32;
        }
    }
    Some(out)
}

/// `numpy.percentile` with linear interpolation: sort a copy, take index
/// `q_percent/100*(n-1)`, interpolate between floor/ceil. Does not mutate input.
///
/// Matches `reconstruct.cpp::percentile_linear`.
pub fn percentile_linear(v: &[f32], q_percent: f64) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return v[0] as f64;
    }
    let mut s: Vec<f32> = v.to_vec();
    s.sort_unstable_by(|a, b| a.total_cmp(b));
    let idx = (q_percent / 100.0) * (n - 1) as f64;
    let lo = idx.floor();
    let hi = idx.ceil();
    let frac = idx - lo;
    let li = lo as usize;
    let mut hii = hi as usize;
    if hii >= n {
        hii = n - 1;
    }
    s[li] as f64 + frac * (s[hii] as f64 - s[li] as f64)
}

/// Rotation matrix (row-major 3×3) → COLMAP-order quaternion `(qw,qx,qy,qz)`
/// with `qw >= 0`, ported from `read_write_model.py::rotmat2qvec`.
///
/// Builds the symmetric 4×4 Shepperd matrix and recovers the quaternion from
/// its largest-eigenvalue eigenvector via cyclic Jacobi rotations.
pub fn rotmat2qvec(r: &[f32; 9]) -> [f32; 4] {
    // R is row-major: R[row*3+col]. Reference flat order is
    // Rxx,Ryx,Rzx, Rxy,Ryy,Rzy, Rxz,Ryz,Rzz (also row-major).
    let rxx = r[0] as f64;
    let ryx = r[1] as f64;
    let rzx = r[2] as f64;
    let rxy = r[3] as f64;
    let ryy = r[4] as f64;
    let rzy = r[5] as f64;
    let rxz = r[6] as f64;
    let ryz = r[7] as f64;
    let rzz = r[8] as f64;

    // Symmetric 4×4 K matrix (lower triangle filled, mirrored to upper).
    let mut k = [[0.0f64; 4]; 4];
    k[0][0] = (rxx - ryy - rzz) / 3.0;
    k[1][1] = (ryy - rxx - rzz) / 3.0;
    k[2][2] = (rzz - rxx - ryy) / 3.0;
    k[3][3] = (rxx + ryy + rzz) / 3.0;
    k[1][0] = (ryx + rxy) / 3.0;
    k[0][1] = k[1][0];
    k[2][0] = (rzx + rxz) / 3.0;
    k[0][2] = k[2][0];
    k[2][1] = (rzy + ryz) / 3.0;
    k[1][2] = k[2][1];
    k[3][0] = (ryz - rzy) / 3.0;
    k[0][3] = k[3][0];
    k[3][1] = (rzx - rxz) / 3.0;
    k[1][3] = k[3][1];
    k[3][2] = (rxy - ryx) / 3.0;
    k[2][3] = k[3][2];

    // Cyclic Jacobi for the symmetric 4×4 eigenproblem.
    let mut vv = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for _ in 0..100 {
        let mut off = 0.0;
        for p in 0..4 {
            for q in (p + 1)..4 {
                off += k[p][q] * k[p][q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                if k[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (k[q][q] - k[p][p]) / (2.0 * k[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate K (both sides).
                for i in 0..4 {
                    let kip = k[i][p];
                    let kiq = k[i][q];
                    k[i][p] = c * kip - s * kiq;
                    k[i][q] = s * kip + c * kiq;
                }
                for i in 0..4 {
                    let kpi = k[p][i];
                    let kqi = k[q][i];
                    k[p][i] = c * kpi - s * kqi;
                    k[q][i] = s * kpi + c * kqi;
                }
                // Accumulate eigenvectors.
                for i in 0..4 {
                    let vip = vv[i][p];
                    let viq = vv[i][q];
                    vv[i][p] = c * vip - s * viq;
                    vv[i][q] = s * vip + c * viq;
                }
            }
        }
    }

    // Eigenvalues are on the diagonal; pick the column of the largest.
    let mut best = 0;
    let mut best_val = k[0][0];
    for i in 1..4 {
        if k[i][i] > best_val {
            best_val = k[i][i];
            best = i;
        }
    }
    // Reference reorders rows [3,0,1,2] -> (qw,qx,qy,qz).
    let ev = [vv[0][best], vv[1][best], vv[2][best], vv[3][best]];
    let mut q = [ev[3] as f32, ev[0] as f32, ev[1] as f32, ev[2] as f32];
    if q[0] < 0.0 {
        q[0] = -q[0];
        q[1] = -q[1];
        q[2] = -q[2];
        q[3] = -q[3];
    }
    q
}

/// Back-project depth maps into world space with per-pixel colors.
///
/// - `depth`, `conf`: `N*H*W` row-major (frame, row, col).
/// - `k`: per-frame 3×3 intrinsics, row-major.
/// - `ext_w2c`: per-frame 4×4 world-to-camera extrinsics, row-major.
/// - `images_u8`: per-frame slice over `H*W*3` RGB uint8 pixels.
///
/// A pixel is valid when `d` is finite and `> 0` and (if confidence is present)
/// `conf >= conf_thr`. For each valid pixel:
/// `ray = inv(K) @ [u,v,1]`; `Xc = ray*d`; `Xw = (inv(ext) @ [Xc;1])[:3]`.
///
/// Frames whose `K` or `ext` are singular are skipped entirely.
#[allow(clippy::too_many_arguments)]
pub fn back_project(
    depth: &[f32],
    conf: &[f32],
    k: &[[f32; 9]],
    ext_w2c: &[[f32; 16]],
    images_u8: &[&[u8]],
    h: usize,
    w: usize,
    n: usize,
    conf_thr: f32,
) -> WorldPoints {
    let mut wp = WorldPoints::default();
    let plane = h * w;
    let have_conf = !conf.is_empty();

    for i in 0..n {
        let Some(kinv) = inv3(&k[i]) else { continue };
        let Some(c2w) = inv4(&ext_w2c[i]) else {
            continue;
        };

        let d_frame = &depth[i * plane..(i + 1) * plane];
        let c_frame = if have_conf {
            &conf[i * plane..(i + 1) * plane]
        } else {
            &[]
        };
        let img = images_u8[i];

        // f64 copies for the math.
        let ki: [f64; 9] = [
            kinv[0] as f64,
            kinv[1] as f64,
            kinv[2] as f64,
            kinv[3] as f64,
            kinv[4] as f64,
            kinv[5] as f64,
            kinv[6] as f64,
            kinv[7] as f64,
            kinv[8] as f64,
        ];
        let cw: [f64; 16] = [
            c2w[0] as f64,
            c2w[1] as f64,
            c2w[2] as f64,
            c2w[3] as f64,
            c2w[4] as f64,
            c2w[5] as f64,
            c2w[6] as f64,
            c2w[7] as f64,
            c2w[8] as f64,
            c2w[9] as f64,
            c2w[10] as f64,
            c2w[11] as f64,
            c2w[12] as f64,
            c2w[13] as f64,
            c2w[14] as f64,
            c2w[15] as f64,
        ];

        for v in 0..h {
            for u in 0..w {
                let pix = v * w + u;
                let d = d_frame[pix];
                if !d.is_finite() || d <= 0.0 {
                    continue;
                }
                if have_conf && c_frame[pix] < conf_thr {
                    continue;
                }

                // ray = Kinv @ [u,v,1]
                let (uu, vv) = (u as f64, v as f64);
                let rx = ki[0] * uu + ki[1] * vv + ki[2];
                let ry = ki[3] * uu + ki[4] * vv + ki[5];
                let rz = ki[6] * uu + ki[7] * vv + ki[8];
                // Xc = ray * d
                let d64 = d as f64;
                let xc = rx * d64;
                let yc = ry * d64;
                let zc = rz * d64;
                // Xw = (c2w @ [Xc;1])[:3]
                let xw = cw[0] * xc + cw[1] * yc + cw[2] * zc + cw[3];
                let yw = cw[4] * xc + cw[5] * yc + cw[6] * zc + cw[7];
                let zw = cw[8] * xc + cw[9] * yc + cw[10] * zc + cw[11];

                wp.xyz.push(xw as f32);
                wp.xyz.push(yw as f32);
                wp.xyz.push(zw as f32);
                wp.rgb.push(img[3 * pix]);
                wp.rgb.push(img[3 * pix + 1]);
                wp.rgb.push(img[3 * pix + 2]);
                wp.frame.push(i as i32);
                wp.u.push(u as i32);
                wp.v.push(v as i32);
            }
        }
    }
    wp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv3_identity_and_scaled() {
        let id = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = inv3(&id).unwrap();
        for a in inv.iter() {
            assert!((a - 1.0).abs() < 1e-6 || a.abs() < 1e-6);
        }
        // 2*I -> inverse is 0.5*I
        let two = [2.0f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let inv = inv3(&two).unwrap();
        assert!((inv[0] - 0.5).abs() < 1e-6);
        assert!((inv[4] - 0.5).abs() < 1e-6);
        assert!((inv[8] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn inv3_singular_returns_none() {
        let sing = [1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0]; // zero row
        assert!(inv3(&sing).is_none());
    }

    #[test]
    fn inv3_roundtrip() {
        // Random-ish invertible 3x3; M @ inv(M) should be ~I.
        let m = [
            0.80f32, 0.10, 0.30, //
            0.20, 0.90, 0.10, //
            0.10, 0.20, 0.70,
        ];
        let inv = inv3(&m).unwrap();
        // Multiply m * inv in f32 and check closeness to identity.
        let mut prod = [0.0f32; 9];
        for r in 0..3 {
            for c in 0..3 {
                let mut s = 0.0f32;
                for k in 0..3 {
                    s += m[r * 3 + k] * inv[k * 3 + c];
                }
                prod[r * 3 + c] = s;
            }
        }
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0f32 } else { 0.0 };
                assert!((prod[r * 3 + c] - want).abs() < 1e-4, "prod[{r}][{c}] off");
            }
        }
    }

    #[test]
    fn inv4_identity() {
        let id = [
            1.0f32, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let inv = inv4(&id).unwrap();
        for (i, a) in inv.iter().enumerate() {
            let want = if i % 5 == 0 { 1.0f32 } else { 0.0 };
            assert!((a - want).abs() < 1e-6, "inv4 identity wrong at {i}");
        }
    }

    #[test]
    fn inv4_translation() {
        // T = translate(1,2,3); inv should translate(-1,-2,-3).
        let t = [
            1.0f32, 0.0, 0.0, 1.0, //
            0.0, 1.0, 0.0, 2.0, //
            0.0, 0.0, 1.0, 3.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let inv = inv4(&t).unwrap();
        assert!((inv[3] + 1.0).abs() < 1e-6);
        assert!((inv[7] + 2.0).abs() < 1e-6);
        assert!((inv[11] + 3.0).abs() < 1e-6);
    }

    #[test]
    fn inv4_singular_returns_none() {
        let sing = [
            1.0f32, 2.0, 3.0, 4.0, //
            2.0, 4.0, 6.0, 8.0, // 2x row 0
            1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        assert!(inv4(&sing).is_none());
    }

    #[test]
    fn percentile_linear_basic() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_linear(&v, 0.0) - 1.0).abs() < 1e-12);
        assert!((percentile_linear(&v, 100.0) - 5.0).abs() < 1e-12);
        assert!((percentile_linear(&v, 50.0) - 3.0).abs() < 1e-12);
        // 25th percentile: idx = 0.25*(5-1) = 1.0 -> exactly s[1] = 2.0
        // (matches numpy.percentile with linear interpolation).
        assert!((percentile_linear(&v, 25.0) - 2.0).abs() < 1e-12);
        // 75th percentile: idx = 0.75*4 = 3.0 -> s[3] = 4.0.
        assert!((percentile_linear(&v, 75.0) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn percentile_linear_unsorted_input_not_mutated() {
        let v = vec![5.0f32, 1.0, 4.0, 2.0, 3.0];
        let p50 = percentile_linear(&v, 50.0);
        assert!((p50 - 3.0).abs() < 1e-12, "median should be 3, got {p50}");
        // Input order unchanged.
        assert_eq!(v, vec![5.0, 1.0, 4.0, 2.0, 3.0]);
    }

    #[test]
    fn percentile_linear_empty_and_single() {
        let empty: Vec<f32> = vec![];
        assert_eq!(percentile_linear(&empty, 50.0), 0.0);
        let one = vec![7.5f32];
        assert!((percentile_linear(&one, 50.0) - 7.5).abs() < 1e-12);
    }

    #[test]
    fn rotmat2qvec_identity_is_unit_quat() {
        let id = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let q = rotmat2qvec(&id);
        // Identity rotation -> (1,0,0,0).
        assert!((q[0] - 1.0).abs() < 1e-4, "qw should be 1, got {}", q[0]);
        assert!(q[1].abs() < 1e-4);
        assert!(q[2].abs() < 1e-4);
        assert!(q[3].abs() < 1e-4);
    }

    #[test]
    fn rotmat2qvec_180_about_y() {
        // 180° rotation about y: [[-1,0,0],[0,1,0],[0,0,-1]] -> q = (0,0,1,0).
        let r = [
            -1.0f32, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, -1.0,
        ];
        let q = rotmat2qvec(&r);
        // qw ~ 0; qy ~ ±1. Normalize sign so qy is positive for the check.
        let qy = q[2];
        assert!(q[0].abs() < 1e-3, "qw should be ~0, got {}", q[0]);
        assert!(q[1].abs() < 1e-3);
        assert!((qy.abs() - 1.0).abs() < 1e-3, "|qy| should be 1, got {qy}");
        assert!(q[3].abs() < 1e-3);
    }

    #[test]
    fn rotmat2qvec_is_unit_norm() {
        let r = [
            0.0f32, -1.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let q = rotmat2qvec(&r);
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "quat not unit, norm = {norm}");
        assert!(q[0] >= 0.0, "qw should be non-negative");
    }

    #[test]
    fn back_project_identity_camera() {
        // Single frame, 2×2 image, K = identity, ext = identity (camera at origin
        // looking down +z). depth all = 2. Pixel (u,v) -> ray [u,v,1]; Xc =
        // ray*d = [2u, 2v, 2]; Xw = c2w(identity) @ [Xc;1] = (2u, 2v, 2).
        let h = 2;
        let w = 2;
        let depth = vec![2.0f32; h * w];
        let conf: Vec<f32> = vec![];
        let k = [[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]];
        let ext = [[
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]];
        let img: Vec<u8> = (0..(h * w * 3)).map(|i| (i % 256) as u8).collect();
        let images: Vec<&[u8]> = vec![img.as_slice()];
        let wp = back_project(&depth, &conf, &k, &ext, &images, h, w, 1, 0.0);
        assert_eq!(wp.len(), 4);
        // Pixel (0,0) -> (0, 0, 2).
        assert!((wp.xyz[0] - 0.0).abs() < 1e-5);
        assert!((wp.xyz[1] - 0.0).abs() < 1e-5);
        assert!((wp.xyz[2] - 2.0).abs() < 1e-5);
        // Pixel (1,1) -> ray [1,1,1] * d=2 = (2, 2, 2).
        let last = &wp.xyz[9..12];
        assert!((last[0] - 2.0).abs() < 1e-5);
        assert!((last[1] - 2.0).abs() < 1e-5);
        assert!((last[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn back_project_filters_invalid_depth() {
        let h = 1;
        let w = 3;
        // depth: nan, 0, 5 -> only the 5 survives.
        let depth = vec![f32::NAN, 0.0, 5.0];
        let conf: Vec<f32> = vec![];
        let k = [[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]];
        let ext = [[
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]];
        let img = vec![0u8; h * w * 3];
        let images = vec![img.as_slice()];
        let wp = back_project(&depth, &conf, &k, &ext, &images, h, w, 1, 0.0);
        assert_eq!(wp.len(), 1);
        assert_eq!(wp.u[0], 2);
        assert!((wp.xyz[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn back_project_filters_by_confidence() {
        let h = 1;
        let w = 3;
        let depth = vec![2.0f32, 2.0, 2.0];
        let conf = vec![0.5f32, 1.5, 0.9];
        let k = [[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]];
        let ext = [[
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]];
        let img = vec![0u8; h * w * 3];
        let images = vec![img.as_slice()];
        // conf_thr = 1.0 -> only pixels with conf >= 1.0 (index 1).
        let wp = back_project(&depth, &conf, &k, &ext, &images, h, w, 1, 1.0);
        assert_eq!(wp.len(), 1);
        assert_eq!(wp.u[0], 1);
    }
}
