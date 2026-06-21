//! Small dense host linear algebra for the optional ray→pose solver.
//!
//! Faithful Rust port of `src/linalg.cpp`. All matrices are row-major `f64`
//! (the reference ray-pose path is computed in double precision in
//! `ray_utils.py`, and the C++ engine keeps it in `double` end-to-end; we do
//! the same so the RANSAC + eigen results match bit-for-bit at `f32` tolerance).
//!
//! These helpers mirror the exact operations `ray_utils.py` performs
//! (`torch.linalg.qr`, `torch.linalg.svd`, weighted homography) on tiny dense
//! matrices, returning `f32` only at the engine boundary (see [`crate::ray_pose`]).
//!
//! `cam_pose` does **not** use this module — its post-processing is pure
//! quaternion + matrix arithmetic living in [`crate::cam_pose`].

#![allow(clippy::needless_range_loop, clippy::identity_op, missing_docs)]

#[inline]
fn dsign(x: f64) -> f64 {
    (x > 0.0) as i64 as f64 - (x < 0.0) as i64 as f64
}

/// 3×3 determinant (row-major).
pub fn det3(a: &[f64; 9]) -> f64 {
    a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6])
}

/// 3×3 matrix product `c = a · b` (all row-major).
pub fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k * 3 + j];
            }
            c[i * 3 + j] = s;
        }
    }
    c
}

/// 3×3 transpose.
pub fn mat3_transpose(a: &[f64; 9]) -> [f64; 9] {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

/// 3×3 inverse via cofactors. Returns `None` if (near-)singular.
pub fn mat3_inverse(a: &[f64; 9]) -> Option<[f64; 9]> {
    let d = det3(a);
    if d.abs() < 1e-300 {
        return None;
    }
    let inv = 1.0 / d;
    Some([
        (a[4] * a[8] - a[5] * a[7]) * inv,
        (a[2] * a[7] - a[1] * a[8]) * inv,
        (a[1] * a[5] - a[2] * a[4]) * inv,
        (a[5] * a[6] - a[3] * a[8]) * inv,
        (a[0] * a[8] - a[2] * a[6]) * inv,
        (a[2] * a[3] - a[0] * a[5]) * inv,
        (a[3] * a[7] - a[4] * a[6]) * inv,
        (a[1] * a[6] - a[0] * a[7]) * inv,
        (a[0] * a[4] - a[1] * a[3]) * inv,
    ])
}

/// Reduced Householder QR of a square `n × n` row-major matrix `a`:
/// `a = q · r` with `q` orthonormal and `r` upper-triangular.
///
/// Sign convention matches LAPACK / `torch.linalg.qr`
/// (`r[k,k] = -sign(x_k)·‖x‖`). `n-1` reflectors are applied (the final 1×1
/// reflector is left as identity with `tau = 0`, matching torch exactly).
pub fn householder_qr(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut r = a.to_vec();
    let mut q = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            q[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    let mut v = vec![0.0f64; n];
    // n-1 reflectors suffice for a square matrix.
    for k in 0..(n.saturating_sub(1)) {
        let mut normx = 0.0;
        for i in k..n {
            normx += r[i * n + k] * r[i * n + k];
        }
        normx = normx.sqrt();
        if normx < 1e-300 {
            continue;
        }
        let alpha = if r[k * n + k] >= 0.0 { -normx } else { normx };
        for i in 0..n {
            v[i] = 0.0;
        }
        for i in k..n {
            v[i] = r[i * n + k];
        }
        v[k] -= alpha;
        let mut vnorm2 = 0.0;
        for i in k..n {
            vnorm2 += v[i] * v[i];
        }
        if vnorm2 < 1e-300 {
            continue;
        }
        // r <- (I - 2 v v^T / vnorm2) r  : per column j
        for j in 0..n {
            let mut dot = 0.0;
            for i in k..n {
                dot += v[i] * r[i * n + j];
            }
            let f = 2.0 * dot / vnorm2;
            for i in k..n {
                r[i * n + j] -= f * v[i];
            }
        }
        // q <- q (I - 2 v v^T / vnorm2)  : per row r
        for row in 0..n {
            let mut dot = 0.0;
            for i in k..n {
                dot += q[row * n + i] * v[i];
            }
            let f = 2.0 * dot / vnorm2;
            for i in k..n {
                q[row * n + i] -= f * v[i];
            }
        }
    }
    // Clean tiny sub-diagonal noise.
    for i in 0..n {
        for j in 0..i {
            r[i * n + j] = 0.0;
        }
    }
    (q, r)
}

/// 3×3 convenience wrapper around [`householder_qr`].
pub fn qr3x3(a: &[f64; 9]) -> ([f64; 9], [f64; 9]) {
    let (q, r) = householder_qr(a, 3);
    let qv: [f64; 9] = q.try_into().unwrap();
    let rv: [f64; 9] = r.try_into().unwrap();
    (qv, rv)
}

/// QL decomposition exactly as `ray_utils.ql_decomposition`: returns `q`
/// (orthonormal) and `l` (lower-triangular) with `a = q · l`. Sign-normalized
/// by `sign(diag(l))` so the result is invariant to the QR sign ambiguity.
pub fn ql_decomposition(a: &[f64; 9]) -> ([f64; 9], [f64; 9]) {
    // P = anti-identity (reverses index 0 <-> 2).  a_tilde = a @ P  (reverse cols).
    let mut at = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            at[i * 3 + j] = a[i * 3 + (2 - j)];
        }
    }
    let (qt, rt) = qr3x3(&at);
    // q = qt @ P  (reverse cols of qt)
    let mut q = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            q[i * 3 + j] = qt[i * 3 + (2 - j)];
        }
    }
    // l = P @ rt @ P  (reverse rows and cols of rt)
    let mut l = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            l[i * 3 + j] = rt[(2 - i) * 3 + (2 - j)];
        }
    }
    // Sign-normalize by sign(diag(l)).
    let s0 = dsign(l[0]);
    let s1 = dsign(l[4]);
    let s2 = dsign(l[8]);
    let sc = [s0, s1, s2];
    for i in 0..3 {
        q[i * 3 + 0] *= s0;
        q[i * 3 + 1] *= s1;
        q[i * 3 + 2] *= s2;
    }
    for r in 0..3 {
        for c in 0..3 {
            l[r * 3 + c] *= sc[r];
        }
    }
    (q, l)
}

/// Cyclic Jacobi eigendecomposition of a symmetric `n × n` row-major matrix `m`.
///
/// Returns `(eigvals, eigvecs)` where `eigvecs` is `n × n` row-major with
/// eigenvector `j` in **column** `j` (i.e. `eigvecs[i*n + j]`). Eigenvalues are
/// not sorted. `m` is not modified.
pub fn jacobi_eigen_sym(m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = m.to_vec();
    let mut eigvecs = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            eigvecs[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    const MAX_SWEEPS: usize = 100;
    for _ in 0..MAX_SWEEPS {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let mut t = dsign(theta) / (theta.abs() + (theta * theta + 1.0).sqrt());
                if theta == 0.0 {
                    t = 1.0;
                }
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate rows/cols (p, q) of a.
                for i in 0..n {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    a[i * n + p] = c * aip - s * aiq;
                    a[i * n + q] = s * aip + c * aiq;
                }
                for i in 0..n {
                    let api = a[p * n + i];
                    let aqi = a[q * n + i];
                    a[p * n + i] = c * api - s * aqi;
                    a[q * n + i] = s * api + c * aqi;
                }
                // Accumulate eigenvectors (columns).
                for i in 0..n {
                    let vip = eigvecs[i * n + p];
                    let viq = eigvecs[i * n + q];
                    eigvecs[i * n + p] = c * vip - s * viq;
                    eigvecs[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }
    let eigvals = (0..n).map(|i| a[i * n + i]).collect();
    (eigvals, eigvecs)
}

/// Eigenvector of the **smallest** eigenvalue of a symmetric `n × n` matrix.
pub fn smallest_eigvec(m: &[f64], n: usize) -> Vec<f64> {
    let (eval, evec) = jacobi_eigen_sym(m, n);
    let mut imin = 0;
    for i in 1..n {
        if eval[i] < eval[imin] {
            imin = i;
        }
    }
    (0..n).map(|i| evec[i * n + imin]).collect()
}

/// Weighted least-squares homography from `m` point correspondences, matching
/// `find_homography_least_squares_weighted_torch`.
///
/// Builds the `2m × 9` normal-equations system `AtA`, solves for the smallest
/// right-singular vector via Jacobi eigen of `a^T a`, reshapes to `3 × 3` and
/// normalizes by `h[8]`.
///
/// - `src`, `dst`: length `2·m` (`x, y` interleaved).
/// - `w`: length `m` (the confidence; **not** `sqrt`'d by the caller — this
///   function applies `sw = sqrt(max(w, 0))` internally, matching the torch ref).
///
/// Returns `None` if degenerate (`h[8] ≈ 0`).
pub fn homography_weighted(src: &[f64], dst: &[f64], w: &[f64], m: usize) -> Option<[f64; 9]> {
    let mut ata = [0.0f64; 81];
    for k in 0..m {
        let x = src[2 * k];
        let y = src[2 * k + 1];
        let u = dst[2 * k];
        let v = dst[2 * k + 1];
        let sw = if w[k] < 0.0 { 0.0 } else { w[k].sqrt() };
        // Row 1: [-x*sw, -y*sw, -sw, 0,0,0, x*u*sw, y*u*sw, u*sw]
        let r1 = [
            -x * sw,
            -y * sw,
            -sw,
            0.0,
            0.0,
            0.0,
            x * u * sw,
            y * u * sw,
            u * sw,
        ];
        // Row 2: [0,0,0, -x*sw, -y*sw, -sw, x*v*sw, y*v*sw, v*sw]
        let r2 = [
            0.0,
            0.0,
            0.0,
            -x * sw,
            -y * sw,
            -sw,
            x * v * sw,
            y * v * sw,
            v * sw,
        ];
        for i in 0..9 {
            for j in 0..9 {
                ata[i * 9 + j] += r1[i] * r1[j] + r2[i] * r2[j];
            }
        }
    }
    let h = smallest_eigvec(&ata, 9);
    if h[8].abs() < 1e-300 {
        return None;
    }
    let inv = 1.0 / h[8];
    let mut out = [0.0f64; 9];
    for i in 0..9 {
        out[i] = h[i] * inv;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "idx {i}: {x} != {y} (tol {tol})");
        }
    }

    #[test]
    fn identity_det() {
        let i = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!((det3(&i) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn det3_known() {
        // [[2,0,1],[3,1,1],[0,2,2]] -> det = 2*(1*2-1*2) - 0 + 1*(3*2-1*0) = 0 + 6 = 6
        let a = [2.0, 0.0, 1.0, 3.0, 1.0, 1.0, 0.0, 2.0, 2.0];
        assert!((det3(&a) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn transpose_twice_is_identity() {
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tt = mat3_transpose(&mat3_transpose(&a));
        approx_eq(&tt, &a, 1e-12);
    }

    #[test]
    fn mat3_mul_identity() {
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let i = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        approx_eq(&mat3_mul(&a, &i), &a, 1e-12);
        approx_eq(&mat3_mul(&i, &a), &a, 1e-12);
    }

    #[test]
    fn inverse_roundtrip() {
        let a = [4.0, 7.0, 2.0, 3.0, 6.0, 1.0, 2.0, 5.0, 3.0];
        let ai = mat3_inverse(&a).expect("non-singular");
        let prod = mat3_mul(&a, &ai);
        let i = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        approx_eq(&prod, &i, 1e-9);
    }

    #[test]
    fn inverse_singular_is_none() {
        // rank-2: row1 = row2
        let a = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0];
        assert!(mat3_inverse(&a).is_none());
    }

    #[test]
    fn householder_qr_reconstructs() {
        // Well-conditioned, non-symmetric.
        let a = vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0];
        let (q, r) = householder_qr(&a, 3);
        // q must be orthonormal: q q^T = I
        let mut qt = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += q[i * 3 + k] * q[j * 3 + k];
                }
                qt[i * 3 + j] = s;
            }
        }
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        approx_eq(&qt, &eye, 1e-9);
        // r must be upper-triangular
        for i in 0..3 {
            for j in 0..i {
                assert!(r[i * 3 + j].abs() < 1e-12, "r not upper-triangular");
            }
        }
        // a ≈ q · r
        let qr = mat3_mul(&q.try_into().unwrap(), &r.try_into().unwrap());
        approx_eq(&qr, &a, 1e-9);
    }

    #[test]
    fn householder_qr_matches_reference_signs() {
        // The canonical Wikipedia QR example matrix. The first two reflectors
        // (n-1 for a 3x3) fully determine R, including the (2,2) entry's sign.
        // We assert the values produced by the n-1-reflector convention used by
        // the C++ `src/linalg.cpp` (whose comment notes it matches LAPACK/torch:
        // the final 1x1 reflector is identity, so the last diag "keeps its sign"
        // rather than being forced). Wikipedia's textbook R sign-flips the last
        // row to make R[2][2]=+35; the raw Householder path yields -35, which is
        // what this port (and the C++ engine) produces.
        let a = vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0];
        let (_q, r) = householder_qr(&a, 3);
        let expected = [-14.0, -21.0, 14.0, 0.0, -175.0, 70.0, 0.0, 0.0, -35.0];
        approx_eq(&r, &expected, 1e-6);
    }

    #[test]
    fn ql_decomposition_reconstructs_and_sign_normalized() {
        // Use a matrix that exercises the sign-normalization.
        let a = [2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let (q, l) = ql_decomposition(&a);
        // l is lower-triangular
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!(l[i * 3 + j].abs() < 1e-12, "l not lower-triangular");
            }
        }
        // diag(l) >= 0 (sign-normalized)
        for i in 0..3 {
            assert!(l[i * 3 + i] >= -1e-12, "diag(l) negative");
        }
        // q is orthonormal
        let mut qtq = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += q[k * 3 + i] * q[k * 3 + j];
                }
                qtq[i * 3 + j] = s;
            }
        }
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        approx_eq(&qtq, &eye, 1e-9);
        // a ≈ q · l
        let ql = mat3_mul(&q, &l);
        approx_eq(&ql, &a, 1e-9);
    }

    #[test]
    fn jacobi_eigen_of_diagonal() {
        // Already diagonal: eigenvalues are the diagonal entries.
        let m = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let (evals, evecs) = jacobi_eigen_sym(&m, 3);
        // Identity eigenvectors
        let eye = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        approx_eq(&evecs, &eye, 1e-12);
        // eigenvalues match diagonal (any order)
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        approx_eq(&sorted, &[1.0, 2.0, 3.0], 1e-12);
    }

    #[test]
    fn jacobi_eigen_symmetric_known() {
        // [[2,1,0],[1,2,1],[0,1,2]] eigenvalues = 2, 2±sqrt(2)
        let m = vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
        let (evals, _evecs) = jacobi_eigen_sym(&m, 3);
        let mut sorted = evals;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // 2-sqrt2, 2, 2+sqrt2
        approx_eq(
            &sorted,
            &[2.0 - 2.0_f64.sqrt(), 2.0, 2.0 + 2.0_f64.sqrt()],
            1e-9,
        );
    }

    #[test]
    fn jacobi_eigen_reconstruction() {
        // M ≈ V · diag(e) · V^T
        let m = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0];
        let (evals, evecs) = jacobi_eigen_sym(&m, 3);
        // reconstruct
        let mut rec = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += evecs[i * 3 + k] * evals[k] * evecs[j * 3 + k];
                }
                rec[i * 3 + j] = s;
            }
        }
        approx_eq(&rec, &m, 1e-9);
    }

    #[test]
    fn smallest_eigvec_picks_smallest() {
        // [[3,0,0],[0,1,0],[0,0,2]] -> smallest eigenvalue 1, eigvec [0,1,0]
        let m = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let v = smallest_eigvec(&m, 3);
        // sign is ambiguous; compare |v|
        let expected: [f64; 3] = [0.0, 1.0, 0.0];
        for i in 0..3 {
            assert!((v[i].abs() - expected[i].abs()).abs() < 1e-9);
        }
    }

    #[test]
    fn homography_identity_transform() {
        // src == dst for many points -> H should be identity (up to scale).
        let m = 20;
        let mut src = vec![0.0; 2 * m];
        let mut dst = vec![0.0; 2 * m];
        let w = vec![1.0; m];
        for k in 0..m {
            let x = 0.1 * k as f64 - 1.0;
            let y = 0.2 * (k as f64 % 5.0) - 0.4;
            src[2 * k] = x;
            src[2 * k + 1] = y;
            dst[2 * k] = x;
            dst[2 * k + 1] = y;
        }
        let h = homography_weighted(&src, &dst, &w, m).expect("non-degenerate");
        // h = identity (h[8]=1)
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        approx_eq(&h, &eye, 1e-6);
    }

    #[test]
    fn homography_translation() {
        // H = [[1,0,tx],[0,1,ty],[0,0,1]] maps (x,y) -> (x+tx, y+ty).
        let m = 30;
        let tx = 0.3;
        let ty = -0.2;
        let mut src = vec![0.0; 2 * m];
        let mut dst = vec![0.0; 2 * m];
        let w = vec![0.5; m];
        for k in 0..m {
            let x = -1.0 + 0.07 * k as f64;
            let y = -0.5 + 0.05 * (k as f64 % 11.0);
            src[2 * k] = x;
            src[2 * k + 1] = y;
            dst[2 * k] = x + tx;
            dst[2 * k + 1] = y + ty;
        }
        let h = homography_weighted(&src, &dst, &w, m).expect("non-degenerate");
        let expected = [1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0];
        approx_eq(&h, &expected, 1e-6);
    }
}
