//! The camera pose head, matching `src/cam_pose.cpp`.
//!
//! Input: the camera token `[1, 1, D]`. Two ReLU-Linear backbone blocks feed
//! three heads (translation, quaternion, FoV), concatenated into a 9-vector
//! `pose_enc = [Tx,Ty,Tz, qi,qj,qk,qr, fov_h, fov_w]`. The quaternion uses
//! the XYZW (scalar-last) convention; FoV is post-ReLU.

use crate::config::Config;
use crate::gguf::GgufFile;
use crate::weights::load_linear;
use crate::Result;
use candle::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub struct CamPose {
    bb0: Linear,
    bb2: Linear,
    fc_t: Linear,
    fc_q: Linear,
    fc_fov: Linear,
    hidden: usize,
}

impl CamPose {
    pub fn load(vb: &VarBuilder, cfg: &Config, file: &GgufFile, device: &Device) -> Result<Self> {
        let _ = device;
        let d = if cfg.cat_token {
            2 * cfg.embed_dim as usize
        } else {
            cfg.embed_dim as usize
        };
        let cam = vb.pp("cam");
        // Hidden size: read the bb0 weight's output dim from the GGUF tensor
        // info directly (avoids a shape-mismatch against a placeholder load).
        // cam.bb0.weight has candle shape [hidden, d].
        let bb0_info = file
            .tensor_info("cam.bb0.weight")
            .ok_or_else(|| crate::Error::Model("missing cam.bb0.weight for pose head".into()))?;
        let bb0_shape = bb0_info.candle_shape();
        if bb0_shape.len() != 2 || bb0_shape[1] != d {
            return Err(crate::Error::Model(format!(
                "cam.bb0.weight: unexpected shape {:?}, expected [hidden, {d}]",
                bb0_shape
            )));
        }
        let hidden = bb0_shape[0];
        let bb0 = load_linear(&cam, "bb0", hidden, d)?;
        let bb2 = load_linear(&cam, "bb2", hidden, hidden)?;
        let fc_t = load_linear(&cam, "fc_t", 3, hidden)?;
        let fc_q = load_linear(&cam, "fc_q", 4, hidden)?;
        let fc_fov = load_linear(&cam, "fc_fov", 2, hidden)?;
        Ok(Self {
            bb0,
            bb2,
            fc_t,
            fc_q,
            fc_fov,
            hidden,
        })
    }

    /// Run the pose MLP. `cam_token`: `[1, 1, D]`. Returns the raw 9-vector
    /// `pose_enc` as `[Tx,Ty,Tz, qi,qj,qk,qr, fov_h, fov_w]`.
    pub fn forward_enc(&self, cam_token: &Tensor) -> Result<Tensor> {
        // [1,1,D] -> [1,D] for the linear layers.
        let x = cam_token.flatten_to(1)?;
        let h = self.bb0.forward(&x)?.relu()?;
        let h = self.bb2.forward(&h)?.relu()?;
        let t = self.fc_t.forward(&h)?; // [1,3], no relu
        let q = self.fc_q.forward(&h)?; // [1,4], no relu
        let fov = self.fc_fov.forward(&h)?.relu()?; // [1,2], relu
        let tq = Tensor::cat(&[&t, &q], 1)?; // [1,7]
        let out = Tensor::cat(&[&tq, &fov], 1)?; // [1,9]
        Ok(out)
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }
}

/// Decode the 9-vector `pose_enc` into extrinsics `[12]` (3×4 row-major) and
/// intrinsics `[9]` (3×3 row-major), given the processed image size `(W, H)`.
///
/// Mirrors `cam_pose.cpp`'s host post-processing exactly:
/// - quaternion (XYZW) → rotation `R` via `s = 2 / ||q||²`,
/// - extrinsics = `[R^T | -R^T · T]` (affine inverse of c2w),
/// - intrinsics: focal from FoV, principal point at the image center.
pub fn decode_pose(pose_enc: &[f32], w: usize, h: usize) -> Result<([f32; 12], [f32; 9])> {
    if pose_enc.len() < 9 {
        return Err(crate::Error::Model(format!(
            "pose_enc needs 9 values, got {}",
            pose_enc.len()
        )));
    }
    let (tx, ty, tz) = (pose_enc[0], pose_enc[1], pose_enc[2]);
    let (qi, qj, qk, qr) = (pose_enc[3], pose_enc[4], pose_enc[5], pose_enc[6]);
    let (fov_h, fov_w) = (pose_enc[7], pose_enc[8]);

    let norm_sq = qi * qi + qj * qj + qk * qk + qr * qr;
    let s = 2.0 / norm_sq;

    // Rotation matrix (row-major 3x3) from quaternion (XYZW).
    let r = [
        [
            1.0 - s * (qj * qj + qk * qk),
            s * (qi * qj - qk * qr),
            s * (qi * qk + qj * qr),
        ],
        [
            s * (qi * qj + qk * qr),
            1.0 - s * (qi * qi + qk * qk),
            s * (qj * qk - qi * qr),
        ],
        [
            s * (qi * qk - qj * qr),
            s * (qj * qk + qi * qr),
            1.0 - s * (qi * qi + qj * qj),
        ],
    ];

    // R^T (rotation is orthonormal → inverse = transpose).
    let rt = [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ];
    // -R^T · T
    let neg_rt_t = [
        -(rt[0][0] * tx + rt[0][1] * ty + rt[0][2] * tz),
        -(rt[1][0] * tx + rt[1][1] * ty + rt[1][2] * tz),
        -(rt[2][0] * tx + rt[2][1] * ty + rt[2][2] * tz),
    ];

    // Extrinsics 3x4 row-major: [R^T | -R^T·T].
    let ext = [
        rt[0][0],
        rt[0][1],
        rt[0][2],
        neg_rt_t[0],
        rt[1][0],
        rt[1][1],
        rt[1][2],
        neg_rt_t[1],
        rt[2][0],
        rt[2][1],
        rt[2][2],
        neg_rt_t[2],
    ];

    // Intrinsics from FoV. f = (size/2) / max(tan(fov/2), 1e-6).
    let fy = (h as f32 / 2.0) / (fov_h / 2.0).tan().max(1e-6);
    let fx = (w as f32 / 2.0) / (fov_w / 2.0).tan().max(1e-6);
    let intr = [
        fx,
        0.0,
        w as f32 / 2.0,
        0.0,
        fy,
        h as f32 / 2.0,
        0.0,
        0.0,
        1.0,
    ];

    Ok((ext, intr))
}
