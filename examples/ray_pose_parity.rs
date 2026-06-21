//! Ray-pose parity: run the Rust `depth_pose_rays` path on a `--with-aux` GGUF
//! and (optionally) compare the resulting camera pose against a reference dump.
//!
//! This exercises the full aux ray head (`src/dpt_head.rs` `forward_rays`) +
//! RANSAC homography + QL solver (`src/ray_pose.rs`) end to end.
//!
//! # Parity model
//!
//! The reference RANSAC samples candidate groups via `torch.randperm` and
//! randomly subsamples the consensus set, so even the *reference* pose is
//! non-deterministic across runs. The Rust production path uses its own seeded
//! sampling (see [`depth_anything::ray_pose`]); the two therefore consume
//! different point subsets and agree only up to RANSAC consensus variation +
//! f32-vs-f64 numeric differences.
//!
//! Accordingly the comparison here is **loose** (matching `e2e_ray_pose.py`):
//!   - rotation geodesic angle < 1.0°
//!   - focal relative error < 2%
//!   - principal-point relative error < 2%
//!
//! For bit-exact parity, feed the C++ solver's indices through the gated path
//! (not exposed at the example level; see `Engine::depth_pose_rays_image_with`).
//!
//! # Usage
//!
//! Build an aux GGUF (one-time):
//! ```text
//! python scripts/convert_da3_to_gguf.py \
//!     --model models/DA3-BASE \
//!     --output models/depth-anything-base-aux-f32.gguf \
//!     --with-aux
//! ```
//!
//! Run the Rust ray-pose path only:
//! ```text
//! cargo run --release --example ray_pose_parity -- \
//!     --model models/depth-anything-base-aux-f32.gguf \
//!     --input assets/samples/street.jpg
//! ```
//!
//! Compare against a C++ reference pose JSON (`{"extrinsics":[12], "intrinsics":[9]}`):
//! ```text
//! cargo run --release --example ray_pose_parity -- \
//!     --model models/depth-anything-base-aux-f32.gguf \
//!     --input assets/samples/street.jpg \
//!     --ref-pose dumps/cpp_ray_pose.json
//! ```

use std::process::ExitCode;

use depth_anything::Engine;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut input_path = String::new();
    let mut ref_pose_path: Option<String> = None;
    let mut max_angle_deg: f64 = 1.0;
    let mut max_rel_err: f64 = 0.02;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_path = args[i + 1].clone();
                i += 2;
            }
            "--input" => {
                input_path = args[i + 1].clone();
                i += 2;
            }
            "--ref-pose" => {
                ref_pose_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--max-angle-deg" => {
                max_angle_deg = args[i + 1].parse().unwrap_or(1.0);
                i += 2;
            }
            "--max-rel-err" => {
                max_rel_err = args[i + 1].parse().unwrap_or(0.02);
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                print_usage();
                return ExitCode::FAILURE;
            }
        }
    }
    if model_path.is_empty() || input_path.is_empty() {
        print_usage();
        return ExitCode::FAILURE;
    }

    let engine = match Engine::load(&model_path, None) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load model: {e}");
            return ExitCode::FAILURE;
        }
    };
    if !engine.has_aux_head() {
        eprintln!(
            "error: this GGUF has no aux ray head. Rebuild with:\n  \
             python scripts/convert_da3_to_gguf.py --model <m> --output <out> --with-aux"
        );
        return ExitCode::FAILURE;
    }

    let (depth, pose) = match engine.depth_pose_rays_path(&input_path) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("depth_pose_rays: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!(
        "depth: {}x{}, pose inliers={} focal=({:.4},{:.4}) pp=({:.4},{:.4})",
        depth.h,
        depth.w,
        pose.diag.n_inlier_best,
        pose.diag.focal[0],
        pose.diag.focal[1],
        pose.diag.pp[0],
        pose.diag.pp[1],
    );
    println!("  extrinsics (3x4 row-major): {:?}", pose.ext);
    println!("  intrinsics (3x3 row-major): {:?}", pose.intr);

    // Optional comparison against a C++ reference pose JSON.
    if let Some(ref_path) = ref_pose_path {
        match compare_to_ref(&ref_path, &pose, max_angle_deg, max_rel_err) {
            Ok(()) => {
                println!("PASS: pose within tolerance vs {ref_path}");
                return ExitCode::SUCCESS;
            }
            Err(msg) => {
                eprintln!("FAIL: {msg}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}

fn print_usage() {
    eprintln!(
        "usage: ray_pose_parity --model AUX.gguf --input IMG.jpg \
         [--ref-pose pose.json] [--max-angle-deg 1.0] [--max-rel-err 0.02]"
    );
}

/// Load `{extrinsics:[12], intrinsics:[9]}` and compare against `pose`.
///
/// Tolerances (matching `scripts/e2e_ray_pose.py`):
/// - rotation geodesic angle < `max_angle_deg`
/// - focal relative error < `max_rel_err`
/// - principal-point relative error < `max_rel_err`
fn compare_to_ref(
    ref_path: &str,
    pose: &depth_anything::RayPoseOutput,
    max_angle_deg: f64,
    max_rel_err: f64,
) -> Result<(), String> {
    let txt = std::fs::read_to_string(ref_path).map_err(|e| format!("read {ref_path}: {e}"))?;
    // Minimal hand-rolled JSON scan (avoid pulling a JSON dep): look for the
    // two arrays by key. Expects the format written by `pose_export.cpp` /
    // `scripts/e2e_ray_pose.py`: {"extrinsics":[...],"intrinsics":[...]}.
    let ref_ext = parse_array(&txt, "extrinsics").ok_or("missing extrinsics[]")?;
    let ref_int = parse_array(&txt, "intrinsics").ok_or("missing intrinsics[]")?;
    if ref_ext.len() != 12 || ref_int.len() != 9 {
        return Err(format!(
            "bad ref shape: ext={} int={}",
            ref_ext.len(),
            ref_int.len()
        ));
    }

    // Rotation geodesic angle between the two 3x3 rotation blocks (cols 0-2 of
    // the 3x4 extrinsics). R_ref^T R_rust = U; angle = arccos((tr(U)-1)/2).
    let pose_ext_f64: Vec<f64> = pose.ext.iter().map(|&v| v as f64).collect();
    let rot_angle = rotation_geodesic_deg(&ref_ext, &pose_ext_f64);
    if rot_angle > max_angle_deg {
        return Err(format!(
            "rotation geodesic {rot_angle:.3}° > {max_angle_deg}°"
        ));
    }

    // Focal: K[0,0], K[1,1]; principal point: K[0,2], K[1,2].
    let rel = |a: f64, b: f64| (a - b).abs() / b.max(1e-9);
    let fx_err = rel(ref_int[0] as f64, pose.intr[0] as f64);
    let fy_err = rel(ref_int[4] as f64, pose.intr[4] as f64);
    let cx_err = rel(ref_int[2] as f64, pose.intr[2] as f64);
    let cy_err = rel(ref_int[5] as f64, pose.intr[5] as f64);
    println!(
        "  focal rel-err fx={fx_err:.4} fy={fy_err:.4}; pp rel-err cx={cx_err:.4} cy={cy_err:.4}; rot {rot_angle:.4}°"
    );
    if fx_err > max_rel_err || fy_err > max_rel_err {
        return Err(format!("focal rel-err > {max_rel_err}"));
    }
    if cx_err > max_rel_err || cy_err > max_rel_err {
        return Err(format!("principal-point rel-err > {max_rel_err}"));
    }
    Ok(())
}

/// Extract the first numeric array following `"key"`.
fn parse_array(txt: &str, key: &str) -> Option<Vec<f64>> {
    let pat = format!("\"{key}\"");
    let idx = txt.find(&pat)?;
    let bracket_open = txt[idx..].find('[')?;
    let bracket_close = txt[idx + bracket_open..].find(']')?;
    let inner = &txt[idx + bracket_open + 1..idx + bracket_open + bracket_close];
    inner
        .split([',', ' ', '\n', '\t', '\r'])
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<f64>().ok())
        .collect()
}

/// Geodesic angle (degrees) between the rotation blocks of two 3x4 extrinsics.
fn rotation_geodesic_deg(ext_a: &[f64], ext_b: &[f64]) -> f64 {
    let rot = |ext: &[f64]| -> [[f64; 3]; 3] {
        [
            [ext[0], ext[1], ext[2]],
            [ext[4], ext[5], ext[6]],
            [ext[8], ext[9], ext[10]],
        ]
    };
    let a = rot(ext_a);
    let b = rot(ext_b);
    // u = a^T · b  (3x3)
    let mut u = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[k][i] * b[k][j];
            }
            u[i][j] = s;
        }
    }
    let tr = u[0][0] + u[1][1] + u[2][2];
    let cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0);
    cos.acos().to_degrees()
}
