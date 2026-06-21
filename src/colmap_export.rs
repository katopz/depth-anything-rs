//! COLMAP sparse-model exporter, matching `src/colmap_export.cpp::write_colmap`.
//!
//! Mirrors the reference exporter `utils/export/colmap.py` (which drives
//! pycolmap's `reconstruction.write`). Emits a sparse model — `cameras`,
//! `images`, `points3D` — to a directory, in either the little-endian COLMAP
//! binary layout (`cameras.bin` / `images.bin` / `points3D.bin`) or the
//! matching `.txt` variants.
//!
//! Behaviour (faithful to `colmap.py`):
//! - `conf_thr = percentile_linear(conf over all frames, 40)`.
//! - [`back_project`] → world points + colors; point3D ids are `1..=num_points`
//!   in back-projection order.
//! - Per frame (camera_id = image_id = frame+1): PINHOLE model. Intrinsics
//!   rescaled to the original image size: `fx,cx *= orig_w/W`; `fy,cy *=
//!   orig_h/H`. `params = [fx,fy,cx,cy]`; `width = orig_w`; `height = orig_h`.
//! - Extrinsic → `qvec = rotmat2qvec(R = ext[:3,:3])` (COLMAP order
//!   `qw,qx,qy,qz`), `tvec = ext[:3,3]`.
//! - `point2D` `x,y = int-truncate(u*orig_w/W, v*orig_h/H)`, matching the
//!   reference's int32 in-place scaling; linked to the point3D id; the point3D
//!   track records `(image_id, point2D_idx)`.

#![allow(clippy::needless_range_loop)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::reconstruct::{back_project, percentile_linear, rotmat2qvec, WorldPoints};
use crate::Result;

/// Per-frame derived camera/pose data (mirrors the C++ `FrameData` local struct).
#[derive(Debug, Clone)]
struct FrameData {
    orig_w: i64,
    orig_h: i64,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    qvec: [f32; 4], // qw,qx,qy,qz
    tx: f64,
    ty: f64,
    tz: f64,
}

/// A 2D observation in an image, linked to its point3D id.
struct Pt2D {
    x: f64,
    y: f64,
    point3d_id: i64,
}

/// Write a COLMAP sparse model (`cameras`, `images`, `points3D`) to directory
/// `dir` (created if missing). Inputs mirror [`back_project`]:
/// - `depth`, `conf`: `N*H*W` row-major (frame, row, col).
/// - `k`: per-frame 3×3 intrinsics, row-major (at processed size `H×W`).
/// - `ext`: per-frame 4×4 world-to-camera extrinsics, row-major.
/// - `images_u8`: per-frame slice over `H*W*3` RGB uint8 (processed size).
/// - `image_names`: per-frame output image name (basename).
/// - `orig_wh`: per-frame original `(width, height)` for intrinsic rescale.
///
/// When `binary` is true (default), emits `cameras.bin`/`images.bin`/
/// `points3D.bin`; otherwise the `.txt` variants. Returns `Ok(())` on success,
/// or an I/O error.
#[allow(clippy::too_many_arguments)]
pub fn write_colmap<P: AsRef<Path>>(
    dir: P,
    depth: &[f32],
    conf: &[f32],
    k: &[[f32; 9]],
    ext: &[[f32; 16]],
    images_u8: &[&[u8]],
    image_names: &[String],
    orig_wh: &[(i64, i64)],
    h: usize,
    w: usize,
    n: usize,
    binary: bool,
) -> Result<()> {
    std::fs::create_dir_all(&dir)?;

    // 1. Confidence threshold (numpy percentile, linear interp) over all conf.
    let conf_thr = if conf.is_empty() {
        0.0
    } else {
        percentile_linear(conf, 40.0) as f32
    };

    // 2. Back-project -> world points, colors, per-point (frame,u,v).
    let wp: WorldPoints = back_project(depth, conf, k, ext, images_u8, h, w, n, conf_thr);
    let num_points = wp.frame.len();

    // 3. Per-frame derived data (rescaled intrinsics + pose).
    let mut fd: Vec<FrameData> = Vec::with_capacity(n);
    for i in 0..n {
        let (ow, oh) = orig_wh[i];
        let sw = ow as f64 / w as f64;
        let sh = oh as f64 / h as f64;
        let ki = &k[i];
        // R = ext[i][:3,:3] extracted row-major.
        let e = &ext[i];
        let r = [
            e[0], e[1], e[2], //
            e[4], e[5], e[6], //
            e[8], e[9], e[10],
        ];
        fd.push(FrameData {
            orig_w: ow,
            orig_h: oh,
            fx: ki[0] as f64 * sw, // fx, row 0
            cx: ki[2] as f64 * sw, // cx, row 0
            fy: ki[4] as f64 * sh, // fy, row 1
            cy: ki[5] as f64 * sh, // cy, row 1
            qvec: rotmat2qvec(&r),
            tx: e[3] as f64,
            ty: e[7] as f64,
            tz: e[11] as f64,
        });
    }

    // 4. Per-image point2D lists + per-point track (each point observed once).
    //    Back-projection orders points frame-outer, so each frame's points are
    //    contiguous; point2D_idx is the running index within the image.
    //    point3D id = global index + 1.
    let mut img_pts: Vec<Vec<Pt2D>> = (0..n).map(|_| Vec::new()).collect();
    let mut track_image_id: Vec<i32> = vec![0; num_points];
    let mut track_pt2d_idx: Vec<i32> = vec![0; num_points];
    for p in 0..num_points {
        let fr = wp.frame[p] as usize;
        let f = &fd[fr];
        let sw = f.orig_w as f64 / w as f64;
        let sh = f.orig_h as f64 / h as f64;
        // Reference scales an int32 array in place → truncation toward zero.
        let x = ((wp.u[p] as f64) * sw).trunc();
        let y = ((wp.v[p] as f64) * sh).trunc();
        let point3d_id = (p as i64) + 1;
        let idx = img_pts[fr].len() as i32;
        img_pts[fr].push(Pt2D { x, y, point3d_id });
        track_image_id[p] = (fr as i32) + 1;
        track_pt2d_idx[p] = idx;
    }

    if binary {
        write_cameras_bin(&dir, &fd, n)?;
        write_images_bin(&dir, &fd, &img_pts, image_names, n)?;
        write_points3d_bin(&dir, &wp, &track_image_id, &track_pt2d_idx, num_points)?;
    } else {
        write_cameras_txt(&dir, &fd, n)?;
        write_images_txt(&dir, &fd, &img_pts, image_names, n)?;
        write_points3d_txt(&dir, &wp, &track_image_id, &track_pt2d_idx, num_points)?;
    }
    Ok(())
}

fn write_cameras_bin<P: AsRef<Path>>(dir: P, fd: &[FrameData], n: usize) -> Result<()> {
    let mut o = BufWriter::new(File::create(dir.as_ref().join("cameras.bin"))?);
    write_u64(&mut o, n as u64)?;
    for (i, f) in fd.iter().enumerate() {
        write_i32(&mut o, (i as i32) + 1)?; // camera id
        write_i32(&mut o, 1)?; // model_id = PINHOLE
        write_u64(&mut o, f.orig_w as u64)?;
        write_u64(&mut o, f.orig_h as u64)?;
        write_f64(&mut o, f.fx)?;
        write_f64(&mut o, f.fy)?;
        write_f64(&mut o, f.cx)?;
        write_f64(&mut o, f.cy)?;
    }
    o.flush()?;
    Ok(())
}

fn write_images_bin<P: AsRef<Path>>(
    dir: P,
    fd: &[FrameData],
    img_pts: &[Vec<Pt2D>],
    image_names: &[String],
    n: usize,
) -> Result<()> {
    let mut o = BufWriter::new(File::create(dir.as_ref().join("images.bin"))?);
    write_u64(&mut o, n as u64)?;
    for (i, f) in fd.iter().enumerate() {
        write_i32(&mut o, (i as i32) + 1)?; // image id
        write_f64(&mut o, f.qvec[0] as f64)?;
        write_f64(&mut o, f.qvec[1] as f64)?;
        write_f64(&mut o, f.qvec[2] as f64)?;
        write_f64(&mut o, f.qvec[3] as f64)?;
        write_f64(&mut o, f.tx)?;
        write_f64(&mut o, f.ty)?;
        write_f64(&mut o, f.tz)?;
        write_i32(&mut o, (i as i32) + 1)?; // camera id
        let nm = &image_names[i];
        o.write_all(nm.as_bytes())?;
        o.write_all(b"\0")?;
        let pts = &img_pts[i];
        write_u64(&mut o, pts.len() as u64)?;
        for p in pts {
            write_f64(&mut o, p.x)?;
            write_f64(&mut o, p.y)?;
            write_i64(&mut o, p.point3d_id)?;
        }
    }
    o.flush()?;
    Ok(())
}

fn write_points3d_bin<P: AsRef<Path>>(
    dir: P,
    wp: &WorldPoints,
    track_image_id: &[i32],
    track_pt2d_idx: &[i32],
    num_points: usize,
) -> Result<()> {
    let mut o = BufWriter::new(File::create(dir.as_ref().join("points3D.bin"))?);
    write_u64(&mut o, num_points as u64)?;
    for p in 0..num_points {
        write_u64(&mut o, (p as u64) + 1)?;
        write_f64(&mut o, wp.xyz[3 * p] as f64)?;
        write_f64(&mut o, wp.xyz[3 * p + 1] as f64)?;
        write_f64(&mut o, wp.xyz[3 * p + 2] as f64)?;
        o.write_all(&[wp.rgb[3 * p], wp.rgb[3 * p + 1], wp.rgb[3 * p + 2]])?;
        write_f64(&mut o, 0.0)?; // error
        write_u64(&mut o, 1)?; // track length (single observation)
        write_i32(&mut o, track_image_id[p])?;
        write_i32(&mut o, track_pt2d_idx[p])?;
    }
    o.flush()?;
    Ok(())
}

fn write_cameras_txt<P: AsRef<Path>>(dir: P, fd: &[FrameData], n: usize) -> Result<()> {
    let mut o = BufWriter::new(File::create(dir.as_ref().join("cameras.txt"))?);
    writeln!(
        o,
        "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: {n}"
    )?;
    for (i, f) in fd.iter().enumerate() {
        writeln!(
            o,
            "{} PINHOLE {} {} {:.17e} {:.17e} {:.17e} {:.17e}",
            i + 1,
            f.orig_w,
            f.orig_h,
            f.fx,
            f.fy,
            f.cx,
            f.cy,
        )?;
    }
    o.flush()?;
    Ok(())
}

fn write_images_txt<P: AsRef<Path>>(
    dir: P,
    fd: &[FrameData],
    img_pts: &[Vec<Pt2D>],
    image_names: &[String],
    n: usize,
) -> Result<()> {
    let mut o = BufWriter::new(File::create(dir.as_ref().join("images.txt"))?);
    let mean_obs: f64 = if n > 0 {
        let total: usize = img_pts.iter().map(|p| p.len()).sum();
        total as f64 / n as f64
    } else {
        0.0
    };
    writeln!(
        o,
        "# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n# Number of images: {n}, mean observations per image: {mean_obs}"
    )?;
    for (i, f) in fd.iter().enumerate() {
        writeln!(
            o,
            "{} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {} {}",
            i + 1,
            f.qvec[0] as f64,
            f.qvec[1] as f64,
            f.qvec[2] as f64,
            f.qvec[3] as f64,
            f.tx,
            f.ty,
            f.tz,
            i + 1,
            image_names[i],
        )?;
        let pts = &img_pts[i];
        let mut first = true;
        for p in pts {
            if !first {
                o.write_all(b" ")?;
            }
            first = false;
            write!(o, "{:.17e} {:.17e} {}", p.x, p.y, p.point3d_id)?;
        }
        writeln!(o)?;
    }
    o.flush()?;
    Ok(())
}

fn write_points3d_txt<P: AsRef<Path>>(
    dir: P,
    wp: &WorldPoints,
    track_image_id: &[i32],
    track_pt2d_idx: &[i32],
    num_points: usize,
) -> Result<()> {
    let mut o = BufWriter::new(File::create(dir.as_ref().join("points3D.txt"))?);
    let mean_track: f64 = if num_points > 0 { 1.0 } else { 0.0 };
    writeln!(
        o,
        "# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n# Number of points: {num_points}, mean track length: {mean_track}"
    )?;
    for p in 0..num_points {
        writeln!(
            o,
            "{} {:.17e} {:.17e} {:.17e} {} {} {} {:.17e} {} {}",
            (p as u64) + 1,
            wp.xyz[3 * p] as f64,
            wp.xyz[3 * p + 1] as f64,
            wp.xyz[3 * p + 2] as f64,
            wp.rgb[3 * p],
            wp.rgb[3 * p + 1],
            wp.rgb[3 * p + 2],
            0.0f64,
            track_image_id[p],
            track_pt2d_idx[p],
        )?;
    }
    o.flush()?;
    Ok(())
}

// ---- little-endian binary writers (host endianness assumed irrelevant) -----

fn write_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_i64<W: Write>(w: &mut W, v: i64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_i32<W: Write>(w: &mut W, v: i32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f64<W: Write>(w: &mut W, v: f64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_ext() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn identity_k() -> [f32; 9] {
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn write_colmap_binary_roundtrip_header() {
        let h = 2;
        let w = 2;
        let depth = vec![2.0f32; h * w];
        let conf: Vec<f32> = vec![];
        let k = vec![identity_k()];
        let ext = vec![identity_ext()];
        let img = vec![10u8; h * w * 3];
        let images: Vec<&[u8]> = vec![img.as_slice()];
        let names = vec!["frame0001.png".to_string()];
        let orig_wh = vec![(100i64, 100i64)];

        let dir = std::env::temp_dir().join("da3_rust_colmap_bin_test");
        let _ = std::fs::remove_dir_all(&dir);
        write_colmap(
            &dir, &depth, &conf, &k, &ext, &images, &names, &orig_wh, h, w, 1, true,
        )
        .unwrap();

        // cameras.bin: u64 count, then per-camera records.
        let cam_bytes = std::fs::read(dir.join("cameras.bin")).unwrap();
        assert_eq!(u64::from_le_bytes(cam_bytes[0..8].try_into().unwrap()), 1);
        // camera id (i32), model id (i32 == 1 PINHOLE)
        let cam_id = i32::from_le_bytes(cam_bytes[8..12].try_into().unwrap());
        let model_id = i32::from_le_bytes(cam_bytes[12..16].try_into().unwrap());
        assert_eq!(cam_id, 1);
        assert_eq!(model_id, 1); // PINHOLE
                                 // width, height (u64 each) == orig_wh (100, 100)
        let cw = u64::from_le_bytes(cam_bytes[16..24].try_into().unwrap());
        let ch = u64::from_le_bytes(cam_bytes[24..32].try_into().unwrap());
        assert_eq!(cw, 100);
        assert_eq!(ch, 100);

        // images.bin: u64 count == 1.
        let img_bytes = std::fs::read(dir.join("images.bin")).unwrap();
        assert_eq!(u64::from_le_bytes(img_bytes[0..8].try_into().unwrap()), 1);

        // points3D.bin: u64 count == 4 (2×2 all-valid).
        let pts_bytes = std::fs::read(dir.join("points3D.bin")).unwrap();
        assert_eq!(u64::from_le_bytes(pts_bytes[0..8].try_into().unwrap()), 4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_colmap_text_has_expected_lines() {
        let h = 1;
        let w = 2;
        let depth = vec![1.5f32, 3.0];
        let conf: Vec<f32> = vec![];
        let k = vec![identity_k()];
        let ext = vec![identity_ext()];
        let img = vec![200u8; h * w * 3];
        let images: Vec<&[u8]> = vec![img.as_slice()];
        let names = vec!["shot.jpg".to_string()];
        let orig_wh = vec![(4i64, 4i64)];

        let dir = std::env::temp_dir().join("da3_rust_colmap_txt_test");
        let _ = std::fs::remove_dir_all(&dir);
        write_colmap(
            &dir, &depth, &conf, &k, &ext, &images, &names, &orig_wh, h, w, 1, false,
        )
        .unwrap();

        let cam = std::fs::read_to_string(dir.join("cameras.txt")).unwrap();
        assert!(cam.contains("PINHOLE"));
        assert!(!cam.contains("shot.jpg")); // name is in images.txt, not cameras
        let camera_line = cam.lines().find(|l| !l.starts_with('#')).unwrap();
        // "1 PINHOLE 4 4 <fx> <fy> <cx> <cy>"
        assert!(camera_line.starts_with("1 PINHOLE 4 4 "));

        let images = std::fs::read_to_string(dir.join("images.txt")).unwrap();
        assert!(images.contains("shot.jpg"));

        let pts = std::fs::read_to_string(dir.join("points3D.txt")).unwrap();
        // Two valid points -> two data lines.
        let data_lines: Vec<&str> = pts.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(data_lines.len(), 2);
        // Color should be (200, 200, 200).
        assert!(data_lines[0].contains(" 200 200 200 "));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
