//! glTF 2.0 binary (.glb) point-cloud + camera-frustum exporter, matching
//! `src/glb_export.cpp::write_glb`.
//!
//! Emits a single-mesh .glb with up to two primitives:
//! - a **POINTS** primitive of the back-projected, aligned world points (colored
//!   from the source RGB), and
//! - an optional **LINES** primitive of camera frustums (one wireframe per
//!   frame, HSV-colored by frame index).
//!
//! Geometry is aligned to the first camera in glTF coordinates and centered on
//! the per-axis median of the point cloud (mirroring the reference's
//! `_compute_alignment_transform_first_cam_glTF_center_by_points`). The
//! reference downsamples with `np.random.choice` (nondeterministic); this port
//! keeps ALL points so the output is deterministic and byte-faithful to the
//! per-point geometry.

// Mirrors `glb_export.cpp` line-for-line; the index expressions like `0*4+c`
// and the 10-argument `write_glb` are intentional for diffing against the C++.
#![allow(
    clippy::needless_range_loop,
    clippy::erasing_op,
    clippy::identity_op,
    clippy::too_many_arguments
)]

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use crate::reconstruct::{back_project, inv3, inv4, percentile_linear, WorldPoints};
use crate::Result;

/// Options controlling .glb (glTF-2.0 binary) export. Mirrors `GlbOptions` in
/// `glb_export.hpp`.
#[derive(Debug, Clone)]
pub struct GlbOptions {
    /// Maximum number of points retained. `<= 0` or `>= the produced point
    /// count` keeps ALL points. NOTE: the reference downsamples with
    /// `np.random.choice` (nondeterministic); this port ignores the cap and
    /// keeps ALL points for deterministic, byte-faithful geometry.
    pub num_max_points: i32,
    /// Whether to emit camera-frustum wireframes (a LINES primitive).
    pub show_cameras: bool,
    /// Camera wireframe scale as a fraction of the estimated scene diagonal.
    pub camera_size: f32,
    /// GLB adaptive confidence threshold base (see `conf_thresh_*`).
    pub conf_thresh: f32,
    pub conf_thresh_percentile: f32,
    pub ensure_thresh_percentile: f32,
}

impl Default for GlbOptions {
    fn default() -> Self {
        Self {
            num_max_points: 1_000_000,
            show_cameras: true,
            camera_size: 0.03,
            conf_thresh: 1.05,
            conf_thresh_percentile: 40.0,
            ensure_thresh_percentile: 90.0,
        }
    }
}

// Apply a row-major 4×4 to a 3D point, returning the `[:3]` of the homogeneous
// result. The last row of `a` is assumed `[0,0,0,1]` in our usage; the w-divide
// is omitted to match `trimesh.transform_points` for affine matrices.
fn apply44(a: &[f64; 16], x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    (
        a[0] * x + a[1] * y + a[2] * z + a[3],
        a[4] * x + a[5] * y + a[6] * z + a[7],
        a[8] * x + a[9] * y + a[10] * z + a[11],
    )
}

// HSV→RGB matching the reference `_hsv_to_rgb` (h,s,v in [0,1]).
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let mut i = (h * 6.0) as i64;
    let f = h * 6.0 - i as f64;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    i = ((i % 6) + 6) % 6;
    match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

// `_index_color_rgb(i,n)`: HSV at hue `(i+0.5)/max(n,1)`, s=0.85, v=0.95 → u8 rgb.
fn index_color_rgb(i: i32, n: i32) -> [u8; 3] {
    let h = (i as f64 + 0.5) / n.max(1) as f64;
    let (r, g, b) = hsv_to_rgb(h, 0.85, 0.95);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

// Port of `_camera_frustum_lines`: returns 8 segments (16 world-frame points,
// pairs of [a,b]) for one camera, BEFORE the alignment transform A.
// `out` is cleared and filled with 16×3 = 48 doubles.
fn camera_frustum_lines(
    k: &[f32; 9],
    ext: &[f32; 16],
    w: i32,
    h: i32,
    scale: f64,
    out: &mut Vec<f64>,
) {
    out.clear();
    let Some(kinv) = inv3(k) else { return };
    let Some(c2w) = inv4(ext) else { return };

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

    // Camera center in world: c2w @ [0,0,0,1] == translation column.
    let cw_center = [cw[3], cw[7], cw[11]];

    let corners = [
        (0.0f64, 0.0, 1.0),
        ((w - 1) as f64, 0.0, 1.0),
        ((w - 1) as f64, (h - 1) as f64, 1.0),
        (0.0, (h - 1) as f64, 1.0),
    ];
    let mut plane_w = [[0.0f64; 3]; 4];
    for c in 0..4 {
        let (cx, cy, cz) = corners[c];
        let rx = ki[0] * cx + ki[1] * cy + ki[2] * cz;
        let ry = ki[3] * cx + ki[4] * cy + ki[5] * cz;
        let rz = ki[6] * cx + ki[7] * cy + ki[8] * cz;
        let z = if rz == 0.0 { 1.0 } else { rz };
        let px = (rx / z) * scale;
        let py = (ry / z) * scale;
        let pz = (rz / z) * scale;
        // to world: c2w @ [px,py,pz,1]
        plane_w[c][0] = cw[0] * px + cw[1] * py + cw[2] * pz + cw[3];
        plane_w[c][1] = cw[4] * px + cw[5] * py + cw[6] * pz + cw[7];
        plane_w[c][2] = cw[8] * px + cw[9] * py + cw[10] * pz + cw[11];
    }

    let mut push = |a: &[f64; 3], b: &[f64; 3]| {
        out.extend_from_slice(a);
        out.extend_from_slice(b);
    };
    // center to corners
    for k in 0..4 {
        push(&cw_center, &plane_w[k]);
    }
    // rectangle edges: 0-1,1-2,2-3,3-0
    let order = [0usize, 1, 2, 3, 0];
    for e in 0..4 {
        push(&plane_w[order[e]], &plane_w[order[e + 1]]);
    }
}

// Little-endian f32 appender into a Vec<u8>.
fn put_f32(b: &mut Vec<u8>, f: f32) {
    b.extend_from_slice(&f.to_le_bytes());
}

// Format a float for a glTF accessor min/max hint. Matches the C++ `%.9g` output
// closely enough for these (advisory) fields — exact byte parity is not needed
// here because min/max are geometry bounds, not part of the binary payload.
fn fmt_f(v: f32) -> String {
    // `{:?}` is Rust's shortest round-trippable repr, close to C's `%g`.
    let s = format!("{:?}", v);
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Write a glTF-2.0 binary point cloud (POINTS primitive) plus optional camera
/// frustums (LINES primitive) to `path`.
///
/// Inputs mirror [`back_project`]:
/// - `depth`, `conf`: `N*H*W` row-major (frame, row, col).
/// - `k`: per-frame 3×3 intrinsics, row-major.
/// - `ext`: per-frame 4×4 world-to-camera extrinsics, row-major.
/// - `images_u8`: per-frame slice over `H*W*3` RGB uint8 image.
///
/// Returns `Ok(())` on success, or an I/O error.
pub fn write_glb<P: AsRef<Path>>(
    path: P,
    depth: &[f32],
    conf: &[f32],
    k: &[[f32; 9]],
    ext: &[[f32; 16]],
    images_u8: &[&[u8]],
    h: usize,
    w: usize,
    n: usize,
    opt: &GlbOptions,
) -> Result<()> {
    // 1) Adaptive confidence threshold over ALL conf values (no sky mask):
    //    lower = percentile(conf, conf_thresh_percentile)
    //    upper = percentile(conf, ensure_thresh_percentile)
    //    thr   = min(max(conf_thresh, lower), upper)
    // (matches `glb_export.cpp::write_glb`).
    let conf_thr = if conf.is_empty() {
        opt.conf_thresh
    } else {
        let lower = percentile_linear(conf, opt.conf_thresh_percentile as f64) as f32;
        let upper = percentile_linear(conf, opt.ensure_thresh_percentile as f64) as f32;
        opt.conf_thresh.max(lower).min(upper)
    };

    // 2) Back-project to world points + colors.
    let wp: WorldPoints = back_project(depth, conf, k, ext, images_u8, h, w, n, conf_thr);
    let npts = wp.xyz.len() / 3;

    // 3) glTF alignment transform A (in f64).
    //    w2c0 = ext[0]; M = diag(1,-1,-1,1); A_no_center = M @ w2c0.
    let mut a_no_center = [
        1.0f64, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    if n > 0 {
        let w0 = &ext[0];
        for c in 0..4 {
            a_no_center[0 * 4 + c] = w0[0 * 4 + c] as f64; // row 0
            a_no_center[1 * 4 + c] = -(w0[1 * 4 + c] as f64); // row 1 flipped
            a_no_center[2 * 4 + c] = -(w0[2 * 4 + c] as f64); // row 2 flipped
            a_no_center[3 * 4 + c] = w0[3 * 4 + c] as f64; // row 3
        }
    }
    // pts_tmp = A_no_center @ [X;1]; center = per-axis median (50th percentile).
    let mut center = [0.0f64; 3];
    let mut tmpx = Vec::with_capacity(npts);
    let mut tmpy = Vec::with_capacity(npts);
    let mut tmpz = Vec::with_capacity(npts);
    for p in 0..npts {
        let (ox, oy, oz) = apply44(
            &a_no_center,
            wp.xyz[3 * p] as f64,
            wp.xyz[3 * p + 1] as f64,
            wp.xyz[3 * p + 2] as f64,
        );
        // Store as float to mirror numpy float32 point cloud for the median.
        tmpx.push(ox as f32);
        tmpy.push(oy as f32);
        tmpz.push(oz as f32);
    }
    if npts > 0 {
        center[0] = percentile_linear(&tmpx, 50.0);
        center[1] = percentile_linear(&tmpy, 50.0);
        center[2] = percentile_linear(&tmpz, 50.0);
    }
    // A = T(-center) @ A_no_center: since last row of A_no_center is [0,0,0,1],
    // this only subtracts `center` from the translation column of rows 0..2.
    let mut a = a_no_center;
    a[3] -= center[0];
    a[7] -= center[1];
    a[11] -= center[2];

    // Final aligned points = A @ [X;1] = pts_tmp - center.
    let mut pos: Vec<f32> = Vec::with_capacity(npts * 3);
    for p in 0..npts {
        pos.push((tmpx[p] as f64 - center[0]) as f32);
        pos.push((tmpy[p] as f64 - center[1]) as f32);
        pos.push((tmpz[p] as f64 - center[2]) as f32);
    }
    // (num_max_points downsample intentionally disabled for determinism.)

    // 4) Optional camera frustums (LINES), aligned by A.
    let mut line_pos: Vec<f32> = Vec::new();
    let mut line_col: Vec<u8> = Vec::new();
    if opt.show_cameras && n > 0 {
        // Scene scale: p5/p95 per-axis diagonal of the ALIGNED points.
        let mut scene_scale = 1.0f64;
        if npts >= 2 {
            let mut ax = vec![0.0f32; npts];
            let mut ay = vec![0.0f32; npts];
            let mut az = vec![0.0f32; npts];
            for p in 0..npts {
                ax[p] = pos[3 * p];
                ay[p] = pos[3 * p + 1];
                az[p] = pos[3 * p + 2];
            }
            let lo = [
                percentile_linear(&ax, 5.0),
                percentile_linear(&ay, 5.0),
                percentile_linear(&az, 5.0),
            ];
            let hi = [
                percentile_linear(&ax, 95.0),
                percentile_linear(&ay, 95.0),
                percentile_linear(&az, 95.0),
            ];
            let dx = hi[0] - lo[0];
            let dy = hi[1] - lo[1];
            let dz = hi[2] - lo[2];
            let diag = (dx * dx + dy * dy + dz * dz).sqrt();
            if diag.is_finite() && diag > 0.0 {
                scene_scale = diag;
            }
        }
        let scale = scene_scale * opt.camera_size as f64;

        let mut segs: Vec<f64> = Vec::new();
        for i in 0..n {
            camera_frustum_lines(&k[i], &ext[i], w as i32, h as i32, scale, &mut segs);
            let col = index_color_rgb(i as i32, n as i32);
            for kk in 0..(segs.len() / 3) {
                let (ox, oy, oz) = apply44(&a, segs[3 * kk], segs[3 * kk + 1], segs[3 * kk + 2]);
                line_pos.push(ox as f32);
                line_pos.push(oy as f32);
                line_pos.push(oz as f32);
                line_col.push(col[0]);
                line_col.push(col[1]);
                line_col.push(col[2]);
                line_col.push(255);
            }
        }
    }
    let nlines = line_pos.len() / 3;
    let have_lines = nlines > 0;
    let have_points = npts > 0;

    // 5) Build BIN buffer with 4-byte-aligned bufferViews. Sections (points,
    // lines) are emitted only when non-empty, and bufferView/accessor indices
    // are assigned dynamically — a zero-length bufferView/accessor/buffer is
    // invalid per glTF 2.0, so an empty point cloud must omit them entirely.
    let mut bin: Vec<u8> = Vec::new();
    let align4 = |bin: &mut Vec<u8>| {
        while bin.len() % 4 != 0 {
            bin.push(0)
        }
    };

    #[derive(Clone, Copy)]
    struct Bv {
        off: u32,
        len: u32,
    }
    let mut bvs: Vec<Bv> = Vec::new();
    let mut bv_ppos: i32 = -1;
    let mut bv_pcol: i32 = -1;
    let mut bv_lpos: i32 = -1;
    let mut bv_lcol: i32 = -1;

    if have_points {
        let off = bin.len() as u32;
        for &v in &pos {
            put_f32(&mut bin, v);
        }
        bvs.push(Bv {
            off,
            len: bin.len() as u32 - off,
        });
        bv_ppos = (bvs.len() - 1) as i32;
        align4(&mut bin);
        let off = bin.len() as u32;
        for p in 0..npts {
            bin.extend_from_slice(&wp.rgb[3 * p..3 * p + 3]);
            bin.push(255);
        }
        bvs.push(Bv {
            off,
            len: bin.len() as u32 - off,
        });
        bv_pcol = (bvs.len() - 1) as i32;
        align4(&mut bin);
    }
    if have_lines {
        let off = bin.len() as u32;
        for &v in &line_pos {
            put_f32(&mut bin, v);
        }
        bvs.push(Bv {
            off,
            len: bin.len() as u32 - off,
        });
        bv_lpos = (bvs.len() - 1) as i32;
        align4(&mut bin);
        let off = bin.len() as u32;
        bin.extend_from_slice(&line_col);
        bvs.push(Bv {
            off,
            len: bin.len() as u32 - off,
        });
        bv_lcol = (bvs.len() - 1) as i32;
        align4(&mut bin);
    }

    // POSITION min/max for accessors.
    let mut pmin = [0.0f32; 3];
    let mut pmax = [0.0f32; 3];
    if have_points {
        pmin.copy_from_slice(&pos[..3]);
        pmax.copy_from_slice(&pos[..3]);
        for p in 0..npts {
            for c in 0..3 {
                pmin[c] = pmin[c].min(pos[3 * p + c]);
                pmax[c] = pmax[c].max(pos[3 * p + c]);
            }
        }
    }
    let mut lmin = [0.0f32; 3];
    let mut lmax = [0.0f32; 3];
    if have_lines {
        lmin.copy_from_slice(&line_pos[..3]);
        lmax.copy_from_slice(&line_pos[..3]);
        for p in 0..nlines {
            for c in 0..3 {
                lmin[c] = lmin[c].min(line_pos[3 * p + c]);
                lmax[c] = lmax[c].max(line_pos[3 * p + c]);
            }
        }
    }

    // Build accessors + mesh primitives with dynamic indices.
    let mut acc = String::new();
    let mut prim = String::new();
    let mut nacc = 0i32;
    let mut nprim = 0i32;
    if have_points {
        let a_pos = nacc;
        nacc += 1;
        let a_col = nacc;
        nacc += 1;
        acc.push_str(&format!(
            "{{\"bufferView\":{},\"componentType\":5126,\"count\":{},\"type\":\"VEC3\",\"min\":[{},{},{}],\"max\":[{},{},{}]}}",
            bv_ppos, npts,
            fmt_f(pmin[0]), fmt_f(pmin[1]), fmt_f(pmin[2]),
            fmt_f(pmax[0]), fmt_f(pmax[1]), fmt_f(pmax[2]),
        ));
        acc.push_str(&format!(
            ",{{\"bufferView\":{},\"componentType\":5121,\"normalized\":true,\"count\":{},\"type\":\"VEC4\"}}",
            bv_pcol, npts,
        ));
        prim.push_str(&format!(
            "{{\"attributes\":{{\"POSITION\":{},\"COLOR_0\":{}}},\"mode\":0}}",
            a_pos, a_col,
        ));
        nprim += 1;
    }
    if have_lines {
        let a_pos = nacc;
        nacc += 1;
        let a_col = nacc;
        nacc += 1;
        if nacc > 2 {
            acc.push(',');
        }
        acc.push_str(&format!(
            "{{\"bufferView\":{},\"componentType\":5126,\"count\":{},\"type\":\"VEC3\",\"min\":[{},{},{}],\"max\":[{},{},{}]}}",
            bv_lpos, nlines,
            fmt_f(lmin[0]), fmt_f(lmin[1]), fmt_f(lmin[2]),
            fmt_f(lmax[0]), fmt_f(lmax[1]), fmt_f(lmax[2]),
        ));
        acc.push_str(&format!(
            ",{{\"bufferView\":{},\"componentType\":5121,\"normalized\":true,\"count\":{},\"type\":\"VEC4\"}}",
            bv_lcol, nlines,
        ));
        if nprim > 0 {
            prim.push(',');
        }
        prim.push_str(&format!(
            "{{\"attributes\":{{\"POSITION\":{},\"COLOR_0\":{}}},\"mode\":1}}",
            a_pos, a_col,
        ));
        nprim += 1;
    }

    // 6) Build JSON.
    let mut js = String::new();
    js.push_str("{\"asset\":{\"version\":\"2.0\",\"generator\":\"depth-anything.rs\"}");
    if nprim > 0 {
        js.push_str(",\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],");
        js.push_str("\"bufferViews\":[");
        for (i, bv) in bvs.iter().enumerate() {
            if i > 0 {
                js.push(',');
            }
            js.push_str(&format!(
                "{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{},\"target\":34962}}",
                bv.off, bv.len,
            ));
        }
        js.push_str("],\"accessors\":[");
        js.push_str(&acc);
        js.push_str("],");
        js.push_str("\"meshes\":[{\"primitives\":[");
        js.push_str(&prim);
        js.push_str("]}],");
        js.push_str(&format!("\"buffers\":[{{\"byteLength\":{}}}]}}", bin.len()));
    } else {
        // Nothing to draw: emit a minimal valid glTF — empty scene, no buffers.
        js.push_str(",\"scene\":0,\"scenes\":[{\"nodes\":[]}]}");
    }

    // Pad JSON to 4 bytes with spaces (0x20).
    while js.len() % 4 != 0 {
        js.push(' ');
    }
    // Pad BIN to 4 bytes with zeros.
    while bin.len() % 4 != 0 {
        bin.push(0);
    }

    let json_len = js.len() as u32;
    let bin_len = bin.len() as u32;
    let emit_bin = bin_len > 0;
    let total = 12 + 8 + json_len + if emit_bin { 8 + bin_len } else { 0 };

    let mut f = BufWriter::new(File::create(path)?);
    // 12-byte header.
    put_u32_buf(&mut f, 0x46546C67)?; // "glTF"
    put_u32_buf(&mut f, 2)?; // version
    put_u32_buf(&mut f, total)?;
    // JSON chunk header + payload.
    put_u32_buf(&mut f, json_len)?;
    put_u32_buf(&mut f, 0x4E4F534A)?; // "JSON"
    f.write_all(js.as_bytes())?;
    if emit_bin {
        put_u32_buf(&mut f, bin_len)?;
        put_u32_buf(&mut f, 0x004E4942)?; // "BIN\0"
        f.write_all(&bin)?;
    }
    f.flush()?;
    Ok(())
}

// Write a u32 little-endian to a generic writer.
fn put_u32_buf<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
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
    fn hsv_to_rgb_red_and_green() {
        // hue=0 -> red (v, t, p) = (1, 0, 0) when s=1, v=1.
        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 1e-9);
        assert!(g.abs() < 1e-9);
        assert!(b.abs() < 1e-9);
        // hue=1/3 -> green.
        let (r, g, b) = hsv_to_rgb(1.0 / 3.0, 1.0, 1.0);
        assert!(r.abs() < 1e-9);
        assert!((g - 1.0).abs() < 1e-9);
        assert!(b.abs() < 1e-9);
    }

    #[test]
    fn index_color_rgb_in_range() {
        // Different hues must produce different colors, and all channels are
        // valid u8 by construction.
        let c0 = index_color_rgb(0, 10);
        let c5 = index_color_rgb(5, 10);
        assert_eq!(c0.len(), 3);
        assert_eq!(c5.len(), 3);
        assert_ne!(c0, c5, "distinct hues should produce distinct colors");
    }

    #[test]
    fn write_glb_minimal_scene_with_cameras() {
        // 2×2 image, depth=2 everywhere, identity K and ext, single frame.
        // Expect 4 points + 8 frustum segments (16 line vertices).
        let h = 2;
        let w = 2;
        let depth = vec![2.0f32; h * w];
        let conf: Vec<f32> = vec![];
        let k = vec![identity_k()];
        let ext = vec![identity_ext()];
        let img: Vec<u8> = (0..(h * w * 3)).map(|i| (i % 200) as u8).collect();
        let images: Vec<&[u8]> = vec![img.as_slice()];

        let dir = std::env::temp_dir().join("da3_rust_glb_test.glb");
        let opt = GlbOptions {
            show_cameras: true,
            ..GlbOptions::default()
        };
        write_glb(&dir, &depth, &conf, &k, &ext, &images, h, w, 1, &opt).unwrap();
        let bytes = std::fs::read(&dir).unwrap();
        // GLB header: magic (0..4), version (4..8), total length (8..12).
        assert_eq!(&bytes[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
        let total = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        assert_eq!(total, bytes.len());
        // JSON chunk header: chunk length (12..16), chunk type "JSON" (16..20).
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(&bytes[16..20], b"JSON");
        let json = std::str::from_utf8(&bytes[20..20 + json_len]).unwrap();
        // Must mention both a POINTS (mode 0) and a LINES (mode 1) primitive.
        assert!(json.contains("\"mode\":0"), "json missing POINTS: {json}");
        assert!(json.contains("\"mode\":1"), "json missing LINES: {json}");
        // BIN chunk header immediately follows the JSON payload.
        let bin_off = 20 + json_len;
        assert_eq!(&bytes[bin_off + 4..bin_off + 8], b"BIN\0");
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn write_glb_empty_scene_is_valid() {
        // All-zero depth -> no valid points; with no cameras shown either this
        // should be a minimal empty glTF.
        let h = 2;
        let w = 2;
        let depth = vec![0.0f32; h * w]; // all invalid
        let conf: Vec<f32> = vec![];
        let k = vec![identity_k()];
        let ext = vec![identity_ext()];
        let img = vec![0u8; h * w * 3];
        let images: Vec<&[u8]> = vec![img.as_slice()];
        let dir = std::env::temp_dir().join("da3_rust_glb_empty_test.glb");
        let opt = GlbOptions {
            show_cameras: false,
            ..GlbOptions::default()
        };
        write_glb(&dir, &depth, &conf, &k, &ext, &images, h, w, 1, &opt).unwrap();
        let bytes = std::fs::read(&dir).unwrap();
        assert_eq!(&bytes[0..4], b"glTF");
        // 12-byte header + 8-byte JSON chunk header + json payload (padded to 4).
        assert!(bytes.len() >= 20);
        assert_eq!(&bytes[16..20], b"JSON");
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let json = std::str::from_utf8(&bytes[20..20 + json_len]).unwrap();
        // Empty scene: nodes array empty, no bufferViews/accessors.
        assert!(json.contains("\"scenes\":[{\"nodes\":[]}]"));
        assert!(!json.contains("bufferViews"));
        let _ = std::fs::remove_file(&dir);
    }
}
