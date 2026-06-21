//! Depth-export activation post-processing and PFM/PNG writers, matching
//! `src/depth_export.cpp`.
//!
//! Activation selection (no `is_metric` flag — implicit from `output_dim`,
//! `max_depth`, and the presence of a sky head):
//!
//! | condition                              | depth    | conf        | sky   |
//! |----------------------------------------|----------|-------------|-------|
//! | `output_dim == 2` (DA3 base)           | `exp(x)` | `exp(x)+1`  | —     |
//! | `output_dim == 1` + sky head (metric)  | `exp(x)` | —           | relu  |
//! | `output_dim == 1` + `max_depth > 0`    | `σ(x)·m` | —           | —     |
//! | `output_dim == 1` + relative (DA2)     | `relu`   | —           | —     |

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::Result;

/// One decoded depth map and its associated outputs, in `(H, W)` row-major
/// (w fastest).
#[derive(Debug, Clone)]
pub struct DepthMaps {
    pub depth: Vec<f32>,
    pub conf: Vec<f32>,
    pub sky: Vec<f32>,
}

/// Apply the activation to raw logits `[W, H, output_dim]` (w fastest in the
/// inner dimension, matching the ggml `[W,H,C,1]` layout), producing `(depth,
/// conf, sky)` each `[H*W]` row-major over `(h, w)`.
pub fn activate(
    logits: &[f32],
    sky_logits: Option<&[f32]>,
    hw: usize,
    output_dim: usize,
    max_depth: f32,
) -> DepthMaps {
    let mut depth = vec![0.0f32; hw];
    let mut conf = Vec::new();
    let mut sky = Vec::new();

    if output_dim >= 2 {
        // DA3 base: ch0 depth = exp, ch1 conf = exp+1.
        for i in 0..hw {
            depth[i] = logits[i].exp();
        }
        conf = (0..hw).map(|i| logits[hw + i].exp() + 1.0).collect();
    } else if let Some(sky_l) = sky_logits {
        // Metric DA3: depth = exp, sky = relu.
        for i in 0..hw {
            depth[i] = logits[i].exp();
            sky.push(sky_l[i].max(0.0));
        }
    } else if max_depth > 0.0 {
        // DA2 metric: depth = sigmoid * max_depth.
        for i in 0..hw {
            depth[i] = sigmoid(logits[i]) * max_depth;
        }
    } else {
        // DA2 relative: depth = relu.
        for i in 0..hw {
            depth[i] = logits[i].max(0.0);
        }
    }
    DepthMaps { depth, conf, sky }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Write a single-channel float map to a little-endian PFM file (rows written
/// bottom-to-top, per the PFM spec). Uses the `Pf` (single-channel) magic,
/// matching `src/depth_export.cpp`'s `write_pfm`.
pub fn write_pfm<P: AsRef<Path>>(path: P, depth: &[f32], w: usize, h: usize) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    write!(f, "Pf\n{w} {h}\n-1.0\n")?;
    // PFM stores rows bottom-to-top; our depth is row-major top-to-bottom, so
    // emit rows in reverse for upright viewers.
    let mut row = vec![0.0f32; w];
    for y in (0..h).rev() {
        row.copy_from_slice(&depth[y * w..y * w + w]);
        let bytes: &[u8] = bytemuck_cast(&row);
        f.write_all(bytes)?;
    }
    Ok(())
}

/// Write an 8-bit grayscale PNG of a depth map, min-max normalized with
/// `invert = true` (closer = brighter), matching `write_depth_png`.
pub fn write_depth_png<P: AsRef<Path>>(
    path: P,
    depth: &[f32],
    w: usize,
    h: usize,
    invert: bool,
) -> Result<()> {
    let dmin = depth.iter().copied().fold(f32::INFINITY, f32::min);
    let dmax = depth.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = dmax - dmin;
    let mut buf = image::GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let v = depth[y * w + x];
            let mut t = if range > 0.0 { (v - dmin) / range } else { 0.0 };
            if invert {
                t = 1.0 - t;
            }
            let px = (t * 255.0).round().clamp(0.0, 255.0) as u8;
            buf.put_pixel(x as u32, y as u32, image::Luma([px]));
        }
    }
    buf.save(path)?;
    Ok(())
}

/// Cast a `&[f32]` to `&[u8]` without pulling in bytemuck.
fn bytemuck_cast(slice: &[f32]) -> &[u8] {
    let ptr = slice.as_ptr() as *const u8;
    let len = std::mem::size_of_val(slice);
    // SAFETY: f32 is a plain-old-data type; we reinterpret its bytes.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Write the extrinsics (3×4) and intrinsics (3×3) as a JSON file.
pub fn write_pose_json<P: AsRef<Path>>(path: P, ext: &[f32; 12], intr: &[f32; 9]) -> Result<()> {
    let mut f = File::create(path)?;
    let fmt = |v: &[f32]| {
        v.iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
            .join(", ")
    };
    writeln!(
        f,
        "{{\n  \"extrinsics\": [\n    [{}],\n    [{}],\n    [{}]\n  ],\n  \"intrinsics\": [\n    [{}],\n    [{}],\n    [{}]\n  ]\n}}",
        fmt(&ext[0..4]),
        fmt(&ext[4..8]),
        fmt(&ext[8..12]),
        fmt(&intr[0..3]),
        fmt(&intr[3..6]),
        fmt(&intr[6..9]),
    )?;
    Ok(())
}
