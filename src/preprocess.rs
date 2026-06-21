//! Image preprocessing: cv2-faithful resize + ImageNet normalize, matching
//! `src/preprocess.cpp::preprocess_real`.
//!
//! The DA3 resize policy (`upper_bound_resize`):
//! 1. Resize the longest side to `img_resize_target` (cv2 `INTER_CUBIC` if
//!    upscaling, else `INTER_AREA`), rounding with Python's `round`
//!    (ties-to-even).
//! 2. Round each dim to the nearest multiple of `patch_size` via a second
//!    resize (same cubic/area rule).
//! 3. Convert to CHW float and ImageNet-normalize.
//!
//! Each resize step quantizes to `uint8` (matching cv2 → PIL → ToTensor).

use crate::config::Config;
use crate::Result;

/// An RGB image in HWC uint8 layout.
#[derive(Debug, Clone)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

/// The preprocessed input tensor in CHW float, plus the bookkeeping needed to
/// map outputs back to the original image.
#[derive(Debug, Clone)]
pub struct Preprocessed {
    pub h: usize,
    pub w: usize,
    /// CHW float, `[3, h, w]`.
    pub chw: Vec<f32>,
    pub orig_w: usize,
    pub orig_h: usize,
    pub scale_w: f32,
    pub scale_h: f32,
    /// Resized-but-not-normalized HWC RGB uint8 (for the glb/COLMAP exporters).
    pub rgb_u8: Vec<u8>,
}

impl Image {
    /// Load a JPEG/PNG from disk as an HWC RGB uint8 [`Image`].
    pub fn load(path: &str) -> Result<Self> {
        let img = image::open(path)?;
        let rgb = img.to_rgb8();
        let w = rgb.width() as usize;
        let h = rgb.height() as usize;
        Ok(Self {
            w,
            h,
            rgb: rgb.into_raw(),
        })
    }

    #[inline]
    fn px(&self, y: usize, x: usize, c: usize) -> u8 {
        self.rgb[(y * self.w + x) * 3 + c]
    }
}

/// Python-style round: half-to-even (banker's rounding), matching `cvRound`.
#[inline]
fn py_round(x: f64) -> i32 {
    let r = x.round();
    // `f64::round` rounds half away from zero; convert ties to even.
    let diff = (x - r).abs();
    if !(0.5 - 1e-12..=0.5 + 1e-12).contains(&diff) {
        return r as i32;
    }
    // Exactly halfway: round to even.
    let rint = r as i64;
    if rint & 1 == 1 {
        // Round toward even (opposite side from away-from-zero).
        if x > 0.0 {
            (rint - 1) as i32
        } else {
            (rint + 1) as i32
        }
    } else {
        rint as i32
    }
}

/// Nearest multiple of `p` to `x` (ties go to the larger multiple).
#[inline]
fn nearest_multiple(x: i32, p: i32) -> i32 {
    let down = (x / p) * p;
    let up = down + p;
    if (up - x).abs() <= (x - down).abs() {
        up
    } else {
        down
    }
}

/// cv2 `saturate_cast<uchar>`: round half-to-even then clamp to `[0, 255]`.
#[inline]
fn sat_u8(v: f32) -> u8 {
    let r = (v.round_ties_even()) as i32;
    r.clamp(0, 255) as u8
}

/// Catmull–Rom cubic kernel, `a = -0.75` (cv2 `INTER_CUBIC`).
#[inline]
fn cubic_w(x: f32) -> f32 {
    const A: f32 = -0.75;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        A * (((x - 5.0) * x + 8.0) * x - 4.0)
    } else {
        0.0
    }
}

#[inline]
fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

/// cv2 `INTER_CUBIC` resize (separable bicubic, single saturate at the end).
pub fn resize_cubic(src: &Image, dw: usize, dh: usize) -> Image {
    let mut dst = Image {
        w: dw,
        h: dh,
        rgb: vec![0; dw * dh * 3],
    };
    let (sw, sh) = (src.w as i32, src.h as i32);
    if dw == 0 || dh == 0 || sw <= 0 || sh <= 0 {
        return dst;
    }
    let sx = sw as f64 / dw as f64;
    let sy = sh as f64 / dh as f64;

    // Precompute x taps.
    let mut xidx = vec![0i32; dw * 4];
    let mut xwt = vec![0.0f32; dw * 4];
    for x in 0..dw {
        let fx = (x as f64 + 0.5) * sx - 0.5;
        let ix = fx.floor() as i32;
        let t = (fx - ix as f64) as f32;
        let w = [
            cubic_w(t + 1.0),
            cubic_w(t),
            cubic_w(t - 1.0),
            cubic_w(t - 2.0),
        ];
        for k in 0..4 {
            xidx[x * 4 + k] = clamp_i(ix - 1 + k as i32, 0, sw - 1);
            xwt[x * 4 + k] = w[k];
        }
    }

    // Horizontal pass: sh × dw × 3 float.
    let mut tmp = vec![0.0f32; (sh as usize) * dw * 3];
    for y in 0..sh as usize {
        for x in 0..dw {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for k in 0..4 {
                    let s = xidx[x * 4 + k] as usize;
                    acc += xwt[x * 4 + k] * src.px(y, s, c) as f32;
                }
                tmp[(y * dw + x) * 3 + c] = acc;
            }
        }
    }

    // Vertical pass + saturate.
    for y in 0..dh {
        let fy = (y as f64 + 0.5) * sy - 0.5;
        let iy = fy.floor() as i32;
        let t = (fy - iy as f64) as f32;
        let w = [
            cubic_w(t + 1.0),
            cubic_w(t),
            cubic_w(t - 1.0),
            cubic_w(t - 2.0),
        ];
        let yi: [i32; 4] = std::array::from_fn(|k| clamp_i(iy - 1 + k as i32, 0, sh - 1));
        for x in 0..dw {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for k in 0..4 {
                    acc += w[k] * tmp[(yi[k] as usize * dw + x) * 3 + c];
                }
                dst.rgb[(y * dw + x) * 3 + c] = sat_u8(acc);
            }
        }
    }
    dst
}

/// One tap of the cv2 `INTER_AREA` decimation table.
struct AreaTap {
    di: usize,
    si: usize,
    alpha: f32,
}

/// cv2 `computeResizeAreaTab`.
fn area_tab(ssize: i32, dsize: i32) -> Vec<AreaTap> {
    let mut tab = Vec::new();
    let scale = ssize as f64 / dsize as f64;
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell_width = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i32;
        let mut sx2 = fsx2.floor() as i32;
        sx2 = sx2.min(ssize - 1);
        sx1 = sx1.min(sx2);
        if (sx1 as f64 - fsx1) > 1e-3 {
            tab.push(AreaTap {
                di: dx as usize,
                si: (sx1 - 1) as usize,
                alpha: ((sx1 as f64 - fsx1) / cell_width) as f32,
            });
        }
        for sx in sx1..sx2 {
            tab.push(AreaTap {
                di: dx as usize,
                si: sx as usize,
                alpha: (1.0 / cell_width) as f32,
            });
        }
        if (fsx2 - sx2 as f64) > 1e-3 {
            tab.push(AreaTap {
                di: dx as usize,
                si: sx2 as usize,
                alpha: (((fsx2 - sx2 as f64).min(1.0)).min(cell_width) / cell_width) as f32,
            });
        }
    }
    tab
}

/// cv2 `INTER_AREA` resize (separable, single saturate).
pub fn resize_area(src: &Image, dw: usize, dh: usize) -> Image {
    let (sw, sh) = (src.w as i32, src.h as i32);
    if dw >= sw as usize && dh >= sh as usize {
        // cv2 falls back to bilinear on upscale; DA3 never asks for this.
        let hwc = resize_bilinear_hwc(src, dw, dh);
        let mut rgb = vec![0u8; dw * dh * 3];
        for i in 0..rgb.len() {
            rgb[i] = sat_u8(hwc[i]);
        }
        return Image { w: dw, h: dh, rgb };
    }
    let xtab = area_tab(sw, dw as i32);
    let ytab = area_tab(sh, dh as i32);

    // Horizontal pass.
    let mut tmp = vec![0.0f32; (sh as usize) * dw * 3];
    for y in 0..sh as usize {
        for t in &xtab {
            for c in 0..3 {
                tmp[(y * dw + t.di) * 3 + c] += t.alpha * src.px(y, t.si, c) as f32;
            }
        }
    }
    // Vertical pass.
    let mut acc = vec![0.0f32; dh * dw * 3];
    for t in &ytab {
        for x in 0..dw {
            for c in 0..3 {
                acc[(t.di * dw + x) * 3 + c] += t.alpha * tmp[(t.si * dw + x) * 3 + c];
            }
        }
    }
    let mut rgb = vec![0u8; dw * dh * 3];
    for i in 0..acc.len() {
        rgb[i] = sat_u8(acc[i]);
    }
    Image { w: dw, h: dh, rgb }
}

/// Legacy bilinear resize (HWC uint8 → HWC float), matching the C++ path used
/// only when `INTER_AREA` is asked to upscale.
fn resize_bilinear_hwc(src: &Image, dw: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dw * dh * 3];
    let sx = src.w as f32 / dw as f32;
    let sy = src.h as f32 / dh as f32;
    for y in 0..dh {
        let fy = (y as f32 + 0.5) * sy - 0.5;
        let y0 = fy.floor() as i32;
        let wy = fy - y0 as f32;
        let y0c = clamp_i(y0, 0, src.h as i32 - 1) as usize;
        let y1c = clamp_i(y0 + 1, 0, src.h as i32 - 1) as usize;
        for x in 0..dw {
            let fx = (x as f32 + 0.5) * sx - 0.5;
            let x0 = fx.floor() as i32;
            let wx = fx - x0 as f32;
            let x0c = clamp_i(x0, 0, src.w as i32 - 1) as usize;
            let x1c = clamp_i(x0 + 1, 0, src.w as i32 - 1) as usize;
            for c in 0..3 {
                let top = src.px(y0c, x0c, c) as f32 * (1.0 - wx) + src.px(y0c, x1c, c) as f32 * wx;
                let bot = src.px(y1c, x0c, c) as f32 * (1.0 - wx) + src.px(y1c, x1c, c) as f32 * wx;
                dst[(y * dw + x) * 3 + c] = top * (1.0 - wy) + bot * wy;
            }
        }
    }
    dst
}

/// The production DA3 preprocessing pipeline (matches `preprocess_real`).
pub fn preprocess_real(img: &Image, cfg: &Config) -> Result<Preprocessed> {
    if img.w == 0 || img.h == 0 || cfg.img_mean.len() < 3 || cfg.img_std.len() < 3 {
        return Err(crate::Error::Model(
            "preprocess: bad image or missing img.mean/img.std".into(),
        ));
    }
    let patch = cfg.patch_size as i32;
    let target = if cfg.img_resize_target > 0 {
        cfg.img_resize_target as i32
    } else {
        504
    };
    let upper = cfg.img_resize_mode != crate::config::ResizeMode::LowerBound;
    let (ow, oh) = (img.w as i32, img.h as i32);

    let mut cur = img.clone();

    // Step 1: boundary resize (longest/shortest side -> target).
    let bound = if upper {
        cur.w.max(cur.h) as i32
    } else {
        cur.w.min(cur.h) as i32
    };
    if bound != target {
        let scale = target as f64 / bound as f64;
        let nw = (py_round(cur.w as f64 * scale)).max(1);
        let nh = (py_round(cur.h as f64 * scale)).max(1);
        cur = if scale > 1.0 {
            resize_cubic(&cur, nw as usize, nh as usize)
        } else {
            resize_area(&cur, nw as usize, nh as usize)
        };
    }

    // Step 2: round each dim to a multiple of patch.
    let nw = (nearest_multiple(cur.w as i32, patch)).max(1);
    let nh = (nearest_multiple(cur.h as i32, patch)).max(1);
    if nw != cur.w as i32 || nh != cur.h as i32 {
        let upscale = nw > cur.w as i32 || nh > cur.h as i32;
        cur = if upscale {
            resize_cubic(&cur, nw as usize, nh as usize)
        } else {
            resize_area(&cur, nw as usize, nh as usize)
        };
    }

    let (h, w) = (cur.h, cur.w);
    let mut chw = vec![0.0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let v = cur.rgb[(y * w + x) * 3 + c] as f32 / 255.0;
                chw[(c * h + y) * w + x] = (v - cfg.img_mean[c]) / cfg.img_std[c];
            }
        }
    }
    Ok(Preprocessed {
        h,
        w,
        chw,
        orig_w: ow as usize,
        orig_h: oh as usize,
        scale_w: w as f32 / ow as f32,
        scale_h: h as f32 / oh as f32,
        rgb_u8: cur.rgb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_round_is_bankers() {
        assert_eq!(py_round(0.5), 0);
        assert_eq!(py_round(1.5), 2);
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(-0.5), 0);
        assert_eq!(py_round(0.4), 0);
        assert_eq!(py_round(0.6), 1);
        assert_eq!(py_round(2.49999), 2);
    }

    #[test]
    fn nearest_multiple_ties_up() {
        assert_eq!(nearest_multiple(35, 14), 42); // exactly halfway → up
        assert_eq!(nearest_multiple(28, 14), 28); // already a multiple
        assert_eq!(nearest_multiple(30, 14), 28); // closer to down
        assert_eq!(nearest_multiple(37, 14), 42); // closer to up
    }

    #[test]
    fn sat_u8_clamps() {
        assert_eq!(sat_u8(-5.0), 0);
        assert_eq!(sat_u8(300.0), 255);
        assert_eq!(sat_u8(127.5), 128); // ties to even
        assert_eq!(sat_u8(128.5), 128); // ties to even
    }
}
