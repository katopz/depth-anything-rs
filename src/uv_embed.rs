//! DPT-head UV positional embedding, matching `src/uv_posembed.cpp`.
//!
//! For a feature map of size `(pw, ph)` (width × height) and an even channel
//! count `C`, this returns a flat `ph*pw*C` vector in **(H, W, C)** row-major
//! order: element `(y, x, c)` lives at index `(y*pw + x)*C + c`. Channels
//! `[0, C/2)` encode the **x** coordinate and `[C/2, C)` the **y** coordinate,
//! each as `[sin(F), cos(F)]` with `F = C/4` sinusoid frequencies over a
//! normalized, aspect-ratio-aware box. This is the raw embedding; the `× 0.1`
//! scale used by the decoder is applied at cache-fill time by the caller.
//!
//! The README calls this out as the ~90 ms single-threaded hot path at full
//! resolution, so it is cached per `(W, H, C, aspect, ratio)`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type Key = (u32, u32, u32, u32, u32); // (W, H, C, aspect_bits, ratio_bits)

static UV_CACHE: Mutex<Option<HashMap<Key, Arc<Vec<f32>>>>> = Mutex::new(None);

/// Compute the raw UV positional embedding for a `(pw, ph, C)` feature map,
/// given the aspect ratio `aspect = W / H` of the **original image**.
///
/// Returns `(H, W, C)` row-major floats (NOT pre-scaled by `ratio`).
pub fn uv_pos_embed(pw: usize, ph: usize, c: usize, aspect: f32) -> Vec<f32> {
    assert!(c % 2 == 0, "UV embed channel count must be even, got {c}");
    let d = c / 2; // per-axis depth (e.g. 32 for C=64)
    let f = d / 2; // frequencies per axis (e.g. 16)

    let aspect = aspect as f64;
    let diag = (aspect * aspect + 1.0).sqrt();
    let span_x = aspect / diag;
    let span_y = 1.0 / diag;
    let left_x = -span_x * (pw as f64 - 1.0) / pw as f64;
    let right_x = span_x * (pw as f64 - 1.0) / pw as f64;
    let top_y = -span_y * (ph as f64 - 1.0) / ph as f64;
    let bottom_y = span_y * (ph as f64 - 1.0) / ph as f64;

    // Linspace x and y coords.
    let x_coords: Vec<f64> = (0..pw)
        .map(|i| {
            if pw == 1 {
                left_x
            } else {
                left_x + i as f64 * (right_x - left_x) / (pw as f64 - 1.0)
            }
        })
        .collect();
    let y_coords: Vec<f64> = (0..ph)
        .map(|j| {
            if ph == 1 {
                top_y
            } else {
                top_y + j as f64 * (bottom_y - top_y) / (ph as f64 - 1.0)
            }
        })
        .collect();

    let omega0: f64 = 100.0;
    let omega: Vec<f64> = (0..f)
        .map(|j| {
            let e = j as f64 / (d as f64 / 2.0); // == j / f
            1.0 / omega0.powf(e)
        })
        .collect();

    let mut out = vec![0.0f32; ph * pw * c];
    for (y, yc) in y_coords.iter().enumerate() {
        for (x, xc) in x_coords.iter().enumerate() {
            let base = (y * pw + x) * c;
            // X half: channels [0, D) = [sin(F), cos(F)]
            for j in 0..f {
                let o = xc * omega[j];
                out[base + j] = o.sin() as f32;
                out[base + f + j] = o.cos() as f32;
            }
            // Y half: channels [D, 2D=C) = [sin(F), cos(F)]
            for j in 0..f {
                let o = yc * omega[j];
                out[base + d + j] = o.sin() as f32;
                out[base + d + f + j] = o.cos() as f32;
            }
        }
    }
    out
}

/// Cached, pre-scaled UV embedding laid out as `(C, H, W)` for direct use as
/// a candle `[1, C, H, W]` input added to a feature map.
///
/// Returns an `Arc<Vec<f32>>` so callers that only need `&[f32]` (e.g. the
/// fast DPT head's parallel `+=` add) pay just an atomic increment instead of
/// deep-copying what can be a 40+ MiB buffer at full resolution. Callers that
/// require ownership (e.g. `Tensor::from_vec`) can call `(*arc).clone()` or
/// `arc.to_vec()`.
///
/// `ratio` (the decoder's `× 0.1` scale) is folded into the cached buffer.
/// Cache key includes the raw IEEE-754 bit patterns of `aspect` and `ratio`
/// so equal floats collide exactly and NaN-keyed entries are impossible.
pub fn uv_embed_chw_cached(w: usize, h: usize, c: usize, aspect: f32, ratio: f32) -> Arc<Vec<f32>> {
    let key: Key = (
        w as u32,
        h as u32,
        c as u32,
        aspect.to_bits(),
        ratio.to_bits(),
    );
    let mut guard = UV_CACHE.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(v) = map.get(&key) {
        return Arc::clone(v);
    }

    // Raw embedding is (H, W, C). Transpose to (C, H, W) and scale by ratio.
    let uv = uv_pos_embed(w, h, c, aspect);
    let mut buf = vec![0.0f32; w * h * c];
    for ch in 0..c {
        for hh in 0..h {
            for ww in 0..w {
                buf[ch * h * w + hh * w + ww] = ratio * uv[(hh * w + ww) * c + ch];
            }
        }
    }
    let arc = Arc::new(buf);
    map.insert(key, Arc::clone(&arc));
    arc
}

/// Clear the global UV-embed cache.
pub fn clear_cache() {
    *UV_CACHE.lock().unwrap() = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uv_shape_and_layout() {
        let pw = 4;
        let ph = 3;
        let c = 8; // d=4, f=2
        let uv = uv_pos_embed(pw, ph, c, 1.0);
        assert_eq!(uv.len(), ph * pw * c);
        // Channels 0..f are sin(x), f..d are cos(x).
        // At x=0 (aspect-symmetric), sin=0 → out[base+0..f] all 0.
        // With aspect=1, span_x = 1/sqrt(2); x_coord[0] = -span*(pw-1)/pw != 0,
        // so we only sanity-check the layout: each pixel's block is independent.
        // Compute flat offsets in the (y, x, c) layout: base = (y * pw + x) * c.
        // Pixel (0, 0) and pixel (1, 0) share the same x → x-half channels match.
        let base_a = 0; // pixel (0, 0)
        let base_b = pw * c; // pixel (1, 0)
        for j in 0..c / 2 {
            assert!(
                (uv[base_a + j] - uv[base_b + j]).abs() < 1e-5,
                "x-half must match same column"
            );
        }
    }
}
