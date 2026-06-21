//! Backbone positional-embedding bicubic interpolation, matching
//! `src/dino_backbone.cpp::interp_pos_embed`.
//!
//! The stored `vit.pos_embed` tensor has `1 + M*M` rows (CLS + an `M×M` patch
//! grid), each of width `embed_dim`. At inference the target patch grid is
//! `gh × gw`, generally different from `M × M`, so the patch rows are
//! resampled with a Catmull–Rom bicubic kernel (`a = -0.75`, matching
//! PyTorch's `bicubic` + `align_corners=False`). The CLS row (row 0) is
//! copied verbatim.
//!
//! This is input-independent, so it is cached per `(model, gh, gw)`.

use std::collections::HashMap;
use std::sync::Mutex;

/// Cache key: (pos_embed data pointer as u64, gh, gw). Mirrors the C++ cache.
type Key = (usize, usize, usize);

static PE_CACHE: Mutex<Option<HashMap<Key, Vec<f32>>>> = Mutex::new(None);

/// Catmull–Rom cubic kernel with `a = -0.75` (matches PyTorch `bicubic`).
fn cubic(x: f32) -> f32 {
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

/// Interpolate `pos_embed` (row-major `[1 + M*M, embed]`) to a `gh × gw`
/// target patch grid, returning a flat `[1 + gh*gw, embed]` row-major vector
/// (CLS row first, then patch rows in row-major `(row, col)` order).
///
/// `interp_offset` is added to the target grid dims before computing the
/// scale (DA3 default 0.1).
pub fn interp_pos_embed(
    pos_embed: &[f32],
    embed: usize,
    m_grid: usize,
    gh: usize,
    gw: usize,
    interp_offset: f32,
) -> Vec<f32> {
    let n_out = 1 + gh * gw;
    let mut out = vec![0.0f32; n_out * embed];

    // CLS row (row 0) is copied verbatim.
    out[..embed].copy_from_slice(&pos_embed[..embed]);

    let sx = (gw as f32 + interp_offset) / m_grid as f32;
    let sy = (gh as f32 + interp_offset) / m_grid as f32;

    for oy in 0..gh {
        let iy = (oy as f32 + 0.5) / sy - 0.5;
        let y0 = iy.floor() as i32;
        let fy = iy - y0 as f32;
        for ox in 0..gw {
            let ix = (ox as f32 + 0.5) / sx - 0.5;
            let x0 = ix.floor() as i32;
            let fx = ix - x0 as f32;

            // Output row index: 1 + oy*gw + ox (skip the CLS row).
            let orow = 1 + oy * gw + ox;
            let wy: [f32; 4] = [
                cubic(fy - (-1.0)),
                cubic(fy - 0.0),
                cubic(fy - 1.0),
                cubic(fy - 2.0),
            ];
            let wx: [f32; 4] = [
                cubic(fx - (-1.0)),
                cubic(fx - 0.0),
                cubic(fx - 1.0),
                cubic(fx - 2.0),
            ];
            for ch in 0..embed {
                let mut acc = 0.0f32;
                for (mi, &w_y) in wy.iter().enumerate().take(4) {
                    if w_y == 0.0 {
                        continue;
                    }
                    let yy = clamp(y0 + mi as i32 - 1, 0, m_grid as i32 - 1) as usize;
                    for (ni, &w_x) in wx.iter().enumerate().take(4) {
                        if w_x == 0.0 {
                            continue;
                        }
                        let xx = clamp(x0 + ni as i32 - 1, 0, m_grid as i32 - 1) as usize;
                        // Source row 1+yy*M+xx (skip CLS), column ch.
                        let src = pos_embed[(1 + yy * m_grid + xx) * embed + ch];
                        acc += w_y * w_x * src;
                    }
                }
                out[orow * embed + ch] = acc;
            }
        }
    }
    out
}

/// Cached version of [`interp_pos_embed`]. The cache is keyed on the
/// `pos_embed` data address plus `(gh, gw)` — identical inputs across
/// forwards at the same resolution therefore hit the cache.
pub fn interp_pos_embed_cached(
    pos_embed: &[f32],
    embed: usize,
    m_grid: usize,
    gh: usize,
    gw: usize,
    interp_offset: f32,
) -> Vec<f32> {
    let key: Key = (pos_embed.as_ptr() as usize, gh, gw);
    let mut guard = PE_CACHE.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(v) = map.get(&key) {
        return v.clone();
    }
    let v = interp_pos_embed(pos_embed, embed, m_grid, gh, gw, interp_offset);
    map.insert(key, v.clone());
    v
}

/// Clear the global pos-embed cache. Useful when a model is unloaded.
pub fn clear_cache() {
    *PE_CACHE.lock().unwrap() = None;
}

#[inline]
fn clamp(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_kernel_matches_spec() {
        // cubic(0) = 1, cubic(±1) = 0, cubic(±2) = 0.
        assert!((cubic(0.0) - 1.0).abs() < 1e-6);
        assert!(cubic(1.0).abs() < 1e-6);
        assert!(cubic(2.0).abs() < 1e-6);
        assert!(cubic(-1.0).abs() < 1e-6);
    }

    #[test]
    fn identity_interp_when_grid_matches() {
        // If gh=gw=M, the interpolated patch rows should equal the source rows
        // (a bicubic with a delta-aligned integer grid is exact).
        let embed = 4;
        let m = 3;
        let mut pe = vec![0.0f32; (1 + m * m) * embed];
        for (i, v) in pe.iter_mut().enumerate() {
            *v = i as f32 * 0.1;
        }
        let out = interp_pos_embed(&pe, embed, m, m, m, 0.1);
        // CLS row matches exactly.
        for ch in 0..embed {
            assert!((out[ch] - pe[ch]).abs() < 1e-4, "CLS row mismatch at {ch}");
        }
        // Note: with interp_offset=0.1 the grid is slightly scaled, so the
        // patch rows are NOT exactly the source rows. Re-run with offset 0 for
        // an exact identity check.
        let out0 = interp_pos_embed(&pe, embed, m, m, m, 0.0);
        for i in 0..(1 + m * m) * embed {
            assert!((out0[i] - pe[i]).abs() < 1e-4, "identity mismatch at {i}");
        }
    }
}
