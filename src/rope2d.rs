//! 2D rotary position embeddings (RoPE), matching `src/rope2d.cpp` exactly.
//!
//! The head dimension is split into two halves: the first half rotates by the
//! **y** patch coordinate, the second half by the **x** coordinate. Within
//! each half, `rotate_half` produces `[-h[quart:], h[:quart]]`, where
//! `half = head_dim/2` and `quart = half/2`.
//!
//! Two position sets are built per forward pass:
//! - **local**: each patch token's coord is its `(row+1, col+1)` in the grid.
//! - **nodiff**: every patch token gets `(1, 1)` (a constant-angle rotation).
//!
//! The backbone selects the table per block: local blocks use `local`;
//! "global" blocks (odd-indexed, `i >= alt_start`) use `nodiff`.

/// One RoPE table (cos and sin), shape `[tokens, head_dim]`, row-major.
pub struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

/// Build the cos/sin tables for a given set of `(y, x)` positions.
///
/// - `pos_yx`: flat `[y0, x0, y1, x1, ...]` (length `2 * tokens`).
/// - `head_dim`: full attention head dim (e.g. 64). Must be divisible by 4.
/// - `freq`: RoPE base (theta). Default 100.
pub fn build_rope_tables(pos_yx: &[f32], head_dim: usize, freq: f32) -> RopeTable {
    assert!(
        head_dim % 4 == 0,
        "head_dim must be divisible by 4 for 2D RoPE, got {head_dim}"
    );
    let tokens = pos_yx.len() / 2;
    let half = head_dim / 2;
    let quart = half / 2;
    let mut cos = vec![0.0f32; tokens * head_dim];
    let mut sin = vec![0.0f32; tokens * head_dim];

    for t in 0..tokens {
        let y = pos_yx[t * 2] as f64;
        let x = pos_yx[t * 2 + 1] as f64;
        for j in 0..quart {
            // theta_j = freq ^ (-2j / half).  Note the denominator is `half`,
            // not `head_dim`.
            let invf = (freq as f64).powf(-2.0 * j as f64 / half as f64);
            let ay = y * invf;
            let ax = x * invf;
            let cy = ay.cos() as f32;
            let sy = ay.sin() as f32;
            let cx = ax.cos() as f32;
            let sx = ax.sin() as f32;
            // y-block [0, half): pair j and j+quart share theta_j.
            cos[t * head_dim + j] = cy;
            cos[t * head_dim + j + quart] = cy;
            sin[t * head_dim + j] = sy;
            sin[t * head_dim + j + quart] = sy;
            // x-block [half, head_dim): pair j and j+quart share theta_j.
            cos[t * head_dim + half + j] = cx;
            cos[t * head_dim + half + j + quart] = cx;
            sin[t * head_dim + half + j] = sx;
            sin[t * head_dim + half + j + quart] = sx;
        }
    }
    RopeTable { cos, sin }
}

/// Build the two position sets used by the DA3 backbone.
///
/// Token 0 (CLS/camera) is `(0, 0)` in both sets (no rotation). Patch tokens
/// are `(row+1, col+1)` for the local set and `(1, 1)` for the nodiff set.
/// Returns `(local_table, nodiff_table)`.
pub fn build_backbone_tables(
    n_tokens: usize,
    grid_w: usize,
    head_dim: usize,
    freq: f32,
) -> (RopeTable, RopeTable) {
    let mut pos_local = vec![0.0f32; 2 * n_tokens];
    let mut pos_nodiff = vec![0.0f32; 2 * n_tokens];
    for t in 1..n_tokens {
        let idx = t - 1;
        let row = idx / grid_w;
        let col = idx % grid_w;
        pos_local[t * 2] = (row + 1) as f32;
        pos_local[t * 2 + 1] = (col + 1) as f32;
        pos_nodiff[t * 2] = 1.0;
        pos_nodiff[t * 2 + 1] = 1.0;
    }
    (
        build_rope_tables(&pos_local, head_dim, freq),
        build_rope_tables(&pos_nodiff, head_dim, freq),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_have_correct_shape() {
        let (local, nodiff) = build_backbone_tables(1 + 4 * 4, 4, 64, 100.0);
        assert_eq!(local.cos.len(), (1 + 16) * 64);
        assert_eq!(local.sin.len(), (1 + 16) * 64);
        // Token 0 (CLS) is all-zero positions → cos=1, sin=0.
        for d in 0..64 {
            assert!((local.cos[d] - 1.0).abs() < 1e-5, "CLS cos not 1 at {d}");
            assert!(local.sin[d].abs() < 1e-5, "CLS sin not 0 at {d}");
        }
        // nodiff: every patch token gets (1,1) so rows 1.. all equal.
        for t in 1..17 {
            for d in 0..64 {
                assert!((nodiff.cos[t * 64 + d] - nodiff.cos[64 + d]).abs() < 1e-6);
            }
        }
    }
}
