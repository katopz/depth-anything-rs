//! K-quant dequantization (q4_K / q5_K / q6_K / q8_K) to f32.
//!
//! This is a direct port of ggml's `dequantize_row_q{n}_K` in
//! `third_party/ggml/src/ggml-quants.c`. We only need the *dequant* direction:
//! the Rust engine works entirely in f32, so quantized weights are expanded to
//! f32 at load time (see [`crate::gguf::GgufFile::tensor_f32`]).
//!
//! All blocks span `QK_K = 256` elements. The byte layouts match the
//! `block_q{n}_K` structs in `ggml-common.h`:
//!
//! | dtype | layout (bytes)                                        | total |
//! |-------|--------------------------------------------------------|-------|
//! | q4_K  | `d:f16, dmin:f16, scales[12], qs[128]`                |  144  |
//! | q5_K  | `d:f16, dmin:f16, scales[12], qh[32], qs[128]`        |  176  |
//! | q6_K  | `ql[128], qh[64], scales[16]:i8, d:f16`               |  210  |
//! | q8_K  | `d:f32, qs[256]:i8, bsums[16]:i16`                    |  292  |
//!
//! `scales` for q4_K/q5_K pack 8 (scale, min) pairs into 12 bytes via
//! [`get_scale_min_k4`]; q6_K stores them as raw int8; q8_K has no per-block
//! scales (just the f32 `d`).

// This module is a faithful port of ggml's `dequantize_row_q{n}_K`. The bit
// arithmetic and loop structure deliberately mirror the C source so the two
// can be diffed side-by-side; we therefore silence two clippy lints that would
// otherwise push the code away from that 1:1 correspondence:
//   - `identity_op`: the C source writes `>> 0`, `+ 0`, `is + 0` etc. to make
//     the per-quant scale/bias symmetry explicit (q1..q4 use offsets 0/2/4/6).
//   - `needless_range_loop`: the C source uses indexed `for (int l = 0; l < 32; ++l)`
//     loops over interleaved ql/qh arrays; iterator rewriting would obscure
//     the offset math.
#![allow(clippy::identity_op, clippy::needless_range_loop)]

use half::f16;

/// Elements per K-quant super-block.
pub const QK_K: usize = 256;
/// Number of bytes used to pack the 8×(6-bit scale, 6-bit min) pairs in q4_K/q5_K.
const K_SCALE_SIZE: usize = 12;

/// Decode the `j`-th (scale, min) pair from a q4_K/q5_K `scales[12]` array.
///
/// Ported verbatim from `get_scale_min_k4` in `ggml-quants.c`. The first 4
/// pairs are stored directly (6 bits each in `scales[0..4]` and `scales[4..8]`);
/// the remaining 4 pairs steal their high bits from the top 2 bits of
/// `scales[0..4]`.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// q4_K block: `d:f16, dmin:f16, scales[12], qs[128]`.
const Q4_K_BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2;

/// Dequantize `n_blocks` contiguous q4_K blocks into `out`.
///
/// Each block produces 256 f32 values: 8 sub-blocks of 32, each with its own
/// `(d·sc, dmin·m)` affine transform applied to 4-bit quants (low nibble for
/// the first 32 of each pair, high nibble for the second).
pub fn dequantize_q4_k(blocks: &[u8], n_blocks: usize, out: &mut [f32]) {
    for b in 0..n_blocks {
        let base = b * Q4_K_BLOCK_BYTES;
        let d = f16::from_le_bytes([blocks[base], blocks[base + 1]]).to_f32();
        let min = f16::from_le_bytes([blocks[base + 2], blocks[base + 3]]).to_f32();
        let scales = &blocks[base + 4..base + 4 + K_SCALE_SIZE];
        let qs = &blocks[base + 4 + K_SCALE_SIZE..base + Q4_K_BLOCK_BYTES];

        let out_base = b * QK_K;
        let mut is = 0; // sub-block index for scale lookup
        let mut q_off = 0; // offset into qs; advances by 32 each pair of sub-blocks
        for j in (0..QK_K).step_by(64) {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;

            let q = &qs[q_off..q_off + 32];
            for l in 0..32 {
                out[out_base + j + l] = d1 * (q[l] & 0x0F) as f32 - m1;
            }
            for l in 0..32 {
                out[out_base + j + 32 + l] = d2 * (q[l] >> 4) as f32 - m2;
            }
            q_off += 32;
            is += 2;
        }
    }
}

/// q5_K block: `d:f16, dmin:f16, scales[12], qh[32], qs[128]`.
const Q5_K_BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;

/// Dequantize `n_blocks` contiguous q5_K blocks into `out`.
///
/// Same sub-block/scale scheme as q4_K, but each quant is 5 bits: the low 4
/// bits come from `qs` and the high bit from `qh`. The high bit for element `l`
/// of sub-block group `g` (g = 0..4) is bit `2g` (low half) or `2g+1` (high
/// half) of `qh[l]`.
pub fn dequantize_q5_k(blocks: &[u8], n_blocks: usize, out: &mut [f32]) {
    let qh_off = 4 + K_SCALE_SIZE;
    let qs_off = qh_off + QK_K / 8;
    for b in 0..n_blocks {
        let base = b * Q5_K_BLOCK_BYTES;
        let d = f16::from_le_bytes([blocks[base], blocks[base + 1]]).to_f32();
        let min = f16::from_le_bytes([blocks[base + 2], blocks[base + 3]]).to_f32();
        let scales = &blocks[base + 4..base + 4 + K_SCALE_SIZE];
        let qh = &blocks[base + qh_off..base + qh_off + QK_K / 8];
        let qs = &blocks[base + qs_off..base + Q5_K_BLOCK_BYTES];

        let out_base = b * QK_K;
        let mut is = 0;
        let mut q_off = 0;
        // u1/u2 select which bit of qh[l] contributes the +16 term for the
        // low-/high-nibble half of each 64-element group; they shift left by 2
        // each iteration (one bit per half, two halves per group).
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..QK_K).step_by(64) {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;

            let q = &qs[q_off..q_off + 32];
            for l in 0..32 {
                let h = if qh[l] & u1 != 0 { 16 } else { 0 };
                out[out_base + j + l] = d1 * ((q[l] & 0x0F) as f32 + h as f32) - m1;
            }
            for l in 0..32 {
                let h = if qh[l] & u2 != 0 { 16 } else { 0 };
                out[out_base + j + 32 + l] = d2 * ((q[l] >> 4) as f32 + h as f32) - m2;
            }
            q_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// q6_K block: `ql[128], qh[64], scales[16]:i8, d:f16`.
const Q6_K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;

/// Dequantize `n_blocks` contiguous q6_K blocks into `out`.
///
/// 6-bit quants: low 4 bits from `ql`, high 2 bits from `qh`, biased by -32 to
/// land in `[-32, 31]`. 16 per-block int8 scales; element `l` picks scale
/// `is = l/16` (within the 8-scale window active for each 128-element chunk).
pub fn dequantize_q6_k(blocks: &[u8], n_blocks: usize, out: &mut [f32]) {
    let ql_off = 0;
    let qh_off = QK_K / 2;
    let sc_off = QK_K / 2 + QK_K / 4;
    let d_off = QK_K / 2 + QK_K / 4 + QK_K / 16;
    for b in 0..n_blocks {
        let base = b * Q6_K_BLOCK_BYTES;
        let d = f16::from_le_bytes([blocks[base + d_off], blocks[base + d_off + 1]]).to_f32();
        let ql = &blocks[base + ql_off..base + ql_off + QK_K / 2];
        let qh = &blocks[base + qh_off..base + qh_off + QK_K / 4];
        let sc = &blocks[base + sc_off..base + sc_off + QK_K / 16];

        let out_base = b * QK_K;
        let mut ql_off_n = 0; // advances by 64 per 128-element chunk
        let mut qh_off_n = 0; // advances by 32
        let mut sc_off_n = 0; // advances by 8
        for n in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                // Each quant is 6 bits in [0,63]; subtract 32 → [-32,31].
                let q1 = (((ql[ql_off_n + l] & 0x0F) as i32)
                    | (((qh[qh_off_n + l] >> 0) & 3) as i32) << 4)
                    - 32;
                let q2 = (((ql[ql_off_n + l + 32] & 0x0F) as i32)
                    | (((qh[qh_off_n + l] >> 2) & 3) as i32) << 4)
                    - 32;
                let q3 = (((ql[ql_off_n + l] >> 4) as i32)
                    | (((qh[qh_off_n + l] >> 4) & 3) as i32) << 4)
                    - 32;
                let q4 = (((ql[ql_off_n + l + 32] >> 4) as i32)
                    | (((qh[qh_off_n + l] >> 6) & 3) as i32) << 4)
                    - 32;
                out[out_base + n + l + 0] = d * (sc[sc_off_n + is + 0] as i8 as f32) * q1 as f32;
                out[out_base + n + l + 32] = d * (sc[sc_off_n + is + 2] as i8 as f32) * q2 as f32;
                out[out_base + n + l + 64] = d * (sc[sc_off_n + is + 4] as i8 as f32) * q3 as f32;
                out[out_base + n + l + 96] = d * (sc[sc_off_n + is + 6] as i8 as f32) * q4 as f32;
            }
            ql_off_n += 64;
            qh_off_n += 32;
            sc_off_n += 8;
        }
    }
}

/// q8_K block: `d:f32, qs[256]:i8, bsums[16]:i16`.
const Q8_K_BLOCK_BYTES: usize = 4 + QK_K + QK_K / 16 * 2;

/// Dequantize `n_blocks` contiguous q8_K blocks into `out`.
///
/// q8_K is normally only an intermediate dot-product type, but if it appears in
/// a GGUF the dequant is just `d * qs[j]` (the `bsums` field is not needed).
pub fn dequantize_q8_k(blocks: &[u8], n_blocks: usize, out: &mut [f32]) {
    for b in 0..n_blocks {
        let base = b * Q8_K_BLOCK_BYTES;
        let d = f32::from_le_bytes([
            blocks[base],
            blocks[base + 1],
            blocks[base + 2],
            blocks[base + 3],
        ]);
        let qs = &blocks[base + 4..base + 4 + QK_K];
        let out_base = b * QK_K;
        for j in 0..QK_K {
            out[out_base + j] = d * (qs[j] as i8 as f32);
        }
    }
}

/// Byte size of one block of the given K-quant dtype.
pub fn block_bytes(dtype: crate::gguf::GgufDType) -> usize {
    match dtype {
        crate::gguf::GgufDType::Q4_K => Q4_K_BLOCK_BYTES,
        crate::gguf::GgufDType::Q5_K => Q5_K_BLOCK_BYTES,
        crate::gguf::GgufDType::Q6_K => Q6_K_BLOCK_BYTES,
        crate::gguf::GgufDType::Q8_K => Q8_K_BLOCK_BYTES,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a q4_K block with known, hand-computable values.
    /// We zero the scales/mins except for sub-block 0, set d=2, dmin=1, and
    /// put distinct 4-bit quants in qs so we can predict every output exactly.
    fn make_q4_block() -> Vec<u8> {
        let mut blk = vec![0u8; Q4_K_BLOCK_BYTES];
        // d = 2.0, dmin = 1.0
        blk[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        blk[2..4].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        // scales[0..4] hold scale6 for sub-blocks 0..3; scales[4..8] hold min6.
        // Sub-block 0: scale = 3, min = 5. (fits in 6 bits, j<4 direct path)
        blk[4] = 3;
        blk[8] = 5;
        // qs[0..32]: low nibble = element value. Use 0,1,2,...,31 mod 16.
        for i in 0..32 {
            blk[16 + i] = (i % 16) as u8; // low nibble = i%16, high nibble = 0
        }
        blk
    }

    #[test]
    fn q4_k_known_block() {
        let blk = make_q4_block();
        let mut out = vec![0.0f32; QK_K];
        dequantize_q4_k(&blk, 1, &mut out);
        // Sub-block 0: d1 = d*scale = 2*3 = 6, m1 = dmin*min = 1*5 = 5.
        // out[l] = 6*(l%16) - 5, for l in 0..32.
        for l in 0..32 {
            let want = 6.0 * (l % 16) as f32 - 5.0;
            assert!(
                (out[l] - want).abs() < 1e-6,
                "q4_k out[{l}] = {} want {want}",
                out[l]
            );
        }
        // Sub-blocks 1..7 have scale=0,min=0 → output = 0*quant - 0 = 0.
        for l in 32..QK_K {
            assert!(out[l].abs() < 1e-6, "q4_k out[{l}] = {} want 0", out[l]);
        }
    }

    #[test]
    fn q4_k_block_byte_size() {
        // d(2) + dmin(2) + scales(12) + qs(128) = 144
        assert_eq!(Q4_K_BLOCK_BYTES, 144);
    }

    #[test]
    fn q5_k_block_byte_size() {
        // d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176
        assert_eq!(Q5_K_BLOCK_BYTES, 176);
    }

    #[test]
    fn q6_k_block_byte_size() {
        // ql(128) + qh(64) + scales(16) + d(2) = 210
        assert_eq!(Q6_K_BLOCK_BYTES, 210);
    }

    #[test]
    fn q8_k_block_byte_size() {
        // d(4) + qs(256) + bsums(32) = 292
        assert_eq!(Q8_K_BLOCK_BYTES, 292);
    }

    /// get_scale_min_k4: j<4 reads the low 6 bits of scales[j] / scales[j+4].
    #[test]
    fn scale_min_k4_direct_path() {
        let mut s = [0u8; K_SCALE_SIZE];
        s[0] = 0b101011; // 43
        s[4] = 0b110001; // 49
        let (d, m) = get_scale_min_k4(0, &s);
        assert_eq!(d, 43);
        assert_eq!(m, 49);
    }

    /// get_scale_min_k4: j>=4 reassembles 6 bits from scales[j+4] low nibble
    /// + top 2 bits of scales[j-4].
    #[test]
    fn scale_min_k4_packed_path() {
        let mut s = [0u8; K_SCALE_SIZE];
        // j = 4: d = (s[8] & 0xF) | ((s[0] >> 6) << 4)
        //            = 0b1011 | (0b11 << 4) = 0b111011 = 59
        //        m = (s[8] >> 4) | ((s[4] >> 6) << 4)
        //            = 0b0010 | (0b10 << 4) = 0b100010 = 34
        s[0] = 0b11_000000; // top 2 bits = 0b11
        s[4] = 0b10_000000; // top 2 bits = 0b10
        s[8] = 0b0010_1011; // low nibble 0b1011, high nibble 0b0010
        let (d, m) = get_scale_min_k4(4, &s);
        assert_eq!(d, 0b111011);
        assert_eq!(m, 0b100010);
    }

    /// q8_K is trivial: d * qs[j]. Verify with d=0.5 and qs = -2,..,-1,0,1,2,...
    #[test]
    fn q8_k_trivial() {
        let mut blk = vec![0u8; Q8_K_BLOCK_BYTES];
        blk[0..4].copy_from_slice(&0.5f32.to_le_bytes());
        let qs = &mut blk[4..4 + QK_K];
        for j in 0..QK_K {
            // ramp from -128..127 cyclically isn't possible in i8 range cleanly;
            // use j as i8 via wrapping cast by setting raw byte = j mod 256.
            qs[j] = j as u8;
        }
        let mut out = vec![0.0f32; QK_K];
        dequantize_q8_k(&blk, 1, &mut out);
        for j in 0..QK_K {
            let want = 0.5 * (j as i8 as f32);
            assert!(
                (out[j] - want).abs() < 1e-6,
                "q8_k out[{j}] = {} want {want}",
                out[j]
            );
        }
    }

    /// q6_K: set d=1, one scale window, quants to a known value.
    #[test]
    fn q6_k_known_block() {
        let mut blk = vec![0u8; Q6_K_BLOCK_BYTES];
        // d = 1.0 (at offset 208)
        let d_off = QK_K / 2 + QK_K / 4 + QK_K / 16;
        blk[d_off..d_off + 2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        // scales[0..16] = all 2
        for i in 0..QK_K / 16 {
            blk[d_off - QK_K / 16 + i] = 2; // int8 scale = 2
        }
        // Set ql[0]=0x12, qh[0]=0x10 so q1 = (0x2 | (0b00<<4)) - 32 = 2-32 = -30
        // (qh[0] bits [1:0] = 0b00)
        blk[0] = 0x12; // low nibble 2
        let qh_off = QK_K / 2;
        blk[qh_off] = 0x10;
        let mut out = vec![0.0f32; QK_K];
        dequantize_q6_k(&blk, 1, &mut out);
        // out[0] = d * sc[0] * q1 = 1 * 2 * (-30) = -60
        assert!(
            (out[0] - (-60.0)).abs() < 1e-6,
            "q6_k out[0] = {} want -60",
            out[0]
        );
    }
}
