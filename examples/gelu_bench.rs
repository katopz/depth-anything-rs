//! Micro-benchmark for the FFN GELU-erf activation at the DA3-BASE shape.
//!
//! Measures the AVX2 batch kernel (`gelu_erf_slice`) vs a scalar reference
//! loop, so the speedup from the bit-manipulation `exp` + branchless erf is
//! visible in isolation (without the surrounding GEMM noise).
//!
//! ```sh
//! RAYON_NUM_THREADS=24 cargo run --release --example gelu_bench
//! ```

use depth_anything::fast_block;
use rayon::prelude::*;
use std::time::Instant;

fn bench_scalar(label: &str, x: &mut Vec<f32>, hidden: usize, iters: usize) -> f64 {
    let n = x.len() / hidden;
    // Warmup.
    for _ in 0..3 {
        x.par_chunks_mut(hidden).for_each(|row| {
            for v in row.iter_mut() {
                *v = fast_block::gelu_erf(*v);
            }
        });
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        x.par_chunks_mut(hidden).for_each(|row| {
            for v in row.iter_mut() {
                *v = fast_block::gelu_erf(*v);
            }
        });
        // keep `n` referenced so the optimiser can't drop the loop.
        black_box(&n);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per = elapsed * 1000.0 / iters as f64;
    eprintln!("  {label:<28} : {per:>8.3} ms");
    per
}

fn bench_avx2(label: &str, x: &mut Vec<f32>, hidden: usize, iters: usize) -> f64 {
    let n = x.len() / hidden;
    // Warmup.
    for _ in 0..3 {
        x.par_chunks_mut(hidden).for_each(|row| {
            fast_block::gelu_erf_slice(row);
        });
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        x.par_chunks_mut(hidden).for_each(|row| {
            fast_block::gelu_erf_slice(row);
        });
        black_box(&n);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per = elapsed * 1000.0 / iters as f64;
    eprintln!("  {label:<28} : {per:>8.3} ms");
    per
}

#[inline(always)]
fn black_box<T>(x: &T) {
    unsafe {
        std::ptr::read_volatile(x as *const T);
    }
}

fn main() {
    eprintln!(
        "GELU-erf bench (RAYON_NUM_THREADS={})",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into())
    );

    // DA3-BASE FFN shapes at 504×336 input: n=865, hidden=3072 (×12 blocks).
    let n = 865_usize;
    let hidden = 3072_usize;
    let blocks = 12_usize;

    // Random-ish input in [-6, 6] (post-layernorm activation range).
    let mut rng: u64 = 0x1234_5678_9abc_def0;
    let mut make = || {
        let mut r = rng;
        let v: Vec<f32> = (0..n * hidden)
            .map(|_| {
                r ^= r << 13;
                r ^= r >> 7;
                r ^= r << 17;
                ((r as f32) / (u32::MAX as f32) - 0.5) * 12.0
            })
            .collect();
        rng = r;
        v
    };

    let iters = 200;

    eprintln!();
    eprintln!("--- single block (n={n}, hidden={hidden}, parallel) ---");
    let mut x_scalar = make();
    let mut x_avx2 = x_scalar.clone();
    let s1 = bench_scalar("scalar gelu_erf loop", &mut x_scalar, hidden, iters);
    let a1 = bench_avx2("AVX2 gelu_erf_slice", &mut x_avx2, hidden, iters);
    eprintln!("  speedup                       : {:.2}×", s1 / a1);

    eprintln!();
    eprintln!("--- full backbone ({blocks} blocks, serial ×{blocks}) ---");
    // The real FFN runs the GELU once per block; multiply the single-block
    // time by 12 to get the backbone contribution. (Each bench iter already
    // processes the full n×hidden slab once; the ×12 factor reflects the
    // backbone loop count.)
    let backbone_scalar_ms = s1 * blocks as f64;
    let backbone_avx2_ms = a1 * blocks as f64;
    eprintln!(
        "  scalar ×{blocks}                  : {:.3} ms",
        backbone_scalar_ms
    );
    eprintln!(
        "  AVX2 ×{blocks}                    : {:.3} ms",
        backbone_avx2_ms
    );
    eprintln!(
        "  per-forward savings           : {:.2} ms",
        backbone_scalar_ms - backbone_avx2_ms
    );
}
