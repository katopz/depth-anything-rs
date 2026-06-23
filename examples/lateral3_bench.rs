//! Microbenchmark: measure the lateral 3 conv3x3_stride2 GEMM shape (768×216 @ 6912)
//! with different DA_TINYBLAS_AXIS settings, to validate K-blocking.
//!
//! Run with:
//!   RAYON_NUM_THREADS=32 cargo run --release --example lateral3_bench

use depth_anything::tinyblas;
use std::time::Instant;

fn bench(label: &str, m: usize, n: usize, k: usize, iters: usize) {
    let a = vec![0.123f32; m * k];
    let b = vec![0.456f32; k * n];
    let mut c = vec![0.0f32; m * n];
    // Warmup
    for _ in 0..3 {
        tinyblas::gemm_nn_into(m, n, k, &a, &b, &mut c);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        tinyblas::gemm_nn_into(m, n, k, &a, &b, &mut c);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let flops = 2.0 * (m * n * k) as f64 * iters as f64;
    let gflops = flops / elapsed / 1e9;
    let ms = elapsed * 1000.0 / iters as f64;
    eprintln!(
        "  {:<24} [{}×{}] @ [{}×{}]  -> {:>7.3} ms  {:>7.1} GFLOP/s",
        label, m, k, k, n, ms, gflops
    );
}

fn main() {
    let axis = std::env::var("DA_TINYBLAS_AXIS").unwrap_or_else(|_| "auto".into());
    let kc = std::env::var("DA_TINYBLAS_KC").unwrap_or_else(|_| "default".into());
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    eprintln!(
        "[lateral3_bench] threads={} DA_TINYBLAS_AXIS={} DA_TINYBLAS_KC={}",
        threads, axis, kc
    );

    // Lateral 3 resize3 conv shape: M=768 (oc), N=216 (12*18), K=6912 (768*9)
    eprintln!("=== Lateral 3 conv3x3_stride2 GEMM (M=768, N=216, K=6912) ===");
    bench("lateral3 shape", 768, 216, 6912, 30);

    // For comparison: FFN fc1 (large N, moderate K — now K-blocked too)
    eprintln!();
    eprintln!("=== FFN fc1 (M=864, N=3072, K=768) — should use K-blocking (N > 3·K) ===");
    bench("fc1 shape", 864, 3072, 768, 20);

    // For comparison: FFN fc2 (large K, large N)
    eprintln!();
    eprintln!("=== FFN fc2 (M=864, N=768, K=3072) — should NOT use K-blocking ===");
    bench("fc2 shape", 864, 768, 3072, 20);
}
