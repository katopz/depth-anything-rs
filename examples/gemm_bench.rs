//! Micro-benchmark for tinyBLAS GEMM at the DA3 attention / FFN shapes.
//!
//! Measures raw GFLOP/s for the hot matmul shapes so we can see which ones
//! leave the most room for a specialised kernel.

use depth_anything::tinyblas;
use std::time::Instant;

fn bench(label: &str, m: usize, n: usize, k: usize, iters: usize) {
    let a = vec![0.123f32; m * k];
    let b = vec![0.456f32; k * n];
    let mut c = vec![0.0f32; m * n];
    // Warmup
    tinyblas::gemm_nn_into(m, n, k, &a, &b, &mut c);
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
    eprintln!("{label:<24} [{m}×{k}] @ [{k}×{n}]  -> {ms:>8.3} ms  {gflops:>8.1} GFLOP/s");
}

fn main() {
    eprintln!(
        "tinyblas GEMM bench (RAYON_NUM_THREADS={})",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into())
    );

    // Attention shapes (per head): n=864, hd=64, heads=12
    let n = 864;
    let hd = 64;
    bench("QK^T (per head)", n, n, hd, 50);
    bench("AV  (per head)", n, hd, n, 50);

    // Full QKV / proj / FFN shapes (n=864, embed=768)
    let embed = 768;
    bench("QKV proj", n, 3 * embed, embed, 20);
    bench("attn proj", n, embed, embed, 30);
    bench("FFN fc1", n, 3072, embed, 20);
    bench("FFN fc2", n, embed, 3072, 20);

    // DPT head conv shapes (lateral proj, resize3 GEMM).
    bench("lat3 proj", 768, 864, 768, 20);
    bench("lat3 resize", 768, 216, 6912, 20);
}
