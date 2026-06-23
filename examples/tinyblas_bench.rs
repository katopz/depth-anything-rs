//! Benchmark: tinyblas vs candle matmul on the DA3 hot shapes.
//!
//! Run with:
//!   RAYON_NUM_THREADS=N cargo run --release --example tinyblas_bench
//!
//! Each shape is run with both backends; output is a JSON line on stdout
//! and a human-readable table on stderr.

use candle::{Device, Tensor};
use depth_anything::tinyblas;
use std::time::{Duration, Instant};

struct Shape {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
    /// If `true`, the "Linear" convention: weight stored as [n, k], compute A@W^T.
    nt: bool,
}

fn hot_shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "QKV",
            m: 673,
            n: 2304,
            k: 768,
            nt: true,
        },
        Shape {
            name: "FC1",
            m: 673,
            n: 3072,
            k: 768,
            nt: true,
        },
        Shape {
            name: "FC2",
            m: 673,
            n: 768,
            k: 3072,
            nt: true,
        },
        Shape {
            name: "Proj",
            m: 673,
            n: 768,
            k: 768,
            nt: true,
        },
        Shape {
            name: "QK^T",
            m: 673,
            n: 673,
            k: 64,
            nt: false,
        },
        Shape {
            name: "AV",
            m: 673,
            n: 64,
            k: 673,
            nt: false,
        },
        Shape {
            name: "QK-batch",
            m: 8076,
            n: 673,
            k: 64,
            nt: false,
        }, // 12×673 rows
        Shape {
            name: "AV-batch",
            m: 8076,
            n: 64,
            k: 673,
            nt: false,
        },
    ]
}

fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (1u64 << 31) as f32 - 1.0
        })
        .collect()
}

fn bench<F: FnMut()>(repeats: usize, warmup: usize, mut f: F) -> Duration {
    for _ in 0..warmup {
        f();
    }
    let t = Instant::now();
    for _ in 0..repeats {
        f();
    }
    t.elapsed() / repeats as u32
}

fn main() {
    let device = Device::Cpu;
    eprintln!(
        "[tinyblas_bench] threads={}, has_avx2_fma={}",
        rayon::current_num_threads(),
        tinyblas::has_avx2_fma()
    );
    eprintln!();
    eprintln!(
        "{:<12} {:>6} {:>4} {:>10} {:>10} {:>9}",
        "shape", "nt?", "gflop", "candle_ms", "tiny_ms", "ratio"
    );
    eprintln!("{}", "-".repeat(60));

    let shapes = hot_shapes();
    let mut json = String::from("{\"shapes\":[");
    for (i, s) in shapes.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }

        let a = pseudo_random(100 + i as u64, s.m * s.k);
        let b = if s.nt {
            pseudo_random(200 + i as u64, s.n * s.k)
        } else {
            pseudo_random(200 + i as u64, s.k * s.n)
        };

        let gflop = (s.m as f64 * s.n as f64 * s.k as f64 * 2.0) / 1e9;

        // candle matmul. For NT (Linear), we compute A @ B^T by transposing B
        // up front (so the call is a clean NN matmul, matching what tinyblas
        // also pays for inside matmul_nt). This is the fair comparison.
        let ca = Tensor::from_slice(&a, (s.m, s.k), &device).unwrap();
        let cb =
            Tensor::from_slice(&b, if s.nt { (s.n, s.k) } else { (s.k, s.n) }, &device).unwrap();
        let cb_t = if s.nt {
            // Use B^T so we measure pure NN matmul time (matches what
            // tinyblas pays for after packing).
            cb.t().unwrap().contiguous().unwrap()
        } else {
            cb
        };

        // Decide repeats so each shape runs for >= ~1s total.
        let repeats = (1000.0 / (gflop * 0.5).max(0.05)).clamp(5.0, 200.0) as usize;
        let warmup = 3;

        let candle_dur = bench(repeats, warmup, || {
            let _ = ca.matmul(&cb_t).unwrap();
        });
        let candle_ms = candle_dur.as_secs_f64() * 1000.0;

        let tiny_ms = if s.nt {
            let d = bench(repeats, warmup, || {
                let _ = tinyblas::matmul_nt(s.m, s.n, s.k, &a, &b);
            });
            d.as_secs_f64() * 1000.0
        } else {
            let d = bench(repeats, warmup, || {
                let _ = tinyblas::matmul_nn(s.m, s.n, s.k, &a, &b);
            });
            d.as_secs_f64() * 1000.0
        };

        let ratio = tiny_ms / candle_ms;
        eprintln!(
            "{:<12} {:>6} {:>4.1} {:>10.2} {:>10.2} {:>8.2}x",
            s.name,
            if s.nt { "nt" } else { "nn" },
            gflop,
            candle_ms,
            tiny_ms,
            ratio
        );

        json.push_str(&format!(
            "{{\"name\":\"{}\",\"m\":{},\"n\":{},\"k\":{},\"nt\":{},\"gflop\":{:.3},\"candle_ms\":{:.3},\"tinyblas_ms\":{:.3},\"ratio\":{:.3}}}",
            s.name, s.m, s.n, s.k, s.nt, gflop, candle_ms, tiny_ms, ratio
        ));
    }
    json.push_str("]}");
    println!("{json}");
}
