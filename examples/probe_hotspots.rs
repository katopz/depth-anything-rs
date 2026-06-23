//! Probe: measure candle's per-op cost on the exact DA3 attention shapes.
//!
//! Run with:
//!   RAYON_NUM_THREADS=N cargo run --release --example probe_hotspots
//!
//! Output is a JSON object on stdout plus a human-readable summary on stderr.

use candle::{Device, Tensor};
use std::time::Instant;

fn time_ms<F: FnMut()>(label: &str, repeats: usize, warmup: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let t = Instant::now();
    for _ in 0..repeats {
        f();
    }
    let total = t.elapsed().as_secs_f64() * 1000.0;
    let per = total / repeats as f64;
    eprintln!("  {label:<42} : {per:>8.3} ms");
    per
}

fn main() {
    let device = Device::Cpu;
    let n = 673; // sequence length (31*21 + 1 CLS) at 504x336/16
    let embed = 768;
    let heads = 12;
    let head_dim = 64;
    let mlp_hidden = 3072;

    // Random-ish input data (avoid all-ones which some kernels fast-path).
    let mut rng_state: u64 = 0x1234_5678_9abc_def0;
    let mut next_f32 = || {
        // xorshift64 -> (0,1]
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f32) / (u32::MAX as f32) - 0.5
    };

    let mut mk = |len: usize| -> Vec<f32> { (0..len).map(|_| next_f32()).collect() };

    eprintln!(
        "[probe] device={device:?}, threads={}",
        rayon::current_num_threads()
    );
    eprintln!("[probe] shapes based on DA3-BASE @ 504x336: n={n} embed={embed} heads={heads} head_dim={head_dim}");
    eprintln!();

    // --- Per-block backbone ops (12 blocks, ×12 for total) ---
    eprintln!("--- per-block ops (×12 in backbone) ---");

    let x_v = mk(n * embed);
    let w_qkv_v = mk(embed * 3 * embed);
    let w_proj_v = mk(embed * embed);
    let w_fc1_v = mk(embed * mlp_hidden);
    let w_fc2_v = mk(mlp_hidden * embed);

    let x = Tensor::from_slice(&x_v, (n, embed), &device).unwrap();
    let w_qkv = Tensor::from_slice(&w_qkv_v, (embed, 3 * embed), &device).unwrap();
    let w_proj = Tensor::from_slice(&w_proj_v, (embed, embed), &device).unwrap();
    let w_fc1 = Tensor::from_slice(&w_fc1_v, (embed, mlp_hidden), &device).unwrap();
    let w_fc2 = Tensor::from_slice(&w_fc2_v, (mlp_hidden, embed), &device).unwrap();

    let r = 50;
    let warm = 5;

    let qkv_ms = time_ms("QKV matmul [n,e]@[e,3e]", r, warm, || {
        let _ = x.matmul(&w_qkv).unwrap();
    });
    let fc1_ms = time_ms("FC1 matmul [n,e]@[e,h]", r, warm, || {
        let _ = x.matmul(&w_fc1).unwrap();
    });

    // FC2 input shape: [n, mlp_hidden]
    let h_v = mk(n * mlp_hidden);
    let h = Tensor::from_slice(&h_v, (n, mlp_hidden), &device).unwrap();
    let fc2_ms = time_ms("FC2 matmul [n,h]@[h,e]", r, warm, || {
        let _ = h.matmul(&w_fc2).unwrap();
    });
    let proj_ms = time_ms("Proj matmul [n,e]@[e,e]", r, warm, || {
        let _ = x.matmul(&w_proj).unwrap();
    });

    // --- Attention ops (per head, then ×12) ---
    eprintln!();
    eprintln!("--- attention ops (×12 heads, ×12 blocks) ---");

    // Q,K,V per-head: [n, head_dim]
    let q_v = mk(n * head_dim);
    let q = Tensor::from_slice(&q_v, (n, head_dim), &device).unwrap();
    let k = q.clone();
    let v = q.clone();

    let qk_ms = time_ms("Q·K^T [n,d]@[d,n] per head", r, warm, || {
        let _ = q.matmul(&k.t().unwrap()).unwrap();
    });

    // AV: scores [n,n] @ V [n,d]
    let s_v = mk(n * n);
    let scores = Tensor::from_slice(&s_v, (n, n), &device).unwrap();
    let av_ms = time_ms("A·V [n,n]@[n,d] per head", r, warm, || {
        let _ = scores.matmul(&v).unwrap();
    });

    // Batched attention: [heads, n, d] @ [heads, d, n] -> [heads, n, n]
    let q_b_v = mk(heads * n * head_dim);
    let q_b = Tensor::from_slice(&q_b_v, (heads, n, head_dim), &device).unwrap();
    let k_b = q_b.clone();
    let v_b = q_b.clone();
    let qk_batch_ms = time_ms("BATCH Q·K^T [h,n,d]@[h,d,n]", r, warm, || {
        let _ = q_b.matmul(&k_b.t().unwrap()).unwrap();
    });

    let scores_b_v = mk(heads * n * n);
    let scores_b = Tensor::from_slice(&scores_b_v, (heads, n, n), &device).unwrap();
    let av_batch_ms = time_ms("BATCH A·V [h,n,n]@[h,n,d]", r, warm, || {
        let _ = scores_b.matmul(&v_b).unwrap();
    });

    // --- Overhead ops (no GEMM) ---
    eprintln!();
    eprintln!("--- candle overhead ops (no matmul) ---");

    // QKV output [1, n, 3, heads, head_dim] — split + transpose + contiguous
    let qkv_out_v = mk(n * 3 * heads * head_dim);
    let qkv_out = Tensor::from_slice(&qkv_out_v, (1, n, 3, heads, head_dim), &device).unwrap();
    let split_transpose_ms = time_ms("qkv split+transpose+contiguous", r, warm, || {
        let qkv = qkv_out.reshape((1, n, 3, heads, head_dim)).unwrap();
        let q = qkv.narrow(2, 0, 1).unwrap().squeeze(2).unwrap();
        let k = qkv.narrow(2, 1, 1).unwrap().squeeze(2).unwrap();
        let _ = q.transpose(1, 2).unwrap().contiguous().unwrap();
        let _ = k.transpose(1, 2).unwrap().contiguous().unwrap();
    });

    // Softmax on [1, heads, n, n]
    let scores_4d_v = mk(heads * n * n);
    let scores_4d = Tensor::from_slice(&scores_4d_v, (1, heads, n, n), &device).unwrap();
    let softmax_ms = time_ms("softmax_last_dim [1,h,n,n]", r, warm, || {
        let _ = candle_nn::ops::softmax_last_dim(&scores_4d).unwrap();
    });

    // --- Extrapolation to full backbone ---
    eprintln!();
    eprintln!("--- extrapolation ---");
    let per_block_gemm = qkv_ms + fc1_ms + fc2_ms + proj_ms + qk_batch_ms + av_batch_ms;
    let per_block_overhead = split_transpose_ms + softmax_ms;
    let blocks = 12;
    eprintln!("  per-block gemm total   : {per_block_gemm:>8.3} ms");
    eprintln!("  per-block overhead est : {per_block_overhead:>8.3} ms  (split+softmax only)");
    eprintln!(
        "  12-block gemm total    : {:>8.1} ms",
        per_block_gemm * blocks as f64
    );
    eprintln!(
        "  12-block overhead est  : {:>8.1} ms  (lower bound; real is higher with rope/ln/residual)",
        per_block_overhead * blocks as f64
    );

    println!(
        "{{\"threads\":{},\"qkv_ms\":{:.3},\"fc1_ms\":{:.3},\"fc2_ms\":{:.3},\"proj_ms\":{:.3},\"qk_per_head_ms\":{:.3},\"av_per_head_ms\":{:.3},\"qk_batch_ms\":{:.3},\"av_batch_ms\":{:.3},\"split_transpose_ms\":{:.3},\"softmax_ms\":{:.3},\"per_block_gemm_ms\":{:.3},\"per_block_overhead_ms\":{:.3}}}",
        rayon::current_num_threads(),
        qkv_ms, fc1_ms, fc2_ms, proj_ms, qk_ms, av_ms, qk_batch_ms, av_batch_ms,
        split_transpose_ms, softmax_ms, per_block_gemm, per_block_overhead
    );
}
