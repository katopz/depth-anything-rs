//! Microbenchmark: measure fast_conv (Winograd 3×3 + 1×1 GEMM) kernel times
//! for the exact DA3-BASE DPT head conv shapes.
//!
//! Run with:
//!   RAYON_NUM_THREADS=16 cargo run --release --example fast_conv_bench

use depth_anything::fast_conv;
use rayon::prelude::*;
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
    eprintln!("  {label:<50} : {per:>8.3} ms");
    per
}

fn bench_conv3x3(ic: usize, oc: usize, h: usize, w: usize, repeats: usize) -> f64 {
    let n = 1;
    let pad = 1;
    let x: Vec<f32> = (0..n * ic * h * w).map(|i| (i as f32) * 0.0001).collect();
    let wt: Vec<f32> = (0..oc * ic * 9).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = vec![0.1; oc];
    eprintln!("[conv3x3] IC={ic} OC={oc} H={h} W={w} (hw={})", h * w);
    // The Winograd policy (F(2,3) vs F(4,3)) is auto-selected based on shape,
    // or overridden via DA_FAST_CONV3X3_WINO=f2|f4|auto (read once per process).
    // To compare policies, run this example twice with different env vars.
    let wino = time_ms("  winograd auto (f2/f4)", repeats, 3, || {
        let _ = fast_conv::conv3x3_pad1(&x, &wt, &b, n, ic, h, w, oc);
    });
    let _ = pad;
    wino
}

fn bench_conv1x1(ic: usize, oc: usize, h: usize, w: usize, repeats: usize) -> f64 {
    let n = 1;
    let x: Vec<f32> = (0..n * ic * h * w).map(|i| (i as f32) * 0.0001).collect();
    let wt: Vec<f32> = (0..oc * ic).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = vec![0.1; oc];
    eprintln!("[conv1x1] IC={ic} OC={oc} H={h} W={w} (hw={})", h * w);
    time_ms("  1x1 GEMM (transpose+gemm+transpose)", repeats, 3, || {
        let _ = fast_conv::conv1x1(&x, &wt, &b, n, ic, h, w, oc);
    })
}

/// Microbench the upsample+out2a fusion: 64→32 @ (h_lo,w_lo)→(h,w).
fn bench_fused_upsample(
    ic: usize,
    oc: usize,
    h_lo: usize,
    w_lo: usize,
    h: usize,
    w: usize,
    repeats: usize,
) {
    let n = 1;
    let x: Vec<f32> = (0..n * ic * h_lo * w_lo)
        .map(|i| (i as f32) * 0.0001)
        .collect();
    let wt: Vec<f32> = (0..oc * ic * 9).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = vec![0.1; oc];
    eprintln!(
        "[fused_upsample] IC={ic} OC={oc} ({h_lo}×{w_lo}) → ({h}×{w}) (hw_up={})",
        h * w
    );
    // Fused path: no materialised upsample.
    let _ = time_ms("  fused (upsample-in-wino)", repeats, 3, || {
        let _ = fast_conv::conv3x3_pad1_relu_out_upsample(
            &x, &wt, &b, n, ic, h_lo, w_lo, oc, h, w, None,
        );
    });
}

fn bench_conv1x1_upsample(
    ic: usize,
    oc: usize,
    h_lo: usize,
    w_lo: usize,
    h: usize,
    w: usize,
    repeats: usize,
) {
    let n = 1;
    let x: Vec<f32> = (0..n * ic * h_lo * w_lo)
        .map(|i| (i as f32) * 0.0001)
        .collect();
    let wt: Vec<f32> = (0..oc * ic).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = vec![0.1; oc];
    eprintln!(
        "[conv1x1_upsample] IC={ic} OC={oc} ({h_lo}×{w_lo}) → ({h}×{w}) (hw_up={})",
        h * w
    );
    // Unfused: materialise upsample + conv1x1. Uses a rayon-parallelised
    // `align_corners=true` bilinear upsample matching the production
    // `upsample_bilinear_ac` (the fast_dpt helper is private). This gives a
    // fair A/B vs the fused path.
    let t_unfused = time_ms("  unfused (upsample + conv1x1)", repeats, 3, || {
        let mut up = vec![0.0f32; ic * h * w];
        let sy = if h > 1 {
            (h_lo - 1) as f32 / (h - 1) as f32
        } else {
            0.0
        };
        let sx = if w > 1 {
            (w_lo - 1) as f32 / (w - 1) as f32
        } else {
            0.0
        };
        // Precompute index/weight tables (shared across channels).
        let xw: Vec<(usize, usize, f32)> = (0..w)
            .map(|ox| {
                let fx = ox as f32 * sx;
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(w_lo - 1);
                let wx = fx - x0 as f32;
                (x0, x1, wx)
            })
            .collect();
        let yw: Vec<(usize, usize, f32)> = (0..h)
            .map(|oy| {
                let fy = oy as f32 * sy;
                let y0 = fy.floor() as usize;
                let y1 = (y0 + 1).min(h_lo - 1);
                let wy = fy - y0 as f32;
                (y0, y1, wy)
            })
            .collect();
        up.par_chunks_mut(h * w)
            .enumerate()
            .for_each(|(ci, out_ch)| {
                let in_off = ci * h_lo * w_lo;
                for oy in 0..h {
                    let (y0, y1, wy) = yw[oy];
                    for ox in 0..w {
                        let (x0, x1, wx) = xw[ox];
                        let p00 = x[in_off + y0 * w_lo + x0];
                        let p01 = x[in_off + y0 * w_lo + x1];
                        let p10 = x[in_off + y1 * w_lo + x0];
                        let p11 = x[in_off + y1 * w_lo + x1];
                        let top = p00 * (1.0 - wx) + p01 * wx;
                        let bot = p10 * (1.0 - wx) + p11 * wx;
                        out_ch[oy * w + ox] = top * (1.0 - wy) + bot * wy;
                    }
                }
            });
        let _ = fast_conv::conv1x1(&up, &wt, &b, n, ic, h, w, oc);
    });
    // Fused path: upsample computed inside the GEMM.
    let t_fused = time_ms("  fused (upsample-in-gemm)", repeats, 3, || {
        let _ = fast_conv::conv1x1_upsample(&x, &wt, &b, n, ic, h_lo, w_lo, oc, h, w);
    });
    eprintln!(
        "  Δ = {:.2} ms ({:.1}% of unfused)",
        t_fused - t_unfused,
        100.0 * t_fused / t_unfused
    );
}

fn main() {
    eprintln!(
        "[fast_conv_bench] threads = {}",
        rayon::current_num_threads()
    );
    eprintln!();

    let r = 10;
    let mut total_3x3 = 0.0;
    let mut total_1x1 = 0.0;

    // DPT head conv shapes (from the debug output of the actual forward pass).
    eprintln!("=== 3x3 convs (Winograd) ===");
    // layer_rn + residual conv units at each fusion level.
    // rn4 level: H=24, W=36 (2 convs per RCU, 1 RCU)
    total_3x3 += 2.0 * bench_conv3x3(128, 128, 24, 36, r);
    // rn3 level: H=48, W=72 (2 convs per RCU, 2 RCUs)
    total_3x3 += 4.0 * bench_conv3x3(128, 128, 48, 72, r);
    // rn2 level: H=96, W=144 (2 convs per RCU, 2 RCUs)
    total_3x3 += 4.0 * bench_conv3x3(128, 128, 96, 144, r);
    // rn1 level: H=192, W=288 (2 convs per RCU, 2 RCUs) — estimated
    // (actual resolution depends on pyramid structure)
    total_3x3 += 4.0 * bench_conv3x3(128, 128, 192, 288, r);
    // layer_rn for each stage (4 stages, different IC → 128)
    total_3x3 += bench_conv3x3(96, 128, 96, 144, r); // stage 0 lateral rn
    total_3x3 += bench_conv3x3(192, 128, 48, 72, r); // stage 1 lateral rn
    total_3x3 += bench_conv3x3(384, 128, 24, 36, r); // stage 2 lateral rn
    total_3x3 += bench_conv3x3(768, 128, 12, 18, r); // stage 3 lateral rn
                                                     // output convs
    total_3x3 += bench_conv3x3(128, 64, 192, 288, r); // out1
    total_3x3 += bench_conv3x3(64, 32, 336, 504, r); // out2a
                                                     // sky head (if present)
    total_3x3 += bench_conv3x3(64, 1, 336, 504, r); // sky conv1

    eprintln!();
    eprintln!("=== upsample+out2a fusion ===");
    // out2a at (192×288) → (336×504). Same shape as the production output stage.
    // Compare:
    //   - unfused: conv3x3_pad1_relu_out @ 336×504, reading from materialised upsample
    //   - fused: conv3x3_pad1_relu_out_upsample, computing upsample on the fly
    // (The unfused time is the same as the out2a row above + the upsample pass
    // ~3.7ms, which isn't measured by this bench — add it manually.)
    bench_fused_upsample(64, 32, 192, 288, 336, 504, r);

    eprintln!();
    eprintln!("=== fusion-stage upsample+conv1x1 ===");
    // rn1 fusion stage: 128×128 conv1x1, upsample 96×144 → 192×288 (2× scale).
    // This is the primary target for the conv1x1_upsample fusion.
    bench_conv1x1_upsample(128, 128, 96, 144, 192, 288, r);
    // rn2 fusion stage: smaller — 48×72 → 96×144.
    bench_conv1x1_upsample(128, 128, 48, 72, 96, 144, r);
    // rn3 fusion stage: smallest — 24×36 → 48×72.
    bench_conv1x1_upsample(128, 128, 24, 36, 48, 72, r);

    eprintln!();
    eprintln!("=== 1x1 convs (GEMM) ===");
    // proj convs (each stage projected to 128 channels)
    total_1x1 += bench_conv1x1(96, 128, 96, 144, r); // proj stage 0
    total_1x1 += bench_conv1x1(192, 128, 48, 72, r); // proj stage 1
    total_1x1 += bench_conv1x1(384, 128, 24, 36, r); // proj stage 2
    total_1x1 += bench_conv1x1(768, 128, 12, 18, r); // proj stage 3
                                                     // outc convs at each fusion level
    total_1x1 += bench_conv1x1(128, 128, 24, 36, r); // rn4 outc
    total_1x1 += bench_conv1x1(128, 128, 48, 72, r); // rn3 outc
    total_1x1 += bench_conv1x1(128, 128, 96, 144, r); // rn2 outc
    total_1x1 += bench_conv1x1(128, 128, 192, 288, r); // rn1 outc
                                                       // out2b
    total_1x1 += bench_conv1x1(32, 2, 336, 504, r); // out2b
                                                    // sky conv2
    total_1x1 += bench_conv1x1(1, 1, 336, 504, r); // sky conv2

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!("  total 3x3 conv time : {:.1} ms", total_3x3);
    eprintln!("  total 1x1 conv time : {:.1} ms", total_1x1);
    eprintln!("  total conv time     : {:.1} ms", total_3x3 + total_1x1);
    eprintln!();
    eprintln!("(Note: this measures raw conv kernels only, without tensor copy");
    eprintln!(" overhead from read_conv_tensors or Tensor construction.)");
}
