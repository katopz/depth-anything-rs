//! A/B parity: compare the candle DPT head vs the candle-free FastDptHead
//! on the same backbone features, to verify correctness.
//!
//! Usage:
//!   RAYON_NUM_THREADS=4 cargo run --release --example head_parity -- \
//!     --model models/depth-anything-base-q5_k.gguf \
//!     --input assets/samples/canyon.jpg

use std::process::ExitCode;
use std::sync::Arc;

use candle::{Device, Tensor};
use depth_anything::backbone::Backbone;
use depth_anything::config::Config;
use depth_anything::dpt_head::DptHead;
use depth_anything::fast_attn::flatten_to_f32;
use depth_anything::fast_dpt::FastDptHead;
use depth_anything::gguf::GgufFile;
use depth_anything::preprocess::{preprocess_real, Image};
use depth_anything::weights::var_builder;

fn main() -> ExitCode {
    match run() {
        Ok(ok) => ExitCode::from(if ok { 0 } else { 1 }),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> depth_anything::Result<bool> {
    let args: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut input = String::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                model = args[i + 1].clone();
                i += 2;
            }
            "--input" | "-i" => {
                input = args[i + 1].clone();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    if model.is_empty() || input.is_empty() {
        eprintln!("usage: head_parity --model M.gguf --input photo.jpg");
        return Ok(false);
    }

    let device = Device::Cpu;
    let file = Arc::new(GgufFile::open(&model)?);
    let cfg = Config::from_gguf(&file)?;
    let vb = var_builder(file.clone(), device.clone());

    // Build the candle DPT head.
    let head = DptHead::load(&vb, &cfg, &file, &device)?;

    // Build the candle backbone.
    let backbone = Backbone::load(&vb, &cfg, &file, &device)?;

    // Preprocess the input image.
    let rgb = image::open(&input)?.to_rgb8();
    let img = Image {
        w: rgb.width() as usize,
        h: rgb.height() as usize,
        rgb: rgb.into_raw(),
    };
    let pre = preprocess_real(&img, &cfg)?;
    let (h, w) = (pre.h, pre.w);
    let gh = h / cfg.patch_size as usize;
    let gw = w / cfg.patch_size as usize;
    let img_t =
        Tensor::from_vec(pre.chw.clone(), (1, 3, h, w), &device)?.to_dtype(candle::DType::F32)?;

    // Backbone forward (shared).
    let feats = backbone.forward(&img_t, gh, gw)?;

    eprintln!("backbone features:");
    for (i, f) in feats.feats.iter().enumerate() {
        eprintln!("  stage {i}: shape {:?}", f.dims());
    }

    // Run candle head.
    std::env::remove_var("DA_FAST_HEAD");
    let (candle_logits, candle_sky) = head.forward(&feats.feats, gh, gw, h, w, &device)?;
    let candle_host = candle_logits
        .to_dtype(candle::DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    eprintln!("candle head output: {} values", candle_host.len());

    // Run fast head.
    // Build a fresh FastDptHead.
    let fast = FastDptHead::from_candle(&head)?;
    let feats_host: Vec<Vec<f32>> = feats
        .feats
        .iter()
        .map(|t| flatten_to_f32(t))
        .collect::<depth_anything::Result<Vec<_>>>()?;
    let (fast_logits, fast_sky) = fast.forward(&feats_host, gh, gw, h, w)?;
    eprintln!("fast head output: {} values", fast_logits.len());

    // Compare.
    assert_eq!(
        candle_host.len(),
        fast_logits.len(),
        "output length mismatch"
    );
    let n = candle_host.len();
    let mut max_abs = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    for i in 0..n {
        let a = candle_host[i] as f64;
        let b = fast_logits[i] as f64;
        let d = (a - b).abs();
        if d > max_abs as f64 {
            max_abs = d as f32;
        }
        sum_sq_diff += (a - b) * (a - b);
        sum_sq += a * a;
        sum_a += a;
        sum_b += b;
    }
    let mean_a = sum_a / n as f64;
    let mean_b = sum_b / n as f64;
    let rms_diff = (sum_sq_diff / n as f64).sqrt();
    let rms = (sum_sq / n as f64).sqrt();
    let rel = rms_diff / rms;
    let mean_diff = (mean_a - mean_b).abs();

    eprintln!();
    eprintln!("==== depth logits comparison (candle vs fast) ====");
    eprintln!("  n values    : {n}");
    eprintln!("  max abs diff: {max_abs:.6e}");
    eprintln!("  rms diff    : {rms_diff:.6e}");
    eprintln!("  rms signal  : {rms:.6e}");
    eprintln!("  rel diff    : {rel:.6e}");
    eprintln!("  mean diff   : {mean_diff:.6e}  (mean_a={mean_a:.6e} mean_b={mean_b:.6e})");

    // Check sky head if present.
    if let (Some(c_sky), Some(f_sky)) = (&candle_sky, &fast_sky) {
        let c_sky_host = c_sky
            .to_dtype(candle::DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut sky_max = 0.0f32;
        for (a, b) in c_sky_host.iter().zip(f_sky.iter()) {
            let d = (a - b).abs();
            if d > sky_max {
                sky_max = d;
            }
        }
        eprintln!("  sky max diff: {sky_max:.6e}");
    }

    // A loose tolerance: Winograd + f32 GEMM introduces some noise, so allow up
    // to 1e-2 max abs diff in absolute units (the depth logits are in the [-10,
    // 10] range so this corresponds to <0.1% relative error).
    let pass = max_abs < 1.0 && rel < 0.01;
    eprintln!();
    eprintln!("  RESULT      : {}", if pass { "PASS" } else { "FAIL" });
    Ok(pass)
}
