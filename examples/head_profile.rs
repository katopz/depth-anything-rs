//! Per-stage timing breakdown of the FastDptHead forward pass.
//!
//! Usage:
//!   RAYON_NUM_THREADS=16 DA_FAST_HEAD_PROFILE=1 cargo run --release --example head_profile -- \
//!     --model models/depth-anything-base-q5_k.gguf \
//!     --input assets/samples/canyon.jpg

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

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
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> depth_anything::Result<()> {
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
        eprintln!("usage: head_profile --model M.gguf --input photo.jpg");
        return Ok(());
    }

    let device = Device::Cpu;
    let file = Arc::new(GgufFile::open(&model)?);
    let cfg = Config::from_gguf(&file)?;
    let vb = var_builder(file.clone(), device.clone());

    let head = DptHead::load(&vb, &cfg, &file, &device)?;
    let backbone = Backbone::load(&vb, &cfg, &file, &device)?;
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

    let feats = backbone.forward(&img_t, gh, gw)?;

    // Warm up the fast head (build FastDptHead, prime caches).
    let feats_host_init: Vec<Vec<f32>> = feats
        .feats
        .iter()
        .map(flatten_to_f32)
        .collect::<depth_anything::Result<Vec<_>>>()?;
    let fast = FastDptHead::from_candle(&head)?;
    let _ = fast.forward(&feats_host_init, gh, gw, h, w)?;

    let repeat = 10;
    let mut total_extract = 0.0f64;
    let mut total_head = 0.0f64;

    for _ in 0..repeat {
        let t0 = Instant::now();
        let feats_host: Vec<Vec<f32>> = feats
            .feats
            .iter()
            .map(flatten_to_f32)
            .collect::<depth_anything::Result<Vec<_>>>()?;
        total_extract += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let _ = fast.forward(&feats_host, gh, gw, h, w)?;
        total_head += t0.elapsed().as_secs_f64() * 1000.0;
    }

    eprintln!("==== FastDptHead forward breakdown ({repeat} iters) ====");
    eprintln!(
        "  feat extract (4 tensors): {:.2} ms",
        total_extract / repeat as f64
    );
    eprintln!(
        "  fast head total          : {:.2} ms",
        total_head / repeat as f64
    );
    Ok(())
}
