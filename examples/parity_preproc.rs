//! Preprocessing parity: run Rust's `preprocess_real` on the reference PNG and
//! compare the resulting CHW tensor against the gold `proc_image` from
//! `dumps/reference_preproc_real.gguf` (produced by `scripts/dump_preproc_real.py`
//! via the genuine upstream cv2 InputProcessor).
//!
//! Usage:
//!   cargo run --release --example parity_preproc -- \
//!     --model models/depth-anything-base-f32.gguf \
//!     --png dumps/preproc_real_input.png \
//!     --ref dumps/reference_preproc_real.gguf

use std::process::ExitCode;
use std::sync::Arc;

use depth_anything::config::Config;
use depth_anything::gguf::GgufFile;
use depth_anything::preprocess::{preprocess_real, Image};

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
    let mut png = String::new();
    let mut ref_path = String::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                model = args[i + 1].clone();
                i += 2;
            }
            "--png" => {
                png = args[i + 1].clone();
                i += 2;
            }
            "--ref" => {
                ref_path = args[i + 1].clone();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    if model.is_empty() || png.is_empty() || ref_path.is_empty() {
        eprintln!("usage: parity_preproc --model M --png P --ref R");
        return Ok(false);
    }

    let file = GgufFile::open(&model)?;
    let cfg = Config::from_gguf(&file)?;
    let ref_file = Arc::new(GgufFile::open(&ref_path)?);
    let ref_img = ref_file.tensor_f32("proc_image")?;

    let img = Image::load(&png)?;
    eprintln!("input png: {}x{}", img.w, img.h);
    let pre = preprocess_real(&img, &cfg)?;
    eprintln!(
        "rust preprocessed: {}x{} (chw len {})",
        pre.w,
        pre.h,
        pre.chw.len()
    );
    eprintln!("reference proc_image: {} elems", ref_img.len());

    let n = pre.chw.len().min(ref_img.len());
    let mut max_abs = 0.0f32;
    let mut mean_abs = 0.0f32;
    let mut count_match = 0usize;
    for (a, b) in pre.chw[..n].iter().zip(ref_img[..n].iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        mean_abs += d;
        if d < 1e-3 {
            count_match += 1;
        }
    }
    mean_abs /= n as f32;
    eprintln!(
        "proc_image vs rust chw: max|d|={max_abs:.4e} mean|d|={mean_abs:.4e} match(<1e-3)={}/{n}",
        count_match
    );
    Ok(max_abs < 1e-2)
}
