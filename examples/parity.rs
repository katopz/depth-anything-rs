//! Parity diagnostic: run the Rust forward on the EXACT reference input from
//! `dumps/reference.gguf` (the PyTorch DA3 gold dump on a fixed 224×224 random
//! image, produced by `scripts/dump_reference.py`) and compare each stage
//! (backbone features, head fused feature, final depth) against the reference.
//!
//! Usage:
//!   cargo run --release --example parity -- --model M.gguf --ref dumps/reference.gguf
//!
//! The reference input bypasses Rust preprocessing (it is fed straight into the
//! backbone as the normalized CHW tensor), so any divergence is purely in the
//! Rust forward pass, not the resize/normalize policy.

use std::process::ExitCode;
use std::sync::Arc;

use candle::{DType, Device, Tensor};
use depth_anything::backbone::Backbone;
use depth_anything::config::Config;
use depth_anything::dpt_head::DptHead;
use depth_anything::gguf::GgufFile;
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
    let mut ref_path = String::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                model = args[i + 1].clone();
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
    if model.is_empty() || ref_path.is_empty() {
        eprintln!("usage: parity --model M.gguf --ref dumps/reference.gguf");
        return Ok(false);
    }

    let device = Device::Cpu;
    let file = Arc::new(GgufFile::open(&model)?);
    let cfg = Config::from_gguf(&file)?;
    let ref_file = Arc::new(GgufFile::open(&ref_path)?);
    let vb = var_builder(file.clone(), device.clone());

    // Derive H,W from a sibling manifest JSON next to the reference GGUF.
    // Prefer a manifest whose stem matches the GGUF stem (e.g.
    // reference_real_depth.gguf -> manifest_real_depth.json), then fall back to
    // manifest.json. Trivial scan for '"H": <n>' / '"W": <n>' avoids a JSON dep.
    let in_vec = ref_file.tensor_f32("input_image")?;
    let ref_p = std::path::Path::new(&ref_path);
    let stem = ref_p
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("reference");
    // reference -> manifest ; reference_real_depth -> manifest_real_depth
    let stem_suffix = stem.strip_prefix("reference").unwrap_or("");
    let preferred = ref_p.with_file_name(format!("manifest{stem_suffix}.json"));
    let scan_hw = |text: &str| -> Option<(usize, usize)> {
        let find = |key: &str| -> Option<usize> {
            let pat = format!("\"{key}\":");
            let i = text.find(&pat)?;
            let rest = &text[i + pat.len()..];
            let n: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            n.parse().ok()
        };
        Some((find("H")?, find("W")?))
    };
    let (h, w): (usize, usize) = [preferred, ref_p.with_file_name("manifest.json")]
        .iter()
        .find_map(|p| std::fs::read_to_string(p).ok().and_then(|t| scan_hw(&t)))
        .unwrap_or_else(|| {
            let side = ((in_vec.len() / 3) as f64).sqrt() as usize;
            (side, side)
        });
    let img = Tensor::from_vec(in_vec, (1, 3, h, w), &device)?.to_dtype(DType::F32)?;
    let (gh, gw) = (h / 14, w / 14);
    eprintln!("input {h}x{w} grid {gh}x{gw}");

    let report = |name: &str, got: &Tensor, ref_name: &str| -> depth_anything::Result<()> {
        let g = got.flatten_all()?.to_vec1::<f32>()?;
        let r = ref_file.tensor_f32(ref_name)?;
        let n = g.len().min(r.len());
        if n == 0 {
            eprintln!("  {name:<22} (ref {ref_name}): EMPTY");
            return Ok(());
        }
        let mut max_abs = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_r = 0.0f32;
        let mut sum_gg = 0.0f32;
        let mut sum_rr = 0.0f32;
        let mut sum_gr = 0.0f32;
        for k in 0..n {
            let (gv, rv) = (g[k], r[k]);
            let d = (gv - rv).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum_g += gv;
            sum_r += rv;
            sum_gg += gv * gv;
            sum_rr += rv * rv;
            sum_gr += gv * rv;
        }
        let nf = n as f32;
        let mean_g = sum_g / nf;
        let mean_r = sum_r / nf;
        let cov = (sum_gr / nf) - mean_g * mean_r;
        let var_g = (sum_gg / nf) - mean_g * mean_g;
        let var_r = (sum_rr / nf) - mean_r * mean_r;
        let corr = cov / (var_g.sqrt() * var_r.sqrt() + 1e-12);
        eprintln!(
            "  {name:<22} (ref {ref_name:<14}): max|d|={max_abs:.4e}  corr={corr:+.5}  n={n}"
        );
        Ok(())
    };

    eprintln!("=== backbone stage ===");
    // Load the pose head FIRST (as the engine does, sharing the same VarBuilder /
    // weight cache), to confirm pose loading does not perturb the shared cache.
    let _pose = depth_anything::cam_pose::CamPose::load(&vb, &cfg, &file, &device).ok();
    let backbone = Backbone::load(&vb, &cfg, &file, &device)?;
    let feats = backbone.forward(&img, gh, gw)?;

    // feat_5/7/9/11 are [1,1,256,1536]; the Rust feats are [1,256,1536].
    for (idx, layer) in cfg.out_layers.iter().enumerate() {
        let f = feats.feats.get(idx).ok_or_else(|| {
            depth_anything::Error::Model(format!("missing feat for layer {layer}"))
        })?;
        // Squeeze the leading [1] batch dim to match [1,256,1536] reference shape.
        let f = f.squeeze(0)?;
        report(&format!("feat_{layer}"), &f, &format!("feat_{layer}"))?;
    }

    eprintln!("=== head stage ===");
    let head = DptHead::load(&vb, &cfg, &file, &device)?;
    let (logits, _sky) = head.forward(&feats.feats, gh, gw, h, w, &device)?;
    // logits: [1, output_dim, H, W] -> [output_dim, H, W]; reference head_depth is
    // already activated (exp). Apply exp to channel 0 and compare.
    let out_dim = logits.dim(1)?;
    let h = logits.dim(2)?;
    let w = logits.dim(3)?;
    eprintln!("  logits shape [1,{out_dim},{h},{w}]");
    let logits_host = logits
        .to_dtype(DType::F32)?
        .permute((0, 2, 3, 1))?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let hw = h * w;
    let depth: Vec<f32> = (0..hw).map(|i| logits_host[i * out_dim].exp()).collect();
    let depth_t = Tensor::from_vec(depth.clone(), (h, w), &device)?;
    report("head_depth(exp)", &depth_t, "head_depth")?;

    // Overall pass/fail: accept if the depth correlation > 0.99.
    let g = depth;
    let r = ref_file.tensor_f32("head_depth")?;
    let n = g.len().min(r.len());
    let (mut sg, mut sr, mut sgg, mut srr, mut sgr) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for k in 0..n {
        sg += g[k];
        sr += r[k];
        sgg += g[k] * g[k];
        srr += r[k] * r[k];
        sgr += g[k] * r[k];
    }
    let nf = n as f32;
    let cov = sgr / nf - (sg / nf) * (sr / nf);
    let varg = sgg / nf - (sg / nf).powi(2);
    let varr = srr / nf - (sr / nf).powi(2);
    let corr = cov / (varg.sqrt() * varr.sqrt() + 1e-12);
    eprintln!();
    eprintln!("FINAL head_depth correlation: {corr:+.6}");
    Ok(corr > 0.99)
}
