//! K-quant dequant parity: load a quantized GGUF (q4_k / q5_k / q6_k / q8_k)
//! with the Rust K-quant dequantizer and compare every tensor against the
//! same tensor in the f32 reference GGUF.
//!
//! This validates that `src/kquants.rs` reproduces ggml's dequantization. The
//! f32 reference holds the *pre-quantization* weights, so we expect small
//! per-tensor differences (the quantization error introduced by `da3-cli
//! quantize`), NOT bit-exact equality. A dequant bug would show up as wild
//! values (near-zero correlation, huge magnitudes), so we gate on correlation.
//!
//! Usage:
//!   cargo run --release --example kquant_parity -- \
//!       --ref models/depth-anything-base-f32.gguf \
//!       --quant models/depth-anything-base-q4_k.gguf
//!
//! Expected: every quantized tensor correlates >0.95 with its f32 original
//! (q4_k typically >0.99, q6_k >0.999).

use std::process::ExitCode;
use std::sync::Arc;

use depth_anything::gguf::{GgufDType, GgufFile};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut ref_path = String::new();
    let mut quant_path = String::new();
    let mut min_corr: f32 = 0.95;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ref" => {
                ref_path = args[i + 1].clone();
                i += 2;
            }
            "--quant" => {
                quant_path = args[i + 1].clone();
                i += 2;
            }
            "--min-corr" => {
                min_corr = args[i + 1].parse().unwrap_or(0.95);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    if ref_path.is_empty() || quant_path.is_empty() {
        eprintln!("usage: kquant_parity --ref F32.gguf --quant QX_K.gguf [--min-corr 0.95]");
        return ExitCode::FAILURE;
    }

    let ref_file = Arc::new(match GgufFile::open(&ref_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("open ref: {e}");
            return ExitCode::FAILURE;
        }
    });
    let quant_file = Arc::new(match GgufFile::open(&quant_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("open quant: {e}");
            return ExitCode::FAILURE;
        }
    });

    // Walk every tensor in the quantized file; for those that are actually
    // quantized (not f32), compare against the f32 reference.
    let mut n_checked = 0usize;
    let mut n_passed = 0usize;
    let mut worst_corr = 1.0f32;
    let mut worst_name = String::new();

    let names: Vec<String> = quant_file.tensor_names().cloned().collect();
    for name in &names {
        let info = match quant_file.tensor_info(name) {
            Some(i) => i,
            None => continue,
        };
        if info.dtype == GgufDType::F32 || info.dtype == GgufDType::F16 {
            continue; // not quantized; identical to ref by construction
        }

        let got = match quant_file.tensor_f32(name) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  dequant {name} ({:?}): FAIL: {e}", info.dtype);
                return ExitCode::FAILURE;
            }
        };
        let want = match ref_file.tensor_f32(name) {
            Ok(v) => v,
            Err(_) => {
                // Tensor not in the f32 ref (e.g. converter renamed it); skip.
                continue;
            }
        };
        let n = got.len().min(want.len());
        if n == 0 {
            continue;
        }

        let stats = corr_and_err(&got[..n], &want[..n]);
        n_checked += 1;
        if stats.corr >= min_corr {
            n_passed += 1;
        }
        if stats.corr < worst_corr {
            worst_corr = stats.corr;
            worst_name = name.clone();
        }
        eprintln!(
            "  {:<40} {:?}  n={:<8} corr={:+.5}  max|d|={:.4e}  rel_L2={:.4e}",
            name, info.dtype, n, stats.corr, stats.max_abs, stats.rel_l2
        );
    }

    eprintln!();
    eprintln!(
        "summary: {n_passed}/{n_checked} tensors corr >= {min_corr}; \
         worst = {worst_corr:+.5} ({worst_name})"
    );
    if n_passed == n_checked {
        eprintln!("PASS");
        ExitCode::SUCCESS
    } else {
        eprintln!("FAIL");
        ExitCode::FAILURE
    }
}

struct Stats {
    corr: f32,
    max_abs: f32,
    rel_l2: f32,
}

fn corr_and_err(got: &[f32], want: &[f32]) -> Stats {
    let n = got.len() as f32;
    let mut max_abs = 0.0f32;
    let mut sg = 0.0f32;
    let mut sw = 0.0f32;
    let mut sgg = 0.0f32;
    let mut sww = 0.0f32;
    let mut sgw = 0.0f32;
    let mut diff_sq = 0.0f32;
    for k in 0..got.len() {
        let (g, w) = (got[k], want[k]);
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
        }
        diff_sq += (g - w) * (g - w);
        sg += g;
        sw += w;
        sgg += g * g;
        sww += w * w;
        sgw += g * w;
    }
    let mg = sg / n;
    let mw = sw / n;
    let cov = sgw / n - mg * mw;
    let varg = sgg / n - mg * mg;
    let varw = sww / n - mw * mw;
    let corr = cov / (varg.sqrt() * varw.sqrt() + 1e-12);
    let rel_l2 = (diff_sq / (sww + 1e-12)).sqrt();
    Stats {
        corr,
        max_abs,
        rel_l2,
    }
}
