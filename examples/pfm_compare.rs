//! Compare two single-channel PFM files for parity (max abs diff, RMS, etc).
//!
//! Usage: pfm_compare a.pfm b.pfm

use std::fs::File;
use std::io::{BufReader, Read};

fn read_pfm(path: &str) -> std::io::Result<(Vec<f32>, usize, usize)> {
    let mut f = BufReader::new(File::open(path)?);
    // Header: "<magic>\n<w> <h>\n<scale>\n" — magic is "PF" (RGB) or "Pf" (gray).
    let mut magic = String::new();
    loop {
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte)?;
        if byte[0] == b'\n' {
            break;
        }
        magic.push(byte[0] as char);
    }
    let mut dims = String::new();
    loop {
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte)?;
        if byte[0] == b'\n' {
            break;
        }
        dims.push(byte[0] as char);
    }
    let mut scale_s = String::new();
    loop {
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte)?;
        if byte[0] == b'\n' {
            break;
        }
        scale_s.push(byte[0] as char);
    }
    let parts: Vec<&str> = dims.split_whitespace().collect();
    let w: usize = parts[0].parse().unwrap();
    let h: usize = parts[1].parse().unwrap();
    let scale: f32 = scale_s.trim().parse().unwrap();
    let channels = if magic.starts_with("PF") { 3 } else { 1 };
    let count = w * h * channels;
    let mut bytes = vec![0u8; count * 4];
    f.read_exact(&mut bytes)?;
    let mut data = vec![0.0f32; count];
    let little = scale < 0.0;
    for i in 0..count {
        let b = &bytes[i * 4..i * 4 + 4];
        data[i] = if little {
            f32::from_le_bytes(b.try_into().unwrap())
        } else {
            f32::from_be_bytes(b.try_into().unwrap())
        };
    }
    Ok((data, w, h))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: pfm_compare a.pfm b.pfm");
        std::process::exit(1);
    }
    let (a, aw, ah) = read_pfm(&args[1]).expect("read a");
    let (b, bw, bh) = read_pfm(&args[2]).expect("read b");
    if (aw, ah) != (bw, bh) {
        eprintln!("dimension mismatch: a={}x{} b={}x{}", aw, ah, bw, bh);
        std::process::exit(1);
    }
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut max_abs = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum_ab = 0.0f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut a_min = f32::INFINITY;
    let mut a_max = f32::NEG_INFINITY;
    for i in 0..n {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        let d = (av - bv).abs();
        if d > max_abs as f64 {
            max_abs = d as f32;
        }
        if a[i] < a_min {
            a_min = a[i];
        }
        if a[i] > a_max {
            a_max = a[i];
        }
        sum_sq_diff += (av - bv) * (av - bv);
        sum_sq += av * av;
        sum_a += av;
        sum_b += bv;
        sum_ab += av * bv;
    }
    let rms_diff = (sum_sq_diff / n as f64).sqrt();
    let rms = (sum_sq / n as f64).sqrt();
    let rel = rms_diff / rms;
    let mean_a = sum_a / n as f64;
    let mean_b = sum_b / n as f64;
    let var_a = (sum_sq - sum_a * sum_a / n as f64) / n as f64;
    let var_b = var_a; // assume same
    let cov = (sum_ab - sum_a * sum_b / n as f64) / n as f64;
    let corr = cov / (var_a.sqrt() * var_b.sqrt());

    eprintln!("n values    : {n}");
    eprintln!("depth range : [{:.6}, {:.6}]", a_min, a_max);
    eprintln!("max abs diff: {:.6e}", max_abs);
    eprintln!("rms diff    : {:.6e}", rms_diff);
    eprintln!("rms signal  : {:.6e}", rms);
    eprintln!("rel diff    : {:.6e}", rel);
    eprintln!("correlation : {:.8}", corr);
    eprintln!("mean a      : {:.6e}", mean_a);
    eprintln!("mean b      : {:.6e}", mean_b);
}
