//! Benchmark harness: sustained inference latency and per-stage timings for
//! the Rust/candle engine.
//!
//! ```sh
//! cargo run --release --example bench -- --model M.gguf --input photo.jpg --repeat 25
//! ```
//!
//! Results are printed as a JSON object on stdout (one line) plus a
//! human-readable summary on stderr, matching the fields the C++ benchmark
//! reports (load_ms, infer_ms, etc.) so the two can be compared directly.

use std::process::ExitCode;
use std::time::Instant;

use depth_anything::{Engine, Timings};

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
    let mut repeat: usize = 10;
    let mut warmup: usize = 3;
    let mut want_pose = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                model = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--input" | "-i" => {
                input = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--repeat" => {
                repeat = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(10);
                i += 2;
            }
            "--warmup" => {
                warmup = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3);
                i += 2;
            }
            "--pose" => {
                want_pose = true;
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!("usage: bench --model M --input I [--repeat N] [--warmup N] [--pose]");
                return Ok(());
            }
            other => {
                return Err(depth_anything::Error::Model(format!(
                    "unknown arg '{other}'"
                )));
            }
        }
    }
    if model.is_empty() || input.is_empty() {
        return Err(depth_anything::Error::Model(
            "usage: bench --model M --input I [--repeat N] [--warmup N] [--pose]".into(),
        ));
    }

    eprintln!("[bench] device: {:?}", depth_anything::default_device()?);

    let t0 = Instant::now();
    let engine = Engine::load_with_timings(&model, None)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let cfg = engine.config();
    eprintln!(
        "[bench] model loaded: {} (load incl. weight decode: {:.1} ms)",
        cfg.checkpoint_name, load_ms
    );
    eprintln!(
        "[bench] arch={:?} embed={} blocks={} heads={} head_dim={} ffn={:?} cat_token={} pose_head={}",
        cfg.arch,
        cfg.embed_dim,
        cfg.depth,
        cfg.num_heads,
        cfg.head_dim,
        cfg.ffn_type,
        cfg.cat_token,
        engine.has_pose_head()
    );

    // Warmup.
    for _ in 0..warmup {
        if want_pose {
            let _ = engine.depth_pose_path(&input)?;
        } else {
            let _ = engine.depth_path(&input)?;
        }
    }

    // Timed loop.
    let mut infer_ms = Vec::with_capacity(repeat);
    let mut stage_sums = Timings::default();
    for _ in 0..repeat {
        let t0 = Instant::now();
        if want_pose {
            let _ = engine.depth_pose_path(&input)?;
        } else {
            let _ = engine.depth_path(&input)?;
        }
        infer_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let t = engine.last_timings();
        stage_sums.preprocess_ms += t.preprocess_ms;
        stage_sums.backbone_ms += t.backbone_ms;
        stage_sums.head_ms += t.head_ms;
        stage_sums.pose_ms += t.pose_ms;
        stage_sums.activate_ms += t.activate_ms;
    }

    infer_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = infer_ms.iter().sum::<f64>() / repeat as f64;
    let median = infer_ms[infer_ms.len() / 2];
    let p50 = median;
    let p95 = infer_ms[((repeat as f64 - 1.0) * 0.95).round() as usize];
    let p99 = infer_ms[((repeat as f64 - 1.0) * 0.99).round() as usize];
    let min = infer_ms[0];
    let max = *infer_ms.last().unwrap();

    let n = repeat as f64;
    let stage = |x: f64| x / n;

    // Human-readable summary on stderr.
    eprintln!();
    eprintln!("==== depth-anything.rs (candle) benchmark ====");
    eprintln!("load_ms                : {:.1}", load_ms);
    eprintln!("infer_ms  mean/median  : {:.1} / {:.1}", mean, median);
    eprintln!(
        "infer_ms  p50/p95/p99  : {:.1} / {:.1} / {:.1}",
        p50, p95, p99
    );
    eprintln!("infer_ms  min/max      : {:.1} / {:.1}", min, max);
    eprintln!("-- per-stage mean (ms) --");
    eprintln!("  preprocess : {:.2}", stage(stage_sums.preprocess_ms));
    eprintln!("  backbone  : {:.2}", stage(stage_sums.backbone_ms));
    eprintln!("  head      : {:.2}", stage(stage_sums.head_ms));
    if want_pose {
        eprintln!("  pose      : {:.2}", stage(stage_sums.pose_ms));
    }
    eprintln!("  activate  : {:.2}", stage(stage_sums.activate_ms));

    // Machine-readable JSON on stdout (single line).
    let dims = (cfg.embed_dim, cfg.depth, cfg.num_heads, cfg.head_dim);
    println!(
        "{{\"engine\":\"rust-candle\",\"arch\":\"{:?}\",\"dims\":[{}, {}, {}, {}],\"load_ms\":{:.3},\"infer_mean_ms\":{:.3},\"infer_p50_ms\":{:.3},\"infer_p95_ms\":{:.3},\"infer_p99_ms\":{:.3},\"infer_min_ms\":{:.3},\"infer_max_ms\":{:.3},\"repeat\":{},\"preprocess_ms\":{:.3},\"backbone_ms\":{:.3},\"head_ms\":{:.3},\"pose_ms\":{:.3},\"activate_ms\":{:.3}}}",
        cfg.arch,
        dims.0, dims.1, dims.2, dims.3,
        load_ms, mean, p50, p95, p99, min, max, repeat,
        stage(stage_sums.preprocess_ms),
        stage(stage_sums.backbone_ms),
        stage(stage_sums.head_ms),
        stage(stage_sums.pose_ms),
        stage(stage_sums.activate_ms),
    );

    // If per-op profiling was enabled, dump the accumulated backbone/head
    // breakdown to stderr.
    depth_anything::fast_profile::dump_and_reset();

    Ok(())
}
