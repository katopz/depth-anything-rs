//! Preprocess microbenchmark: measure resize_cubic, resize_area, and CHW
//! normalize stages separately for the DA3 default image size.
//!
//! Run with:
//!   RAYON_NUM_THREADS=32 cargo run --release --example preprocess_bench

use depth_anything::config::Config;
use depth_anything::preprocess::{preprocess_real, resize_area, resize_cubic, Image};
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
    eprintln!("  {label:<55} : {per:>8.3} ms");
    per
}

fn main() {
    eprintln!(
        "[preprocess_bench] threads = {}",
        rayon::current_num_threads()
    );

    // Load the canyon test image (typical input).
    let img = Image::load("assets/samples/canyon.jpg").expect("load canyon.jpg");
    eprintln!("input image: {}x{} ({} bytes)", img.w, img.h, img.rgb.len());

    // Simulate the DA3 default resize path:
    // - longest side → 518 (target=518, patch=14 → 518 = 37*14)
    // - round to nearest multiple of 14
    let target = 518;
    let patch = 14;
    let longest = img.w.max(img.h);
    let scale = target as f64 / longest as f64;
    let nw1 = (scale * img.w as f64).round().max(1.0) as usize;
    let nh1 = (scale * img.h as f64).round().max(1.0) as usize;
    eprintln!("step1 resize: {nw1}x{nh1} (scale={scale:.4})");

    let r = 20;

    // Step 1 resize (area if downscale, cubic if upscale).
    let step1 = if scale < 1.0 {
        let _t = time_ms("step1 resize_area (full→step1)", r, 3, || {
            let _ = resize_area(&img, nw1, nh1);
        });
        resize_area(&img, nw1, nh1)
    } else {
        let _t = time_ms("step1 resize_cubic (full→step1)", r, 3, || {
            let _ = resize_cubic(&img, nw1, nh1);
        });
        resize_cubic(&img, nw1, nh1)
    };
    let _ = scale; // silence
    eprintln!("step1 result: {}x{}", step1.w, step1.h);

    // Step 2: round to multiple of patch.
    let nw2 = ((step1.w as f64 / patch as f64).round() * patch as f64) as usize;
    let nh2 = ((step1.h as f64 / patch as f64).round() * patch as f64) as usize;
    eprintln!("step2 resize: {nw2}x{nh2}");
    let step2 = if nw2 > step1.w || nh2 > step1.h {
        let _t = time_ms("step2 resize_cubic (step1→step2)", r, 3, || {
            let _ = resize_cubic(&step1, nw2, nh2);
        });
        resize_cubic(&step1, nw2, nh2)
    } else {
        let _t = time_ms("step2 resize_area (step1→step2)", r, 3, || {
            let _ = resize_area(&step1, nw2, nh2);
        });
        resize_area(&step1, nw2, nh2)
    };

    // CHW + normalize.
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    time_ms("CHW + normalize", r, 3, || {
        let (h, w) = (step2.h, step2.w);
        let mut chw = vec![0.0f32; 3 * h * w];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let v = step2.rgb[(y * w + x) * 3 + c] as f32 / 255.0;
                    chw[(c * h + y) * w + x] = (v - mean[c]) / std[c];
                }
            }
        }
    });

    // Full preprocess_real for reference.
    let mut cfg = Config::default();
    cfg.img_resize_target = 518;
    cfg.patch_size = 14;
    cfg.img_mean = mean.to_vec();
    cfg.img_std = std.to_vec();
    let _ = &mut cfg;
    time_ms("preprocess_real (full)", r, 3, || {
        let _ = preprocess_real(&img, &cfg).unwrap();
    });
}
