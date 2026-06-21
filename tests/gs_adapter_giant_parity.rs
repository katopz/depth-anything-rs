//! Integration parity test for [`GsAdapter`] against a dumped Giant baseline.
//!
//! Mirrors `tests/test_gs_adapter.cpp`: feeds the dumped Giant raw_gs/gs_conf/
//! depth/extrinsics/intrinsics into the pure-host [`GsAdapter`] and gates each
//! output attribute (means / scales / rotations / harmonics / opacities) against
//! the dumped reference at `2e-3` (atol + rtol).
//!
//! # Environmental gate
//!
//! This test is **skipped** (via early-return, not `#[ignore]`, so it shows up
//! in `cargo test` output) unless both environment variables are set:
//!
//! - `DA_TEST_BASELINE_GIANT` — path to the baseline GGUF dumped by
//!   `scripts/dump_reference.py --giant` (contains the `raw_gs`, `depth_g`,
//!   `extrinsics_g`, `intrinsics_g`, `gs_conf`, and `gs_*` reference tensors).
//!
//! No GGUF model is required — the adapter is pure host math, so the baseline
//! GGUF alone is sufficient.
//!
//! # When it runs
//!
//! ```sh
//! DA_TEST_BASELINE_GIANT=dumps/reference_giant.gguf \
//!     cargo test --release --test gs_adapter_giant_parity -- --nocapture
//! ```

#![allow(clippy::needless_range_loop)]

use depth_anything::{Gaussians, GsAdapter};

/// atol + rtol used by the C++ parity test (`tests/test_gs_adapter.cpp`).
const ATOL: f64 = 2e-3;
const RTOL: f64 = 2e-3;

/// Load a tensor from the baseline GGUF as a flat `Vec<f32>` (ggml natural order:
/// fastest-varying dim first). Mirrors `da_parity::load_baseline`.
fn load_baseline_f32(file: &depth_anything::GgufFile, name: &str) -> Option<Vec<f32>> {
    let info = file.tensor_info(name)?;
    // The adapter math doesn't care about the on-disk dtype — `tensor_f32`
    // dequantizes F16 / Q8_0 / Q4_K / Q5_K / Q6_K to f32 transparently.
    file.tensor_f32(name)
        .ok()
        .inspect(|v| debug_assert_eq!(v.len(), info.n_elems()))
}

/// Compare got vs ref element-wise under `atol + rtol * |ref|`. Returns
/// `(max_abs_diff, mean_abs_diff, n_violations)` and prints a one-line summary
/// to stderr matching the C++ `da_parity::compare` format.
fn compare(got: &[f32], ref_: &[f32], label: &str) -> (f64, f64, usize) {
    assert_eq!(
        got.len(),
        ref_.len(),
        "[{label}] size mismatch got={} ref={}",
        got.len(),
        ref_.len()
    );
    if got.is_empty() {
        eprintln!("[{label}] empty vectors (got=ref=0) -> FAIL");
        return (f64::INFINITY, f64::INFINITY, 1);
    }
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut n_viol = 0usize;
    let mut worst_i = 0usize;
    for (i, (g, r)) in got.iter().zip(ref_.iter()).enumerate() {
        let d = ((g - r) as f64).abs();
        sum_abs += d;
        if d > max_abs {
            max_abs = d;
            worst_i = i;
        }
        let tol = ATOL + RTOL * (*r as f64).abs();
        if d > tol {
            n_viol += 1;
        }
    }
    let mean = sum_abs / got.len() as f64;
    let ok = n_viol == 0;
    eprintln!(
        "[{label}] n={} max|d|={:.3e} mean|d|={:.3e} viol={} (worst@{} got={:.5} ref={:.5}) -> {}",
        got.len(),
        max_abs,
        mean,
        n_viol,
        worst_i,
        got[worst_i],
        ref_[worst_i],
        if ok { "OK" } else { "FAIL" }
    );
    (max_abs, mean, n_viol)
}

/// Assert every attribute of `got` matches `ref_` within tolerance; abort on
/// the first attribute that fails (mirrors the C++ `ok &= ...` aggregate).
fn assert_attribute(got: &[f32], ref_: &[f32], label: &str) {
    let (max_abs, mean, n_viol) = compare(got, ref_, label);
    assert!(
        n_viol == 0,
        "[{label}] {n_viol} / {} elements violate atol={ATOL} rtol={RTOL} \
         (max|d|={max_abs:.3e}, mean|d|={mean:.3e})",
        got.len()
    );
}

#[test]
fn gs_adapter_matches_giant_baseline() {
    let Some(baseline_path) = std::env::var_os("DA_TEST_BASELINE_GIANT") else {
        eprintln!(
            "[gs_adapter_giant_parity] SKIP: DA_TEST_BASELINE_GIANT not set \
             (no Giant baseline available)."
        );
        return;
    };

    let baseline = depth_anything::GgufFile::open(&baseline_path).unwrap_or_else(|e| {
        panic!(
            "[gs_adapter_giant_parity] failed to open baseline `{}`: {e}",
            baseline_path.to_string_lossy()
        )
    });

    // Required inputs.
    let raw_gs = load_baseline_f32(&baseline, "raw_gs").expect("baseline missing `raw_gs`");
    let gs_conf = load_baseline_f32(&baseline, "gs_conf").expect("baseline missing `gs_conf`");
    let depth = load_baseline_f32(&baseline, "depth_g").expect("baseline missing `depth_g`");
    let ext_v =
        load_baseline_f32(&baseline, "extrinsics_g").expect("baseline missing `extrinsics_g`");
    let intr_v =
        load_baseline_f32(&baseline, "intrinsics_g").expect("baseline missing `intrinsics_g`");

    // The C++ test hardcodes 224x224 (the giant fixture size).
    const H: usize = 224;
    const W: usize = 224;
    assert_eq!(
        raw_gs.len(),
        H * W * 37,
        "raw_gs size mismatch (expected H*W*37 = {})",
        H * W * 37
    );
    assert_eq!(depth.len(), H * W, "depth_g size mismatch");
    assert_eq!(gs_conf.len(), H * W, "gs_conf size mismatch");
    assert_eq!(ext_v.len(), 12, "extrinsics_g must have 12 elements (3x4)");
    assert_eq!(intr_v.len(), 9, "intrinsics_g must have 9 elements (3x3)");

    let mut ext = [0.0f32; 12];
    let mut intr = [0.0f32; 9];
    ext.copy_from_slice(&ext_v);
    intr.copy_from_slice(&intr_v);

    let ad = GsAdapter::default();
    let g: Gaussians = ad
        .build(&raw_gs, &depth, &gs_conf, &ext, &intr, H, W)
        .expect("GsAdapter::build failed");

    // Reference outputs (each one is independently optional in the C++ test
    // but treated as required here — a missing tensor is a baseline bug).
    let ref_means = load_baseline_f32(&baseline, "gs_means").expect("baseline missing `gs_means`");
    let ref_scales =
        load_baseline_f32(&baseline, "gs_scales").expect("baseline missing `gs_scales`");
    let ref_rotations =
        load_baseline_f32(&baseline, "gs_rotations").expect("baseline missing `gs_rotations`");
    let ref_harmonics =
        load_baseline_f32(&baseline, "gs_harmonics").expect("baseline missing `gs_harmonics`");
    let ref_opacities =
        load_baseline_f32(&baseline, "gs_opacities").expect("baseline missing `gs_opacities`");

    assert_attribute(&g.means, &ref_means, "gs_means");
    assert_attribute(&g.scales, &ref_scales, "gs_scales");
    assert_attribute(&g.rotations, &ref_rotations, "gs_rotations");
    assert_attribute(&g.harmonics, &ref_harmonics, "gs_harmonics");
    assert_attribute(&g.opacities, &ref_opacities, "gs_opacities");
}
