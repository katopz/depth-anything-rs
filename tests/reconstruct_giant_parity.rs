//! End-to-end integration test for the 3D-Gaussian reconstruction pipeline
//! against a DA3-Giant GGUF.
//!
//! Mirrors the C++ `da3-cli reconstruct` path: load the Giant GGUF through
//! [`Engine`], run [`Engine::reconstruct_path`] on a sample image, and gate
//! the per-attribute Gaussian output against a baseline GGUF dumped by
//! `scripts/dump_reference.py --giant` (the same one used by the C++
//! `tests/test_gs_adapter.cpp`).
//!
//! # Environmental gate
//!
//! This test is **skipped** unless **both** environment variables are set:
//!
//! - `DA_TEST_GGUF_GIANT` — path to a DA3-Giant GGUF (must include the
//!   `gs.*` head).
//! - `DA_TEST_BASELINE_GIANT` — path to the baseline GGUF containing the
//!   `gs_means` / `gs_scales` / `gs_rotations` / `gs_harmonics` /
//!   `gs_opacities` reference tensors.
//!
//! An optional sample image is read from `DA_TEST_INPUT` (defaults to
//! `assets/samples/street.jpg` if unset).
//!
//! # When it runs
//!
//! ```sh
//! DA_TEST_GGUF_GIANT=models/depth-anything-giant-f32.gguf \
//! DA_TEST_BASELINE_GIANT=dumps/reference_giant.gguf \
//! cargo test --release --test reconstruct_giant_parity -- --nocapture
//! ```
//!
//! # Tolerance
//!
//! Same as the C++ suite: `atol = rtol = 2e-3` per element on every Gaussian
//! attribute. The full forward pass introduces f32 accumulation noise beyond
//! the host-only adapter path covered by `gs_adapter_giant_parity`, so this
//! is the stricter end-to-end gate.

use depth_anything::{Engine, GgufFile};

/// atol + rtol used by the C++ parity tests.
const ATOL: f64 = 2e-3;
const RTOL: f64 = 2e-3;

/// Per-element parity check. Returns `(max_abs, mean_abs, n_viol)`.
fn compare(got: &[f32], ref_: &[f32], label: &str) -> (f64, f64, usize) {
    assert_eq!(
        got.len(),
        ref_.len(),
        "[{label}] size mismatch got={} ref={}",
        got.len(),
        ref_.len()
    );
    if got.is_empty() {
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
fn reconstruct_matches_giant_baseline() {
    let (Some(gguf_path), Some(baseline_path)) = (
        std::env::var_os("DA_TEST_GGUF_GIANT"),
        std::env::var_os("DA_TEST_BASELINE_GIANT"),
    ) else {
        eprintln!(
            "[reconstruct_giant_parity] SKIP: DA_TEST_GGUF_GIANT and/or \
             DA_TEST_BASELINE_GIANT not set (no Giant model + baseline available)."
        );
        return;
    };

    let input_path =
        std::env::var("DA_TEST_INPUT").unwrap_or_else(|_| "assets/samples/street.jpg".to_string());

    let engine = Engine::load(gguf_path.to_str().unwrap(), None).unwrap_or_else(|e| {
        panic!(
            "[reconstruct_giant_parity] failed to load GGUF `{}`: {e}",
            gguf_path.to_string_lossy()
        )
    });
    assert!(
        engine.has_gs_head(),
        "GGUF `{}` has no GSDPT head (`gs.*`); need a DA3-Giant GGUF",
        gguf_path.to_string_lossy()
    );

    let out = engine
        .reconstruct_path(&input_path)
        .unwrap_or_else(|e| panic!("[reconstruct_giant_parity] reconstruct_path failed: {e}"));
    eprintln!(
        "[reconstruct_giant_parity] reconstructed {} Gaussians (H={}, W={})",
        out.gaussians.n, out.h, out.w
    );

    let baseline = GgufFile::open(&baseline_path).unwrap_or_else(|e| {
        panic!(
            "[reconstruct_giant_parity] failed to open baseline `{}`: {e}",
            baseline_path.to_string_lossy()
        )
    });

    let g = &out.gaussians;
    let ref_means = baseline
        .tensor_f32("gs_means")
        .expect("baseline missing `gs_means`");
    let ref_scales = baseline
        .tensor_f32("gs_scales")
        .expect("baseline missing `gs_scales`");
    let ref_rotations = baseline
        .tensor_f32("gs_rotations")
        .expect("baseline missing `gs_rotations`");
    let ref_harmonics = baseline
        .tensor_f32("gs_harmonics")
        .expect("baseline missing `gs_harmonics`");
    let ref_opacities = baseline
        .tensor_f32("gs_opacities")
        .expect("baseline missing `gs_opacities`");

    assert_attribute(&g.means, &ref_means, "gs_means");
    assert_attribute(&g.scales, &ref_scales, "gs_scales");
    assert_attribute(&g.rotations, &ref_rotations, "gs_rotations");
    assert_attribute(&g.harmonics, &ref_harmonics, "gs_harmonics");
    assert_attribute(&g.opacities, &ref_opacities, "gs_opacities");
}
