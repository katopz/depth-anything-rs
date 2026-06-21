# depth-anything.rs — Rust/candle port status

This is a from-scratch Rust port of [`depth-anything.cpp`](../README.md) using
the [candle](https://github.com/huggingface/candle) tensor library. The goal is
a real, comparable Rust implementation of the Depth Anything 3 forward pass —
not an FFI wrapper over the C++ engine — so that `Rust/candle` vs `C++/ggml` is
an honest engine-vs-engine benchmark.

## What's implemented

### Core forward path (DA3-BASE depth + pose) — ✅ complete

Every module on the DA3-BASE production path is ported and unit-tested:

| Module | C++ source | Rust module | Status |
|---|---|---|---|
| GGUF v3 reader (metadata + tensors, mmap) | `model_loader.cpp` | `src/gguf.rs` | ✅ |
| Config from GGUF KV | `model_loader.hpp` | `src/config.rs` | ✅ |
| GGUF → candle `VarBuilder` bridge | — | `src/weights.rs` | ✅ |
| Image preprocessing (cv2-faithful resize + ImageNet norm) | `preprocess.cpp` | `src/preprocess.rs` | ✅ |
| Bicubic positional embedding (cached) | `dino_backbone.cpp` | `src/pos_embed.rs` | ✅ |
| 2D RoPE (local + nodiff tables) | `rope2d.cpp` | `src/rope2d.rs` | ✅ |
| UV positional embedding (cached, the ~90 ms hot path) | `uv_posembed.cpp` | `src/uv_embed.rs` | ✅ |
| Multi-head attention (qknorm + 2D RoPE + sdpa) | `attention.cpp` | `src/attention.rs` | ✅ |
| ViT transformer block (pre-norm, LayerScale, GELU-erf / SwiGLu) | `vit_block.cpp` | `src/vit_block.rs` | ✅ |
| DINO backbone (patch embed, camera-token inject, feat assembly) | `dino_backbone.cpp` | `src/backbone.rs` | ✅ |
| DPT depth head (projectors, resize pyramid, refinenets, UV add) | `dpt_head.cpp`, `dpt_blocks.cpp` | `src/dpt_head.rs` | ✅ |
| Camera pose head (MLP → 9-vec → ext/intr decode) | `cam_pose.cpp` | `src/cam_pose.rs` | ✅ |
| Depth activation + PFM/PNG/JSON writers | `depth_export.cpp`, `pose_export.cpp` | `src/depth_export.rs` | ✅ |
| Engine orchestrator (load + forward + timings) | `engine.cpp` | `src/engine.rs` | ✅ |

### Other DA3 paths — ✅ all ported

| Path | Status |
|---|---|
| DA2 relative/metric depth (`depth_relative`) | ✅ activation handled (`relu` / `sigmoid·max_depth`); routes through the same head |
| Mono DA3 (depth + sky) | ✅ sky head present; activation `exp`/`relu` |
| Metric DA3 ViT-L (`cat_token=false`) | ✅ handled by config branch |
| F16 / Q8_0 GGUF | ✅ dequantized to F32 at load |
| K-quants (Q4_K, Q5_K, Q6_K, Q8_K) | ✅ load-time dequant to F32 (`src/kquants.rs`, ported from ggml's `dequantize_row_q{n}_K`) — verified against the C++ `quantize` tool's output |
| 3D-Gaussian reconstruction (Giant + GS head) | ✅ ported (`src/gs_head.rs` + `src/gs_adapter.rs` + `Engine::reconstruct_{image,path}`). Math unit-tested (22 new tests: mat3_inverse, quaternion<->matrix round-trips, wigner_D identities, focal/sigmoid/SH masking, hand-derived means/scales/rotations/harmonics/opacities); end-to-end on a DA3-Giant GGUF pending model availability (see Verification) |
| Nested metric (two-branch alignment) | ✅ ported (`src/nested.rs`, `Engine::load_nested` + `depth_metric_{image,path}`); metric-branch tensor aliasing (`m_vit.*`/`m_head.*` → `vit.*`/`head.*`) in `src/weights.rs`. Math unit-tested; end-to-end on a nested-metric GGUF pending model availability (see Verification) |
| Ray-based pose (aux head + solver) | ✅ ported (`src/ray_pose.rs` + `src/linalg.rs`); aux ray head in `src/dpt_head.rs::forward_rays`; engine entry `Engine::depth_pose_rays_{image,path}`. Solver math verified; end-to-end on a `--with-aux` GGUF pending a model rebuild (see Verification) |
| Multi-view batched forward | ✅ ported (`Backbone::forward_mv` in `src/backbone.rs`; engine entry `Engine::depth_pose_multi_{image,path}`). Cross-view global attention at odd blocks ≥ `alt_start`; saddle-balanced reference-view selection for S ≥ 3 |
| glb / COLMAP exporters | ✅ ported (`src/glb_export.rs`, `src/colmap_export.rs`, `src/reconstruct.rs`); engine entry `Engine::depth_pose_export_{image,path}` + `ExportSpec`. End-to-end verified against the C++ engine on `street.jpg` (see Verification) |
| Gaussian `.ply` exporter | ✅ ported (`src/ply_export.rs::write_gaussian_ply`); binary-little-endian 3DGS-compatible output, logit-opacity + log-scale transforms unit-tested with hand-computed references (7 new tests) |

### Quantization

The Rust port dequantizes **all** supported GGUF dtypes (F16, Q8_0, Q4_K,
Q5_K, Q6_K, Q8_K) to `f32` at load time. K-quants are decoded by
`src/kquants.rs`, a direct port of ggml's `dequantize_row_q{n}_K`. This means:

- **Inference numerical path is identical** regardless of source quant — all
  matmuls run in `f32`, same as the C++ engine's parity-preserving path.
- **Load time** is *slower* for quantized GGUFs (we expand q8_0/q4_k → f32 in
  Rust; the C++ engine keeps tensors quantized and dequants inside each matmul).
- **Memory footprint** is the f32 size for every quant (the C++ q8_0/q4_k
  memory advantage does not apply here).
- **Quantization accuracy is preserved**: q4_k end-to-end depth correlates
  **0.9968** with the f32 baseline, q5_k **0.9995**, q6_k **0.9999** on the
  street sample (the rest of the error is the quantizer's, not the dequantizer's).

So a `q4_k` benchmark will favor C++ on load time and memory, and the
inference-time comparison is effectively `candle-f32` vs `ggml-f32` (with
ggml running its on-the-fly q4_k dequant). An apples-to-apples inference
comparison is `Rust/f32 GGUF` vs `C++/f32 GGUF`.

## Verification

The port is **end-to-end parity-verified against the C++ engine** on a real
DA3-BASE f32 GGUF (converted from HuggingFace `depth-anything/DA3-BASE`).
Measured on an i7-13700K (Windows, clang build), PFM-vs-PFM on the 4 bundled
sample images, all at the native 504×336 processing resolution:

| sample    | corr (Rust vs C++) | max \|d\| |
|-----------|--------------------|-----------|
| street    | 0.999972           | 2.97e-02  |
| canyon    | 0.999994           | 6.22e-02  |
| desk      | 0.999978           | 3.48e-02  |
| mountains | 0.999986           | 4.74e-02  |

All samples correlate **0.99997+**; the residual diff is f32 accumulation
noise across 12 ViT blocks + the DPT decoder. See [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)
for the full latency + parity table.

The forward pass was also checked stage-by-stage against the PyTorch reference
(`scripts/dump_reference.py`, fixed 224×224 input): backbone features and head
depth both correlate **0.99999+** with the PyTorch gold tensors.

Run the built-in parity harnesses any time the forward/preprocess code changes:

```sh
# Generate the gold references (needs the DA3 Python env — see scripts/da3_reference.py).
python scripts/dump_reference.py           # dumps/reference.gguf (forward)
python scripts/dump_preproc_real.py        # dumps/reference_preproc_real.gguf (preprocess)

# Forward parity (Rust forward vs PyTorch gold, on the 224×224 fixture).
cargo run --release --example parity -- \
  --model models/depth-anything-base-f32.gguf --ref dumps/reference.gguf
# expect: head_depth correlation +0.99988

# Preprocess parity (Rust resize+normalize vs the genuine cv2 InputProcessor).
cargo run --release --example parity_preproc -- \
  --model models/depth-anything-base-f32.gguf \
  --png dumps/preproc_real_input.png --ref dumps/reference_preproc_real.gguf
# expect: max|d| = 0.0 (bit-exact)
```

K-quant dequant parity (Rust `src/kquants.rs` vs the f32 pre-quantization
weights). This is the regression test for the K-quant dequantizer — any
bit-manipulation bug would crater the correlations well below 0.95.

```sh
# Produce the quantized GGUFs from the f32 model with the C++ quantizer.
./build/examples/cli/da3-cli quantize models/depth-anything-base-f32.gguf \
    models/depth-anything-base-q4_k.gguf q4_k
# repeat for q5_k, q6_k as desired

# Verify every quantized tensor dequantizes back to within quantization error.
cargo run --release --example kquant_parity -- \
  --ref   models/depth-anything-base-f32.gguf \
  --quant models/depth-anything-base-q4_k.gguf
# expect: 53/53 tensors corr >= 0.95; worst (q4_k) ≈ +0.9965
```

Ray-pose solver parity. The deterministic solver math
(`src/ray_pose.rs` + `src/linalg.rs`) is covered by 22 unit tests
(RANSAC cloud build, weighted homography, Householder QR / QL, cyclic Jacobi
Eigen, bit-exact `mt19937`). The full aux ray head + solver end-to-end needs a
`--with-aux` GGUF (not currently in `models/`):

```sh
# Build an aux GGUF (one-time; needs the DA3 Python env + torch).
python scripts/convert_da3_to_gguf.py \
  --model models/DA3-BASE \
  --output models/depth-anything-base-aux-f32.gguf --with-aux

# Run the Rust ray-pose path (optionally vs a C++ reference pose JSON).
cargo run --release --example ray_pose_parity -- \
  --model models/depth-anything-base-aux-f32.gguf \
  --input assets/samples/street.jpg \
  --ref-pose dumps/cpp_ray_pose.json   # optional, loose tolerances
```

The `--ref-pose` comparison is **loose by design** (rotation geodesic < 1.0°,
focal/pp rel-err < 2%): the reference RANSAC samples via `torch.randperm` and
is itself non-deterministic, while the Rust production path uses its own
seeded sampling. For bit-exact parity, inject the C++ solver's indices through
the gated `Engine::depth_pose_rays_image_with` (see `src/ray_pose.rs`'s
parity note).

### 3D exporters (glb / COLMAP)

The exporter pipeline (`src/reconstruct.rs` geometry primitives +
`src/glb_export.rs` + `src/colmap_export.rs`) is end-to-end verified against
the C++ engine on the bundled `street.jpg` (DA3-BASE f32, 504×336 processed,
camera-pose head present):

```sh
# Rust
cargo run --release --example da3 -- depth \
    --model models/depth-anything-base-f32.gguf \
    --input assets/samples/street.jpg --glb target/street.glb --colmap target/street_colmap/

# C++ (same image, same model)
./build/examples/cli/da3-cli.exe depth \
    --model models/depth-anything-base-f32.gguf \
    --input assets/samples/street.jpg --glb build/cpp_street.glb --colmap build/cpp_street_colmap/
```

Measured parity (Rust vs C++):

| metric                       | result   | note |
|------------------------------|----------|------|
| COLMAP point count           | **101,606 == 101,606** | identical conf-boundary filtering (same algorithm) |
| Camera intrinsics `fx`,`fy`  | rel-err **1.0e-4**      | engine f32 noise (depth correlates 0.99997+) |
| Principal point `cx`,`cy`    | **bit-exact**          | `512.0, 340.0` in both |
| First back-projected point   | \|Δxyz\| **2e-4**        | engine f32 noise |
| conf-boundary pixel flips    | ±22,928 (symmetric)    | expected: conf carries f32 noise, so the 40th-percentile threshold lands at a slightly different value |

The exporter **math** (back-projection, `inv3`/`inv4`, `rotmat2qvec`, percentile,
the glTF/COLMAP binary layouts) is unit-tested with hand-computed references in
`src/reconstruct.rs`, `src/glb_export.rs`, `src/colmap_export.rs` (21 new tests).
The GLB output is a structurally valid glTF 2.0 (POINTS + LINES primitives);
the COLMAP `.bin` files round-trip through a binary parser. The remaining
point-set divergence versus C++ is the documented engine-level f32 noise
propagating through the confidence threshold — **not an exporter bug**.

To compare Rust vs C++ end-to-end on any image:

```sh
# C++ reference
./build/examples/cli/da3-cli depth --model models/depth-anything-base-f32.gguf \
    --input assets/samples/street.jpg --pfm build/cpp_depth.pfm

# Rust port
cargo run --release --example da3 -- depth \
    --model models/depth-anything-base-f32.gguf \
    --input assets/samples/street.jpg --pfm target/rs_depth.pfm

# Compare (correlation should be ≈ 1.0)
python -c "import numpy as np; \
  a=np.fromfile('build/cpp_depth.pfm',dtype='f32',offset=15); \
  b=np.fromfile('target/rs_depth.pfm',dtype='f32',offset=15); \
  print('corr', np.corrcoef(a[:b.size],b)[0,1])"
```

### Nested metric alignment

The alignment math (`quantile`, `process_mono_sky`, `NestedAligner::align`) is
unit-tested with hand-computed references in `src/nested.rs` (23 new tests):
least-squares `scale_factor = Σ(a·b)/Σ(b·b)`, the focal/300 metric rescale,
the median-confidence align mask, the 99th-percentile sky-fill with the 200
cap, and the translation-column rescale `ext[3,7,11] *= scale_factor`. The
metric-branch tensor aliasing (`m_vit.*`/`m_head.*` → `vit.*`/`head.*`) is
implemented in `src/weights.rs::GgufBackend::new_metric` and mirrors the alias
map `ModelLoader::load` inserts in `src/model_loader.cpp`.

**End-to-end C++ parity is pending** a nested-metric GGUF pair (anyview GIANT +
metric ViT-L). `Engine::load_nested` correctly rejects a non-metric GGUF with
`gguf error: missing m_vit.embed_dim` (verified by passing the base DA3 GGUF as
`--metric-model`). Once a metric GGUF is available, run:

```sh
# Rust nested metric depth
cargo run --release --example da3 -- depth \
    --model models/depth-anything-giant-f32.gguf \
    --metric-model models/depth-anything-metric-f32.gguf \
    --input assets/samples/street.jpg --pfm target/rs_nested.pfm

# C++ reference
./build/examples/cli/da3-cli depth \
    --model models/depth-anything-giant-f32.gguf \
    --metric-model models/depth-anything-metric-f32.gguf \
    --input assets/samples/street.jpg --pfm build/cpp_nested.pfm

# scale_factor + depth should match within engine f32 noise.
python -c "import numpy as np; \
  a=np.fromfile('build/cpp_nested.pfm',dtype='f32',offset=15); \
  b=np.fromfile('target/rs_nested.pfm',dtype='f32',offset=15); \
  print('corr', np.corrcoef(a[:b.size],b)[0,1])"
```

### 3D-Gaussian reconstruction

The adapter geometry (`GsAdapter::build` in `src/gs_adapter.rs`) is unit-tested
with hand-computed references (22 new tests): `mat3_inverse` identity / diagonal
/ singular / product-is-identity, `quat_xyzw_to_mat` identity / 180°-about-z,
`mat_to_quat_xyzw` identity / 180°-about-y round-trips, `wigner_D` identities
for l=1 and l=2 plus an alpha-rotation gate, and end-to-end `build` cases for
opacity identity, means unprojection, focal/sigmoid scale, SH DC passthrough,
the camera→world quaternion pipeline, and the `pred_offset_{depth,xy}` switches.
The `.ply` writer is unit-tested bit-exact (`src/ply_export.rs`, 7 new tests):
header format, length guards, one-record bit-pattern, opacity clamp-then-logit,
scale clamp-then-log, multi-record layout.

**End-to-end C++ parity is pending** a DA3-Giant GGUF that includes the GSDPT
head (`gs.*`). `Engine::reconstruct_image` correctly rejects a non-GS GGUF with
`reconstruct: this model has no GSDPT head (gs.*)` (verified by passing the
base DA3 GGUF as `--model`).

Two env-gated integration tests mirror the C++ `tests/test_gs_adapter.cpp` /
`tests/test_gs_head.cpp` flow and SKIP automatically when the Giant artifacts
aren't present (so they run in every `cargo test` without noise):

```sh
# Adapter-only parity (host math; needs just the baseline, no GGUF model).
DA_TEST_BASELINE_GIANT=dumps/reference_giant.gguf \
    cargo test --release --test gs_adapter_giant_parity -- --nocapture

# End-to-end parity (needs both the GGUF model and the baseline).
DA_TEST_GGUF_GIANT=models/depth-anything-giant-f32.gguf \
DA_TEST_BASELINE_GIANT=dumps/reference_giant.gguf \
    cargo test --release --test reconstruct_giant_parity -- --nocapture
```

Both gate every Gaussian attribute (means / scales / rotations / harmonics /
opacities) at `atol = rtol = 2e-3`, matching the C++ tolerance.

Alternatively, compare Rust vs C++ `.ply` outputs directly:

```sh
# Rust 3D-Gaussian .ply
cargo run --release --example da3 -- reconstruct \
    --model models/depth-anything-giant-f32.gguf \
    --input assets/samples/street.jpg --ply target/rs_gaussians.ply

# C++ reference
./build/examples/cli/da3-cli reconstruct \
    --model models/depth-anything-giant-f32.gguf \
    --input assets/samples/street.jpg --output-ply build/cpp_gaussians.ply

# Per-attribute correlation (means, scales, rotations, harmonics, opacities).
python -c "import numpy as np, struct; \
  def load(p): \
    b=open(p,'rb').read(); \
    i=b.find(b'end_header')+len('end_header\n'); \
    return np.frombuffer(b[i:],dtype='f32').reshape(-1,14); \
  a=load('build/cpp_gaussians.ply'); b=load('target/rs_gaussians.ply'); \
  for k,name in enumerate(['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity', \
                           'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']): \
    print(name, np.corrcoef(a[:,k],b[:,k])[0,1])"
```

## Building

```sh
cargo build --release --examples
```

CUDA: `cargo build --release --examples --features cuda` (needs the CUDA
toolkit; otherwise falls back to CPU). Force CPU with `DA_DEVICE=cpu`.

## Running

```sh
# Inference CLI (depth, depth+pose, info)
cargo run --release --example da3 -- info   --model models/depth-anything-base-f32.gguf
cargo run --release --example da3 -- depth  --model models/... --input photo.jpg \
    --png depth.png --pfm depth.pfm --pose pose.json

# 3D exports (single-view; requires a camera-pose head)
cargo run --release --example da3 -- depth --model models/... --input photo.jpg \
    --glb scene.glb --colmap sparse/            # COLMAP defaults to binary .bin
cargo run --release --example da3 -- depth --model models/... --input photo.jpg \
    --colmap sparse/ --colmap-text              # text .txt variant

# Nested metric-scale depth (anyview GIANT + metric ViT-L branches)
cargo run --release --example da3 -- depth \
    --model models/depth-anything-giant-f32.gguf \
    --metric-model models/depth-anything-metric-f32.gguf \
    --input photo.jpg --pfm metric_depth.pfm --pose pose.json

# Multi-view depth + pose (one backbone pass; saddle-balanced ref-view selection).
# Repeat --input for each view.
cargo run --release --example da3 -- multi \
    --model models/depth-anything-giant-f32.gguf \
    --input a.jpg --input b.jpg --input c.jpg

# 3D-Gaussian reconstruction (requires a GSDPT head; DA3-Giant).
cargo run --release --example da3 -- reconstruct \
    --model models/depth-anything-giant-f32.gguf \
    --input photo.jpg --ply gaussians.ply

# Benchmark (sustained latency, p50/p95/p99, per-stage timings)
cargo run --release --example bench -- --model models/... --input photo.jpg \
    --repeat 25 --warmup 3 [--pose]
```

## Expected benchmark outcome

**Measured, not predicted** — see [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)
for the full table. Summary (DA3-BASE depth @504×336, i7-13700K, f32 GGUF,
median over 25 iters):

| threads | C++/ggml | Rust/candle | Rust ÷ C++ |
|--------:|---------:|------------:|-----------:|
| 1       | 3075 ms  | 4042 ms     | 1.31×      |
| 8       | 486 ms   | 2141 ms     | 4.40×      |
| 16      | 403 ms   | 2195 ms     | 5.45×      |

The C++/ggml engine is heavily tuned (tinyBLAS AVX2 GEMM, Winograd conv, fused
flash attention) and parallelizes far better than candle's generic CPU backend.
At 1 thread the gap is small (**1.3×**); it widens with threads because ggml's
kernels scale and candle's plateau (candle even slows from 8→16 threads here).
This is the honest **engine gap, not a language gap** — the Rust port is
numerically faithful (0.99997+ correlation with C++) and the gap lives entirely
in the tensor-library kernels.
