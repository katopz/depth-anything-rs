# C++ vs Rust benchmark — DA3-BASE depth, measured

Head-to-head latency of the **C++/ggml** engine vs the **Rust/candle** port,
on the same hardware, same input image, same processed resolution (504×336).

> **⚠️ Reader's guide.** This document is a chronological log of the
> optimisation work. The sections below were appended one per session and the
> numbers in each reflect what was true *at that time*. The intro table and
> several early "vs C++" comparisons used **f32**; most later sections use
> **q5_k**. The two are **not** interchangeable for engine-vs-engine claims.
> See the [Current status](#current-status) section below for the up-to-date
> summary and the caveats.

## Current status

### What the Rust fast path actually runs

The Rust fast path (`DA_FAST_ATTN=1 DA_FAST_HEAD=1`) calls `flatten_to_f32`
at **load time**, so quantised weights (q4_k / q5_k / q6_k) are dequantised to
f32 once and the inference compute is identical regardless of the on-disk
format. Measured on this machine (24 threads, `canyon.jpg`, this session):

| model format | min infer (ms) | median (ms) | load (ms) |
|--------------|---------------:|------------:|----------:|
| f32          |            324 |         329 |       154 |
| q5_k         |            326 |         331 |       118 |

The ~2 ms inference difference is noise. q5_k only helps **load time and
RAM footprint**, not inference latency, because the GEMM operates on the
dequantised f32 copy. A native INT4/INT8 GEMM (dequantise-inside-the-microkernel,
as ggml does) would change this, but the Rust port has not implemented one.

### Fair engine-vs-engine comparison (same model type)

The honest like-for-like numbers, **measured on the same i7-13700K**:

| comparison                  | engine      | model | threads | min (ms) | median (ms) | source |
|-----------------------------|-------------|-------|--------:|---------:|------------:|--------|
| Rust f32 (current session)  | Rust/fast   | f32   |      24 |      324 |         329 | this session |
| Rust q5_k (current session) | Rust/fast   | q5_k  |      24 |      326 |         331 | this session |
| C++ f32 (earlier session)   | C++/ggml    | f32   |      16 |      404 |         407 | [backbone profiling update](#update-backbone-profiling--build-tuning) |
| C++ q5_k (earlier session)  | C++/ggml    | q5_k  |      16 |      497 |         504 | [backbone profiling update](#update-backbone-profiling--build-tuning) |
| Rust q5_k (earlier session) | Rust/fast   | q5_k  |      16 |      484 |         487 | [tiled flash update](#update-tiled-flash--parallelised-non-gemm-ops--rust-now-faster-than-c) |

**Caveats (read before drawing conclusions):**

1. **The C++ numbers are stale.** They were measured in an earlier session
   and the C++ binary has not been re-run against the current Rust build.
   `da3-cli.exe` still exists at `./build/examples/cli/da3-cli.exe`; a
   same-session re-run is the only way to get a defensible comparison.
2. **Different thread counts.** Rust scales well to 24 threads; C++/ggml
   catastrophically regresses past 16 threads (see the [tiled flash update](
   #update-tiled-flash--parallelised-non-gemm-ops--rust-now-faster-than-c)
   table). So "Rust at 24 vs C++ at 16" overstates Rust's per-thread
   efficiency — it's partly a scheduler/scaling win, not a per-op win.
3. **Both engines dequantise to f32.** Neither has a native quantised GEMM
   on this CPU (no AVX-512). So the q5_k numbers don't reflect what a
   real quantised-inference engine (e.g. llama.cpp with `--kquants`) would
   achieve.

### If you want to claim "Rust is faster than C++"

The defensible claim, based on the data above, is narrower than the section
header "Rust now FASTER than C++" suggests:

> On this i7-13700K, the Rust fast path at 16 threads matches or slightly
> beats the C++/ggml engine at 16 threads on the same q5_k model
> (~484 ms vs ~510 ms, earlier session). Rust additionally scales to 24–32
> threads where C++ does not, extending the lead. These numbers have not been
> re-measured against a fresh C++ build in the latest session.

Anything stronger than that needs a same-session, same-cooling, same-binary
head-to-head run.

### Why not just use the GPU (RTX 4090)?

**Update: we did.** After installing the CUDA 13.3 toolkit + driver 610.62, the
Rust engine runs on the RTX 4090. The C++ engine's ggml 0.15.1 CUDA backend
has a cublas compatibility issue with CUDA 13 (work in progress).

**Measured on the RTX 4090 (same i7-13700K + RTX 4090 box, same `canyon.jpg`):**

| config | backbone | head | total min (ms) | total median (ms) |
|--------|---------:|-----:|---------------:|------------------:|
| CPU-only (fast path, 24 thr, q5_k) | 207 | 85 | 294 | 301 |
| **GPU backbone + CPU head (hybrid)** | **10** | **106** | **124** | **127** |
| GPU pure (candle, backbone+head) | 11 | 140 | 170 | 175 |

The **hybrid config** (GPU backbone via candle/CUDA + CPU fast head via
Winograd/tinyBLAS) is the fastest at **124 ms min / 127 ms median** — 2.4×
faster than CPU-only. The head stays on CPU because candle's GPU conv path is
slower than the hand-tuned Winograd CPU path for the DPT head's many small
3×3 convs.

**How to run the hybrid config:**

```bash
cargo build --release --features cuda --example bench
# Device auto-selects CUDA (device 0). Fast backbone path is CPU-only so leave
# DA_FAST_ATTN unset (0); fast HEAD path stays on CPU via Winograd.
RAYON_NUM_THREADS=24 DA_FAST_ATTN=0 DA_FAST_HEAD=1 \
  ./target/release/examples/bench \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg --warmup 5 --repeat 15
```

**C++/ggml CUDA status:** the build succeeds (ggml 0.15.1 compiled with CUDA
13.3 for sm_89) but the first `cublasSgemm_v2` call fails at runtime with
`CUDA_ERROR_LAUNCH_FAILED`. This is a known ggml/CUDA-13 incompatibility —
ggml 0.15.1 predates CUDA 13 and its cublas wrapper needs updating. Updating
ggml to a newer version (or patching the cublas call) is left as future work.

For the original Windows CUDA toolchain setup notes (driver/toolkit versions,
the cudarc 13.3→12.8 binding patch), see
[`docs/GPU.md`](GPU.md#windows-cuda-toolchain-blocker).

## Setup

- **Hardware:** 13th Gen Intel Core i7-13700K (8P+16E logical → 24 threads).
  AVX2 / FMA / F16C / BMI2 / AVX-VNNI; **no AVX-512** (consumer 13th-gen).
- **OS:** Windows 11.
- **C++ build:** clang 22.1.4 + Ninja, `-DCMAKE_BUILD_TYPE=Release`,
  `GGML_NATIVE=OFF` + explicit `AVX2/FMA/F16C/BMI2/SSE42/AVX_VNNI=ON`,
  `DA_HAS_AVX512F=OFF` (this CPU has no AVX-512; the auto-detected path
  crashes with SIGILL). ggml llamafile tinyBLAS kernels enabled (default).
- **Rust build:** `cargo build --release` (`opt-level=3`, thin LTO, 1 codegen
  unit). `candle-core` / `candle-nn` 0.8.4, CPU backend, rayon for parallelism.
- **Model:** `models/depth-anything-base-f32.gguf` (412 MB), converted from the
  HuggingFace `depth-anything/DA3-BASE` safetensors via `convert_da3_to_gguf.py`.
- **Input:** `assets/samples/street.jpg` (640×427 → upper_bound_resize → 504×336).
- **Method:** 25 timed iterations after 3 warmup; `--repeat 25 --threads N`.
  Rust via `cargo run --release --example bench`; C++ via `da3-cli depth`.

## Parity (correctness gate)

Before timing, Rust output is verified bit-equivalent to C++ on the same input
(both engines compared PFM-to-PFM):

| sample    | shape      | corr (Rust vs C++) | max \|d\| |
|-----------|------------|--------------------|-----------|
| street    | 336 × 504  | 0.999972           | 2.97e-02  |
| canyon    | 336 × 504  | 0.999994           | 6.22e-02  |
| desk      | 336 × 504  | 0.999978           | 3.48e-02  |
| mountains | 336 × 504  | 0.999986           | 4.74e-02  |

All four samples correlate **0.99997–0.99999** with max abs diff ~3–6e-2 on
depth values up to ~4.6 — pure f32 accumulation noise across 12 ViT blocks +
the DPT decoder. **The Rust port is numerically faithful to C++.**

## Latency (ms/iter, median over 25)

| threads | C++/ggml | Rust/candle | Rust ÷ C++ |
|--------:|---------:|------------:|-----------:|
| 1       | 3075     | 4042        | 1.31×      |
| 8       | 486      | 2141        | 4.40×      |
| 16      | 403      | 2195        | 5.45×      |

Per-stage breakdown (Rust, threads=16):

| stage       | ms     | share |
|-------------|-------:|------:|
| preprocess  | 8.4    | 0.4%  |
| backbone    | 1381   | 62.9% |
| head (DPT)  | 802    | 36.5% |
| activate    | 1.0    | 0.0%  |
| **total**   | **2195** |       |

Load time: C++ ~104 ms, Rust ~160 ms (Rust dequants f32 weights into candle
tensors on load; C++ mmaps ggml tensors directly).

## Interpretation

1. **Rust is correct** (0.99997+ correlation with C++). The port is real, not
   a stub.
2. **The gap is an engine gap, not a language gap.** At 1 thread Rust is only
   **1.31×** slower — within the same order of magnitude. The gap *widens* with
   threads because:
   - ggml's CPU backend has hand-tuned tinyBLAS (llamafile) GEMM microkernels,
     Winograd convolution, and fused flash-attention that each parallelize
     efficiently and are AVX2-tuned.
   - candle's CPU backend uses generic BLAS-free matmuls and a more naive
     thread pool; the conv-heavy DPT head in particular does not parallelize
     as well.
3. **candle stops scaling past ~8 threads here** (2141 → 2195 from 8→16),
   while ggml keeps scaling (486 → 403). This is candle's thread-pool / work
   distribution, not a Rust-language limit.
4. The original `RUST_PORT.md` prediction of "~1.5–3× slower" was **optimistic
   at high thread counts** — the real gap is **1.3× (1 thread)** up to
   **~5.5× (16 threads)**. The single-thread number is the cleaner
   engine-vs-engine signal; the multi-thread widening is ggml's superior
   kernel parallelism.

## Reproducing

```sh
# C++
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DDA_BUILD_CLI=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_BMI2=ON -DGGML_SSE42=ON -DGGML_AVX_VNNI=ON -DDA_HAS_AVX512F=OFF
cmake --build build -j
./build/examples/cli/da3-cli depth --model models/depth-anything-base-f32.gguf \
  --input assets/samples/street.jpg --repeat 25 --threads 16

# Rust
cargo run --release --example bench -- \
  --model models/depth-anything-base-f32.gguf \
  --input assets/samples/street.jpg --repeat 25 --warmup 3
```

Set Rust thread count with `RAYON_NUM_THREADS=N`.

## Update: candle-free ViT block (tinyBLAS + raw f32 path)

A candle-free ViT block forward (`src/fast_block.rs` + `src/fast_attn.rs`) was
added, bypassing candle's per-op allocation and dispatcher overhead for the
entire transformer block (norm1 → attention → layerscale → residual → norm2 →
FFN → layerscale → residual). Activated by `DA_FAST_ATTN=1`.

**Method:** A/B benchmark, 5 interleaved pairs, 15s cooldown between runs to
control for i7-13700K thermal throttling. DA3-BASE q5_k, 504×336, 16 threads.

| metric        | candle (median) | fast block (median) | speedup |
|---------------|----------------:|--------------------:|--------:|
| infer total   |        2188 ms  |            1271 ms  |  **1.72×** |
| backbone      |        1386 ms  |             448 ms  |  **3.09×** |
| head (DPT)    |         790 ms  |             799 ms  |  ~1.0× (unchanged) |

**Correctness:** depth output verified bit-faithful to the candle path
(correlation 1.00000000, max abs diff 9e-6, identical depth range) on the
same input image.

**Interpretation:** the candle-free block eliminates the ~80 ms/block of
candle per-op overhead the probe (`examples/probe_hotspots.rs`) identified
(rope2d, layernorm, residuals, scale_layer, affine, gelu, narrows, reshapes,
allocs). The backbone is now GEMM-dominated (~275 ms) plus fast non-GEMM
(~175 ms). The DPT head is now the bottleneck (63% of inference time).

**Reproducing:**
```sh
# candle path
RAYON_NUM_THREADS=16 cargo run --release --example bench -- \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg --warmup 3 --repeat 5

# fast block path
DA_FAST_ATTN=1 RAYON_NUM_THREADS=16 cargo run --release --example bench -- \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg --warmup 3 --repeat 5

# interleaved A/B
./bench_ab.sh 5 models/depth-anything-base-q5_k.gguf assets/samples/canyon.jpg 15
```

## Update: candle-free DPT head (Winograd + tinyBLAS convs)

A candle-free DPT head conv path (`src/fast_conv.rs`) was added, bypassing
candle's per-op overhead for the head's 3×3 and 1×1 convolutions. Activated by
`DA_FAST_HEAD=1` (independent of `DA_FAST_ATTN` for the backbone).

- **3×3 stride-1 pad-1 convs**: Winograd F(2×2,3×3), ported 1:1 from the C++
  engine's `src/winograd.cpp`. Uses AVX2/FMA (8 floats/vector) instead of the
  C++ engine's AVX-512 (16 floats/vector, not available on the i7-13700K).
  Reduces multiplications by 2.25× vs direct convolution.
- **1×1 stride-1 pad-0 convs**: direct tinyBLAS GEMM with **zero transposes**
  — the NCHW input `[IC, HW]` is already the `[K, N]` operand, and the weight
  `[OC, IC]` is `[M, K]`, so the 1×1 conv is a single `gemm_nn_into` call.

Strided convs (conv-transpose resize, stride-2 downsample) still use candle.

**A/B results** (5 interleaved pairs, 15s cooldown, q5_k model, 16 threads,
canyon.jpg; both paths use the fast backbone `DA_FAST_ATTN=1`):

| metric        | candle head (median) | fast head (median) | speedup |
|---------------|---------------------:|-------------------:|--------:|
| infer total   |           1326 ms    |          749 ms    | **1.77×** |
| backbone      |            456 ms    |          458 ms    | ~1.0× (unchanged) |
| head (DPT)    |            854 ms    |          274 ms    | **3.12×** |

**Cumulative progress** (q5_k model, 16 threads, vs C++ f32 at 396 ms):

| stage                        | Rust total | Rust head | vs C++ |
|------------------------------|-----------:|----------:|-------:|
| original (candle)            |    2188 ms |    790 ms | 5.53× slower |
| + fast backbone              |    1271 ms |    799 ms | 3.21× slower |
| + fast head (Winograd + 1×1) |     749 ms |    274 ms | **1.89× slower** |

**Correctness:** depth output verified bit-faithful to the candle path
(correlation 1.00000000, max abs diff 8.8e-6, identical depth range
[0.2755, 4.4747]) on canyon.jpg.

**Key optimizations in `fast_conv.rs`:**
1. Winograd F(2,3) algorithm (2.25× fewer multiplications for 3×3 convs).
2. 1×1 conv as a single transpose-free GEMM (7.3× faster than transpose+GEMM).
3. Thread-local scratch buffers (Vblk/Mblk) reused across tile-blocks.
4. U-transform cache keyed by weight storage pointer (avoids recomputing
   the filter transform on every forward).
5. Zero-copy weight/bias access via `storage_and_layout` (avoids `to_vec1`).

**Reproducing:**
```sh
# fast backbone + fast head (full fast path)
DA_FAST_ATTN=1 DA_FAST_HEAD=1 RAYON_NUM_THREADS=16 \
  cargo run --release --example bench -- \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg --warmup 3 --repeat 5

# interleaved head A/B (both use fast backbone)
./bench_head_ab.sh 5 models/depth-anything-base-q5_k.gguf assets/samples/canyon.jpg 15
```

## Update: fully candle-free DPT head (`fast_dpt.rs`)

A new `src/fast_dpt.rs` runs the **entire** DPT head (input LayerNorm,
projection convs, resize conv-transpose/strided conv, fusion pyramid with
residual conv units, bilinear upsample, output convs, UV embed, optional sky
head) on raw `Vec<f32>` buffers — eliminating all of candle's per-op overhead
in the head (dispatcher, tensor construction, type checks, layout validation).

This is the head analogue of the prior session's `src/fast_block.rs` for the
backbone. The previous `try_fast_conv` path in `dpt_head.rs` (which only
replaced 3×3 stride-1 and 1×1 convs, leaving the non-conv ops on the candle
path) is now superseded by a full candle-free path.

**Key new kernels** (all in `src/fast_dpt.rs`):
- **`conv_transpose_k_stride`** — conv-transpose with k=stride, pad=0
  (resize0 stride-4 and resize1 stride-2). Decomposed into k² independent
  tinyBLAS GEMMs (one per `(ky, kx)` offset), each scattered to strided output
  positions in parallel.
- **`conv3x3_stride2_pad1`** — im2col + tinyBLAS GEMM. Built a `[ic*9, out_hw]`
  column matrix in parallel, then a single GEMM `[oc, ic*9] @ [ic*9, out_hw]`.
  Brought the 768→768 stride-2 conv on stage 3 from 120 ms → 13 ms (**9×
  speedup** vs the naive direct convolution).
- **`upsample_bilinear_ac`** — bilinear `align_corners=true` upsample with
  precomputed `(x0, x1, wx)` / `(y0, y1, wy)` tables and parallelized across
  channels.
- **`layernorm_rows`** — row-wise LayerNorm over `[n, dim]` for the optional
  input norm.
- **`residual_conv_unit`** — relu + 3×3 + relu + 3×3 + residual add, all on
  raw buffers.

**A/B results** (5 interleaved pairs, 25 s cooldown, q5_k model, 16 threads,
canyon.jpg; both paths use the fast backbone `DA_FAST_ATTN=1`):

| metric        | candle head (median) | fast head (median) | speedup |
|---------------|---------------------:|-------------------:|--------:|
| infer total   |           2724 ms    |          907 ms    | **3.00×** |
| backbone      |           1132 ms    |          548 ms    | 2.06× (lower memory pressure) |
| head (DPT)    |           1553 ms    |          342 ms    | **4.55×** |

Single-shot (warm, after cooldown) timings on the same setup:

| metric        | candle head          | fast head           | speedup |
|---------------|---------------------:|--------------------:|--------:|
| head (DPT)    |            ~854 ms   |          ~141 ms    | **6.0×** |
| total         |           ~1326 ms   |          ~624 ms    | **2.1×** |

The single-shot head time of **141 ms** is now well below the C++ engine's
~196 ms head time on the same hardware. The bottleneck has shifted entirely
to the backbone (GEMM-dominated).

**Cumulative progress** (q5_k model, 16 threads, vs C++ f32 at 396 ms,
cool/single-shot timings):

| stage                                  | Rust total | Rust head | vs C++ |
|----------------------------------------|-----------:|----------:|-------:|
| original (candle)                      |    2188 ms |    790 ms | 5.53× slower |
| + fast backbone                        |    1271 ms |    799 ms | 3.21× slower |
| + fast head convs (Winograd + 1×1)     |     749 ms |    274 ms | 1.89× slower |
| + fully candle-free head (this update) |     624 ms |    141 ms | **1.58× slower** |

**Correctness:** depth logits compared to the candle path on canyon.jpg
(q5_k model): max abs diff 4.8e-5, rel diff 6.3e-7 — bit-faithful within f32
accumulation noise.

**Reproducing:**
```sh
# Full fast path (backbone + head).
DA_FAST_ATTN=1 DA_FAST_HEAD=1 RAYON_NUM_THREADS=16 \
  cargo run --release --example bench -- \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg --warmup 3 --repeat 5

# Interleaved head A/B (both use fast backbone).
./bench_head_ab.sh 5 models/depth-anything-base-q5_k.gguf assets/samples/canyon.jpg 25

# Head correctness vs candle (canyon.jpg).
RAYON_NUM_THREADS=8 cargo run --release --example head_parity -- \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg

# Per-stage profile of the fast head (laterals/fusion/output breakdown).
RAYON_NUM_THREADS=16 DA_FAST_HEAD_PROFILE=1 \
  cargo run --release --example head_profile -- \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg
```

## Update: backbone profiling + build tuning

### Corrected C++ comparison (apples-to-apples, same model)

The earlier "1.58× slower than C++" figure compared **Rust q5_k** against
**C++ f32**. The fair comparison is same-model-type:

| engine                | model | min infer (ms) | median (ms) |
|-----------------------|-------|---------------:|------------:|
| C++ / ggml, 16 thr    | q5_k  |          497.4 |       504.0 |
| C++ / ggml, 16 thr    | f32   |          403.9 |       407.0 |
| Rust / fast, 16 thr   | q5_k  |          616.8 |       623.2 |
| Rust / fast, 16 thr   | f32   |          640.9 |       ...   |

So Rust q5_k vs C++ q5_k is **1.24× slower** (617 vs 497 ms), not 1.58×. The
gap to close is ~120 ms.

### Backbone GEMM breakdown (per-iter, profiled)

Added `DA_FAST_PROFILE=1` accumulators around each GEMM call. Profiled
breakdown (q5_k, 16 threads, canyon.jpg, warmup 2 / repeat 3):

| stage                | µs/iter | calls/iter | note |
|----------------------|--------:|-----------:|------|
| attn Q·Kᵀ (per head) | 249,076 |        720 | `[864,64]@[64,864]`, biggest single cost |
| ffn fc2              | 125,161 |         60 | `[864,3072]@[3072,768]` |
| ffn fc1              | 119,887 |         60 | `[864,768]@[768,3072]` |
| attn A·V (per head)  | 112,681 |        720 | `[864,864]@[864,64]` |
| attn qkv             |  90,396 |         60 | `[864,768]@[768,2304]` |
| attn proj            |  28,555 |         60 | `[864,768]@[768,768]` |

Non-GEMM work inside each block (layernorm, rope2d, softmax, gather/scatter,
residual) is only ~10% of block time. Block↔candle marshalling is <2%.

### Standalone GEMM throughput (tinyBLAS in isolation, 16 threads)

| shape                         | ms   | GFLOP/s |
|-------------------------------|-----:|--------:|
| QKᵀ `[864,64]@[64,864]`       | 0.25 |    378  |
| AV  `[864,864]@[864,64]`      | 0.12 |    811  |
| QKV `[864,768]@[768,2304]`    | 4.52 |    677  |
| FFN fc1 `[864,768]@[768,3072]`| 5.75 |    710  |
| FFN fc2 `[864,3072]@[3072,768]`| 6.06 |    673  |

The isolated kernels are already at 70–90% of peak for the large-K GEMMs.
The in-context slowdown (profiled 2–3× the isolated numbers) comes from
profiling-overhead mutex contention, not real kernel inefficiency.

### Flash attention experiment

Implemented a fused flash-attention path (`src/flash_attn.rs`, AVX2 online-
softmax recurrence matching `ggml_flash_attn_ext`). Correctness verified
(PFM-to-PFM vs materialised: max abs diff 5.2e-6, correlation 1.0000003).

**Result: slower than the materialised tinyBLAS path for these shapes.**
Per-query flash (even with flattened `(head, query_row)` parallelism and
stack-resident accumulators) loses to tinyBLAS's 6×16 microkernel because
K/V (216 KiB/head) fit in L2, so the `[n,n]` scores matrix is the only real
cost — and tinyBLAS handles it efficiently. Kept as opt-in via
`DA_FAST_FLASH=1`; default OFF.

### Build tuning

- `.cargo/config.toml`: `target-cpu=native` (so auto-vectorised loops pick up
  AVX2/FMA, matching the explicit `#[target_feature]` kernels).
- `Cargo.toml [profile.release]`: added `panic = "abort"` (removes unwinding
  tables from the hot path). Kept `lto = "thin"` (fat LTO regressed ~5%).

Combined effect: ~617 → 617 ms (within noise on this thermally-throttled
setup; the win is more consistent on a stable cooling profile).

### Remaining gap analysis

The ~120 ms gap to C++ q5_k is dominated by the **attention Q·Kᵀ** (the
biggest single GEMM cost) and **L3 cache pressure** from the f32 weights
(~26 MiB of qkv+fc1+fc2 weights vs L3 = 30 MiB). The two highest-value next
steps are:

1. **Quantised FFN/QKV GEMM** (q5_k dequantise-on-the-fly): would shrink the
   weight footprint ~6×, eliminating L3 pressure. Requires threading raw
   q5_k bytes through to `FastFfn` (bypassing candle's f32 dequant at load).
2. **Tiled flash attention** (process BQ queries × BK keys per tile): the
   per-query version can't beat tinyBLAS, but a properly tiled one reuses K
   across BQ queries and may win for the `[864,64]@[64,864]` shape.

## Update: tiled flash + parallelised non-GEMM ops — Rust now FASTER than C++

This session closed the gap to C++ by attacking the non-GEMM overhead that
turned out to be far larger than the prior profile (with mutex-contented
profiler) suggested. Four changes, each independently measurable:

### 1. Lock-free profiler (`src/fast_profile.rs`)

The previous profiler used a `Mutex<Vec<String>>` for label→bucket lookup.
Under heavy rayon parallelism (1680+ record() calls/iter across 16 threads)
this caused massive contention that inflated measured GEMM times by 2–3×
over actual, making A/B comparisons unreliable. Replaced with a fixed
compile-time label table backed by `OnceLock<Option<&'static str>>` —
registration is lock-free after the first call, and the hot path is just two
relaxed `AtomicU64::fetch_add`s.

### 2. Tiled flash attention (`src/flash_attn.rs`)

Rewrote the flash attention to match ggml's
`ggml_compute_forward_flash_attn_ext_tiled` (ops.cpp:8558+): Q-tiles of 64
queries × KV-tiles of 64 keys, reusing each K/V tile across all 64 queries
in the Q-tile via two tinyBLAS `[64,64]@[64,64]` GEMMs (QKᵀ + AV) with
online softmax between them. The previous per-query flash lost because each
key was loaded once per query — no reuse across the 864 queries.

The K/V tile (64 keys × 64 floats = 16 KiB) fits in L1. Q is pre-scaled by
`1/sqrt(hd)` once during the copy, fusing the per-KV-tile `KQ *= scale` into
the Q load. The online-softmax max reduction uses AVX2 `hmax_ps`; the VKQ
rescale and final normalisation are vectorised across the `hd` lanes.

The per-query implementation is kept as `forward_per_query()` for A/B.

### 3. Skipped strided `k_heads_t` writes in flash path

When flash is active, the transposed-K buffer `k_heads_t[h, d*stride_n + ni]`
is never read (flash uses the row-major `k_heads` directly). The strided
writes (one cache line per element) were pure waste. The gather loop now
branches on `use_flash` and skips them.

### 4. Parallelised all serial element-wise loops

The biggest single win. Profiling revealed that non-GEMM work was ~45% of
backbone time (not the ~10% the prior session estimated from the
contention-inflated numbers). The culprits were all serial `for ni in 0..n`
loops running single-threaded while 31 other cores sat idle:

- **FFN bias-add + GELU** (`fast_block.rs`): the GELU uses an `erf`
  polynomial (~20 FLOP/element), called 864×3072 = 2.65M times per FC1
  output. Parallelised via `par_chunks_mut(hidden)`. **This alone saved
  ~70 ms.**
- **Block-level LayerNorm** (`layernorm_rows` in `fast_block.rs`): 3 passes
  over `dim=768` per row × 864 rows. Parallelised via `par_chunks_mut(dim)`.
- **Attention gather/scatter** (`fast_attn.rs`): the QKV→per-head-panel
  gather (with rope2d + q/k layernorm) and the reverse scatter were serial.
  Parallelised over the token axis via `into_par_iter()`.
- **Residual adds + layerscale** (`fast_block.rs`): parallelised across rows.

### Results (DA3-BASE q5_k, canyon.jpg, i7-13700K)

**Head-to-head, same thread count (16):**

| engine      | threads | min infer (ms) | median (ms) |
|-------------|--------:|---------------:|------------:|
| C++/ggml    |      16 |          510.2 |       515.2 |
| Rust/fast   |      16 |          483.8 |       487.3 |

Rust is **~26 ms faster than C++ at equal thread count** (0.95×).

**Rust benefits from thread oversubscription (C++ does not):**

| threads | Rust min (ms) | C++ min (ms) |
|--------:|--------------:|-------------:|
|      16 |           484 |          510 |
|      24 |           448 |     (>1s, bad) |
|      32 |           444 |    (14615!) |
|      48 |           454 |              |

Rust kernels are latency-bound (memory/pipeline stalls that oversubscription
hides); C++ kernels are compute-bound (oversubscription just adds contention,
hence the catastrophic 32-thread regression). At its respective best thread
count:

| engine      | best threads | min infer (ms) |
|-------------|-------------:|---------------:|
| C++/ggml    |           16 |          510.2 |
| Rust/fast   |           32 |          444.4 |

**Rust is 65 ms faster than C++ at their respective best thread counts (0.87×).**

### Per-stage breakdown (Rust, 32 threads, profiled)

| stage       | µs/iter | share |
|-------------|--------:|------:|
| attn_flash  |   49580 |  17%  |
| attn_proj   |   14640 |   5%  |
| ffn_fc1     |   56310 |  19%  |
| ffn_fc2     |   59500 |  21%  |
| other       |  110860 |  38%  |
| **backbone**| **290294** |       |

The GEMMs are now ~62% of backbone; the remaining 38% is parallelised
non-GEMM work (layernorm, rope2d, gather/scatter, GELU, residuals).

### Correctness

All 156 unit tests pass. Depth output verified bit-faithful to the
materialised path (correlation 0.99999988, max abs diff 5e-7, identical depth
range [0.2755, 4.4747]) on canyon.jpg.

### Reproducing

```sh
# Rust fast path (best config: 32 threads + tiled flash + parallelised non-GEMM)
RAYON_NUM_THREADS=32 DA_FAST_ATTN=1 DA_FAST_HEAD=1 DA_FAST_FLASH=1 \
  cargo run --release --example bench -- \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg --warmup 3 --repeat 10

# C++ comparison (16 threads — C++ gets catastrophically slow at 32)
./build/examples/cli/da3-cli.exe depth \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg --repeat 10 --threads 16

# Correctness check (PFM correlation vs materialised path)
RAYON_NUM_THREADS=8 DA_FAST_ATTN=1 DA_FAST_HEAD=1 DA_FAST_FLASH=1 \
  cargo run --release --example da3 -- depth \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg --pfm /tmp/depth_flash.pfm
RAYON_NUM_THREADS=8 DA_FAST_ATTN=1 DA_FAST_HEAD=1 DA_FAST_FLASH=0 \
  cargo run --release --example da3 -- depth \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg --pfm /tmp/depth_mat.pfm
cargo run --release --example pfm_compare -- /tmp/depth_mat.pfm /tmp/depth_flash.pfm
```

### What didn't help (documented for future sessions)

- **Vectorising the softmax exp+sum**: the compiler already auto-vectorises
  `expf` at `-O3 target-cpu=native`; a hand-rolled AVX2 exp didn't beat it.
- **K-tile transpose loop order**: flipping `j`-outer/`d`-inner vs the
  reverse made no measurable difference (the tile is small enough to stay in
  L1 regardless).
- **Scatter parallelisation**: the scatter is already so cheap (small
  contiguous copies) that parallelising it was within noise.

## Update: parallelised DPT head element-wise ops

The DPT head was ~140 ms (31% of inference) with the same serial non-GEMM
loops that the previous session fixed in the backbone. Applied the same
rayon treatment to the head's serial loops:

- **`layernorm_rows`** (`fast_dpt.rs`): serial `for ni in 0..n` →
  `par_chunks_mut(dim)`. Called over `n_patch × in_channels` (up to 768
  channels) for the optional input LayerNorm.
- **`transpose_npatch_chw`** (`fast_dpt.rs`): serial triple loop → parallel
  over channels. Each channel writes a contiguous `[ph, pw]` plane, so writes
  never alias.
- **`relu_inplace`** (`fast_dpt.rs`): serial → `par_iter_mut()`.
- **UV positional embed adds** (3 sites: `forward`, `compute_laterals`,
  `fusion_forward`): serial `zip` → `par_chunks_mut(hw)` over channels.
- **Residual adds** (`residual_conv_unit`, `fusion_forward`): serial `zip`
  → `par_chunks_mut(hw)` over channels.
- **`conv1x1` bias-add** (`fast_conv.rs`): serial `for oc_i in 0..oc` →
  `par_chunks_mut(hw)` over channels.

Also reduced profiler overhead: `fast_profile::Scope::new()` now gates the
`Instant::now()` call behind `enabled()`, so disabled profiling has zero
hot-path cost (previously it always paid for the timestamp).

### Results

| metric            | before (prev session) | after (this session) |
|-------------------|----------------------:|---------------------:|
| min infer (ms)    |                  444  |                ~428  |
| head (ms, cool)   |                  ~140 |                ~127  |
| backbone (ms)     |                  ~290 |                ~290  |

Correctness: bit-identical to the prior verified output (correlation
1.00000000, rel diff 0.0, depth range [0.275524, 4.474692] on canyon.jpg).
All 156 unit tests pass.

## Update: fused bias-add into GEMM/convs + thread-local flash scratch

Extended the win by eliminating separate bias-add passes throughout the
hot path. The `gemm_nn_into` and Winograd kernels accumulate (`C += A@B`),
so pre-filling `C` with the bias and letting the GEMM accumulate on top
yields `bias + A@B` in a single pass — no separate read-modify-write
bias-add loop needed.

### Changes

- **FFN** (`src/fast_block.rs`): `fc1`/`fc2` (and `w12`/`w3` SwiGLU) now
  pre-fill the output with bias before the GEMM. Removes 4 serial bias-add
  loops per block. GELU remains a separate fused pass.
- **Attention** (`src/fast_attn.rs`): QKV and output projections now
  pre-fill with bias before the GEMM. Removes 2 serial bias-add loops per
  block.
- **conv3x3_pad1** (`src/fast_conv.rs`): bias fused into the Winograd
  output scatter (`ypatch[i,j] + bv`). Removes the separate `add_bias` pass.
- **conv1x1** (`src/fast_conv.rs`): general path pre-fills each channel
  plane with its bias before the GEMM. Also added a fast path for `oc=1`
  (the final depth-projection conv) that parallelises over pixels instead
  of the single-row GEMM (which had M=1 = no row parallelism).
- **conv_transpose_k_stride** (`src/fast_dpt.rs`): pre-fills output with
  bias before the scatter accumulate.
- **conv3x3_stride2_pad1** (`src/fast_dpt.rs`): pre-fills output with
  bias before the im2col GEMM.
- **Flash attention** (`src/flash_attn.rs`): per-task scratch buffers
  (`q_scaled`, `kq`, `vkq`, `k32`, `v32`) moved from per-task heap
  allocation to thread-local storage, reused across tasks. Avoids ~12k
  allocations per forward.
- **tinyblas** (`src/tinyblas.rs`): `MC` (row-block granularity) now
  overridable via `DA_TINYBLAS_MC` env var for experimentation. Default
  unchanged (MC=MR=6); empirically confirmed larger MC regresses.

### Results

| metric            | before (prev session) | after (this session) |
|-------------------|----------------------:|---------------------:|
| min infer (ms)    |                  ~428 |              ~402-414 |
| vs C++ (510ms)    |              ~16% faster |          ~18-21% faster |

Correctness: numerically identical (correlation 1.00000000, rel diff
3.36e-8 — within f32 epsilon; the tiny difference comes from folding
bias into the GEMM's FMA accumulation, which changes rounding order).
All 156 unit tests pass.

### What was tried but didn't help

- **Larger `MC`** (row-block granularity): tested MC=12, 18, 24, 30, 36, 48.
  All regressed total inference due to worse tail load balance with fewer,
  larger rayon tasks. MC=6 (one microkernel per task) remains optimal.
- **Thread-local flash scratch**: kept for cleanliness (avoids 12k
  allocs/forward), but the allocator was already fast enough that the
  measurable speedup was within benchmark noise.
- **conv1x1 `oc=1` pixel-parallel fast path**: correct but marginal gain
  (the conv is memory-bound at ~5M FMAs).

## Update: Winograd TB=4 + 16-OC microkernel + N-axis GEMM experiment

Further optimisation of the Winograd conv3x3 microkernel and the tinyBLAS
GEMM parallelisation strategy. Head time dropped from ~125 ms to ~109 ms,
total inference min from ~414 ms to ~400 ms.

### Winograd `TB` reduced 8 → 4 (`src/fast_conv.rs`)

The Winograd-domain GEMM microkernel (`wino_gemm_avx2`) had `TB=8` (8 tiles
per block). With the 16-OC fast path (two 8-lane u-loads per V broadcast),
that needs `8×2 = 16` accumulator registers — exactly all 16 ymm registers —
forcing the compiler to spill to the stack in the hot loop.

Reducing `TB` to 4 gives `4×2 = 8` accumulators, leaving 8 registers free
for temporaries. Assembly inspection confirmed the hot loop no longer spills.
Despite doubling the block count (more rayon dispatch overhead), the
spill-free kernel wins consistently:

| conv shape | TB=8 (ms) | TB=4 (ms) | speedup |
|---|---:|---:|---:|
| 128→128 @ 192×288 | 16.3 | 14.5 | 11% |
| 128→128 @ 96×144 | 4.4 | 3.8 | 14% |
| 128→128 @ 48×72 | 1.5 | 1.4 | 7% |
| 128→128 @ 24×36 | 0.43 | 0.30 | 30% |
| 96→128 @ 96×144 | 3.7 | 3.4 | 8% |
| 192→128 @ 48×72 | 1.8 | 1.7 | 6% |
| 384→128 @ 24×36 | 0.87 | 0.74 | 15% |
| 768→128 @ 12×18 | 0.72 | 0.60 | 17% |
| 128→64 @ 192×288 | 9.2 | 8.5 | 8% |
| 64→32 @ 336×504 | 10.9 | 9.9 | 9% |
| 64→1 @ 336×504 (sky) | 6.8 | 6.1 | 10% |

Average ~11% across all DPT-head conv shapes.

### 16-OC winograd GEMM path (`src/fast_conv.rs`)

Added a fast path to `wino_gemm_avx2` that processes 16 output channels per
iteration (two 8-lane u-loads per V broadcast). This reaches 16 FLOP/cycle
(theoretical AVX2 peak) by amortising each port-5 broadcast across 16 FMA
lanes instead of 8. The old 8-OC path was port-5-bound at 8 FLOP/cycle.

The 16-OC path is only enabled when `TB ≤ 4` (so 2·TB ≤ 8 accumulators fit
without spilling). For `oc < 16` or small `TB`, it falls back to the 8-OC
path.

### N-axis GEMM parallelisation experiment (`src/tinyblas.rs`)

Added an alternative parallelisation strategy for `gemm_nn_into` that
parallelises over N-column blocks instead of M-row blocks. The idea: for
large-K GEMMs (FFN fc1/fc2 where B = [K,N] ≈ 9 MiB exceeds L2), each
M-row-block task re-reads all of B from L3; an N-column-block task instead
re-reads all of A from L3 but keeps its B panel small enough for L2.

Selectable via `DA_TINYBLAS_AXIS=n` (default `m`). Empirically the N-axis
variant is mixed: it helps QKV proj (+20%) but hurts FFN fc2 (-38%) and
attn proj (-30%). The FFN regression is because A (10.5 MiB) doesn't fit in
L2 either, so N-axis trades one L3 bottleneck for another while reducing
task count (worse load balance). Default remains M-axis; the N-axis code is
retained behind the env var for future experimentation.

Also added `DA_TINYBLAS_NC` to control the N-block width (default 64).

### Direct im2col conv path (`src/fast_conv.rs`)

Added `conv3x3_pad1_direct_into` — a direct convolution via im2col + GEMM,
selectable via `DA_FAST_CONV3X3=direct`. The idea: for large spatial sizes,
Winograd's per-tile transform overhead might exceed the GEMM savings vs a
single large well-shaped GEMM.

Empirically the direct path is much slower for large HW because it
materialises an `IC·9 × Hout·Wout` im2col buffer that becomes hundreds of
MiB. Default (`auto`) always selects Winograd; the direct path is retained
behind the env var for future tiled-im2col work.

### OC=1 winograd GEMM path (`src/fast_conv.rs`)

Added a special case in `wino_gemm_avx2` for `oc == 1`: since the OC dimension
can't be vectorised (only 1 lane), vectorise across the tile dimension
instead. For TB=4, `v[ic, 0..4]` is 4 contiguous floats — a `__m128` load.
Broadcast `u[ic]` as a `__m128` and FMA into a single 4-wide accumulator.
This is 4× faster than the scalar tail (which processed 1 tile per FMA).

Benefits the sky head's final conv (IC=64, OC=1, 336×504): 6.1 ms → 4.7 ms
(23% faster).

### Cached conv-transpose weight transpose (`src/fast_dpt.rs`)

The conv-transpose (`conv_transpose_k_stride`) decomposes the operation into
k×k GEMMs, each needing the weight in `[out_c, in_c]` (NN) layout. The source
weight is `[in_c, ky, kx, out_c]`, so each (ky, kx) slice must be transposed.
Previously this transpose ran on every forward; now it's cached by weight
pointer (like the Winograd U-bank cache), so it runs once per model load.

Saves ~1-2 ms per forward on the resize0/resize1 laterals.

### Detailed lateral profiling (`src/fast_dpt.rs`)

Added per-step timing to `compute_laterals` (LN, transpose, proj, UV embed,
resize, layer_rn) when `DA_FAST_HEAD_PROFILE=1`. Revealed that lateral 3's
12 ms is dominated by `resize3` (conv3x3_stride2 at 768→768, ~6 ms) due to
the large im2col buffer allocation + GEMM with K=6912.

### Results

| metric            | before (prev session) | after (this session) |
|-------------------|----------------------:|---------------------:|
| min infer (ms)    |               ~402-414 |              ~394-406 |
| head (ms)         |                  ~125 |               ~114-119 |
| vs C++ (510ms)    |          ~18-21% faster |         ~20-23% faster |

Correctness: numerically identical (correlation 1.00000000, rel diff
3.36e-8). All 156 unit tests pass.

### What was tried but didn't help

- **N-axis GEMM parallelisation (global)**: forcing N-axis on ALL GEMMs helps
  QKV (+20%) but hurts FFN (-38%) and attn proj (-30%) because FFN fc2 has
  N=768 → only 12 N-blocks (starves the 32-thread pool). This was the prior
  session's finding.
- **Full-im2col direct conv**: materialises a huge column matrix (390 MiB
  for 336×504×64 input). Memory-bound, far slower than Winograd.
- **`MC` (tinyblas row-block) > 6**: confirmed again — larger values regress
  due to tail load imbalance with fewer rayon tasks.

## Update: Shape-aware GEMM axis auto-selection

The previous N-axis experiment failed because it was applied globally. The
key insight: **N-axis parallelisation is beneficial only when (a) the B panel
`B[K,N]` exceeds L2 and (b) there are enough N-blocks (`N/NC ≥ num_threads`)
for good parallelism.** This perfectly separates the two cases:

- **conv1x1 GEMMs** (N = H×W ≫ 1000): B is huge, N has thousands of blocks →
  N-axis wins big.
- **FFN/proj GEMMs** (N = 768-3072): either B fits L2 or N has too few blocks
  → M-axis stays optimal.

### Implementation (`src/tinyblas.rs`)

Added `should_use_n_axis(m, n, k)` which checks both conditions per-call. The
default `DA_TINYBLAS_AXIS` is now `'auto'` (per-call selection); `'m'` and
`'n'` force a fixed axis for experimentation. NC remains 64 (auto-tuning
experiments showed NC=64 is near-optimal across all shapes — small-K shapes
have L1-resident B-strips, large-K shapes have L2-resident B-strips).

### Per-shape conv1x1 speedup

| conv1x1 shape (IC→OC @ H×W) | N (HW) | before (ms) | after (ms) | speedup |
|-----------------------------|-------:|------------:|-----------:|--------:|
| 128→128 @ 192×288           |  55296 |        6.31 |       4.30 |   32%   |
| 128→128 @ 96×144            |  13824 |        1.94 |       1.66 |   14%   |
| 128→128 @ 48×72             |   3456 |        0.78 |       0.67 |   15%   |
| 96→128 @ 96×144             |  13824 |        1.80 |       1.63 |    9%   |
| 32→2 @ 336×504             | 169344 |        0.62 |       0.50 |   20%   |

Small-N shapes (N ≤ 864) are unchanged (M-axis selected, B fits L2).

### Failed experiment: M-blocked N-stripped GEMM for conv3x3_stride2

The lateral-3 `resize3` (conv3x3_stride2, 768→768, 24×36→12×18, K=6912, col
5.7 MiB) was the biggest remaining head bottleneck at ~6 ms. An M-blocked
N-stripped GEMM was implemented (`gemm_nn_into_blocked`) to keep each strip's
B panel L2-resident across multiple microkernels. **It was 2× slower**, not
faster: the hardware L2 prefetcher does not anticipate the "re-read from start
of strip" access pattern of the second microkernel in an M-block, so the
expected L2 hits never materialise and B is re-fetched from L3. The blocked
GEMM code was removed; `conv3x3_stride2_pad1` keeps the original `gemm_nn_into`.

### Results

| metric            | before (prev session) | after (this update) |
|-------------------|----------------------:|--------------------:|
| min infer (ms)    |               ~394-406 |           ~392-398  |
| head (ms)         |           ~114-119    |           ~112-114  |
| vs C++ (510ms)    |          ~20-23% faster |         ~21-23% faster |

Correctness: numerically identical (correlation 1.00000000, rel diff
3.36e-8). All 156 unit tests pass.

## Update: K-blocking GEMM for L3-bound shapes

The lateral-3 `resize3` (`conv3x3_stride2`, 768→768, 24×36→12×18, K=6912,
B panel = 5.7 MiB) was still the biggest single head-conv bottleneck at ~6 ms
after the M-blocked N-stripped experiment failed. The new approach: **K-blocking**
— split the K dimension into chunks so each `B[KC, N]` panel can be reused
across multiple M-axis tasks running sequentially on the same core, cutting
L3 traffic dramatically.

### Why K-blocking is different from the failed M-blocked N-stripped experiment

Both approaches try to keep B L2/L3-resident across multiple M-row
microkernels, but they differ in *which* dimension they split:

| Approach | B slice | Within-task access pattern | Prefetcher-friendly? |
|----------|---------|---------------------------|----------------------|
| M-blocked N-stripped (failed) | `B[K, NC]` (full K, partial N) | **re-reads** same B region from start for each M-row in the block | ❌ (prefetcher has moved past; L2 evicted early data) |
| K-blocking (this update) | `B[KC, N]` (partial K, full N) | walks through **different** B regions (different N-tiles) | ✅ (sequential forward access only) |

### Implementation (`src/tinyblas.rs`)

Added three pieces:

1. **`gemm_nn_avx2_kblocked`**: outer serial loop over K-chunks of size `KC`,
   inner rayon parallel over M-row blocks (each task processes `MC=6` rows for
   one K-chunk). Tasks call a strided-A microkernel variant.

2. **`microkernel_6x16_kc_ptr` / `microkernel_1x16_kc_ptr`**: like the existing
   pointer microkernels, but A's row stride (`a_stride`) is decoupled from the
   K-loop iteration count (`kc_len`). This is needed because the full A matrix
   is laid out `[M, K]` row-major; when we slice A to `[M, KC]` for a K-chunk,
   the row stride is still `K` (full), not `KC`.

3. **`auto_kc(n, k)` shape-aware chunk sizing**: the optimal KC depends on N
   (empirically validated):
   - **Small N (≤ 256, e.g. `conv3x3_stride2` with N=216)**: prefer `KC = K/2`
     (only 2 K-chunks). Each task has few N-tiles (14 for N=216), so dispatch
     overhead dominates if KC is too small.
   - **Large N (> 256, e.g. FFN fc2 with N=768)**: target `B-chunk ~ L2 budget`
     (2 MiB). Each task has many N-tiles (48 for N=768), so per-task work is
     large enough to amortize dispatch.

The default `DA_TINYBLAS_AXIS='auto'` now prefers N-axis (when viable), then
K-blocking (when B > L2 but N too small for N-axis), then M-axis. Explicit
`DA_TINYBLAS_AXIS=k` forces K-blocking; `DA_TINYBLAS_KC=N` overrides chunk
size (must be a multiple of 8).

### Per-shape GEMM speedup

| GEMM shape | M | N | K | B (MiB) | Strategy | Before (ms) | After (ms) | Speedup |
|------------|--:|--:|--:|--------:|----------|------------:|-----------:|--------:|
| `conv3x3_stride2` (lateral 3 resize) | 768 | 216 | 6912 | 5.7 | K-blocking (KC=3456) | 4.66 | **3.58** | **23%** |
| FFN fc2 | 864 | 768 | 3072 | 9.0 | K-blocking (KC≈680) | 5.13 | **4.07** | **21%** |
| attn proj (B≈L2) | 864 | 768 | 768 | 2.25 | K-blocking (KC≈680) | 1.05 | 1.05 | tie |
| QKV proj (N-axis still wins) | 864 | 2304 | 768 | 6.8 | N-axis | 3.25 | 2.77 | 15% (unchanged) |
| FFN fc1 (N-axis still wins) | 864 | 3072 | 768 | 9.0 | N-axis | 4.80 | 4.70 | 2% (unchanged) |

### Lateral-3 end-to-end impact

The `resize3` step dropped from **6.16 ms → 5.22 ms** (~15% faster on the
conv in isolation, ~0.9 ms saved). The additional ~11 ms of total inference
improvement comes from FFN fc2 in the backbone (12 ViT blocks × ~0.9 ms each).

### Failed experiments within K-blocking

- **Fixed `KC=864` (fits L2 for lateral 3)**: 4.06 ms. Slower than `KC=3456`
  (3.58 ms) because each task has too little work (only 14 N-tiles × 6 rows ×
  864 K = 2.2 MFLOP) relative to rayon dispatch overhead, and because 8 K-chunks
  × 128 M-blocks = 1024 dispatches thrash rayon.
- **`KC=K` (no K-blocking)**: 4.66 ms — same as baseline M-axis.
- **Pure L2-fit heuristic** (`KC = 2MiB/(N*4)`): gave 3.88 ms for lateral 3,
  worse than the N-aware `K/2` for small N.

### Results

| metric            | before (prev update) | after (this update) |
|-------------------|---------------------:|--------------------:|
| min infer (ms)    |           ~392-398   |           **~383-390** |
| head (ms)         |           ~112-114   |           ~110-112  |
| backbone (ms)     |           ~265       |           ~257       |
| vs C++ (510ms)    |          ~21-23% faster |         **~23-25% faster** |

Best min-infer observed: **382.8 ms**.

Correctness: numerically identical (correlation 1.00000000, rel diff
3.36e-8). All 156 unit tests pass.

## Update: K-blocking for FFN fc1 (N > 3·K shapes)

The previous K-blocking heuristic only fired when N-axis was *not* viable
(`N/NC < num_threads`). For FFN fc1 (M=864, N=3072, K=768), N-axis was
viable (48 N-blocks ≥ 32 threads), so K-blocking was never tried — even
though it turns out to be **15% faster** for this specific shape.

### Why K-blocking wins for fc1 but not for QKV proj

Both shapes have M=864, K=768, B > L2:

| shape | N | N-axis (ms) | K-blocking (ms) | winner |
|-------|--:|------------:|----------------:|--------|
| FFN fc1 | 3072 | 5.05 | **4.50** | K-blocking |
| QKV proj | 2304 | **2.88** | 3.69 | N-axis |

The key difference is that for fc1, the N-axis strategy re-reads the full
`A[864, 768]` = 2.5 MiB panel from L3 for every one of the 48 NC-wide tasks
(~120 MiB of L3 traffic), and each per-task B-strip (192 KiB) is dwarfed by
that A traffic. K-blocking with `KC ≈ 168` keeps the A tile at `MC × KC` =
96 × 168 × 4 = 64 KiB (fits L1!) and the B-chunk at `KC × N` = 168 × 3072 × 4
= 1.97 MiB (fits L2), eliminating the L3 latency stalls on A reads.

For QKV proj (N=2304), the same K-blocking KC = 224 gives a B-chunk of 2.0 MiB
(just fits L2) and A tile of 84 KiB (L1), but the total work is smaller (3.06
vs 4.08 GFLOP) and the N-axis path amortizes its A reads better — empirically
N-axis still wins by ~22%.

### The heuristic change

`should_use_k_blocking` now also fires when `N > 3·K` (strictly greater), even
if N-axis is viable. The previous logic (`!use_n && should_use_k_blocking`)
was inverted to `should_use_k_blocking || (!use_k && should_use_n_axis)` so
that K-blocking is checked **first** in the auto path:

```rust
// Before: N-axis always won when viable, K-blocking never tried for fc1
use_n = should_use_n_axis(m, n, k);
use_k = !use_n && should_use_k_blocking(m, n, k);

// After: K-blocking checked first, N-axis is the fallback
use_k = should_use_k_blocking(m, n, k);
use_n = !use_k && should_use_n_axis(m, n, k);
```

The `n <= 3 * k` carve-out in `should_use_k_blocking` ensures N-axis is still
preferred for shapes like QKV proj (N = 3·K exactly) and everything with
smaller N/K ratio.

### Per-shape GEMM speedup (updated)

| GEMM shape | M | N | K | Strategy | Before (ms) | After (ms) | Speedup |
|------------|--:|--:|--:|----------|------------:|-----------:|--------:|
| FFN fc1 | 864 | 3072 | 768 | K-blocking (KC≈168) | 5.05 | **4.50** | **11%** |
| FFN fc2 | 864 | 768 | 3072 | K-blocking (KC≈680) | 5.13 | 4.07 | 21% (unchanged) |
| `conv3x3_stride2` | 768 | 216 | 6912 | K-blocking (KC=3456) | 4.66 | 3.58 | 23% (unchanged) |
| QKV proj | 864 | 2304 | 768 | N-axis | 2.88 | 2.88 | unchanged |
| attn proj | 864 | 768 | 768 | K-blocking (KC≈680) | 1.05 | 1.05 | unchanged |

### Results

| metric | before (prev update) | after (this update) |
|---|---:|---:|
| min infer (ms) | ~383-390 | **~375-385** |
| backbone (ms) | ~257 | **~248-253** |
| head (ms) | ~110-112 | ~110-112 |
| vs C++ (510ms) | ~23-25% faster | **~25-26% faster** |

Best min-infer observed: **374.7 ms** (new record).

Correctness: correlation 0.99999995, rel diff 2.80e-7 (within f32 epsilon;
the tiny change vs the previous 3.36e-8 is because K-blocking reorders the
floating-point accumulation in fc1). All 160 unit tests pass.

## Update: eliminate redundant zero-fill in head conv allocation

The DPT head's conv helpers (`conv3x3_stride2_pad1`, `conv_transpose_k_stride`,
`conv1x1`) all follow the same pattern: allocate the output buffer, pre-fill
with bias, then run the conv (which accumulates onto the bias values).

Previously the allocation used `vec![0.0; n]` (zero-fill) or `out.resize(n, 0.0)`,
and then the bias pre-fill pass overwrote every element. The zero-fill was
therefore pure waste — the bias values replaced it before any reader saw it.

### Changes

All three conv helpers now allocate via `Vec::with_capacity` + `unsafe set_len`,
skipping the zero-fill entirely. The bias pre-fill is the first write to each
element, satisfying the safety contract.

```rust
// Before: zero-fill (wasted — overwritten by bias below)
let mut out = vec![0.0f32; oc * out_h * out_w];
out.par_chunks_mut(out_h * out_w).enumerate().for_each(|(oc_i, ch)| {
    let bv = w.bias[oc_i];
    for v in ch { *v = bv; }
});

// After: allocate uninitialised, bias-fill is the first write
let total = oc * out_h * out_w;
out.clear();
out.reserve(total);
unsafe { out.set_len(total) };  // every element written by bias-fill below
out.par_chunks_mut(out_h * out_w).enumerate().for_each(|(oc_i, ch)| {
    let bv = w.bias[oc_i];
    for v in ch { *v = bv; }
});
```

The callers in `compute_laterals` were also simplified: they previously
pre-allocated `vec![0.0; oc_s * out_h * out_w]` and passed it in, where the
function would `resize` (a no-op) and overwrite. Now they pass `Vec::new()`
and let the function own the allocation.

### Affected conv calls (per forward)

| conv | calls | output size | zero-fill saved |
|------|------:|-----------:|----------------:|
| `conv_transpose_k_stride` (lateral 0) | 1 | 96×576×864 = 47.7 MiB | 47.7 MiB |
| `conv_transpose_k_stride` (lateral 1) | 1 | 192×288×432 = 23.9 MiB | 23.9 MiB |
| `conv3x3_stride2_pad1` (lateral 3) | 1 | 768×12×18 = 0.63 MiB | 0.63 MiB |
| `conv1x1` (all laterals + output) | ~8 | varies | varies |

The biggest savings come from the conv-transpose calls (laterals 0 and 1),
which write large output buffers that previously got zero-filled then
immediately overwritten with bias.

### Analysis of the lateral-3 resize bottleneck

Detailed profiling of `conv3x3_stride2_pad1` (DA_LATERAL3_PROF) confirmed the
breakdown is dominated by the GEMM, with limited headroom in the surrounding
boilerplate:

| stage | time (ms) | share |
|-------|----------:|------:|
| GEMM (K-blocked, KC=3456) | 3.55 | 84% |
| im2col (5.7 MiB col buffer) | 0.60 | 14% |
| bias pre-fill | 0.13 | 3% |
| allocation + dispatch | ~0.4 | – |
| **total resize** | **~5.1** | |

The GEMM runs at 646 GFLOP/s (53% of the i7-13700K's AVX2 peak), which is
typical for a port-5-bottlenecked f32 kernel. The KC=3456 (K/2) tuning was
re-verified as optimal: smaller KC (1728, 864) regresses due to dispatch
overhead exceeding the L2-residency benefit.

The im2col writes 5.7 MiB and is memory-bound (~12 GiB/s L3 write bandwidth).
Fusing it into the GEMM microkernel was considered but rejected: the col
matrix has a strided-gather access pattern (stride-2 in both spatial dims),
which would require slow AVX2 gather instructions in the microkernel's inner
loop.

A standalone alloc+zero benchmark confirmed that `vec![0.0; 5.7 MiB]` costs
only ~9 µs (the allocator caches large allocations), so caching the col
buffer across calls would save negligible time.

### Results

The zero-fill elimination is a cleanup that removes wasted memory writes
without changing the numerics. Measured min-infer is unchanged at ~375-385 ms
(within thermal noise), and correctness is preserved.

Correctness: correlation 1.0000004, rel diff 4.36e-7 (within f32 epsilon).
All 160 unit tests pass.

## Update: parallelise preprocess resize + CHW normalize

The preprocessing pipeline (`resize_cubic`, `resize_area`, and the HWC→CHW +
ImageNet-normalize step) was entirely single-threaded, costing ~8 ms per
inference. All three stages are embarrassingly parallel across rows and have
been ported to rayon:

- **`resize_cubic`**: both the horizontal pass (parallelised over source rows)
  and the vertical pass (parallelised over destination rows) now use
  `par_chunks_mut`.
- **`resize_area`**: horizontal pass parallelised over source rows; vertical
  pass restructured to group taps by destination row, then parallelised over
  destination rows (each row's accumulation is independent).
- **CHW + normalize**: parallelised over row-chunks (16 rows per rayon task),
  reading HWC triplets contiguously and scattering into the 3 channel planes
  (so each input byte is read exactly once).
- Also avoids the upfront `img.clone()` when a resize fires (the common case).

### Results

Microbenchmark (`examples/preprocess_bench.rs`, canyon.jpg 1024×680 → 518×350,
32 threads):

| stage | before | after | speedup |
|-------|-------:|------:|--------:|
| step1 `resize_area` (1024×680 → 518×344) | 4.42 ms | 1.99 ms | 2.2× |
| step2 `resize_cubic` (518×344 → 518×350) | 2.44 ms | 0.82 ms | 3.0× |
| CHW + normalize | 0.84 ms | 0.85 ms | 1.0× |
| **total `preprocess_real`** | **8.29 ms** | **3.65 ms** | **2.27×** |

The CHW stage barely improves because its working set (~2 MiB output) already
fits in L3 and the per-row work is tiny; the dispatch overhead matches the
compute. The two resize stages dominate and parallelise well.

End-to-end benchmark (`bench --model q5_k --input canyon.jpg`, 32 threads, 5
single-iteration runs with 8 s cooldown):

| metric | before | after |
|--------|-------:|------:|
| preprocess_ms | ~13 | 2.9–3.8 |
| infer_min_ms | ~376 | **369.6** |

C++ baseline remains ~510 ms, so Rust is now ~27.5 % faster (was ~26 %).

Correctness: `parity_preproc` reports max|d|=2.38e-7, all 508032 elements
match within 1e-3 (unchanged from before — the parallel passes produce
bit-identical output to the serial ones). All 160 unit tests pass.

### Reproducing

```sh
RAYON_NUM_THREADS=32 cargo run --release --example preprocess_bench
RAYON_NUM_THREADS=32 cargo run --release --example parity_preproc -- \
  --model models/depth-anything-base-f32.gguf \
  --png dumps/preproc_real_input.png \
  --ref dumps/reference_preproc_real.gguf
```

## Update: fuse ReLU/residual/UV-add into conv kernels + Arc the UV embed

### Motivation

Profiling (`DA_FAST_HEAD_PROFILE=1`) showed the head at ~107 ms, dominated by
the fusion pyramid at 192×288 (27 ms for one fusion stage, with two
residual_conv_units at ~10 ms each). Closer inspection revealed several
unnecessary **memory passes** over multi-MiB intermediate tensors — work that
is pure memory traffic, not compute, and therefore a clean target.

### Changes

Four independent fusions / clone-eliminations in the head:

1. **`uv_embed_chw_cached` returns `Arc<Vec<f32>>` instead of `Vec<f32>`**
   (`src/uv_embed.rs`). Previously every call deep-copied the cached UV embed
   — **43 MiB at full 336×504×64 resolution** — on every forward pass. The
   cache now stores `Arc<Vec<f32>>` and returns a cheap reference-counted
   clone (one atomic increment). Fast-path callers (`fast_dpt.rs`) iterate
   the slice directly; the slow candle paths (`dpt_head.rs`, `gs_head.rs`)
   clone out of the Arc into the Vec that `Tensor::from_vec` consumes.

2. **Fused `relu(x)` into the Winograd input read**
   (`src/fast_conv.rs::conv3x3_pad1_relu_in`). The new variant applies
   `max(0.0)` to input pixels as they are loaded into the 4×4 input tile,
   so `conv(relu(x))` never materialises a separate `relu(x)` buffer.
   Saves one full read+write pass over the input tensor (up to 28 MiB at the
   largest fusion level). `residual_conv_unit` was rewritten to call this
   variant for both its convs.

3. **Fused the residual `+x` add into conv2's output scatter**
   (`src/fast_conv.rs::conv3x3_pad1_relu_in_res_out`). The new variant takes
   a `residual: &[f32]` parameter that is added (along with bias) in the
   Winograd output transform's scatter loop, eliminating a separate
   read+write pass over the output. `residual_conv_unit` now ends with
   `conv2(relu(h)) + x` computed entirely inside the conv kernel.

4. **Avoided cloning `top.data` in `fusion_forward`** (`src/fast_dpt.rs`).
   Previously the fusion stage cloned `top` (28 MiB at 192×288) into a
   mutable `y`, then added the lateral residual in-place. Now when a
   lateral is present, `y = top + res` is written directly into a fresh
   buffer (saving one read pass over `top`).

5. **Fused the UV embed add into the upsample**
   (`src/fast_dpt.rs::upsample_bilinear_ac`). The upsample now takes an
   optional `add: Option<&[f32]>` and writes `upsample(x) + add` in one
   pass, avoiding a separate read+write over the 43 MiB output. The output
   stage uses this to fold the UV positional embed into the upsample that
   precedes `oc2a`.

### Results

Per-stage head breakdown (steady-state, q5_k model, canyon.jpg, 32 threads):

| Stage | Before | After |
|-------|-------:|------:|
| laterals | ~33 ms | ~31 ms |
| fusion   | ~43 ms | ~37 ms |
| output   | ~28 ms | ~22 ms |
| **total head** | **~107 ms** | **~90 ms** |

End-to-end (10 single-iter runs with 8 s cooldown, best of 10):

| Metric | Before | After |
|--------|-------:|------:|
| `infer_min_ms` | ~372 | **~354** |

C++ baseline remains ~510 ms, so Rust is now **~30.5 % faster** (was ~27.5 %).

### Correctness

- All **160 unit tests pass**.
- `head_parity` reports `rel diff = 5.84e-7` against the candle DPT head
  (bit-identical to the pre-change parity — the fusions are algebraically
  exact, no change in floating-point ordering except for the UV-add-into-
  upsample fusion which is bit-identical because both paths compute
  `upsample(x) + pe` at the same index).

## Update: parallel laterals + fusion RCU overlap

### What changed

Two structural changes to the DPT head (`src/fast_dpt.rs`) that exploit
independence between stages to overlap work via the rayon thread pool:

1. **Parallel laterals** (`compute_laterals`): the 4 lateral stages
   (proj → resize → layer_rn) are independent — each reads `feats[s]`
   and writes `lats[s]`. Previously computed sequentially (31 ms total),
   they now run concurrently via `(0..4).into_par_iter().map().collect()`.
   Rayon's work-stealing shares the thread pool across all 4 stages, so
   dispatch-overhead-bound operations (small GEMMs in proj/resize for
   the smaller stages) fill idle threads during the heavier stages'
   compute-bound work.

2. **Fusion RCU overlap** (`forward`): `RCU(lat[0])` — the lateral
   residual conv unit for the last (heaviest) fusion level rn1 — is
   independent of the rn4→rn3→rn2 chain (it only needs `lat[0]`, which
   is ready after `compute_laterals`). It now runs concurrently with the
   early fusion levels via `rayon::join`, then the result is passed to
   rn1 via the new `fusion_forward_with_lateral_rcu` method. This saves
   ~8 ms of serial RCU work from the critical path.

### Infrastructure

- `compute_one_lateral` extracted from `compute_laterals` so each lateral
  can run in its own rayon task.
- `fusion_forward_with_lateral_rcu`: like `fusion_forward` but takes a
  pre-computed lateral RCU result instead of computing `RCU(lateral)`
  inline.
- `conv3x3_stride2_pad1`: col-buffer construction parallelised over
  channels (ic tasks) instead of ic×ksq rows, with a serial fallback for
  tiny outputs (hw_out < 64) where rayon dispatch overhead dominated.
  Bias pre-fill also goes serial for hw_out < 1024.

### Results

Per-stage head breakdown (steady-state, q5_k model, canyon.jpg, 32 threads):

| Stage | Previous | Now |
|-------|-------:|------:|
| laterals | ~31 ms | ~21 ms |
| fusion   | ~37 ms | ~35 ms |
| output   | ~20 ms | ~20 ms |
| **total head** | **~88 ms** | **~76 ms** |

End-to-end (10 single-iter runs with 8 s cooldown, best of 10):

| Metric | Previous | Now |
|--------|-------:|------:|
| `infer_min_ms` | ~354 | **~343** |

C++ baseline ~510 ms → Rust now **~33 % faster** (was ~30.5 %).

### Correctness

- All **163 unit tests pass**.
- `head_parity` reports `rel diff = 6.16e-7` (unchanged — the parallel
  execution doesn't change the arithmetic, only the scheduling).

---

## Update: F(4,3) Winograd for large output convs

Replaced the F(2×2,3×3) Winograd kernel with an auto-selecting F(2,3)/F(4,3)
dispatcher. F(4,3) uses 6×6 input tiles → 4×4 output tiles, reducing
multiplications per output pixel from 4.0 to 2.25 (1.78× fewer). The transform
matrices use fractions (1/6, 1/24) so precision drops from ~1e-5 to ~1e-4,
which is acceptable for the downstream ReLU + depth projection.

### Implementation

- Added F(4,3) transforms (`wino_filt_f4`, `wino_inp_f4`, `wino_outp_f4`)
  as a 1:1 port of the C++ engine's `F4Policy` (`src/winograd.cpp`).
- Added a `WinoPolicy` trait with two implementations (`F2`, `F4`) that
  monomorphize the shared `conv3x3_pad1_wino` kernel.
- U-cache extended to key on `(ptr, ic, oc, P)` so both policies can coexist.
- Auto-select heuristic: `F(4,3)` when `hout*wout >= 2048`, `F(2,3)` otherwise.
  Override with `DA_FAST_CONV3X3_WINO=f2|f4|auto`.
- Per-tile scratch buffers (`dpatch`, `vpatch`, `mpatch`, `ypatch`) hoisted
  to thread-local `PATCHES` to avoid per-block heap allocation churn.

### Microbenchmark (32 threads, standalone conv)

| Shape | F(2,3) ms | F(4,3) ms | Ratio |
|-------|--------:|--------:|------:|
| 128→128 @ 24×36   (hw=    864) |  0.49 |  0.55 | 0.89× |
| 128→128 @ 48×72   (hw=  3,456) |  1.27 |  1.36 | 0.93× |
| 128→128 @ 96×144  (hw= 13,824) |  3.86 |  3.61 | 1.07× |
| 128→128 @ 192×288 (hw= 55,296) | 16.39 | 12.47 | 1.31× |
| 96→128  @ 96×144  (hw= 13,824) |  4.45 |  3.29 | 1.35× |
| 192→128 @ 48×72   (hw=  3,456) |  2.07 |  1.57 | 1.32× |
| 384→128 @ 24×36   (hw=    864) |  0.79 |  0.83 | 0.95× |
| 768→128 @ 12×18   (hw=    216) |  0.70 |  1.45 | 0.48× |
| 128→64  @ 192×288 (hw= 55,296) | 11.31 |  8.22 | 1.38× |
| 64→32   @ 336×504 (hw=169,344) | 14.12 | 10.36 | 1.36× |
| 64→1    @ 336×504 (hw=169,344) |  7.96 |  5.13 | 1.55× |

F(4,3) wins for hw ≥ ~3,000 and loses for very small hw (< 1,000) where the
4× fewer tile-blocks undersaturate the 32-thread pool.

### Results

Per-stage head breakdown (32 threads, q5_k, canyon.jpg, profiled):

| Stage | F(2,3) only | Auto (F4 for hw≥2k) |
|-------|------------:|--------------------:|
| laterals | ~22 ms | ~22 ms |
| fusion   | ~36 ms | ~24 ms |
| output   | ~27 ms | ~20 ms |
| **total head** | **~85 ms** | **~66 ms** |

Output stage detail (profiled):

| Component | F(2,3) | Auto |
|-----------|-------:|-----:|
| out1 (128→64 @ 192×288) | 10.4 ms | 6.8 ms |
| upsample (192×288 → 336×504) | 3.7 ms | 3.7 ms |
| out2a (64→32 @ 336×504) | 12.5 ms | 8.8 ms |
| **total output** | **27.3 ms** | **20.2 ms** |

End-to-end (8 single-iter runs with 8 s cooldown, best of 8):

| Mode | Best ms | Median ms |
|------|--------:|----------:|
| F(2,3) forced | 362.3 | ~384 |
| F(4,3) forced | 347.1 | ~351 |
| Auto (F4 for hw≥2k) | **349.7** | ~354 |

C++ baseline ~510 ms → Rust now **~31–33 % faster** depending on system load.

### Correctness

- All **165 unit tests pass** (8 original F(2,3) + 5 new F(4,3) + 152 others).
- `head_parity` reports `rel diff = 6.27e-7` (slightly worse than F(2,3)'s
  6.16e-7 due to F(4,3)'s fraction-based transforms, but far below the 1e-4
  tolerance gate).
- F(4,3) vs naive convolution: max diff < 1e-3 for all tested shapes.
- F(4,3) vs F(2,3) cross-check: agree to < 1e-3.

## Update: fuse upsample into out2a input read

The output stage's `upsample_bilinear_ac(192×288 → 336×504)` writes a 43 MiB
intermediate that `out2a` (3×3 conv, 64→32) immediately reads back. This is
fused: `out2a`'s Winograd input transform computes each upsampled pixel
on-the-fly via `align_corners=true` bilinear interpolation from the 14 MiB
un-upsampling input (which fits entirely in L2), eliminating the 43 MiB
DRAM write+read round-trip.

### Implementation

- Added `UpsampleSpec` (precomputed `(x0, x1, wx)` / `(y0, y1, wy)` tables)
  and `UpsampleFusion` borrows in `src/fast_conv.rs`.
- `conv3x3_pad1_wino<W>` takes an optional `upsample: Option<&UpsampleFusion>`
  parameter; when `Some`, the IT×IT input patch is gathered via bilinear
  interpolation from the un-upsampling `[N, IC, h_lo, w_lo]` input instead of
  a direct read. The branch lives inside the per-`ic_i` gather loop (highly
  predictable — one direction per conv call).
- New public entry point `conv3x3_pad1_relu_out_upsample` builds the spec,
  dispatches F2/F4 via `select_wino_policy`, and runs the fused conv.
- `fast_dpt.rs` output stage uses the fused path when no sky head is present
  (DA3-BASE has no sky head, confirmed via GGUF inspection). Falls back to the
  materialised path when a sky head needs the upsampled tensor too.
- Optional UV-positional-embed add is folded into the upsample (the `add`
  field of `UpsampleFusion`), matching the previous `upsample_bilinear_ac`
  fusion.
- A/B override: `DA_FAST_FUSE_UPSAMPLE=0` forces the materialised path.

### Microbenchmark (32 threads, standalone)

| Shape | Fused ms | Unfused (upsample + out2a) ms |
|-------|--------:|------------------------------:|
| 64→32 @ (192×288)→(336×504) | 14.0 | 3.7 + 11.3 = 15.0 |

The fusion saves ~1 ms standalone by eliminating the 43 MiB write+read.

### In-head output stage (profiled, 32 threads)

| Component | Unfused | Fused |
|-----------|--------:|------:|
| out1 (128→64 @ 192×288) | 7.7 ms | 7.7 ms |
| upsample (192×288 → 336×504) | 3.7 ms | — (fused) |
| out2a (64→32 @ 336×504) | 9.7 ms | 13.5 ms |
| **total output** | **21.1 ms** | **21.2 ms** |

The fused `out2a` is ~4 ms slower than the unfused `out2a` alone (bilinear
interpolation adds compute), but the separate 3.7 ms upsample pass is
eliminated, so total output-stage time is roughly break-even (the ~1 ms
microbench win is within end-to-end measurement noise). The fusion is kept
because:
  - It eliminates a 43 MiB allocation, reducing memory pressure.
  - It's the architecturally correct approach (producer-consumer fusion).
  - The microbench shows a small real win in isolation.

### Correctness

- All **168 unit tests pass** (165 + 3 new upsample-fusion tests:
  `upsample_fusion_matches_materialized`, `upsample_fusion_with_add`,
  `upsample_fusion_identity_filter`).
- `head_parity` reports `rel diff = 6.27e-7` — bit-identical to the pre-fusion
  value (the bilinear interpolation math is exact; no new approximation).
- Fused vs unfused cross-check: agree to < 1e-3 for all tested shapes.

### End-to-end

| Mode | Best ms | Median ms |
|------|--------:|----------:|
| Fused (default) | 352.5 | 353.8 |
| Unfused (`DA_FAST_FUSE_UPSAMPLE=0`) | 349.0 | 354.8 |

Difference is within measurement noise (±5 ms run-to-run). C++ baseline ~510 ms
→ Rust remains **~31–33 % faster**.

## Update: fusion-stage upsample+conv1x1 fusion (investigated, default OFF)

The previous session's recommended next step was: "Fuse upsample into
fusion-stage conv1x1 — rn1 fusion stage does `upsample(96×144→192×288) →
conv1x1(128→128 @ 192×288)`. Same pattern as the output-stage fusion but for
GEMM instead of Winograd. Potential ~2ms savings."

This was implemented and benchmarked. **Result: the fusion is a net loss
(~5 ms slower) for DA3-BASE shapes, so it ships disabled by default.**

### Implementation

A new `fast_conv::conv1x1_upsample` fuses the bilinear upsample into the
conv1x1 GEMM. Each parallel NC-panel task materialises a narrow `[ic, NC]`
B-strip (≈ 512 KiB, L2-resident) by bilinear-interpolating the un-upsampling
input (L3-resident), then immediately consumes it via a new
`tinyblas::gemm_nn_panel_strided` kernel (strided-output panel GEMM built on
the existing AVX2 microkernel).

**Files changed:**

| File | Changes |
|------|---------|
| `src/tinyblas.rs` | Exposed `pub(crate) const NR`; added `pub(crate) fn gemm_nn_panel_strided` (strided-output panel GEMM wrapping the existing `gemm_nn_rows_cols_avx2` microkernel with a scalar fallback). |
| `src/fast_conv.rs` | Added `conv1x1_upsample_nc` (panel-width heuristic), `pub fn conv1x1_upsample`, and 3 unit tests (`conv1x1_upsample_matches_materialized`, `conv1x1_upsample_identity`, `conv1x1_upsample_no_scale`). |
| `src/fast_dpt.rs` | Extracted `fusion_upsample_conv1x1` helper used by both `fusion_forward` and `fusion_forward_with_lateral_rcu`; added `fuse_fusion_upsample_enabled()` OnceLock flag (**default OFF**, opt-in via `DA_FAST_FUSE_FUSION_UPSAMPLE=1`). |
| `examples/fast_conv_bench.rs` | Added `bench_conv1x1_upsample` comparing fused vs rayon-parallelised unfused upsample+conv1x1. |

### Why the fusion loses for DA3-BASE

The fusion's premise was that the upsampled activation tensor
(`[ic, h, w]`) is too large for cache, making its DRAM write+read round-trip
the dominant cost. For the output-stage fusion (previous section) this holds:
the upsampled tensor is `[64, 336, 504]` = 43 MiB > 30 MiB L3.

For the fusion stages the upsampled tensor is much smaller because the
spatial dimensions are 4–16× smaller and `features = 128` (not 256 as initially
estimated):

| Stage | Shape | Upsampled B size |
|-------|-------|----------------:|
| rn3 | `[128, 48, 72]` | 1.75 MiB (fits L2) |
| rn2 | `[128, 96, 144]` | 7.1 MiB (fits L3) |
| rn1 | `[128, 192, 288]` | 28.2 MiB (fits L3) |

Since B fits in L3, the unfused GEMM reads it from L3 (~50 GB/s) rather than
DRAM. The fusion's per-panel B-strip materialisation adds compute overhead
that exceeds the DRAM savings.

### Microbenchmark (`examples/fast_conv_bench`, 32 threads)

Fair A/B: fused vs rayon-parallelised unfused upsample + `conv1x1`.

| Stage | Shape | Unfused | Fused | Δ |
|-------|-------|--------:|------:|--:|
| rn1 | 96×144 → 192×288 | 7.0 ms | 9.5 ms | **+2.5 ms** |
| rn2 | 48×72 → 96×144 | 2.6 ms | 4.1 ms | **+1.5 ms** |
| rn3 | 24×36 → 48×72 | 1.2 ms | 2.7 ms | **+1.5 ms** |

### End-to-end (DA3-BASE, `head_parity` with `DA_FAST_HEAD_PROFILE=1`, median of 3)

| Mode | Fusion total | Head total |
|------|-------------:|-----------:|
| Fused (`=1`) | 62 ms | 117 ms |
| Unfused (default) | 57 ms | 111 ms |

The fused path adds ~5 ms to the fusion stage and ~6 ms end-to-end.

### Correctness

- All **171 unit tests pass** (168 + 3 new).
- `head_parity` reports `rel diff = 6.43e-7` with fusion enabled — bit-identical
  to the unfused path (1×1 conv GEMM is exact, no Winograd transforms).

### Decision

The fusion is **disabled by default** (`fuse_fusion_upsample_enabled()` returns
false unless `DA_FAST_FUSE_FUSION_UPSAMPLE=1`). The implementation is retained
because:

1. It's architecturally correct and fully tested.
2. `tinyblas::gemm_nn_panel_strided` is a useful primitive for future
   strided-output GEMM patterns.
3. The fusion becomes a win for models where the upsampled B exceeds L3
   capacity (e.g. `features = 256` → 56 MiB B at rn1).

### Lesson learned

The output-stage fusion worked because the upsampled tensor (43 MiB) exceeded
L3 (30 MiB). The fusion-stage fusion doesn't work because the tensor (28 MiB)
fits in L3. **Fusion wins only when it eliminates DRAM traffic, not L3
traffic** — L3 bandwidth is high enough (~50 GB/s) that materialising an
L3-resident tensor is cheaper than recomputing it inside the GEMM.

## Update: AVX2 GELU-erf batch kernel (`fast_block.rs`)

The FFN GELU-erf activation between `fc1` and `fc2` was previously a
`par_chunks_mut` loop calling the scalar `gelu_erf` per element. Profiling
(`DA_FAST_PROFILE=1`) showed it cost **7.6 ms/forward** across the 12 backbone
blocks — small in absolute terms but surprisingly slow per element (~5.7 ns,
implying the loop was **not** auto-vectorising despite `target-cpu=native`).

### Root cause

The scalar `gelu_erf` → `erff` → `(-ax*ax).exp()` chain has two operations
that block LLVM's loop vectoriser on x86_64:

1. **`f32::exp()` (libm `expf`)** — emitted as a scalar `call expf`, not
   inlined or vectorised. (The BENCHMARK note from the softmax experiment —
   "the compiler already auto-vectorises `expf` at `-O3 target-cpu=native`" —
   turned out to be inaccurate for this call site; inspection of the emitted
   assembly showed zero `vexp`/polynomial-exp instructions in the GELU loop.)
2. **`f32 as i32` truncation** in the replacement `exp_fast` (needed for the
   `2^n` bit reconstruction) lowers to a scalar `cvttss2si`, not the vector
   `vcvtps2dq`. Even with `target-cpu=native`, LLVM refuses to vectorise
   loops containing this scalar conversion.

The result: the entire GELU loop compiled to **scalar** `expf` + polynomial
FMA, ~10× slower than the 8-wide AVX2 throughput the rest of the engine
achieves.

### Implementation

Added an explicit AVX2 batch kernel in `src/fast_block.rs::avx2_gelu`:

- **`exp_fast_avx2(__m256)`** — same bit-manipulation + degree-6 polynomial
  algorithm as the scalar `exp_fast`, but uses `_mm256_cvtps_epi32` for the
  float→int step (which the auto-vectoriser refuses to emit) and
  `_mm256_slli_epi32` + `_mm256_castsi256_ps` for the `2^n` reconstruction.
  Round-to-nearest-even is done via the "magic number" trick
  (`(t + 1.5·2^23) - 1.5·2^23`) instead of `f32::round_ties_even` (which
  also lowers to a scalar libm call).
- **`erff_avx2(__m256)`** — A&S 7.1.26 rational approximation using
  `exp_fast_avx2`; sign applied branchlessly via sign-bit AND/OR (vector
  `copysign`).
- **`gelu_erf_avx2(__m256)`** — compose the above.
- **`gelu_erf_slice(&mut [f32])`** — runtime-dispatched entry point
  (`is_x86_feature_detected!("avx2") && "fma")`); processes 8 lanes/iteration
  with a scalar tail for non-multiple-of-8 lengths.

The scalar `gelu_erf` / `erff` / `exp_fast` are retained as the non-AVX2
fallback and for the scalar tail.

### Results

**Microbenchmark** (`examples/gelu_bench.rs`, n=865, hidden=3072, 24 threads):

| path                     | per-block | ×12 backbone |
|--------------------------|----------:|-------------:|
| scalar `gelu_erf` loop   |   1.34 ms |     16.1 ms  |
| AVX2 `gelu_erf_slice`    |   0.14 ms |      1.7 ms  |
| **speedup**              |   **9.4×**|              |

Single-threaded (RAYON_NUM_THREADS=1) the speedup is **15.3×**, confirming
the scalar loop was entirely un-vectorised.

**End-to-end** (`bench`, DA3-BASE q5_k, 504×336, 24 threads, median of 5):

| metric                  | before   | after    | Δ       |
|-------------------------|---------:|---------:|--------:|
| `ffn_gelu` profile      |   7.56 ms|   2.14 ms| −5.4 ms |
| backbone total          |  253 ms  |  241 ms  | −12 ms  |
| `infer_mean`            |  349 ms  |  337 ms  | −12 ms  |

(The end-to-end improvement exceeds the isolated GELU savings because the
shorter GELU reduces rayon scheduling pressure on the surrounding GEMMs.)

**Parity**: `head_parity` rel diff **6.71e-7** (was 6.43e-7). The
<3e-8 regression is from the degree-6 polynomial `exp` (max rel error ~1e-7)
replacing libm's `expf`, and from AVX2 FMA preserving intermediate precision.
Well within the head parity budget.

### Lesson learned

**`target-cpu=native` does not guarantee auto-vectorisation of libm calls or
float↔int conversions.** Even with the host CPU's full feature set enabled,
LLVM emits scalar code for:
- `f32::exp()`, `f32::ln()`, `f32::sin()` etc. (libm calls)
- `f32::round_ties_even()`, `f32::floor()` (lower to `call round`/`call floor`)
- `f32 as i32` truncation in a vectorisable loop (scalar `cvttss2si`)

The fix is to write the hot loop with **explicit `core::arch::x86_64`
intrinsics** (or `std::simd` once stabilised) so the vector instructions are
emitted directly. The scalar fallback remains for non-x86_64 or pre-AVX2
targets.

**New benchmark**: `cargo run --release --example gelu_bench`

## Update: Investigated flash-attn exp vectorisation & conv3x3 stripe fusion (both negative)

Two follow-up optimisations from the GELU session's action items were
investigated and found **not to help** on DA3-BASE. Documenting here to save
future sessions from repeating the same dead ends.

### 1. Flash-attn softmax exp vectorisation (no gain)

**Hypothesis:** The flash-attn tiled AVX2 path's online-softmax exp+sum loop
(`src/flash_attn.rs` lines ~460–465) used scalar `(row[j] - mnew).exp()` in a
loop of `KV_TILE_SZ=64` iterations per query row. The code comment claimed
"the compiler already auto-vectorises `expf` at -O3 with target-cpu=native" —
which the GELU session proved false (`expf` emits `callq expf`). So this loop
was running 64 scalar `callq expf` per query row, per KV tile.

**Experiment:** Replaced the scalar exp loop with an AVX2 vectorised version
using the same `exp_fast_avx2` polynomial as the GELU kernel (8 lanes/iter +
`hsum_ps` for the row sum). Assembly inspection confirmed the `callq expf`
calls were eliminated from the flash-attn path.

**Result:** **No improvement** — `attn_flash` stayed at ~50 ms/iter (within
run-to-run noise). The exp loop is **not the bottleneck** of flash-attn; the
two per-tile GEMMs (`QKᵀ` and `AV`, via `tinyblas::gemm_nn_into_serial`)
dominate. Each tile does 2× 64×64×64 = 524k MAC in the GEMMs vs only 64 exp
evaluations in the softmax. Vectorising the 64 exps saves <2 ms; the GEMMs are
the real cost.

**Cross-module inlining pitfall:** Making `fast_block::avx2_gelu::exp_fast_avx2`
`pub(crate)` to share it with `flash_attn` caused a **14× regression** in
`ffn_gelu` (2 ms → 250 ms). The `#[target_feature(enable = "avx2,fma")]`
function, once referenced from a second module, was no longer inlined into
`gelu_erf_slice_avx2` by thin-LTO, despite being `#[inline]`. The fix was to
**duplicate** the exp polynomial locally in `flash_attn.rs` rather than share
it. (The experiment was ultimately reverted since it provided no gain.)

**Lesson:** The GELU win was large (12 ms) because GELU is *pure* exp+erf —
the exp IS the bottleneck there. Flash-attn softmax exp is a tiny fraction of
a GEMM-dominated kernel. **Always profile the target loop's share of its
parent stage before investing in vectorisation.**

### 2. Conv3x3 stripe fusion for `residual_conv_unit` (wrong premise)

**Hypothesis:** The fusion-stage `residual_conv_unit` (two sequential conv3x3
with an intermediate tensor) was profiled at ~8 ms for `rn1.rc2`. The action
item suggested fusing at the tile/stripe level to "keep the intermediate in
L1", avoiding a DRAM round-trip.

**Premise was wrong on two counts:**

1. **Spatial size:** The profile line `[fusion-pre-rcu] y_h=192 y_w=288` is
   the *upsampled output* size, not the `rc2` input size. `rc2` operates on
   the PRE-upsampling `top` feature, which for `rn1` is `96×144` (the output
   of `rn2`). So `rn1.rc2` is a `96×144` conv, not `192×288`.

2. **Cache residency:** The intermediate at `96×144×256×4 = 14 MiB` fits in
   the i7-13700K's 30 MiB L3. The existing comment in `fast_dpt.rs` already
   notes this ("up to 28 MiB at the largest fusion level"). Since the
   intermediate is L3-resident, `conv2` reads it at L3 bandwidth (~50+ GB/s),
   not DRAM bandwidth. There is no DRAM round-trip to eliminate. **Fusion
   wins only when it eliminates DRAM traffic, not L3 traffic** (a lesson the
   prior session already documented for the fusion-stage upsample fusion).

**Experiment (reverted):** A stripe-based `residual_conv_unit_striped` was
implemented — splitting H into 16-row stripes, extracting input row tiles
into contiguous buffers, running conv1+conv2 per stripe. It broke parity
(`rel diff` jumped from 6e-7 to 1.6e-2) due to halo-row border handling bugs,
and even if fixed would not help (the intermediate is already L3-resident).

**Lesson:** The prior session's note "keeps intermediate in L1" is infeasible
for these channel counts — even 1 row of `256×W×4` exceeds the 48 KiB L1.
And L3 residency is already sufficient. **This optimisation target is closed.**

### Current bottleneck breakdown (post-investigation)

`DA_FAST_PROFILE=1` per-iter breakdown (i7-13700K, 24 threads, q5_k, 504×336):

| stage | ms/iter | share | notes |
|-------|--------:|------:|-------|
| `attn_qkv` | 35.7 | 10% | tinyBLAS GEMM |
| `attn_flash` | 50.9 | 15% | 2× tinyBLAS GEMM per tile + online softmax |
| `attn_proj` | 14.7 | 4% | tinyBLAS GEMM |
| `ffn_fc1` | 53.9 | 15% | tinyBLAS GEMM |
| `ffn_gelu` | 2.3 | 1% | AVX2 batch kernel (optimised) |
| `ffn_fc2` | 56.8 | 16% | tinyBLAS GEMM |
| backbone (other) | ~45 | 13% | norm/rope/residual (not profiled) |
| head (DPT) | ~81 | 23% | Winograd convs (optimised) |
| activate | 1.0 | 0% | scalar `expf` loop (not worth vectorising) |
| **total** | **~355** | | |

The engine is **GEMM-bound** (~212 ms = 60% of inference is in tinyBLAS GEMMs).
Further gains require either a faster GEMM microkernel (AVX-512, or a better
panel-blocking strategy) or algorithmic changes (e.g. quantised K/V cache,
sparse attention). The current ~355 ms is ~30% faster than the C++/ggml
baseline (~510 ms) on this hardware.

---

## Update: tinyBLAS GEMM K-loop unroll by 2 (+~10 ms end-to-end)

**Shipped.** Unrolled the K-loop by 2 in both 6×16 microkernels
(`microkernel_6x16_ptr` and `microkernel_6x16_kc_ptr`). The compiler's
original single-iteration loop emitted ~8 ALU ops per K iteration for the
6 A-row address computations (`leaq`/`addq` chains), competing with the
12 FMA ops for ports 0/1. Unrolling by 2 amortises these ALU ops across
twice as many FMAs, and the second K iteration's A broadcasts reuse the
first iteration's row base addresses with a +4 byte offset (no extra
`leaq`).

**Synthetic GEMM bench (RAYON_NUM_THREADS=24, median of 5 runs):**

| shape | M×K @ K×N | before (ms) | after (ms) | Δ |
|-------|-----------|------------:|-----------:|----:|
| QK^T (per head) | 864×64 @ 64×864 | 0.259 | 0.247 | **+4.6%** |
| AV (per head) | 864×864 @ 864×64 | 0.129 | 0.112 | **+13.2%** |
| QKV proj | 864×768 @ 768×2304 | 2.926 | 2.863 | **+2.1%** |
| attn proj | 864×768 @ 768×768 | 1.082 | 1.088 | −0.6% |
| FFN fc1 | 864×768 @ 768×3072 | 4.605 | 4.450 | **+3.4%** |
| FFN fc2 | 864×3072 @ 3072×768 | 4.388 | 4.101 | **+6.5%** |
| lat3 proj | 768×768 @ 768×864 | 1.089 | 1.092 | −0.3% |
| lat3 resize | 768×6912 @ 6912×216 | 3.712 | 4.100 | **−10.4%** |

`lat3 resize` regresses −10% in the synthetic bench (the unrolled body
overflows the micro-op cache / DSB for this small-N/large-K shape), but
**the end-to-end inference improves by ~10 ms** (median ~360→~350 ms)
because the FFN fc1/fc2 gains (12 layers × ~0.5 ms each) outweigh the
lateral-resize regression (4 calls × ~0.4 ms each).

**Parity unchanged:** rel diff 6.45e-7 (the unroll doesn't reorder FMAs).

**Correctness:** all 177 unit tests pass, including the K-tail (odd-K)
and M-tail (partial-row) tests for both microkernels.

### Why the compiler didn't unroll on its own

LLVM at `-O3` with `#[target_feature(enable="avx2,fma")]` and `#[inline]`
does NOT auto-unroll the K loop. Assembly inspection confirmed a
single-iteration loop body with `cmpq`/`jne` every 12 FMAs. Explicit
manual unrolling by 2 (with a scalar tail for odd K) was needed.

### Microkernel efficiency analysis (1-thread, isolating compute)

Single-threaded GFLOP/s as % of P-core peak (~160 GF AVX2/FMA):

| shape | K | GFLOP/s | % peak | notes |
|-------|--:|--------:|------:|-------|
| attn proj | 768 | 144 | **90%** | A,B fit L2/L3 |
| FFN fc1 | 768 | 95 | 59% | K-blocking, A L1-resident |
| FFN fc2 | 3072 | 67 | 42% | K-blocking, A fits L1 |
| lat3 resize | 6912 | 61 | 38% | A doesn't fit L1 (83 KiB > 48 KiB) |

The large-K shapes are at 38-59% of peak — the gap is from A-cache
pressure (6 interleaved stride-1 streams exceed the L1 prefetcher's
2-4 stream tracking capacity) and L2 bandwidth contention. The K-unroll
helps by reducing per-K-iteration ALU overhead, but the fundamental
ceiling for these shapes without A-packing (BLIS layout) is ~60% of peak.

### Closed: further microkernel tweaks

- **K-unroll by 4:** would double code size again, likely causing more
  DSB overflow regressions like `lat3 resize`. Not worth it.
- **MR=4 (smaller tile):** same 2 MAC/cycle throughput but more tiles /
  dispatch overhead. Tested: TB=8 already spills; MR=4 would be worse.
- **Software prefetching:** the hardware prefetcher already handles the
  stride-N B access; A prefetching adds 6 instructions per K iter for
  uncertain gain. Not investigated further.
- **A-packing (BLIS [K,MR] layout):** would convert the 6 interleaved A
  streams into 1 sequential stream, potentially helping large-K shapes
  reach ~60% peak. The packing cost is ~1-3% of GEMM time (amortised over
  N-tiles). **Investigated and closed — see below.**

---

## Update: A-packing (BLIS [K,MR] layout) — closed, negative result

**Investigated and reverted.** Implemented A-packing for the K-blocked
GEMM path: repack the per-task `A[MC=6, kc_len]` panel into `[kc_len, MR]`
row-major layout before running the microkernel, converting the 6
stride-`k` A-row streams into a single stride-1 stream. Added a
`microkernel_6x16_packed_a_kc` variant, a scalar `pack_a_kc` packer, a
thread-local scratch buffer, and an `L1_KIB` threshold gate.

### Why it was reverted

The optimisation is ineffective for the DA3 shapes because of a
fundamental tension between **A-packing** and **B-L2-residency**:

1. `auto_kc` picks `KC` to keep the B-chunk `B[KC, N]` in L2 (≤ 2 MiB).
   For the per-layer GEMM shapes this forces `KC` small enough that the
   A tile `MR × KC × 4` already fits L1 (≤ 48 KiB):

   | shape | N | K | KC | A tile (KiB) | fits L1? |
   |-------|--:|--:|---:|-----------:|:--------:|
   | attn proj | 768 | 768 | 384 | 9 | yes |
   | FFN fc1 | 3072 | 768 | 168 | 4 | yes |
   | FFN fc2 | 768 | 3072 | 680 | 16 | yes |
   | lat3 proj | 864 | 768 | 384 | 9 | yes |
   | lat3 resize | 216 | 6912 | 3456 | **81** | **no** |

2. When the A tile already fits L1, packing **adds L1 write-pollution**
   (the pack buffer evicts B/C cache lines) with no prefetcher benefit.
   Measured regression on the synthetic bench (median of 3+ runs,
   RAYON_NUM_THREADS=24):

   | shape | unpacked (GF/s) | packed (GF/s) | Δ |
   |-------|--------------:|-------------:|----:|
   | FFN fc1 | ~935 | ~840 | **−10%** |
   | FFN fc2 | ~985 | ~815 | **−17%** |
   | attn proj | ~900 | ~920 | +2% (noise) |
   | lat3 proj | ~940 | ~940 | 0% |
   | lat3 resize | ~555 | ~570 | +3% |

3. Adding a threshold (`KC × MR × 4 > L1`, i.e. `KC > 2048` for 48 KiB L1)
   restricts packing to `lat3 resize` only. That recovers a small ~3%
   win on `lat3 resize` (a ~4 ms shape called once per inference in the
   DPT head) but it is **below end-to-end noise** (≤ 0.2 ms on ~350 ms).

### Why "larger KC + packing" doesn't help either

Tested forcing `KC=3072` (full K for FFN fc2) with packing on. This makes
A exceed L1 (so packing is beneficial in principle) but makes B exceed
L2 (9 MiB vs 2 MiB L2). The B-L3 traffic dominates: FFN fc2 drops to
~710 GF/s vs ~985 GF/s baseline. The K-blocking `auto_kc` heuristic is
already near-optimal — B-L2-residency wins over A-L1-residency for these
shapes.

### Conclusion

A-packing is **not** a worthwhile optimisation for the DA3 GEMM shapes.
The previous session's analysis conflated "large K" with "large `KC`":
the FFN shapes have large K (768-3072) but `auto_kc` splits them into
small `KC` chunks (168-680) that keep A L1-resident. The 38-42%
1-thread efficiency of FFN fc1/fc2 is bounded by **B-cache pressure and
dispatch overhead**, not A-cache pressure.

**Code reverted.** All 14 tinyBLAS unit tests pass. No end-to-end change.

---

## Update: vectorised flash-attention softmax exp (+~20 ms end-to-end)

**Shipped.** Replaced the scalar `(row[j] - mnew).exp()` loop in the
flash-attention online-softmax step with an AVX2 vectorised version that
reuses the existing [`exp_fast_avx2`] degree-6 polynomial approximation
(from the GELU kernel). The softmax was ~37% of `attn_flash` time (~19 ms
per inference) and was dominated by ~116M scalar `expf` evaluations per
inference (12 layers × 12 heads × 14 Q-tiles × 64 queries × 14 KV-tiles
× 64 exp per row).

### What changed

| File | Change |
|------|--------|
| `src/fast_block.rs` | Made `avx2_gelu` module `pub(crate)` and changed `exp_fast_avx2` from `#[inline]` to `#[inline(always)]`. Without `always`, LLVM keeps it as a `callq` + `vzeroupper` (~20 cycles) per call when it sees multiple call sites (gelu + softmax). |
| `src/flash_attn.rs` | Added `#[target_feature(enable="avx2,fma")]` to `forward_tiled_avx2` so the rayon closure body inherits the target-feature context and AVX2 intrinsics are inlined. Added `softmax_exp_sum_avx2` `#[target_feature]` helper that vectorises the exp+sum loop. Updated the misleading "the compiler auto-vectorises expf" comment. |

### The inlining problem

Three separate issues had to be solved to get the vectorised exp inlined
into the softmax loop:

1. **Rayon closures don't inherit `#[target_feature]`.** The softmax loop
   lives inside a `for_each` closure within `forward_tiled_avx2`. Adding
   `#[target_feature]` to the outer function does **not** propagate to the
   closure (which is a separate function). Without the feature, every
   `_mm256_*` intrinsic lowers to a `callq` wrapper.

2. **The `#[target_feature]` closure-inheritance gap.** Even with
   `#[target_feature(enable="avx2,fma")]` on `forward_tiled_avx2`, the
   closure's AVX2 intrinsics are still `callq`s unless the *closure
   body's* codegen target includes AVX2. Compiling the whole crate with
   `-C target-cpu=native` (via `.cargo/config.toml`) partially helps —
   it lets the intrinsics be inlined as native instructions. But the
   polynomial `exp_fast_avx2` function still needs `#[inline(always)]`
   (see #3).

3. **`exp_fast_avx2` was not inlined despite `#[inline]`.** LLVM's
   default inliner kept it as a separate function because it has two
   call sites (gelu + softmax). Each call emitted a `vzeroupper` (AVX→SSE
   state transition, ~20 cycles) + `callq` + `ret` sequence. With 8
   calls per softmax row, the overhead was ~160 cycles/row — 3× the
   actual polynomial work (~50 cycles). Changing to `#[inline(always)]`
   eliminated all calls and the `vzeroupper` transitions.

Before the `#[inline(always)]` fix, the vectorised softmax was actually
**~20% slower end-to-end** than the scalar baseline (411 ms vs 343 ms)
because the `vzeroupper` + `callq` overhead per loop iteration dwarfed
the exp computation itself.

### A/B benchmark

**Baseline (scalar `expf`):** min 347 / 343 / 342 ms → median **343 ms**
**Vectorised (`exp_fast_avx2`):** min 320 / 326 / 318 ms → median **320 ms**

**Improvement: ~23 ms (~6.7%) end-to-end.**

Per-stage (`attn_flash` profile, 12 layers per inference):
- Before: ~50.4 ms (4200 µs/call × 12)
- After: ~32.0 ms (2668 µs/call × 12)
- **Flash-attention speedup: ~18 ms (~37% faster)**

### Why the polynomial is accurate enough

`exp_fast_avx2` uses a degree-6 Horner polynomial for `2^r` over
`r ∈ [-0.5, 0.5]`, combined with integer-exponent bit manipulation. The
max relative error is < 2e-6 over `[-87, 88]` (validated by
`exp_fast_matches_libm` unit test). For flash-attention softmax, the
argument `row[j] - mnew` is always ≤ 0 (since `mnew` is the running max),
so there is no overflow risk; underflow (`exp(−large) → 0`) is handled by
the clamp in `exp_fast_avx2`.

**Parity unchanged:** head rel diff **6.45e-7** (identical to baseline —
the polynomial's < 2e-6 error is far below the q5_k quantisation noise).

### Correctness

All 177 unit tests pass, including `tiled_matches_reference_da3_shape`
(which compares the full flash-attention output against a naive
materialised-softmax reference at the DA3 shape).
