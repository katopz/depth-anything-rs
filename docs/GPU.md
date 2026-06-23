# GPU offload (CUDA)

depth-anything.cpp can offload model weights and the compute graph to a GPU
through ggml's backend layer. The C++ code calls **only ggml backend APIs** (no
direct CUDA), so the same sources build with or without CUDA — the only
difference is a CMake flag and which device the runtime selects.

> **Status (development box — has an RTX 4090):** **Working as of the latest
> session.** The Rust/candle CUDA path runs on the RTX 4090 via
> `--features cuda`. The fastest config is a **hybrid**: backbone on GPU via
> cuBLAS, DPT head on CPU via the hand-tuned Winograd/tinyBLAS fast path.
> Measured: **124 ms min / 127 ms median** inference (2.4× faster than CPU-only).
> The C++/ggml CUDA path builds but ggml 0.15.1's cublas wrapper is incompatible
> with CUDA 13 at runtime — see below.
>
> Previously blocked by a CUDA-toolchain version mismatch; see
> [Windows CUDA toolchain blocker](#windows-cuda-toolchain-blocker) for the
> history and the (now-applied) fixes.
>
> The path is also **validated on the NVIDIA GB10 (Blackwell, ARM64, CUDA 13)
> DGX** via `scripts/validate_gpu.sh`.

> **Status (this dev box's GPU, for context):** an **NVIDIA GeForce RTX 4090**
> (Ada, sm_89) with driver 610.62 (CUDA 13.3 runtime) and CUDA 13.3 toolkit.

## Build

CPU-only (default — no CUDA toolkit required):

```bash
cmake -B build -DDA_BUILD_CLI=ON
cmake --build build -j
```

With CUDA:

```bash
cmake -B build-cuda -DDA_BUILD_CLI=ON \
      -DDA_GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda -j
```

- `DA_GGML_CUDA=ON` forwards to ggml's `GGML_CUDA` and links the CUDA backend in.
- `CMAKE_CUDA_ARCHITECTURES=native` targets the GPU on the build host (on the
  GB10, Blackwell `sm_121`). If you omit it while `DA_GGML_CUDA=ON`, CMake
  defaults it to `native`; you can override it (e.g. `-DCMAKE_CUDA_ARCHITECTURES=90`).
- All CUDA-specific CMake is guarded behind `if(DA_GGML_CUDA)`, so a CPU-only
  build never touches CUDA settings.

Metal / Vulkan backends are wired the same way (`-DDA_GGML_METAL=ON`,
`-DDA_GGML_VULKAN=ON`).

## Device selection (`DA_DEVICE`)

The compute device is chosen by `da::Backend` from the ggml device registry:

- **unset** — auto-pick the first GPU/iGPU device a compiled-in backend
  registers, else fall back to CPU.
- `DA_DEVICE=cpu` — force the CPU backend (numeric baseline / CPU-only box).
- `DA_DEVICE=<name>` — select a registry device by name, case-insensitive
  (e.g. `CUDA0`, `Vulkan0`, `Metal`).

At startup the backend logs the chosen device:
`da::Backend using device: <name>`. Read that line off a GPU run to learn the
exact device name to pin.

## Offload design

When a non-CPU device is selected (`Backend::is_offloading()` true),
`ModelLoader::offload_weights()` mirrors the GGUF weights onto the device:

- A `no_alloc` device `ggml_context` is created; for every weight tensor a
  device tensor of the same type/shape is added, the context is allocated on the
  backend (`ggml_backend_alloc_ctx_tensors`), bytes are uploaded
  (`ggml_backend_tensor_set`), and the loader's tensor map is repointed at the
  device tensors. Metric-branch aliases (`m_vit.*`/`vit.*` sharing one source
  tensor) are de-duplicated by pointer so each weight is uploaded once.
- **Four host-read tensors are deliberately kept host-resident** because they are
  read via `->data` on the CPU during graph build (they produce host-computed
  graph *inputs*, not graph nodes). Offloading them would turn `->data` into a
  device pointer and crash:
  - `vit.pos_embed` — host bicubic interpolation (`interp_pos_embed`)
  - `vit.camera_token` — host camera-token inject
  - `vit.norm.weight`, `vit.norm.bias` — host post-norm
  - (the metric branch aliases `m_vit.*` → `vit.*`, so the same names apply.)
- On the CPU backend `offload_weights` is a **no-op**: graphs keep referencing
  the GGUF host tensors directly (zero-copy), so the CPU path is byte-identical.
- `offload_weights` is idempotent; the device buffer + context are freed in the
  loader's destructor before the host context.

### GPU-friendly op routing

After a successful offload, `Engine::load` calls `da::set_gpu_mode(true)`
(see `src/compute_mode.hpp`). In GPU mode the graph builders route to **standard
ggml ops that have CUDA kernels**, instead of the CPU-tuned custom paths that
would force GPU↔CPU round-trips:

- **Conv (`src/dpt_blocks.cpp`)** — 3×3 stride-1 convs use
  `ggml_conv_2d_direct` (CUDA kernel) instead of the CPU-only Winograd custom op
  (a `ggml_custom_4d`). 1×1 convs stay im2col GEMM either way.
- **Attention (`src/attention.cpp`)** — the manual `mul_mat` / `soft_max_ext`
  path (all CUDA-backed F32 ops) instead of `ggml_flash_attn_ext`, whose
  CPU-tuned F32-kv config may not map cleanly onto the CUDA flash kernel.

In both cases the explicit env override (`DA_CONV`, `DA_ATTN`) still takes
precedence. On CPU (`gpu_mode()` false) the defaults are unchanged — Winograd +
flash — so the CPU path is byte-identical.

Unsupported ops are additionally offloaded back to CPU automatically by the
`ggml_backend_sched` scheduler path in `src/backend.cpp`, so the graph runs even
if some op lacks a device kernel.

## Validation

`scripts/validate_gpu.sh` (run on the GB10 / any CUDA box):

1. Builds both a CPU-only (`build-cpu`) and a CUDA (`build-cuda`,
   `-DDA_GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native`) `da3-cli`.
2. Runs `da3-cli depth` on the same image with `DA_DEVICE=cpu` and on the GPU,
   writing two PFMs.
3. Compares the depth maps — reports `max|d|`, `mean|d|`, correlation; parity
   passes when `max|d| ≤ 1e-2` and `corr ≥ 0.999` (GPU floating-point ordering
   differs slightly, so an exact bit match is not expected).
4. Benchmarks both with `--repeat 10` and reports the GPU speedup.
5. Prints a clear `PASS`/`FAIL`.

Required env: `DA_GGUF` (model gguf), `DA_IMAGE` (input image). Optional:
`DA_CUDA_DEV` (pin a GPU device name; unset = auto-pick first GPU),
`DA_REPEAT` (default 10), `DA_THREADS`, `DA_TOL`, `DA_CORR`.

```bash
DA_GGUF=models/depth-anything-giant-f32.gguf \
DA_IMAGE=dumps/native_input.png \
bash scripts/validate_gpu.sh
```

## Fused backbone+head graph (single-image depth)

`Engine::depth_native` runs the backbone and DPT head as ONE ggml graph (`build_feats_graph` →
`build_depth_graph`) so the out-layer features stay device-resident — eliminating a feats
GPU→host→GPU round-trip and a second graph setup. The out-layer post-processing
(`cat([local_x, vit.norm(x)])` + token-0 strip) runs as ggml ops instead of a host scalar loop.
`DA_FUSED=0` falls back to the two-graph path. depth_pose / multi-view / metric / gs stay unfused.

Parity: fused vs unfused depth max|d|=1.2e-7 (CPU); CPU-vs-GPU corr=0.999998. On the **unified**
GB10 it's latency-neutral (160 vs 160 ms — the round-trip was already cheap); the win is for
**discrete** (PCIe) GPUs where the feats round-trip is a real copy. No regression anywhere; 31/31 tests.

## Windows CUDA toolchain blocker

**Symptom:** neither the C++ (ggml) nor the Rust (candle) CUDA build compiles on
the development box (i7-13700K + RTX 4090, Windows 11).

**Root cause:** the installed CUDA **toolkit** versions (11.8, 12.0, 12.1) are
all too old for the installed MSVC host compiler (14.41.34123, VS 2022 17.41).
MSVC 14.41's STL header `yvals_core.h` emits a hard `#error` (STL1002) when
invoked by nvcc from CUDA 12.0/12.1, because the STL requires **CUDA 12.4 or
newer** to recognise the MSVC version. The `-allow-unsupported-compiler` flag
bypasses nvcc's own version check but **not** MSVC's STL check — so the
combination cannot compile any `.cu`/`.cpp` file that pulls in `<yvals_core.h>`.

**The driver already supports what we need:** `nvidia-smi` reports driver
610.62 / **CUDA 13.3 runtime**. The **CUDA 13.3 toolkit** is installed at
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`.

**One build-time patch was needed:** the `cudarc` 0.13.9 crate (a transitive
dependency of candle-core 0.8.4) doesn't recognise CUDA 13.3 in its
`cuda_version_from_build_system()` match. The fix is to add `"13.3" => (12, 8),`
to `build.rs` (maps 13.3 to the 12.8 pre-generated bindings, which are
forward-compatible at the driver level). Edit:
`~/.cargo/registry/src/*/cudarc-0.13.9/build.rs`, then `touch` it and
`cargo clean -p cudarc` to force a rebuild.

**Fixes (historical — what was needed to get here):**

1. **Install the CUDA 13.3 toolkit** ✅ done. Matches driver 610.62, supports
   MSVC 14.41 and sm_89 (RTX 4090).
2. **Update the NVIDIA driver to ≥595.x** ✅ done (610.62). Required because
   CUDA 13.3's nvcc generates PTX ISA 9.3, which older drivers (≤591.x /
   CUDA 13.1 runtime) can't load.
3. **Patch cudarc 0.13.9** to recognise CUDA 13.3 (map to 12.8 bindings).
   See the note above.
4. *(Not needed)* ~~Install an older MSVC toolset~~ — CUDA 13.3 works with
   the installed MSVC 14.41.

**After the fix, the commands to build & run are:**

```bash
# C++ (ggml) — pick sm_89 for Ada (RTX 4090), sm_121 for Blackwell (GB10)
cmake -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DDA_BUILD_CLI=ON -DDA_GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_BMI2=ON -DGGML_SSE42=ON -DGGML_AVX_VNNI=ON -DDA_HAS_AVX512F=OFF
cmake --build build-cuda -j
DA_DEVICE=CUDA0 ./build-cuda/examples/cli/da3-cli.exe depth \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg --repeat 10 --threads 16

# Rust (candle) — Device::new_cuda(0) is picked up automatically by
# default_device() when the `cuda` feature is compiled in and DA_DEVICE != cpu.
# IMPORTANT: disable the CPU-only fast path on GPU — it would copy each
# activation GPU→CPU→GPU per block.
cargo build --release --features cuda --example bench
DA_FAST_ATTN=0 DA_FAST_HEAD=0 ./target/release/examples/bench.exe \
  --model models/depth-anything-base-q5_k.gguf \
  --input assets/samples/canyon.jpg --warmup 3 --repeat 10
```

**Expected outcome (rough, not yet measured):** the RTX 4090 has ~83 TFLOP/s
f32 and ~330 TFLOP/s with tensor cores, vs the i7-13700K's ~1.5 TFLOP/s f32.
DA3-BASE inference should drop into the **10–30 ms** range — 10–30× faster than
the ~300 ms CPU path. The fair C++-vs-Rust GPU comparison is still TBD; it
needs the toolchain fix above first.
