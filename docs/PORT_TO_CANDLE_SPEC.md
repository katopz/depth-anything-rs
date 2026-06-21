# Depth-Anything 3 (C++/ggml) → Rust/Candle Port Spec

This document is the **source of truth** for porting the C++/ggml DA3 inference
engine in this repo to Rust using the `candle` tensor library. It is derived
exclusively from reading the C++ sources (filenames cited inline). No code is
written here — every shape, constant, KV key, tensor name and algorithm
described below is reproduced verbatim from the implementation.

All paths are relative to the repo root, e.g. `src/engine.cpp`.

Conventions used below:

- Tensor shapes are written in **logical / PyTorch order** `[B, C, H, W]`, even
  though ggml stores them in column-major `ne[0..3]` with the fastest axis in
  `ne[0]`. Each section calls out the ggml layout when it matters.
- "Row-major" refers to host `std::vector<float>` output buffers.
- All metadata key strings begin with the literal prefix `depthanything3.`.

---

## 1. Repository map (the modules in scope)

| File | Role |
|------|------|
| `src/engine.{hpp,cpp}` | Top-level `Engine` class. Orchestrates preprocess → backbone → DPT head → cam pose (and GS / nested / multi-view / ray-pose variants). |
| `src/model_loader.{hpp,cpp}` | Loads the GGUF, reads every KV into `struct Config`, builds the tensor name → `ggml_tensor*` map, optionally mirrors weights to a GPU buffer. |
| `src/backend.{hpp,cpp}` | Thin wrapper over `ggml_backend_*`. Builds a graph from a lambda, allocates via persistent `ggml_gallocr` (CPU) or `ggml_backend_sched` (GPU fallback), uploads inputs, computes, reads output back. |
| `src/preprocess.{hpp,cpp}` | Image resize (`preprocess` legacy floor-to-patch bilinear, `preprocess_real` cv2-faithful upper/lower-bound bicubic/area) + ImageNet normalize. |
| `src/image_io.{hpp,cpp}` | stb_image-based JPEG/PNG/etc → `Image { w, h, rgb(HWC uint8) }`. |
| `src/compute_mode.hpp` | One global bool `gpu_mode()` toggled after load; routes graph builders between CPU-tuned custom ops and GPU-friendly standard ops. |
| `include/da_gguf_keys.h` | Auto-generated `#define`s for every KV string. |
| `examples/cli/main.cpp` | CLI entrypoint: parses argv, dispatches to `Engine::*`. |

Adjacent modules that the engine pulls in but were not in the requested list
(consult these when porting the matching sub-system):

- `src/dino_backbone.{hpp,cpp}` — ViT backbone (patch embed, RoPE, blocks, pos-embed interp, multi-view).
- `src/vit_block.{hpp,cpp}` + `src/attention.{hpp,cpp}` — one transformer block + attention.
- `src/dpt_head.{hpp,cpp}` — DualDPT depth (+sky, +aux ray) head.
- `src/dpt_blocks.{hpp,cpp}`, `src/uv_posembed.{hpp,cpp}` — DPT primitives + the UV positional embedding.
- `src/cam_pose.{hpp,cpp}` — camera pose MLP (cam_dec).
- `src/ray_pose.{hpp,cpp}` — aux-ray → extrinsics/intrinsics solver.
- `src/gs_head.{hpp,cpp}` + `src/gs_adapter.{hpp,cpp}` — DA3-GIANT 3D-Gaussian head.
- `src/nested.{hpp,cpp}` — nested metric aligner.
- `src/rope2d.{hpp,cpp}` — 2-D RoPE tables + apply.
- `src/quantize.{hpp,cpp}` — f16/q8_0/q6_k/q5_k/q4_k re-quantizer.

---

## 2. `image_io` — file → `Image`

### 2.1 Data type

```cpp
struct Image { int w=0, h=0; std::vector<unsigned char> rgb; };   // HWC uint8
```

`rgb.size() == w*h*3`, channel order **RGB** (forced via stb_image's
`req_comp=3`).

### 2.2 Entry points

- `bool load_image_rgb(const std::string& path, Image& out);`
  - Calls `stbi_load(path, &w, &h, &c, /*req_comp=*/3)` (forces 3 channels
    regardless of source).
- `bool load_image_rgb_buffer(const unsigned char* bytes, size_t len, Image& out);`
  - Same but `stbi_load_from_memory`.

No orientation flipping, no premultiply, no color-space conversion. Source of
the "uint8 pixels" used by every downstream resizer.

### 2.3 Port note (candle)

Use the `image` crate (or `candle::utils`), decode to RGB8 HWC, then convert to
`Tensor [3,H,W] f32` only inside `preprocess`. **No EXIF orientation handling
exists in the C++** — keep that bit-exact.

---

## 3. `preprocess` — `Image` → `Preprocessed`

### 3.1 Output struct

```cpp
struct Preprocessed {
    int H=0, W=0;
    std::vector<float> chw;          // [3,H,W] f32, C-major (c slowest, w fastest)
    int orig_w=0, orig_h=0;          // original image dims
    float scale_w=1.f, scale_h=1.f;  // W/orig_w, H/orig_h (used to rescale intrinsics)
};
```

### 3.2 Two resize paths

There are **two** resizers. The CLI uses `preprocess_real` by default
("native"); `--legacy-resize` selects `preprocess` (used only by the parity
fixtures). See `examples/cli/main.cpp` lines 183, 197, 205.

#### 3.2.1 `preprocess` (legacy, `src/preprocess.cpp:26`)

1. Floor each dim to a multiple of `patch_size`:
   ```cpp
   int dw = (img.w/patch)*patch, dh = (img.h/patch)*patch;
   if (dw==0) dw=patch;  if (dh==0) dh=patch;
   ```
2. **Bilinear** resize (`resize_bilinear`, `src/preprocess.cpp:8`) with
   coordinate map `src = (dst+0.5)*scale - 0.5` (scale = src/dst), border
   **replicate** (clamped indices).
3. Convert to `float / 255.f`, then per-channel
   `(v - mean[c]) / std[c]` into `[3,H,W]` C-major layout
   (`out.chw[c*H*W + y*W + x]`).
4. `scale_w = dw/orig_w; scale_h = dh/orig_h`.

#### 3.2.2 `preprocess_real` (production, `src/preprocess.cpp:157`)

Implements the **real DA3 `InputProcessor`** ("upper_bound_resize"):

```
target = cfg.img_resize_target (504 default)
upper  = cfg.img_resize_mode does NOT start with "lower"   (default upper_bound)
patch  = cfg.patch_size                                     (14)
```

Two resize steps, each quantizing to **uint8** (cv2 → PIL.fromarray → ToTensor
semantics):

**Step 1 — boundary resize**:
```cpp
int bound = upper ? max(cur.w,cur.h) : min(cur.w,cur.h);
double scale = (double)target / bound;
int nw = max(1, py_round(cur.w*scale));   // Python round-half-to-even
int nh = max(1, py_round(cur.h*scale));
cur = (scale > 1.0) ? resize_cubic(cur,nw,nh) : resize_area(cur,nw,nh);
```

**Step 2 — round to multiple of patch**:
```cpp
int nw = max(1, nearest_multiple(cur.w, patch));
int nh = max(1, nearest_multiple(cur.h, patch));
if (nw!=cur.w || nh!=cur.h){
    bool upscale = (nw>cur.w) || (nh>cur.h);
    cur = upscale ? resize_cubic(cur,nw,nh) : resize_area(cur,nw,nh);
}
```
`nearest_multiple` picks whichever of `floor(x/p)*p` and `+p` is closer
(ties → up).

**Step 3 — ImageNet normalize** (same as legacy path).

`py_round` is `std::nearbyint` (ties-to-even, matches Python's `round`).

#### 3.2.3 `resize_cubic` — `src/preprocess.cpp:61`

- Catmull-Rom bicubic, `a = -0.75` (matches `cv2.INTER_CUBIC`).
- Coordinate map `src = (dst+0.5)*scale - 0.5`, `scale = src_size/dst_size`.
- Separable: horizontal pass producing float intermediate, vertical pass with
  single `saturate_cast<uchar>` at the end.
- Border: **replicate** (`std::clamp(ix-1+k, 0, sw-1)`).
- `sat_u8(v)` = `std::lrintf` (ties-to-even, == cvRound), then clamp to `[0,255]`.

#### 3.2.4 `resize_area` — `src/preprocess.cpp:122`

- Reproduces `cv2.INTER_AREA` decimation: builds `AreaTap {di, si, alpha}`
  tables via `computeResizeAreaTab` (`area_tab` at line 103).
- Separable horizontal then vertical pass, float accumulator, single saturate.
- **Upscale fallback**: if `dw>=sw && dh>=sh`, secretly calls
  `resize_bilinear` (cv2 does the same).

### 3.3 Constants

| Constant | Value | Source |
|----------|-------|--------|
| `patch_size` | 14 (default; from `cfg.patch_size`) | GGUF KV `depthanything3.patch_size` |
| `img_resize_target` | **504** (default; longest side) | GGUF KV `depthanything3.img.resize_target` |
| `img_resize_mode` | `"upper_bound"` | GGUF KV `depthanything3.img.resize_mode` |
| `img_mean` | `[0.485, 0.456, 0.406]` (ImageNet) | GGUF array `depthanything3.img.mean` |
| `img_std` | `[0.229, 0.224, 0.225]` (ImageNet) | GGUF array `depthanything3.img.std` |

> The "504×336" you may have seen is **derived, not stored**: with a 4:3 input
> and `upper_bound` mode, longest side → 504; step 2 then snaps the short side
> to the nearest multiple of 14, which is 336 (= 24·14). 504 = 36·14. Both
> dimensions are always multiples of `patch_size` (14). The patch grid is
> `gh = H/14, gw = W/14`, e.g. 504×336 → `gh=24, gw=36`, `N_patch = 864`.

### 3.4 Caching

No host-side caching at this layer. (Caching of position embeddings happens
inside the backbone / head — see §6 and §8.)

---

## 4. `model_loader` — GGUF → `Config` + tensor map

### 4.1 Loader implementation notes (`src/model_loader.cpp:73`)

- Uses **ggml's built-in gguf API** (`gguf_init_from_file`, `gguf_find_key`,
  `gguf_get_val_*`, `gguf_get_arr_*`, `gguf_get_tensor_name`,
  `ggml_get_tensor`). No custom binary parsing.
- `gguf_init_params{ no_alloc=false, ctx=&ctx_ }` → the loaded `ggml_context`
  owns every weight tensor's data; the loader hands out raw `ggml_tensor*`
  pointers into that buffer (zero-copy on CPU).
- Builds `std::unordered_map<std::string, ggml_tensor*> tensors_` for O(1)
  lookup by name.

### 4.2 Nested-metric aliasing (`src/model_loader.cpp:82-154`)

If the GGUF carries `depthanything3.m_vit.embed_dim`, the loader treats it as a
**nested-metric** GGUF (the metric branch of `DA3NESTED-GIANT-LARGE`) and:

1. Reads every config field from the `m_vit.*` / `m_head.*` KVs.
2. After tensor enumeration, **aliases** every `m_vit.X` → `vit.X` and every
   `m_head.X` → `head.X` so the existing backbone/DPT code (hardcoded to
   `vit.`/`head.` prefixes) works unchanged.
3. `head_pos_embed` defaults to **false** for the metric branch
   (`DA3METRIC` head's `nn.LayerNorm`-free DPT).

### 4.3 `offload_weights(Backend&)` — `src/model_loader.cpp:166`

- **CPU backend (`!is_offloading()`)**: no-op. Graphs keep referencing the
  gguf host tensors in `ctx_` directly.
- **GPU backend**: allocates a `no_alloc` device context, mirrors every tensor
  *except* the four **host-read** tensors listed below, uploads bytes, and
  repoints `tensors_[name]` to the device copies. The host-read tensors stay
  on CPU so the host-side bicubic interp / camera-token injection /
  layernorm keep working.

The host-read tensors (defined in `is_host_read_tensor`,
`src/model_loader.cpp:25`):

```
vit.pos_embed
vit.camera_token
vit.norm.weight
vit.norm.bias
```

### 4.4 Load validation (`src/model_loader.cpp:158`)

Returns `true` iff `embed_dim>0 && depth>0 && num_heads>0 && head_dim>0`.

### 4.5 Full GGUF metadata key catalogue

Every key the loader actually reads. Keys marked **not read by loader** are
written by the converters but currently unused at inference (included for
completeness). All keys are prefixed `depthanything3.`.

#### 4.5.1 Top-level / arch

| Key | C++ type | Loader use | Default |
|-----|----------|-----------|---------|
| `arch` | string | `cfg.arch` (route discriminator; `"depthanything2"` = DA2) | `"depthanything3"` |
| `checkpoint_name` | string | `cfg.checkpoint_name` (printed by `info`) | `""` |
| `patch_size` | u32 | `cfg.patch_size` | 14 |
| `image_size` | u32 | **not read by loader** (informational; the real resize uses `img.resize_target`) | — |
| `task_caps` | (any) | **not read by loader** (bitmask of heads present) | — |

#### 4.5.2 ViT backbone (anyview / DA3) — read from `vit.*` unless metric GGUF, then `m_vit.*`

| Key | C++ type | Field | Default |
|-----|----------|-------|---------|
| `vit.embed_dim` | u32 | `embed_dim` | **required** |
| `vit.depth` | u32 | `depth` (#blocks) | **required** |
| `vit.num_heads` | u32 | `num_heads` | **required** |
| `vit.head_dim` | u32 | `head_dim` | **required** |
| `vit.mlp_hidden` | u32 | `mlp_hidden` | (informational) |
| `vit.ffn_type` | string | `ffn_type` (`"mlp"` or `"swiglu"`) | `"mlp"` |
| `vit.num_register_tokens` | u32 | `num_register` | 0 |
| `vit.init_values` | f32 | `init_values` (informational; ls1/ls2 carry the actual γ) | 0 |
| `vit.alt_start` | i32 | `alt_start` (block index where global attn begins; −1 disables) | −1 |
| `vit.rope_start` | i32 | `rope_start` (block index where RoPE begins; −1 disables) | −1 |
| `vit.qknorm_start` | i32 | `qknorm_start` (informational) | −1 |
| `vit.rope_freq` | f32 | `rope_freq` (RoPE base θ) | 100.0 |
| `vit.cat_token` | bool | `cat_token` (backbone feat = `cat[local_x, norm(x)]`) | `true` |
| `vit.qkv_bias` | bool | `qkv_bias` (informational; bias tensor presence drives the graph) | `true` |
| `vit.ln_eps` | f32 | `ln_eps` (block LN epsilon) | `1e-6` |
| `vit.interpolate_offset` | f32 | `interp_offset` (pos-embed bicubic offset) | 0.1 |
| `vit.interpolate_antialias` | bool | `interp_antialias` (informational) | `false` |
| `vit.pos_embed_grid` | u32 | `pos_embed_grid` (M, where pos_embed has `M*M+1` rows) | 0 |
| `vit.out_layers` | i32[] | `out_layers` (block indices returned as feats; default `[5,7,9,11]`) | `[]` |

#### 4.5.3 Metric ViT-L (nested only) — `m_vit.*`

Identical schema to §4.5.2 with `m_vit.` prefix. The loader uses these **only**
when `depthanything3.m_vit.embed_dim` is present in the GGUF.

#### 4.5.4 Preprocessing

| Key | C++ type | Field | Default |
|-----|----------|-------|---------|
| `img.mean` | f32[] | `img_mean` (length 3) | (required) |
| `img.std` | f32[] | `img_std` (length 3) | (required) |
| `img.resize_mode` | string | `img_resize_mode` | `"upper_bound"` |
| `img.resize_target` | u32 | `img_resize_target` | 504 |

#### 4.5.5 DualDPT main depth head — `head.*` (or `m_head.*` when nested-metric)

| Key | C++ type | Loader use | Default |
|-----|----------|-----------|---------|
| `head.features` | u32 | `cfg.head_features` (fusion width, e.g. 128) | 0 |
| `head.out_channels` | i32[] | `cfg.head_out_channels` (length 4) | `[]` → `[96,192,384,768]` |
| `head.pos_embed` | bool | `cfg.head_pos_embed` (UV pos-embed on/off) | `true` (metric branch default `false`) |
| `head.output_dim` | u32 | **not read by loader** (derived from `head.scratch.out2b.weight` `ne[3]`) | — |
| `head.down_ratio` | u32 | **not read** | — |
| `head.activation` | string | **not read** (depth uses `exp` always; DA2 has its own path) | — |
| `head.conf_activation` | string | **not read** (host post: `exp(logit)+1`) | — |
| `head.sky_activation` | string | **not read** (sky uses `relu(logit)`) | — |
| `head.norm_type` | string | **not read** (presence of `head.norm.weight` gates it) | — |
| `head.max_depth` | f32 | `cfg.head_max_depth` (DA2 metric scale; 0 ⇒ relative) | 0 |

> **Important port note:** `output_dim` is **not** read from the GGUF. The
> engine derives it from the *shape* of `head.scratch.out2b.weight`:
> ```cpp
> const int output_dim = out2b_w ? (int)out2b_w->ne[3] : 2;
> ```
> (`ne[3]` is the conv output-channel count in ggml's column-major layout —
> i.e. PyTorch `out_channels`.) Likewise `head.norm.*` presence (not the
> `norm_type` string) gates the head LayerNorm.

#### 4.5.6 DualDPT auxiliary ray head (`--with-aux` only)

| Key | C++ type | Loader use |
|-----|----------|-----------|
| `head.has_aux` | bool | **not read** (presence of `head.scratch.rn1_aux.out.weight` gates it; see `has_aux()` in `engine.cpp:195`) |
| `head.aux_ray_dim` | u32 | **not read** (hardcoded 6 ray channels + 1 conf) |
| `head.aux_levels` | u32 | **not read** (only the finest level is emitted by the converter) |

#### 4.5.7 Camera pose decoder (`cam_dec`)

| Key | C++ type | Loader use |
|-----|----------|-----------|
| `cam.dim_in` | u32 | **not read** (validated at runtime against `cam.bb0.weight->ne[0]`) |

#### 4.5.8 GSDPT (giant 3D-Gaussian head)

| Key | C++ type | Loader use |
|-----|----------|-----------|
| `gs.output_dim` | u32 | not read by `model_loader` (gs_head reads tensor shapes directly) |
| `gs.features` | u32 | not read |
| `gs.out_channels` | (array) | not read |
| `gs.sh_degree` | u32 | read by `gs_adapter` |
| `gs.scale_min` | f32 | read by `gs_adapter` |
| `gs.scale_max` | f32 | read by `gs_adapter` |
| `gs.pred_offset_depth` | bool | read by `gs_adapter` |
| `gs.pred_offset_xy` | bool | read by `gs_adapter` |
| `gs.pred_color` | bool | read by `gs_adapter` |

### 4.6 Tensor dtypes

Loaded as-is from the GGUF — **f32 by default**, and the supported quantized
types are whatever ggml's `gguf_init_from_file` will materialize (the
quantizer in `src/quantize.cpp:25` accepts `f16 / q8_0 / q6_k / q5_k /
q4_k`). ggml handles quantized-tensor ops natively, so the loader does **no
dequant at load time**; quantized weights stay quantized through `mul_mat` and
are dequantized lazily by ggml kernels.

The **quantizer only touches 2-D matmul weights** — see §10.

---

## 5. `backend` — graph build / compute wrapper

### 5.1 Public surface (`src/backend.hpp`)

```cpp
class Backend {
    Backend();                                  // auto-picks GPU via DA_DEVICE, else CPU
    bool is_offloading() const;                 // true iff non-CPU backend
    ggml_backend_t handle() const;
    void set_n_threads(int n);

    // Register host tensors as graph inputs (F32, or I32 for RoPE positions).
    ggml_tensor* add_graph_input(ctx, pool, host, n);
    ggml_tensor* add_graph_input_nd(ctx, pool, host, ne, n_dims);
    ggml_tensor* add_graph_input_nd_borrow(ctx, host, ne, n_dims);   // no copy
    ggml_tensor* add_int32_input_nd(ctx, pool, host, ne, n_dims);

    // Build -> alloc -> upload inputs -> compute -> read output (f32) back.
    bool compute(build_lambda, std::vector<float>& out);
    bool forward_capture(build_lambda, out);    // honors capture() nodes
    void capture(ggml_tensor* t, std::vector<float>* dst);
    void add_graph_root(ggml_tensor* t);
};
```

`GraphInputPool` (`src/backend.hpp:14`) is just an owning buffer for the host
bytes registered during build; it outlives the compute call.

### 5.2 Device selection (`src/backend.cpp:51`)

- `DA_DEVICE=cpu` → force CPU.
- `DA_DEVICE=<name>` (case-insensitive) → match a registry device name (e.g.
  `CUDA0`, `Vulkan0`, `Metal`).
- unset → first `GGPU`/`IGPU` device, else CPU.

A GPU device is **offloading** (`is_offloading() == true`) only when the
scheduler path is in use (`type != GGML_BACKEND_DEVICE_TYPE_CPU`).

### 5.3 Compute pipeline (`Backend::compute`, `src/backend.cpp:207`)

1. Allocate a no-alloc metadata ctx sized for `kGraphSize = 16384` tensors +
   graph overhead.
2. Clear `pending` / `captures` / `roots`.
3. Invoke `build(ctx)` → output tensor.
4. Mark output + captures as outputs (`ggml_set_output`).
5. Build the cgraph: expand captures first, then registered roots, then the
   output.
6. Decide single-backend vs scheduler:
   - If GPU and **any** node unsupported by the GPU backend → use
     `ggml_backend_sched` over `{GPU, CPU}` (op-level CPU fallback).
   - Else persistent `ggml_gallocr` on the active backend.
7. Upload inputs (`ggml_backend_tensor_set`).
8. `ggml_backend_graph_compute` (or `..._sched_graph_compute`).
9. Read back captured tensors and the final output as f32.

The CPU path is byte-identical to a single-backend run; the scheduler is only
used on a GPU whose backend is missing a kernel for one of this graph's ops.

### 5.4 Port note (candle)

- The `Backend::compute(lambda, out)` pattern maps to: build a `candle::Tensor`
  DAG → `forward()` → `.to_device(Device::Cpu)` → `.flatten().to_vec1()`.
- The "borrow vs copy" distinction (`add_graph_input_nd_borrow`) is essential
  for the cached UV pos-embed (§8.1) — keep the cached buffer alive across
  forwards in Rust too.
- `set_n_threads` is CPU-only; on GPU the thread count is ignored.

---

## 6. Engine orchestration (`src/engine.{hpp,cpp}`)

### 6.1 `Engine::load` — `src/engine.cpp:19`

```cpp
ml_.load(path)                  // parse GGUF, build cfg + tensor map
be_.set_n_threads(n_threads)
ml_.offload_weights(be_)        // no-op on CPU, mirrors to GPU otherwise
set_gpu_mode(be_.is_offloading())
```

`load_nested` (`engine.cpp:28`) does the same for a second `(metric_ml_,
metric_be_)` pair on the metric GGUF.

### 6.2 Routing helpers (`engine.hpp:60`, `engine.cpp:60`)

- `is_mono()` → `head.scratch.out2b.weight` has `ne[3]==1` **and**
  `head.scratch.sky_out2b.weight` is present.
- `is_da2()` → `cfg.arch == "depthanything2"`.
- `has_aux()` → `head.scratch.rn1_aux.out.weight` is present.

### 6.3 The production depth+pose path
(`depth_pose_native`, `src/engine.cpp:173`)

```
Image img
  → preprocess_real(img, cfg, p)             [H, W, 3·H·W f32 C-major]
  → DinoBackbone::forward(p.chw, H, W, feats[4], cam_tokens[4])
        feats[i]     : [N_patch * 2*embed]   token-major (token·C+chan)   (cat_token=true)
        cam_tokens[i]: [2*embed]             (cat[local_x[t0], x[t0]] RAW)
  → DptHead::depth(feats, H, W, depth, conf)
        depth: [H*W] row-major, = exp(logits[..., 0])
        conf : [H*W] row-major, = exp(logits[..., 1]) + 1
  → CamPose::pose(cam_tokens[3], H, W, pe_unused, ext[12], intr[9])
        ext  : 3x4 row-major (12 floats)
        intr : 3x3 row-major (9 floats)
```

`depth_native_*` does the same minus the cam-pose call (and optionally
**fuses** the backbone + DPT head into one graph — §6.5).

### 6.4 All engine entrypoints and their data flow

| Method | Preprocess | Backbone call | Head call | Pose call | Output |
|--------|-----------|---------------|-----------|-----------|--------|
| `depth_native_image` | `preprocess_real` | `bb.forward` or fused | `head.depth` | — | depth, conf |
| `depth_native_fused` | `preprocess_real` | `bb.build_feats_graph` (in-graph) | `head.build_depth_graph` (in-graph) | — | depth, conf |
| `depth_native_unfused` | `preprocess_real` | `bb.forward` | `head.depth` | — | depth, conf |
| `depth_pose_native` | `preprocess_real` | `bb.forward` | `head.depth` | `cam.pose(cam_tokens[3])` | depth, conf, ext, intr |
| `depth_pose_rays_native` | `preprocess_real` | `bb.forward` | `head.depth` + `head.rays` | `solve_ray_pose` | depth, conf, ext, intr |
| `depth_pose` (legacy resize) | `preprocess` | `bb.forward` | `head.depth` | `cam.pose` | depth, conf, ext, intr |
| `depth` / `depth_image` (legacy) | `preprocess` | `bb.forward` | `head.depth` | — | depth, conf |
| `depth_mono` | `preprocess_real` | `bb.forward` | `head.depth_sky` | — | depth, sky |
| `depth_relative` (DA2) | `preprocess_real` | `bb.forward` | `head.depth_relative` | — | depth |
| `depth_pose_multi` | `preprocess` per view | `bb.forward_mv` (cross-view global attn) | per-view `head.depth` | per-view `cam.pose(cam_tokens[3][v])` | `ViewResult[S]` |
| `reconstruct` (giant) | `preprocess` | `bb.forward` | `head.depth` | `cam.pose` | depth, conf, ext, intr, raw_gs, Gaussians |
| `depth_metric` (nested) | `preprocess` | anyview `bb.forward` + metric `bb.forward` (SAME `p.chw`) | anyview `head.depth` + metric `head.depth_sky` | anyview `cam.pose` | `NestedOut` (aligned depth, ext, intr, scale_factor) |

For DA3 main models the head's output activation is applied **on host** after
graph compute:

```cpp
depth_out[i] = exp(logits[i]);
conf_out[i]  = exp(logits[HW + i]) + 1.0f;     // only if output_dim >= 2
```

DA2 path (`dpt_head.cpp:369`):
```cpp
metric   (max_depth>0): depth = sigmoid(logit) * max_depth
relative (max_depth=0): depth = max(0, logit)        // ReLU
```

Monocular (`dpt_head.cpp:363` `depth_sky`):
```cpp
depth = exp(logit);              // sky head produces the same single logit channel
sky   = max(0, sky_logit);       // relu
```

### 6.5 Fused vs unfused depth (`engine.cpp:132`)

The fused single-image path builds **one** graph: backbone feats → DPT head,
so the four feat tensors never leave the device. The same host post-process is
applied to the resulting `[output_dim*H*W]` logits. Restricted to
`cat_token == true` (BASE / GIANT); falls back to unfused otherwise.

The unfused path is two separate `compute()` calls (backbone, then head) with
feats round-tripped through host f32 vectors.

### 6.6 CLI dispatch (`examples/cli/main.cpp`)

`main()` parses argv via `da::cli::parse` into `Parsed` (`src/cli.hpp`):

```cpp
enum class Sub { Info, Depth, Reconstruct, Quantize, Help, None };
struct Parsed {
    Sub sub;
    std::string model, metric_model;             // --metric-model => nested
    std::string input, output_pfm, output_png;
    std::string output_sky, output_pose, output_ply;
    std::string output_glb, output_colmap;
    bool colmap_binary = true;
    std::vector<std::string> inputs;             // repeated --input
    std::string out_prefix;
    std::string q_in, q_out, q_type;             // quantize subcommand
    int n_threads = 0, repeat = 0;
    bool invert = true, legacy_resize = false, ray_pose = false;
};
```

Depth command routing (`cmd_depth`, `main.cpp:148`):

1. If `metric_model` set → `cmd_depth_metric` (loads nested).
2. If `repeat > 0` and one input → `cmd_depth_bench` (timing harness).
3. Load `Engine::load(model, n_threads)`.
4. If `--glb` or `--colmap` → `cmd_depth_export` (single-view native pose +
   export).
5. If multiple `--input` → `cmd_depth_multi` (`depth_pose_multi`).
6. Else by model type:
   - `is_da2()` → `depth_relative_path`.
   - `is_mono()` → `depth_mono_path` (+ optional `--sky out.pfm`).
   - else if `--pose`:
     - `--ray-pose` → `depth_pose_rays_native_path` (requires `has_aux()`).
     - native → `depth_pose_native_path`.
     - `--legacy-resize` → `depth_pose_path`.
   - else: `depth_native` / `depth`.

Outputs (per the CLI):
- `--pfm out.pfm` — float PFM depth.
- `--png out.png` — colorized grayscale depth (turbo), `--invert` toggles polarity.
- `--sky out.pfm` — monocular sky map.
- `--pose out.json` — `{"extrinsics":[12], "intrinsics":[9]}`.
- `--glb out.glb`, `--colmap dir/`, `--ply out.ply` — exporters.
- `info` subcommand prints `checkpoint_name`, `embed_dim`, `depth`,
  `num_heads`.

`n_threads` defaults to **1** when passed as 0.

---

## 7. ViT backbone (`src/dino_backbone.cpp`)

### 7.1 Input / output shapes

Input: `[3, H, W]` f32, C-major (the `Preprocessed.chw` field).

Tokens sequence built inside the graph:
```
img [W,H,3,1]
  → conv2d(patch_embed.weight, stride=patch, kernel=patch) → [gw,gh,embed,1]
  → reshape_2d(N_patch=gw*gh, embed) → transpose → [embed, N_patch]
  → + patch_embed.bias
  → concat(cls_token [embed,1], x, dim=1)            → [embed, 1+N_patch]
  → + interp_pos_embed(gh, gw)                       → [embed, 1+N_patch]
```

For 504×336 with patch=14, embed=768 (BASE): `gw=36, gh=24, N_patch=864,
N_tok=865`.

Output (per out-layer, the 4 indices in `cfg.out_layers`, default
`[5,7,9,11]`):
- `feats[o]`: `[N_patch, C]` flat (`N_patch * C` floats) where
  `C = 2 * embed` if `cat_token` else `embed`. Layout is token-major /
  channel-minor (`flat[token*C + chan]`). The "channel" half #0 is
  `local_x_raw` (last LOCAL block's output), half #1 is
  `layernorm(x, vit.norm.{w,b}, ln_eps)` — token 0 (cls) **stripped**.
- `cam_tokens[o]`: `[2*embed]` flat = `cat([local_x[t0], x[t0]])` RAW (no
  final norm). Unused on metric branch (`cat_token=false`) but produced for
  shape consistency.

### 7.2 Positional embedding interpolation
(`interp_pos_embed`, `src/dino_backbone.cpp:18`)

- Source tensor: `vit.pos_embed` of shape `[embed, 1+M*M]` (token-major),
  where `M = cfg.pos_embed_grid` (37 for BASE → 1370 rows).
- Output: `[embed, 1+gh*gw]` (token-major).
- Row 0 (cls pos-embed) is **copied unchanged** from row 0 of the source.
- Rows 1..gh*gw are filled by **bicubic** interpolation of the M×M grid into
  the gh×gw grid, with:
  - Catmull-Rom kernel, `a = -0.75` (PyTorch `_interpolate_acubic` default).
  - `sx = (gw + interp_offset) / M`, `sy = (gh + interp_offset) / M`
    (`interp_offset = 0.1` default).
  - Coord map `dst_pos = (out_idx + 0.5)/scale - 0.5`.
  - Border **replicate** (clamp).
- **Caching**: a `static std::map` keyed by
  `(pos_embed_ptr, gh, gw)` returns the same `std::vector<float>` on every
  forward at a given resolution (README highlights this as a ~10ms saving).
  Thread-safe via a mutex.

### 7.3 Camera-token injection (`alt_start`)

When `cfg.alt_start >= 0` (BASE: 12, GIANT: 24, etc.), at block
`i == alt_start` the token-0 (cls/cam) slot is **overwritten** with the
learned `vit.camera_token` row 0:

```cpp
// input: cam0 = first row of vit.camera_token (shape [embed])
rest = ggml_view_2d(x, embed, Ntok-1, ...)        // tokens 1..Ntok-1
x    = ggml_concat(ctx, cam_in, rest, dim=1)      // [embed, Ntok]
```

`vit.camera_token` has shape `[embed, 2]`; only row 0 is used in the
single-view path. The multi-view path uses both rows (ref + per-view).

### 7.4 RoPE (2-D) (`src/rope2d.cpp`)

Built **only when** `cfg.rope_start >= 0`. Two position sets over the
`Ntok = 1+N_patch` tokens (BASE: 865):
- `pos_local[t] = (row+1, col+1)` for patches, `(0,0)` for token 0.
- `pos_nodiff[t] = (1,1)` for patches, `(0,0)` for token 0.

For each token, per-axis frequencies:
```
half = head_dim / 2      # y-axis uses [0, half), x-axis uses [half, head_dim)
quart = half / 2
for j in [0, quart):
    invf = rope_freq ** (-2*j / half)        # rope_freq = 100.0
    cos[t, j]            = cos(y * invf);  cos[t, j + quart]    = same
    cos[t, half + j]     = cos(x * invf);  cos[t, half + j + quart] = same
    (likewise sin)
```
`rotate_half` of a half-H is `cat(-H[quart:], H[:quart])`.

Block `i` uses **global** positions (`pos_nodiff`) iff `i >= alt_start && i%2
== 1`, else `pos_local`. The cam-token block (`alt_start`) is always local.

### 7.5 Block loop (single view, `src/dino_backbone.cpp:144`)

```
local_x = x
for i in [0, cfg.depth):
    if alt_start>=0 and i==alt_start: x = concat(cam_in, x[:,1:])
    global  = alt_start>=0 and i>=alt_start and (i%2==1)
    use_rope = rope_start>=0 and i>=rope_start
    cos/sin input: global ? rt_nodiff : rt_local
    x = vit_block(x, load_block(i), heads, hd, eps, cos, sin)
    if not global: local_x = x
    for o in out_layers: if o==i: capture(local_x, x)
```

Captures are read back via `Backend::capture`, then host post-process (§7.1)
produces the final `feats` and `cam_tokens`.

### 7.6 Layer norms

Two different epsilons — do not confuse them:
- Block `norm1` / `norm2`: `eps = cfg.ln_eps` (= `1e-6`).
- Attention `q_norm` / `k_norm`: `eps = 1e-5` (torch `LayerNorm` default;
  see `QK_NORM_EPS` in `src/vit_block.cpp:11`).
- DPT head `head.norm`: `eps = 1e-5` (also torch default).

### 7.7 Multi-view (`forward_mv`, `src/dino_backbone.cpp:399`)

`forward_mv` stacks `S` views along the token axis for cross-view global
attention. Reference view is auto-selected (`saddle_balanced`) iff `S >= 3` at
layer `alt_start-1`; the function then reorders views (ref first) and restores
input order before returning. Feats/cam_tokens shape:
`feats[L=4][S][N_patch*2*embed]`, `cam_tokens[L=4][S][2*embed]`.

---

## 8. DPT depth head (`src/dpt_head.cpp`)

### 8.1 Inputs and outputs

Input: `feats[4]` each `[C, N_patch]` (ne0=channel fastest, ne1=token),
where `C = 2*embed` (cat_token) or `embed` (metric).

Output (graph): `[W, H, output_dim, 1]` logits (ggml column-major), where
`output_dim` is read from `head.scratch.out2b.weight->ne[3]` (2 for depth+conf,
1 for depth-only metric, 1 for DA2, 1 for mono+sky).

Host post-process (per `run()`, `dpt_head.cpp:289`):
```cpp
depth[i] = exp(logits[i]);
conf[i]  = exp(logits[HW + i]) + 1;     // when output_dim>=2
sky[i]   = max(0, sky_logits[i]);       // when sky head present
```

### 8.2 Graph topology (`build_depth_graph`, `dpt_head.cpp:71`)

Per stage `s ∈ {0,1,2,3}` with patch-grid `pw=W/patch, ph=H/patch` and
`oc[s] = cfg.head_out_channels[s]` (default `[96,192,384,768]`):

```
x = feats[s]                                           # [C, N]
if has_head_norm: x = layernorm(x, head.norm.{w,b}, eps=1e-5)
x = transpose(x)                                       # [N, C]
x = reshape_4d(x, pw, ph, C, 1)                        # [pw,ph,C,1]
x = conv2d(head.proj.{s}.{w,b}, k=1, p=0)              # [pw,ph,oc[s],1]
if head_pos_embed: x += 0.1 * uv_pos_embed(pw, ph, oc[s])  # cached (see §8.3)
# resize_layers:
#   s==0: conv_transpose2d(stride=4)  -> [4pw,4ph,oc[0]]
#   s==1: conv_transpose2d(stride=2)  -> [2pw,2ph,oc[1]]
#   s==2: identity
#   s==3: conv2d(stride=2, pad=1)     -> [pw/2,ph/2,oc[3]]      (only if pw,ph even)
l[s] = x
```

Then the pyramid fusion:
```
layer{i}_rn = conv2d(head.scratch.layer{i}_rn.weight, no bias, k=3, p=1)
              : maps l[i] -> head_features (128)
# sizes match the lateral-skip spatial sizes: rn3=(pw,ph), rn2=(2pw,2ph),
# rn1=(4pw,4ph). rn4 has no rc1.
out = refinenet4(l4_rn, size=(pw,ph))
out = refinenet3(out, l3_rn, size=(2pw,2ph))
out = refinenet2(out, l2_rn, size=(4pw,4ph))
out = refinenet1(out, l1_rn, scale_factor=2 -> (8pw,8ph))
out = output_conv1: conv2d(head.scratch.out1, k=3, p=1)  -> [.,.,feat_half=64]
# capture as `fused` here for debug
out = interp_bilinear_ac(out, W, H)                       # upsample to image
feat_map = out
if head_pos_embed: feat_map += 0.1 * uv_pos_embed(W, H, feat_half)
if sky head present:
    sk = relu(conv2d(scratch.sky_out2a, k=3,p=1))
    sk = conv2d(scratch.sky_out2b, k=1,p=0)               # [W,H,1,1]
out = relu(conv2d(scratch.out2a, k=3, p=1))               # [W,H,32,1]
out = conv2d(scratch.out2b, k=1, p=0)                     # [W,H,output_dim,1]  = logits
```

### 8.3 UV positional embedding cache (`add_uv_input`, `dpt_head.cpp:18`)

- The UV embedding depends only on `(W,H,C,aspect,ratio)` so it's
  **memoized** in a `static std::map`. Cached buffer is *borrowed* into the
  graph (`add_graph_input_nd_borrow`) — no per-forward memcpy.
- `aspect = W/H`, `ratio = 0.1` (the UV scale baked into DA3).
- Channel layout converted to `[C,H,W]` C-major before upload.
- README cites this cache as ~90 ms / forward saved.

### 8.4 Channels-last LayerNorm (`layernorm_channels_last`, line 52)

For the aux-ray head's `out2_aux_ln`:
```
permute [W,H,C,N] -> [C,W,H,N]
ggml_norm(eps)             # over ne0 = C
mul by weight[C], add bias[C]
permute back
```

### 8.5 Aux ray head (`rays`, line 313, graph in `build_depth_graph` lines 195–240)

A fully independent pyramid sharing only `l{i}_rn` with the main path. Only
the finest level is used. Final logits: `[W, H, 7, 1]` (6 ray channels + 1
confidence). Aux resolution is `8pw × 8ph`.

Host post:
```cpp
ray_out    = identity(aux[..., 0..5])
ray_conf   = exp(aux[..., 6]) + 1
```

### 8.6 Tensor name suffix catalogue (DPT head)

Every name below is **prefixed `head.`** (or `m_head.` when nested-metric
GGUF). `{s}` ∈ `{0,1,2,3}`, `{i}` ∈ `{1,2,3,4}`, `{j}` ∈ `{1,2}`.

| Tensor | ggml shape (ne) |
|--------|-----------------|
| `head.norm.weight`, `head.norm.bias` | `[C]` (absent on metric) |
| `head.proj.{s}.weight`, `head.proj.{s}.bias` | conv 1×1, `[1,1,oc[s],C]` |
| `head.resize.0.weight`, `head.resize.0.bias` | transposed conv stride 4 |
| `head.resize.1.weight`, `head.resize.1.bias` | transposed conv stride 2 |
| `head.resize.3.weight`, `head.resize.3.bias` | conv stride 2, pad 1 |
| `head.scratch.layer{i}_rn.weight` | conv 3×3 pad 1, no bias |
| `head.scratch.rn{i}.rc{j}.c{1,2}.{weight,bias}` | 3×3 pad 1 |
| `head.scratch.rn{i}.out.{weight,bias}` | 3×3 pad 1 (rn4 has no rc1) |
| `head.scratch.out1.{weight,bias}` | 3×3 pad 1 |
| `head.scratch.out2a.{weight,bias}` | 3×3 pad 1 |
| `head.scratch.out2b.{weight,bias}` | 1×1 (out channels = output_dim) |
| `head.scratch.sky_out2a.{weight,bias}` | 3×3 pad 1 (mono/metric only) |
| `head.scratch.sky_out2b.{weight,bias}` | 1×1 → 1 channel |
| Aux (`--with-aux` only): `head.scratch.rn{i}_aux.rc{j}.c{1,2}.{w,b}`, `head.scratch.rn{i}_aux.out.{w,b}`, `head.scratch.out1_aux.{0..4}.{w,b}`, `head.scratch.out2a_aux.{w,b}`, `head.scratch.out2b_aux.{w,b}`, `head.scratch.out2_aux_ln.{w,b}` | various |

---

## 9. Camera pose MLP (`src/cam_pose.cpp`)

### 9.1 Input / output

- Input: `cam_token` = `cam_tokens[3]` from the backbone, a flat
  `[D]` vector where `D = 2*embed` (1536 for BASE, 3072 for GIANT).
- Output:
  - `pose_enc[9]` = `[Tx, Ty, Tz, qi, qj, qk, qr, fov_h, fov_w]`
    (quaternion **XYZW**, scalar-last).
  - `extrinsics[12]` (3×4 row-major) = `affine_inverse(c2w) = [R^T | -R^T·T]`.
  - `intrinsics[9]` (3×3 row-major) = `[[fx,0,cx],[0,fy,cy],[0,0,1]]` with
    principal point at the image center and focal length derived from FoV:
    ```cpp
    fy = (H / 2) / max(tan(fov_h / 2), 1e-6);
    fx = (W / 2) / max(tan(fov_w / 2), 1e-6);
    cx = W / 2; cy = H / 2;
    ```

### 9.2 Graph (`cam_pose.cpp:21`)

```
feat = relu(linear(cam.bb0, D->D))(cam_token)
feat = relu(linear(cam.bb2, D->D))(feat)
th   = linear(cam.fc_t,   D->3)             # no activation
qh   = linear(cam.fc_q,   D->4)             # no activation
fvh  = relu(linear(cam.fc_fov, D->2))
pe   = concat(th, qh, fvh, dim=0)           # [9]
```

### 9.3 Quaternion → rotation → extrinsics (host math, lines 41–81)

Standard Hamilton-product quaternion-to-matrix with `s = 2 / |q|²`. Intrinsics
as above. The world-to-camera transform is `affine_inverse([R | T])`.

### 9.4 Tensor names (camera head)

All prefixed `cam.`:

| Tensor | Shape |
|--------|-------|
| `cam.bb0.weight`, `cam.bb0.bias` | Linear D→D |
| `cam.bb2.weight`, `cam.bb2.bias` | Linear D→D |
| `cam.fc_t.weight`, `cam.fc_t.bias` | Linear D→3 |
| `cam.fc_q.weight`, `cam.fc_q.bias` | Linear D→4 |
| `cam.fc_fov.weight`, `cam.fc_fov.bias` | Linear D→2 |

---

## 10. Quantization (`src/quantize.cpp`)

### 10.1 Supported types

```cpp
f16, q8_0, q6_k, q5_k, q4_k
```

### 10.2 What gets quantized

Only the **2-D matmul weights** consumed via `ggml_mul_mat`:

```
^vit\.blk\.[0-9]+\.(attn_qkv|attn_proj|mlp_fc1|mlp_fc2|mlp_w12|mlp_w3)\.weight$
^cam\.(bb0|bb2|fc_t|fc_q|fc_fov)\.weight$
```

Everything else (conv kernels, norms, biases, `vit.pos_embed`,
`vit.cls_token`, `vit.camera_token`, all `head.*` and `gs.*` weights) is
**copied through unchanged as f32**. This is a hard rule — quantizing conv
kernels would break the loader / conv / host-read paths.

### 10.3 Load-time vs inference-time

- **Load time**: no dequant. Quantized tensors stay in their packed ggml type
  in the `tensors_` map. The CPU path uses zero-copy pointers into the GGUF
  mmap; the GPU path uploads the packed bytes via
  `ggml_backend_tensor_set`.
- **Inference time**: ggml's `mul_mat` kernels natively consume the packed
  types (k-quants dispatch to dedicated SIMD kernels). The host never sees
  the dequantized weights.

### 10.4 Round-tripping through f32

When re-quantizing an already-quantized GGUF, `dequantize_to_f32` first
dequantizes via `ggml_get_type_traits(t->type)->to_float` (works for any
registered type) before calling `ggml_quantize_chunk` with the new target
type. F16 row-wise dequant uses the dedicated `ggml_fp16_to_fp32_row`.

---

## 11. Compute mode (`src/compute_mode.hpp`)

Two non-member functions:

```cpp
void set_gpu_mode(bool on);   // called once by Engine::load
bool gpu_mode();              // read by every graph builder
```

When `true`, the graph builders (attention, conv, etc.) route to standard ggml
ops with CUDA kernels (`ggml_conv_2d` direct, manual `mul_mat`/`softmax`)
instead of the CPU-tuned custom paths (`winograd` custom op, F32 flash-attn).
The CPU path is byte-identical to before the flag was added.

The port should expose the same switch (e.g. a `ComputeMode::Cpu` /
`ComputeMode::Gpu` enum) so CPU parity is preserved while GPU runs can pick
faster fused kernels.

---

## 12. Tensor naming convention — complete map

All weight tensor names in the GGUF. Names with `vit.` / `head.` / `cam.` /
`gs.` belong to the anyview (DA3) branch; the nested-metric branch duplicates
every `vit.` → `m_vit.` and every `head.` → `m_head.` (the loader aliases them
back to the unprefixed names so graph code is unchanged).

### 12.1 Patch embed / cls / pos-embed / camera token / final norm

| Name | Source (HF) | ggml ne |
|------|-------------|---------|
| `vit.patch_embed.weight` | `patch_embed.proj.weight` | `[1, 1, embed, 3]` conv kernel patch×patch |
| `vit.patch_embed.bias` | `patch_embed.proj.bias` | `[embed]` |
| `vit.cls_token` | `cls_token` | `[embed, 1]` |
| `vit.camera_token` | `camera_token` | `[embed, 2]` (row 0 = single-view) |
| `vit.pos_embed` | `pos_embed` | `[embed, 1 + M*M]` (host-read) |
| `vit.norm.weight`, `vit.norm.bias` | `norm.{w,b}` | `[embed]` (host-read) |

### 12.2 Transformer block `i` (`vit.blk.{i}.*`)

| gguf name | HF name |
|-----------|---------|
| `vit.blk.{i}.norm1.{weight,bias}` | `blocks.{i}.norm1.{w,b}` |
| `vit.blk.{i}.norm2.{weight,bias}` | `blocks.{i}.norm2.{w,b}` |
| `vit.blk.{i}.ls1` | `blocks.{i}.ls1.gamma` (single 1-D tensor) |
| `vit.blk.{i}.ls2` | `blocks.{i}.ls2.gamma` |
| `vit.blk.{i}.attn_qkv.{weight,bias}` | `blocks.{i}.attn.qkv.{w,b}` |
| `vit.blk.{i}.attn_proj.{weight,bias}` | `blocks.{i}.attn.proj.{w,b}` |
| `vit.blk.{i}.attn_qnorm.{weight,bias}` | `blocks.{i}.attn.q_norm.{w,b}` |
| `vit.blk.{i}.attn_knorm.{weight,bias}` | `blocks.{i}.attn.k_norm.{w,b}` |
| `vit.blk.{i}.mlp_fc1.{weight,bias}` | `blocks.{i}.mlp.fc1.{w,b}` (FFN type `"mlp"`) |
| `vit.blk.{i}.mlp_fc2.{weight,bias}` | `blocks.{i}.mlp.fc2.{w,b}` (FFN type `"mlp"`) |
| `vit.blk.{i}.mlp_w12.{weight,bias}` | `blocks.{i}.mlp.w12.{w,b}` (FFN type `"swiglu"`) |
| `vit.blk.{i}.mlp_w3.{weight,bias}` | `blocks.{i}.mlp.w3.{w,b}` (FFN type `"swiglu"`) |

### 12.3 DPT head (`head.*`)

See §8.6.

### 12.4 Camera pose (`cam.*`)

See §9.4.

### 12.5 GSDPT (`gs.*`, giant only)

Same pattern as DPT head but under the `gs.` prefix; plus an `images_merger`
sequence (indices 0/2/4 → renamed `gs.merger.0/1/2`).

| Pattern | Notes |
|---------|-------|
| `gs.merger.{0,1,2}.{weight,bias}` | Conv 3×3 pad 1 |
| `gs.proj.{s}.{weight,bias}` | 1×1 projection |
| `gs.resize.{0,1,3}.{weight,bias}` | resize_layers |
| `gs.scratch.layer{i}_rn.weight` | 3×3 pad 1, no bias |
| `gs.scratch.rn{i}.rc{j}.c{1,2}.{weight,bias}` | refinenet convs |
| `gs.scratch.rn{i}.out.{weight,bias}` | refinenet out_conv |
| `gs.scratch.out1.{weight,bias}` | output_conv1 |
| `gs.scratch.out2a.{weight,bias}` | first conv of output_conv2 |
| `gs.scratch.out2b.{weight,bias}` | last conv of output_conv2 |

`gs_adapter` carries **no learned weights** — only KV (`gs.sh_degree`,
`gs.scale_min/max`, `gs.pred_offset_depth`, `gs.pred_offset_xy`,
`gs.pred_color`).

---

## 13. Engine entry sequence (cheat-sheet)

For the production **single-image depth+pose** path
(`Engine::depth_pose_native`):

```text
1. load_image_rgb(path)                            -> Image {w,h, rgb[w*h*3] uint8}

2. preprocess_real(img, cfg, &pp)
      resize_cubic / resize_area   (cv2-faithful)
      -> Preprocessed { H, W, chw[3*H*W] f32 C-major, scale_w, scale_h }

3. DinoBackbone::forward(pp.chw, H, W, &feats, &cam_tokens)
      patch_embed conv2d (kernel=patch, stride=patch)
      concat(cls_token)
      + interp_pos_embed(gh, gw)                   <- bicubic, cached
      cam-token inject at block alt_start          <- if alt_start>=0
      2D RoPE from block rope_start                <- if rope_start>=0
      block loop (alt global/local routing)
      host post: feats[o]   = cat[local_x, vit.norm(x)], token-0 stripped
                  cam_tok[o]= cat[local_x[t0], x[t0]] raw
      -> feats[4]      each [N_patch * 2*embed]
         cam_tokens[4] each [2*embed]

4. DptHead::depth(feats, H, W, &depth, &conf)
      per-stage proj/resize/UV-pos-embed (cached)
      refinenet pyramid 4->3->2->1
      out1, bilinear upsample to (H,W), UV-pos-embed
      relu+out2a, out2b -> logits [W,H,output_dim,1]
      host post: depth=exp(logits[...,0]); conf=exp(logits[...,1])+1
      -> depth[H*W], conf[H*W] (row-major)

5. CamPose::pose(cam_tokens[3], H, W, _, &ext, &intr)
      bb0->relu->bb2->relu
      fc_t (3), fc_q (4), relu(fc_fov) (2) -> pe[9]
      host: quat->R, K from fov, ext = inv([R|T])
      -> ext[12] row-major 3x4, intr[9] row-major 3x3
```

Optional add-ons (same backbone pass):

- **Ray pose** (`--ray-pose`): instead of step 5, run `DptHead::rays` →
  `[6·HWa] ray + [HWa] conf` at 8×patch resolution; solve via
  `solve_ray_pose` (seeded deterministic RANSAC).
- **GS reconstruct**: also run `GsHead::raw_gaussians` on the same feats and
  pass everything through `GsAdapter::build` to world-space Gaussians.
- **Multi-view**: `forward_mv` runs the block loop over `S` views with
  cross-view global attention, then per-view DPT + cam.
- **Nested metric**: a second `(metric_ml_, metric_be_)` runs
  `forward + depth_sky` on the **same** preprocessed input, then
  `NestedAligner::align` merges the two depth maps.

---

## 14. Gotchas the port must reproduce bit-exactly

1. **CV-faithful resize**. `resize_cubic` must use `a=-0.75` and the
   `(dst+0.5)*scale - 0.5` coordinate map; `resize_area` must replicate
   `cv2.INTER_AREA`'s `computeResizeAreaTab` including its 1e-3 epsilon.
   Python's `round` is round-half-to-even — do **not** use Rust's `as i32`
   truncation.
2. **Two pos-embed caches**. Both the backbone `interp_pos_embed` (bicubic
   on `vit.pos_embed`) and the DPT `add_uv_input` must be cached by their
   full geometry key or the per-forward overhead dominates.
3. **Two RoPE position sets** (`pos_local` vs `pos_nodiff`); block-level
   routing by `(alt_start, i%2)`.
4. **Camera-token injection exactly at block `alt_start`**, not before,
   not after.
5. **Two different LayerNorm epsilons**: `1e-6` for block norms,
   `1e-5` for attention `q_norm`/`k_norm` and the DPT `head.norm`.
6. **`output_dim` derived from tensor shape**, not from KV.
7. **Host post-activations**:
   - depth = `exp(logit)` (DA3)
   - conf  = `exp(logit) + 1` (DA3)
   - sky   = `relu(logit)` (mono / metric)
   - DA2 metric = `sigmoid(logit) * max_depth`
   - DA2 relative = `relu(logit)`
8. **Quantize-only-matmul-weights** rule (§10.2) — conv kernels must stay
   f32.
9. **Channels-last LayerNorm** in the aux ray head: permute to bring C to
   ne0, normalize, permute back (do not implement as a per-pixel
   operation — the reference is a single `nn.LayerNorm(C)`).
10. **Nested aliasing**: `m_vit.X` / `m_head.X` GGUF tensors must be
    reachable under both their original and aliased names.
11. **Host-read tensors stay on CPU** when offloading to GPU
    (`vit.pos_embed`, `vit.camera_token`, `vit.norm.{w,b}`).
12. **Quaternion convention** XYZW (scalar-last); intrinsics' principal point
    is exactly the image center (`cx = W/2`, `cy = H/2`).
13. **Patch grid** is always `gh = H/patch, gw = W/patch` with `patch=14`;
    both H and W are guaranteed multiples of 14 by `preprocess_real`.

---

## 15. Open items to confirm during the port

- The `gs_head` / `gs_adapter` / `ray_pose` / `nested` modules are only
  sketched here at the engine-call level. If the Rust port needs to support
  GIANT reconstruction, ray-pose, or nested metric depth, read those modules
  in full — their internal numerical details (RANSAC sampling seeds,
  Gaussian activation math, sky-fill-before-align) are not reproduced here.
- The `num_register` KV is read but the current backbone code does **not**
  prepend register tokens (DA3-BASE has `num_register=0`). If a future model
  uses register tokens, the token-concat / RoPE / cam-token index logic will
  need updating.
- `DA_DEVICE`, `DA_FUSED`, `DA_PROFILE`, `DA_ATTN`, `DA_ATTN_F16` env vars
  are read by the C++ at runtime. Decide which the Rust port wants to
  preserve.
