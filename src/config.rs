//! Model configuration, parsed from the GGUF metadata KV.

use crate::gguf::{GgufFile, MetaArray, MetaValue};
use crate::Result;

/// Which Depth Anything architecture the GGUF carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// Depth Anything 3 (default).
    Da3,
    /// Depth Anything V2 (depth-only, no pose/confidence).
    Da2,
}

impl Arch {
    pub fn parse(s: &str) -> Self {
        if s == "depthanything2" {
            Arch::Da2
        } else {
            Arch::Da3
        }
    }
}

/// Feed-forward network type used by the ViT blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnType {
    /// GELU-erf MLP: fc1 → GELU → fc2.
    Mlp,
    /// SwiGLU: silu(w12_a) * w12_b → w3.
    Swiglu,
}

impl FfnType {
    pub fn parse(s: &str) -> Self {
        if s == "swiglu" {
            FfnType::Swiglu
        } else {
            FfnType::Mlp
        }
    }
}

/// How the input image is resized to the processing resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// Longest side is bounded above by `img_resize_target` (DA3 default: 504).
    UpperBound,
    /// Shortest side is bounded below by `img_resize_target` (DA2 default: 518).
    LowerBound,
}

impl ResizeMode {
    pub fn parse(s: &str) -> Self {
        if s == "lower_bound" {
            ResizeMode::LowerBound
        } else {
            ResizeMode::UpperBound
        }
    }
}

/// The full DA3 model configuration, mirroring the C++ `Config` struct.
///
/// Every field comes from a GGUF metadata KV (see `include/da_gguf_keys.h`).
/// The defaults match the C++ engine's defaults so a GGUF that omits optional
/// keys behaves identically.
#[derive(Debug, Clone)]
pub struct Config {
    pub patch_size: u32,
    pub embed_dim: u32,
    pub depth: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub mlp_hidden: u32,
    /// Loaded for compatibility but **not consulted** by either engine — DA3
    /// uses a `camera_token` injected at `alt_start` instead of register tokens.
    pub num_register: u32,
    /// Source grid size `M` of the stored positional embedding (`M x M` patches).
    pub pos_embed_grid: u32,
    /// Block index at which the camera token is injected into the CLS slot and
    /// odd-indexed blocks switch to "global" (nodiff) RoPE. `-1` disables.
    pub alt_start: i32,
    /// First block index that applies RoPE. `-1` disables RoPE entirely.
    pub rope_start: i32,
    /// Stored for compatibility but **not consulted**: q/k LayerNorm is applied
    /// whenever the `attn_{q,k}norm` weight tensors are present.
    pub qknorm_start: i32,
    pub init_values: f32,
    /// RoPE base frequency (theta).
    pub rope_freq: f32,
    /// Epsilon for the block LayerNorms and the final backbone norm.
    pub ln_eps: f32,
    /// Scale factor for the positional-embedding bicubic grid mapping.
    pub interp_offset: f32,
    /// If true, backbone out-layer features are `cat[local_x, norm(x)]` over
    /// the channel dim (doubling it). DA3-BASE/giant: true. Metric ViT-L: false.
    pub cat_token: bool,
    pub qkv_bias: bool,
    pub interp_antialias: bool,
    /// If true, the DPT decoder adds the (cached) UV positional embedding to
    /// its feature maps. True for DA3 base / DA2 relative; false for metric DA3.
    pub head_pos_embed: bool,
    pub ffn_type: FfnType,
    pub head_features: u32,
    /// Which transformer block indices are exposed as decoder inputs
    /// (DA3 default: `[5, 7, 9, 11]`).
    pub out_layers: Vec<i32>,
    /// Per-stage projector output channel counts (DA3 default: `[96,192,384,768]`).
    pub head_out_channels: Vec<i32>,
    pub img_mean: Vec<f32>,
    pub img_std: Vec<f32>,
    /// Longest- (upper_bound) or shortest-side (lower_bound) target resolution.
    pub img_resize_target: u32,
    pub img_resize_mode: ResizeMode,
    pub checkpoint_name: String,
    /// DA2 metric depth scale in metres (0 = relative-depth model).
    pub head_max_depth: f32,
    pub arch: Arch,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            patch_size: 14,
            embed_dim: 0,
            depth: 0,
            num_heads: 0,
            head_dim: 0,
            mlp_hidden: 0,
            num_register: 0,
            pos_embed_grid: 0,
            alt_start: -1,
            rope_start: -1,
            qknorm_start: -1,
            init_values: 0.0,
            rope_freq: 100.0,
            ln_eps: 1e-6,
            interp_offset: 0.1,
            cat_token: true,
            qkv_bias: true,
            interp_antialias: false,
            head_pos_embed: true,
            ffn_type: FfnType::Mlp,
            head_features: 0,
            out_layers: Vec::new(),
            head_out_channels: Vec::new(),
            img_mean: Vec::new(),
            img_std: Vec::new(),
            img_resize_target: 504,
            img_resize_mode: ResizeMode::UpperBound,
            checkpoint_name: String::new(),
            head_max_depth: 0.0,
            arch: Arch::Da3,
        }
    }
}

impl Config {
    /// Half of the head dimension — the per-axis feature width used by RoPE.
    pub fn rope_half(&self) -> u32 {
        self.head_dim / 2
    }

    /// Whether this config describes a metric-depth model (DA2 metric, or DA3
    /// metric with a sky head). Drives the final depth activation choice.
    pub fn is_metric(&self) -> bool {
        self.head_max_depth > 0.0
    }

    /// The DPT head's intermediate feature width. Defaults to `head_features`
    /// when set, else `feat_half = 64`.
    pub fn feat_half(&self) -> u32 {
        if self.head_features > 0 {
            self.head_features / 2
        } else {
            64
        }
    }

    /// The four per-stage projector output channel counts, with the DA3 default
    /// fallback `[96, 192, 384, 768]`.
    pub fn head_out_channels_arr(&self) -> [u32; 4] {
        let d = &self.head_out_channels;
        [
            d.first().copied().unwrap_or(96) as u32,
            d.get(1).copied().unwrap_or(192) as u32,
            d.get(2).copied().unwrap_or(384) as u32,
            d.get(3).copied().unwrap_or(768) as u32,
        ]
    }

    /// Parse the configuration from a loaded [`GgufFile`]'s metadata KV.
    ///
    /// Reads the standard `depthanything3.vit.*` / `depthanything3.head.*`
    /// keys. For the nested-metric branch's separate GGUF (whose structural
    /// keys live under `depthanything3.m_vit.*` / `depthanything3.m_head.*`),
    /// use [`Self::from_gguf_metric`] instead.
    pub fn from_gguf(f: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_prefix(f, "vit", "head")
    }

    /// Parse the **metric** branch configuration from a nested-metric GGUF.
    ///
    /// The metric branch stores its ViT keys under `depthanything3.m_vit.*`
    /// and its head pos_embed under `depthanything3.m_head.pos_embed`; the
    /// remaining keys (img.*, checkpoint_name, arch, head.max_depth, …) are
    /// shared with the anyview branch and read from the same locations.
    ///
    /// Mirrors the `metric` branch of `ModelLoader::load` in
    /// `src/model_loader.cpp`.
    pub fn from_gguf_metric(f: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_prefix(f, "m_vit", "m_head")
    }

    fn from_gguf_with_prefix(f: &GgufFile, vit_pfx: &str, head_pfx: &str) -> Result<Self> {
        let mut c = Config::default();

        // head-prefixed key (only `pos_embed` lives under the head prefix;
        // the other head.* keys are shared and read from `depthanything3.head.*`).
        let head_pos_embed_key = format!("depthanything3.{head_pfx}.pos_embed");

        // Look up a ViT key under the chosen prefix: "depthanything3.<vit_pfx>.<field>".
        let vit = |field: &str| -> Option<&MetaValue> {
            f.meta(&format!("depthanything3.{vit_pfx}.{field}"))
        };
        // Look up a shared (non-prefixed) key.
        let get = |k: &str| -> Option<&MetaValue> { f.meta(k) };

        c.patch_size = get("depthanything3.patch_size")
            .and_then(as_u32)
            .unwrap_or(c.patch_size);
        c.embed_dim = vit("embed_dim")
            .and_then(as_u32)
            .ok_or_else(|| crate::Error::Gguf(format!("missing {}.embed_dim", vit_pfx)))?;
        c.depth = vit("depth")
            .and_then(as_u32)
            .ok_or_else(|| crate::Error::Gguf(format!("missing {}.depth", vit_pfx)))?;
        c.num_heads = vit("num_heads")
            .and_then(as_u32)
            .ok_or_else(|| crate::Error::Gguf(format!("missing {}.num_heads", vit_pfx)))?;
        c.head_dim = vit("head_dim")
            .and_then(as_u32)
            .ok_or_else(|| crate::Error::Gguf(format!("missing {}.head_dim", vit_pfx)))?;
        c.mlp_hidden = vit("mlp_hidden").and_then(as_u32).unwrap_or(0);
        c.num_register = vit("num_register_tokens").and_then(as_u32).unwrap_or(0);
        c.pos_embed_grid = vit("pos_embed_grid").and_then(as_u32).unwrap_or(0);
        c.alt_start = vit("alt_start").and_then(as_i32).unwrap_or(-1);
        c.rope_start = vit("rope_start").and_then(as_i32).unwrap_or(-1);
        c.qknorm_start = vit("qknorm_start").and_then(as_i32).unwrap_or(-1);
        c.init_values = vit("init_values").and_then(as_f32).unwrap_or(0.0);
        c.rope_freq = vit("rope_freq").and_then(as_f32).unwrap_or(100.0);
        c.ln_eps = vit("ln_eps").and_then(as_f32).unwrap_or(1e-6);
        c.interp_offset = vit("interpolate_offset").and_then(as_f32).unwrap_or(0.1);
        c.cat_token = vit("cat_token").and_then(as_bool).unwrap_or(true);
        c.qkv_bias = vit("qkv_bias").and_then(as_bool).unwrap_or(true);
        c.interp_antialias = vit("interpolate_antialias")
            .and_then(as_bool)
            .unwrap_or(false);
        // The metric branch omits `head.pos_embed`; default to false (DPT.__init__
        // default). The anyview branch reads `depthanything3.head.pos_embed`
        // (default true).
        c.head_pos_embed = f
            .meta(&head_pos_embed_key)
            .and_then(as_bool)
            .unwrap_or(vit_pfx == "vit");
        c.ffn_type = FfnType::parse(
            &vit("ffn_type")
                .and_then(as_str)
                .unwrap_or_else(|| "mlp".to_string()),
        );
        // head.features / head.out_channels are shared keys (no m_ prefix).
        c.head_features = get("depthanything3.head.features")
            .and_then(as_u32)
            .unwrap_or(0);
        c.out_layers = vit("out_layers").and_then(as_i32_vec).unwrap_or_default();
        c.head_out_channels = get("depthanything3.head.out_channels")
            .and_then(as_i32_vec)
            .unwrap_or_default();
        c.img_mean = get("depthanything3.img.mean")
            .and_then(as_f32_vec)
            .unwrap_or_default();
        c.img_std = get("depthanything3.img.std")
            .and_then(as_f32_vec)
            .unwrap_or_default();
        c.img_resize_target = get("depthanything3.img.resize_target")
            .and_then(as_u32)
            .unwrap_or(504);
        c.img_resize_mode = ResizeMode::parse(
            &get("depthanything3.img.resize_mode")
                .and_then(as_str)
                .unwrap_or_else(|| "upper_bound".to_string()),
        );
        c.checkpoint_name = get("depthanything3.checkpoint_name")
            .and_then(as_str)
            .unwrap_or_default();
        c.head_max_depth = get("depthanything3.head.max_depth")
            .and_then(as_f32)
            .unwrap_or(0.0);
        c.arch = Arch::parse(
            &get("depthanything3.arch")
                .and_then(as_str)
                .unwrap_or_else(|| "depthanything3".to_string()),
        );

        // Validate the head_dim/embed_dim/num_heads relationship.
        if c.num_heads.checked_mul(c.head_dim) != Some(c.embed_dim) {
            return Err(crate::Error::Model(format!(
                "inconsistent dims: num_heads({}) * head_dim({}) != embed_dim({})",
                c.num_heads, c.head_dim, c.embed_dim
            )));
        }

        Ok(c)
    }
}

// ---- metadata value extractors ----------------------------------------

fn as_u32(m: &MetaValue) -> Option<u32> {
    match m {
        MetaValue::U8(v) => Some(*v as u32),
        MetaValue::U16(v) => Some(*v as u32),
        MetaValue::U32(v) => Some(*v),
        MetaValue::I8(v) => Some(*v as u32),
        MetaValue::I16(v) => Some(*v as u32),
        MetaValue::I32(v) => Some(*v as u32),
        MetaValue::U64(v) => Some(*v as u32),
        MetaValue::I64(v) => Some(*v as u32),
        _ => None,
    }
}
fn as_i32(m: &MetaValue) -> Option<i32> {
    match m {
        MetaValue::I8(v) => Some(*v as i32),
        MetaValue::I16(v) => Some(*v as i32),
        MetaValue::I32(v) => Some(*v),
        MetaValue::U8(v) => Some(*v as i32),
        MetaValue::U16(v) => Some(*v as i32),
        MetaValue::U32(v) => Some(*v as i32),
        MetaValue::I64(v) => Some(*v as i32),
        MetaValue::U64(v) => Some(*v as i32),
        _ => None,
    }
}
fn as_f32(m: &MetaValue) -> Option<f32> {
    match m {
        MetaValue::F32(v) => Some(*v),
        MetaValue::F64(v) => Some(*v as f32),
        _ => None,
    }
}
fn as_bool(m: &MetaValue) -> Option<bool> {
    match m {
        MetaValue::Bool(v) => Some(*v),
        _ => None,
    }
}
fn as_str(m: &MetaValue) -> Option<String> {
    match m {
        MetaValue::String(s) => Some(s.clone()),
        _ => None,
    }
}
fn as_i32_vec(m: &MetaValue) -> Option<Vec<i32>> {
    match m {
        MetaValue::Array(MetaArray::I8(v)) => Some(v.iter().map(|x| *x as i32).collect()),
        MetaValue::Array(MetaArray::I32(v)) => Some(v.clone()),
        MetaValue::Array(MetaArray::U8(v)) => Some(v.iter().map(|x| *x as i32).collect()),
        MetaValue::Array(MetaArray::U32(v)) => Some(v.iter().map(|x| *x as i32).collect()),
        MetaValue::Array(MetaArray::I64(v)) => Some(v.iter().map(|x| *x as i32).collect()),
        MetaValue::Array(MetaArray::U64(v)) => Some(v.iter().map(|x| *x as i32).collect()),
        _ => None,
    }
}
fn as_f32_vec(m: &MetaValue) -> Option<Vec<f32>> {
    match m {
        MetaValue::Array(MetaArray::F32(v)) => Some(v.clone()),
        MetaValue::Array(MetaArray::F64(v)) => Some(v.iter().map(|x| *x as f32).collect()),
        _ => None,
    }
}
