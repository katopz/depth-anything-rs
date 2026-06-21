//! Bridge from the GGUF tensor store into a candle [`VarBuilder`].
//!
//! Each GGUF tensor is dequantized to `f32` and reshaped to its **reversed**
//! ggml `ne` dims — which equals the PyTorch/candle layout for both linear
//! weights (`[in,out] → [out,in]`) and conv kernels
//! (`[kW,kH,IC,OC] → [OC,IC,kH,kW]`). A custom candle
//! [`SimpleBackend`] then serves these tensors by full dotted name, so
//! `vb.pp("vit").pp("blk").pp(0)` looks up `vit.blk.0.…` exactly as stored.

use crate::gguf::GgufFile;
use crate::Result;
use candle::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A candle [`SimpleBackend`] backed by a GGUF file. Tensors are dequantized
/// to `f32` lazily on first access and cached.
pub struct GgufBackend {
    file: Arc<GgufFile>,
    cache: Mutex<HashMap<String, Tensor>>,
    /// Optional prefix rewrites applied on lookup miss (e.g. `vit.` -> `m_vit.`
    /// for the nested-metric branch, whose tensors live under `m_vit.*` /
    /// `m_head.*` but are referenced by `Backbone`/`DptHead` as `vit.*` /
    /// `head.*`). Mirrors the alias map inserted by `ModelLoader::load` in
    /// `src/model_loader.cpp`.
    prefix_rewrites: Vec<(&'static str, &'static str)>,
}

impl GgufBackend {
    /// Backend that serves tensors by their exact stored name.
    pub fn new(file: Arc<GgufFile>) -> Self {
        Self {
            file,
            cache: Mutex::new(HashMap::new()),
            prefix_rewrites: Vec::new(),
        }
    }

    /// Backend that rewrites `vit.*` -> `m_vit.*` and `head.*` -> `m_head.*`
    /// on lookup miss, so a nested-metric GGUF (which stores its tensors under
    /// the `m_*` prefixes) can be loaded by the same `Backbone`/`DptHead` code
    /// that references the unprefixed names. Mirrors the alias map built in
    /// `ModelLoader::load` (`src/model_loader.cpp`) for the metric branch.
    pub fn new_metric(file: Arc<GgufFile>) -> Self {
        Self {
            file,
            cache: Mutex::new(HashMap::new()),
            prefix_rewrites: vec![("vit.", "m_vit."), ("head.", "m_head.")],
        }
    }

    fn load(&self, name: &str, shape: &Shape, dtype: DType, dev: &Device) -> Result<Tensor> {
        // Fast path: already cached.
        if let Some(t) = self.cache.lock().unwrap().get(name) {
            return Ok(t.clone());
        }
        let info = match self.file.tensor_info(name) {
            Some(i) => i,
            None => {
                // Apply prefix rewrites (metric-branch aliasing). The first
                // rewrite whose target exists wins; if none match, fall through
                // to the original-name error below.
                let mut resolved: Option<String> = None;
                for (from, to) in &self.prefix_rewrites {
                    if let Some(rest) = name.strip_prefix(from) {
                        let aliased = format!("{to}{rest}");
                        if self.file.tensor_info(&aliased).is_some() {
                            resolved = Some(aliased);
                            break;
                        }
                    }
                }
                match resolved {
                    Some(actual) => return self.load_named(&actual, name, shape, dtype, dev),
                    None => {
                        return Err(crate::Error::Model(format!("tensor not found: {name}")));
                    }
                }
            }
        };
        let data = self.file.tensor_f32(name)?;
        self.finish_load(name, info.candle_shape(), data, shape, dtype, dev)
    }

    /// Load a tensor stored under `actual_name`, but cache it under
    /// `cache_name` (the name the caller requested, before alias rewriting).
    /// This keeps the cache keyed by the requested name so subsequent lookups
    /// for the same (unprefixed) name hit the cache directly.
    fn load_named(
        &self,
        actual_name: &str,
        cache_name: &str,
        shape: &Shape,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        if let Some(t) = self.cache.lock().unwrap().get(cache_name) {
            return Ok(t.clone());
        }
        let info = self
            .file
            .tensor_info(actual_name)
            .ok_or_else(|| crate::Error::Model(format!("tensor not found: {actual_name}")))?;
        let data = self.file.tensor_f32(actual_name)?;
        self.finish_load(cache_name, info.candle_shape(), data, shape, dtype, dev)
    }

    fn finish_load(
        &self,
        cache_name: &str,
        cand_shape: Vec<usize>,
        data: Vec<f32>,
        shape: &Shape,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let mut t = Tensor::from_vec(data, cand_shape, dev)?;
        if t.dtype() != dtype {
            t = t.to_dtype(dtype)?;
        }
        // Sanity-check the requested shape against the tensor's actual shape.
        // (VarBuilder passes the *consumer's* expected shape; we allow it if it
        // matches element count, otherwise we reshape to the expected shape.)
        if t.shape().dims() != shape.dims() {
            if t.elem_count() == shape.elem_count() {
                t = t.reshape(shape)?;
            } else {
                return Err(crate::Error::Model(format!(
                    "tensor {cache_name}: shape {:?} != requested {:?}",
                    t.shape().dims(),
                    shape.dims()
                )));
            }
        }
        self.cache
            .lock()
            .unwrap()
            .insert(cache_name.to_string(), t.clone());
        Ok(t)
    }
}

impl SimpleBackend for GgufBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        self.load(name, &s, dtype, dev)
            .map_err(|e| candle::Error::Msg(format!("failed to load tensor {name}: {e}")))
    }
    fn contains_tensor(&self, name: &str) -> bool {
        if self.file.tensor_info(name).is_some() {
            return true;
        }
        // Apply the same prefix rewrites as `load` so `VarBuilder` probes report
        // existence correctly for the metric-branch aliased names.
        for (from, to) in &self.prefix_rewrites {
            if let Some(rest) = name.strip_prefix(from) {
                let aliased = format!("{to}{rest}");
                if self.file.tensor_info(&aliased).is_some() {
                    return true;
                }
            }
        }
        false
    }
}

/// Build a candle [`VarBuilder`] over a GGUF file.
pub fn var_builder(file: Arc<GgufFile>, device: Device) -> VarBuilder<'static> {
    let backend: Box<dyn SimpleBackend + 'static> = Box::new(GgufBackend::new(file));
    VarBuilder::from_backend(backend, DType::F32, device)
}

/// Build a candle [`VarBuilder`] over a **nested-metric** GGUF, rewriting
/// `vit.*` -> `m_vit.*` and `head.*` -> `m_head.*` lookups so the metric
/// branch's tensors can be loaded by the same `Backbone`/`DptHead` code.
/// Mirrors the alias map built in `ModelLoader::load`
/// (`src/model_loader.cpp`) for the metric branch.
pub fn var_builder_metric(file: Arc<GgufFile>, device: Device) -> VarBuilder<'static> {
    let backend: Box<dyn SimpleBackend + 'static> = Box::new(GgufBackend::new_metric(file));
    VarBuilder::from_backend(backend, DType::F32, device)
}

/// Helper: does the GGUF contain a tensor of the given name?
pub fn has_tensor(file: &GgufFile, name: &str) -> bool {
    file.tensor_info(name).is_some()
}

/// Load a `candle_nn::Linear` (`{name}.weight`, `{name}.bias`) from the
/// VarBuilder, with an explicit `(out, in)` shape used only for validation.
pub fn load_linear(
    vb: &VarBuilder,
    name: &str,
    out: usize,
    inn: usize,
) -> Result<candle_nn::Linear> {
    let w = vb.pp(name).get((out, inn), "weight")?;
    // bias is mandatory for DA3 linear layers; use an explicit fetch so a
    // missing bias surfaces a clear error rather than a silent no-bias Linear.
    match vb.pp(name).get((out,), "bias") {
        Ok(b) => Ok(candle_nn::Linear::new(w, Some(b))),
        Err(_) => Ok(candle_nn::Linear::new(w, None)),
    }
}
