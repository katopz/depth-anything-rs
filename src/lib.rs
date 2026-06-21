//! A pure-Rust inference engine for [Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3),
//! built on the [`candle`](https://github.com/huggingface/candle) tensor library.
//!
//! This crate is a port of the C++/ggml engine [`depth-anything.cpp`](https://github.com/mudler/depth-anything.cpp)
//! to Rust. It loads the same self-contained GGUF model files, reads the same
//! baked-in metadata, and runs the same forward pass (patch embed, bicubic
//! positional embedding, ViT backbone with 2D RoPE attention, DPT depth head,
//! and camera pose head).
//!
//! # Status
//!
//! This is a from-scratch port. The DA3-BASE depth + pose forward path is
//! implemented; see [`docs/RUST_PORT.md`] for the full status matrix. Outputs
//! are numerically faithful to the reference when the same (f32) weights are
//! loaded, but have **not** been bit-exact-verified against PyTorch reference
//! tensors in this repo (no reference fixtures are bundled).
//!
//! # Quick start
//!
//! ```no_run
//! use depth_anything::Engine;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let engine = Engine::load("models/depth-anything-base-f32.gguf", None)?;
//! let out = engine.depth_path("photo.jpg")?;
//! println!("depth: {}x{}", out.h, out.w);
//! # Ok(())
//! # }
//! ```

pub mod attention;
pub mod backbone;
pub mod cam_pose;
pub mod colmap_export;
pub mod config;
pub mod depth_export;
pub mod dpt_head;
pub mod engine;
pub mod gguf;
pub mod glb_export;
pub mod gs_adapter;
pub mod gs_head;
pub mod kquants;
pub mod linalg;
pub mod nested;
pub mod ply_export;
pub mod pos_embed;
pub mod preprocess;
pub mod ray_pose;
pub mod reconstruct;
pub mod rope2d;
pub mod uv_embed;
pub mod vit_block;
pub mod weights;

pub use backbone::select_reference_view_saddle;
pub use config::{Arch, Config, FfnType, ResizeMode};
pub use engine::{
    DepthOutput, Engine, ExportResult, ExportSpec, MultiViewOutput, PoseOutput, RayPoseDiag,
    RayPoseOutput, ReconstructOutput, Timings, ViewResult,
};
pub use gguf::{GgufDType, GgufFile};
pub use gs_adapter::{BuildError as GsBuildError, Gaussians, GsAdapter};
pub use nested::{AnyviewOut, MetricOut, NestedAligner, NestedOut};
pub use ply_export::write_gaussian_ply;

use candle::Device;

/// Error type for all fallible operations in this crate.
#[derive(Debug)]
pub enum Error {
    /// A candle tensor error.
    Candle(candle::Error),
    /// An I/O error (file not found, read failure, etc.).
    Io(std::io::Error),
    /// An image decode error.
    Image(image::ImageError),
    /// The GGUF file is malformed or unsupported.
    Gguf(String),
    /// A required tensor or metadata key was missing, or had an unexpected shape.
    Model(String),
    /// A requested code path is not yet implemented in the Rust port.
    Unimplemented(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Candle(e) => write!(f, "tensor error: {e}"),
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Image(e) => write!(f, "image error: {e}"),
            Error::Gguf(m) => write!(f, "gguf error: {m}"),
            Error::Model(m) => write!(f, "model error: {m}"),
            Error::Unimplemented(m) => write!(f, "not yet implemented: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<candle::Error> for Error {
    fn from(e: candle::Error) -> Self {
        Error::Candle(e)
    }
}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
impl From<image::ImageError> for Error {
    fn from(e: image::ImageError) -> Self {
        Error::Image(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Pick the candle [`Device`] to run on.
///
/// If the `cuda` feature is enabled and a CUDA device is available, it is used;
/// otherwise this falls back to CPU. The `DA_DEVICE=cpu` environment variable
/// always forces CPU (mirrors the C++ engine's `DA_DEVICE`).
pub fn default_device() -> Result<Device> {
    if std::env::var("DA_DEVICE").as_deref() == Ok("cpu") {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return Ok(d);
        }
    }
    Ok(Device::Cpu)
}
