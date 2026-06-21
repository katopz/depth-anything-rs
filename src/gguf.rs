//! A minimal, dependency-light GGUF v3 reader.
//!
//! This parses the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
//! container that `depth-anything.cpp` writes: the metadata KV table, the
//! tensor info table, and (via mmap) the tensor data section. Quantized
//! tensors (`q8_0`, `q4_k`, ...) are dequantized to `f32` at load time so the
//! rest of the engine always works in `f32`; this is simpler than plumbing
//! quantization through candle and is correct, at the cost of losing the
//! memory/throughput benefits of on-the-fly dequant (documented in
//! `docs/RUST_PORT.md`).
//!
//! The tensor data is **not** copied; we hold the mmap and slice into it.

use crate::Result;
use half::f16;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

/// GGUF metadata scalar value types (subset of the spec).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> std::result::Result<Self, ()> {
        Ok(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return Err(()),
        })
    }
}

/// A typed metadata array.
#[derive(Debug, Clone)]
pub enum MetaArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    String(Vec<String>),
}

/// A metadata value.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(MetaArray),
}

/// GGUF tensor element types (subset). Unknown / unsupported types are
/// rejected at load time.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDType {
    F32,
    F16,
    Q8_0,
    Q4_K,
    Q6_K,
    Q5_K,
    Q8_K,
}

impl GgufDType {
    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            14 => Self::Q6_K,
            13 => Self::Q5_K,
            15 => Self::Q8_K,
            _ => return None,
        })
    }

    /// Number of f32 elements each block expands to.
    fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q8_0 => 32,
            Self::Q4_K | Self::Q6_K | Self::Q5_K | Self::Q8_K => 256,
        }
    }
}

/// Info about one tensor: its name, shape (in ggml `ne` order, fastest dim
/// first), element dtype, and byte offset into the data section.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml `ne` dims: `ne[0]` is the fastest-varying (innermost) dimension.
    pub ne: Vec<u64>,
    pub dtype: GgufDType,
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of logical f32 elements.
    pub fn n_elems(&self) -> usize {
        self.ne.iter().product::<u64>() as usize
    }

    /// Shape in **PyTorch/candle order** — i.e. the reverse of ggml's `ne`,
    /// because ggml stores the fastest-varying dimension first.
    pub fn candle_shape(&self) -> Vec<usize> {
        self.ne.iter().rev().map(|d| *d as usize).collect()
    }
}

/// A loaded GGUF file: metadata + mmap of the tensor data section.
pub struct GgufFile {
    meta: HashMap<String, MetaValue>,
    tensors: HashMap<String, TensorInfo>,
    /// The file offset at which the (alignment-padded) tensor data begins.
    data_start: u64,
    mmap: memmap2::Mmap,
}

impl GgufFile {
    /// Open and parse the header of a GGUF file. Tensor data is mmaped but
    /// not decoded until [`tensor_f32`] is called.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let bytes: &[u8] = &mmap[..];

        let mut r = Reader::new(bytes);
        let magic = r.u32()?;
        if magic != GGUF_MAGIC {
            return Err(crate::Error::Gguf(format!("bad magic: 0x{magic:08x}")));
        }
        let version = r.u32()?;
        if version != 3 && version != 2 {
            return Err(crate::Error::Gguf(format!(
                "unsupported GGUF version {version} (need 2 or 3)"
            )));
        }
        let _tensor_count = r.u64()?;
        let meta_kv_count = r.u64()?;

        let mut meta = HashMap::with_capacity(meta_kv_count as usize);
        for _ in 0..meta_kv_count {
            let key = r.string()?;
            let value_type = GgufValueType::from_u32(r.u32()?)
                .map_err(|()| crate::Error::Gguf(format!("unknown value type for key {key}")))?;
            let value = read_value(&mut r, value_type)?;
            meta.insert(key, value);
        }

        // Tensor info table. (tensor_count was already read at the top of the header.)
        let tensor_count = _tensor_count;
        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()?;
            let mut ne = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                ne.push(r.u64()?);
            }
            let dtype_raw = r.u32()?;
            let dtype = GgufDType::from_u32(dtype_raw).ok_or_else(|| {
                crate::Error::Gguf(format!("tensor {name}: unsupported dtype {dtype_raw}"))
            })?;
            let offset = r.u64()?;
            let info = TensorInfo {
                name: name.clone(),
                ne,
                dtype,
                offset,
            };
            tensors.insert(name.clone(), info);
        }

        // Align the data section to `general.alignment` (default 32).
        let alignment = meta
            .get("general.alignment")
            .and_then(|v| match v {
                MetaValue::U32(x) => Some(*x as u64),
                MetaValue::U64(x) => Some(*x),
                _ => None,
            })
            .unwrap_or(32);
        let pos = r.pos as u64;
        let data_start = align_up(pos, alignment);

        Ok(Self {
            meta,
            tensors,
            data_start,
            mmap,
        })
    }

    /// Look up a metadata value by key.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.meta.get(key)
    }

    /// Iterate over all `(key, value)` metadata pairs.
    pub fn meta_iter(&self) -> impl Iterator<Item = (&String, &MetaValue)> {
        self.meta.iter()
    }

    /// Look up a tensor's info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Names of all tensors in the file.
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Decode a tensor to `f32`, dequantizing if necessary. The returned
    /// `Vec` has length `info.n_elems()` and is in ggml's natural order
    /// (fastest dim first); callers reshape it to the reversed candle shape.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| crate::Error::Model(format!("tensor not found: {name}")))?;
        let start = (self.data_start + info.offset) as usize;
        let block = info.dtype.block_size();
        let n_blocks = info.n_elems() / block;
        let raw = &self.mmap[start..];
        let mut out = vec![0.0f32; info.n_elems()];
        match info.dtype {
            GgufDType::F32 => {
                let bytes = take(raw, info.n_elems() * 4)?;
                let src = cast_bytes::<f32>(bytes);
                out.copy_from_slice(src);
            }
            GgufDType::F16 => {
                let bytes = take(raw, info.n_elems() * 2)?;
                let src = cast_bytes::<f16>(bytes);
                for (i, v) in src.iter().enumerate() {
                    out[i] = v.to_f32();
                }
            }
            GgufDType::Q8_0 => {
                // block: 1 x f16 scale, 32 x int8.  value = scale * q
                const Q8_BLOCK_BYTES: usize = 2 + 32;
                let bytes = take(raw, n_blocks * Q8_BLOCK_BYTES)?;
                for b in 0..n_blocks {
                    let base = b * Q8_BLOCK_BYTES;
                    let scale = f16::from_le_bytes([bytes[base], bytes[base + 1]]).to_f32();
                    let qs = &bytes[base + 2..base + Q8_BLOCK_BYTES];
                    let o = b * 32;
                    for i in 0..32 {
                        out[o + i] = scale * (qs[i] as i8 as f32);
                    }
                }
            }
            GgufDType::Q4_K | GgufDType::Q6_K | GgufDType::Q5_K | GgufDType::Q8_K => {
                // K-quants: 256 elements per super-block; the exact byte
                // layout (block_q{n}_K in ggml-common.h) differs per dtype.
                let bpqb = crate::kquants::block_bytes(info.dtype);
                let need = n_blocks * bpqb;
                let bytes = take(raw, need)?;
                match info.dtype {
                    GgufDType::Q4_K => crate::kquants::dequantize_q4_k(bytes, n_blocks, &mut out),
                    GgufDType::Q5_K => crate::kquants::dequantize_q5_k(bytes, n_blocks, &mut out),
                    GgufDType::Q6_K => crate::kquants::dequantize_q6_k(bytes, n_blocks, &mut out),
                    GgufDType::Q8_K => crate::kquants::dequantize_q8_k(bytes, n_blocks, &mut out),
                    _ => unreachable!(),
                }
            }
        }
        Ok(out)
    }
}

// ---- helpers ----------------------------------------------------------

fn read_value(r: &mut Reader, t: GgufValueType) -> Result<MetaValue> {
    Ok(match t {
        GgufValueType::U8 => MetaValue::U8(r.u8()?),
        GgufValueType::I8 => MetaValue::I8(r.i8()?),
        GgufValueType::U16 => MetaValue::U16(r.u16()?),
        GgufValueType::I16 => MetaValue::I16(r.i16()?),
        GgufValueType::U32 => MetaValue::U32(r.u32()?),
        GgufValueType::I32 => MetaValue::I32(r.i32()?),
        GgufValueType::U64 => MetaValue::U64(r.u64()?),
        GgufValueType::I64 => MetaValue::I64(r.i64()?),
        GgufValueType::F32 => MetaValue::F32(r.f32()?),
        GgufValueType::F64 => MetaValue::F64(r.f64()?),
        GgufValueType::Bool => MetaValue::Bool(r.bool()?),
        GgufValueType::String => MetaValue::String(r.string()?),
        GgufValueType::Array => {
            let inner = GgufValueType::from_u32(r.u32()?)
                .map_err(|()| crate::Error::Gguf("array with unknown inner type".into()))?;
            let count = r.u64()? as usize;
            // Specialize the common cases to keep the Vec strongly typed.
            let arr = match inner {
                GgufValueType::U8 => MetaArray::U8(read_array(r, count, |r| r.u8())?),
                GgufValueType::I8 => MetaArray::I8(read_array(r, count, |r| r.i8())?),
                GgufValueType::U16 => {
                    MetaArray::U32(read_array(r, count, |r| r.u16().map(|x| x as u32))?)
                }
                GgufValueType::I16 => {
                    MetaArray::I32(read_array(r, count, |r| r.i16().map(|x| x as i32))?)
                }
                GgufValueType::U32 => MetaArray::U32(read_array(r, count, |r| r.u32())?),
                GgufValueType::I32 => MetaArray::I32(read_array(r, count, |r| r.i32())?),
                GgufValueType::U64 => MetaArray::U64(read_array(r, count, |r| r.u64())?),
                GgufValueType::I64 => MetaArray::I64(read_array(r, count, |r| r.i64())?),
                GgufValueType::F32 => MetaArray::F32(read_array(r, count, |r| r.f32())?),
                GgufValueType::F64 => MetaArray::F64(read_array(r, count, |r| r.f64())?),
                GgufValueType::Bool => MetaArray::Bool(read_array(r, count, |r| r.bool())?),
                GgufValueType::String => MetaArray::String(read_array(r, count, |r| r.string())?),
                GgufValueType::Array => {
                    return Err(crate::Error::Gguf("nested arrays are not supported".into()));
                }
            };
            MetaValue::Array(arr)
        }
    })
}

fn read_array<T, F: FnMut(&mut Reader) -> Result<T>>(
    r: &mut Reader,
    count: usize,
    mut read_one: F,
) -> Result<Vec<T>> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(read_one(r)?);
    }
    Ok(v)
}

fn align_up(pos: u64, align: u64) -> u64 {
    if align == 0 {
        return pos;
    }
    let rem = pos % align;
    if rem == 0 {
        pos
    } else {
        pos + (align - rem)
    }
}

fn take(raw: &[u8], n: usize) -> Result<&[u8]> {
    raw.get(..n)
        .ok_or_else(|| crate::Error::Gguf(format!("tensor data truncated: needed {n} bytes")))
}

fn cast_bytes<T: bytemuck::Pod>(b: &[u8]) -> &[T] {
    bytemuck::cast_slice(b)
}

// We need `Pod` for `cast_slice`. Rather than add bytemuck as a dependency,
// re-implement the cast via a tiny sealed helper using `half`/stdlib only.
mod bytemuck {
    pub trait Pod: Copy + 'static {}
    impl Pod for f32 {}
    impl Pod for half::f16 {}
    pub fn cast_slice<T: Pod>(b: &[u8]) -> &[T] {
        let n = b.len() / std::mem::size_of::<T>();
        let ptr = b.as_ptr() as *const T;
        // SAFETY: bytes come from an mmap of a valid GGUF file; the caller has
        // verified the length is a whole number of `T`s via `take`.
        unsafe { std::slice::from_raw_parts(ptr, n) }
    }
}

/// A little-endian byte reader over a borrowed slice, with bounds checks.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn need(&self, n: usize) -> Result<()> {
        if self.pos + n > self.buf.len() {
            return Err(crate::Error::Gguf(format!(
                "unexpected end of GGUF header at byte {} (needed {n})",
                self.buf.len()
            )));
        }
        Ok(())
    }
    fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }
    fn u16(&mut self) -> Result<u16> {
        self.need(2)?;
        let v = u16::from_le_bytes([self.buf[self.pos], self.buf[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn u32(&mut self) -> Result<u32> {
        self.need(4)?;
        let v = u32::from_le_bytes([
            self.buf[self.pos],
            self.buf[self.pos + 1],
            self.buf[self.pos + 2],
            self.buf[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn u64(&mut self) -> Result<u64> {
        self.need(8)?;
        let v = u64::from_le_bytes([
            self.buf[self.pos],
            self.buf[self.pos + 1],
            self.buf[self.pos + 2],
            self.buf[self.pos + 3],
            self.buf[self.pos + 4],
            self.buf[self.pos + 5],
            self.buf[self.pos + 6],
            self.buf[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn bool(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        self.need(len)?;
        let s = std::str::from_utf8(&self.buf[self.pos..self.pos + len])
            .map_err(|e| crate::Error::Gguf(format!("invalid utf8 in gguf string: {e}")))?
            .to_string();
        self.pos += len;
        Ok(s)
    }
}
