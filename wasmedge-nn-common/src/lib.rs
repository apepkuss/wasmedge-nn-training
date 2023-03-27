extern crate num as num_renamed;
#[macro_use]
extern crate num_derive;

use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

/// Convert a slice of any type to a slice of bytes.
pub fn to_bytes<'a, T>(data: &'a [T]) -> &'a [u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const _,
            data.len() * std::mem::size_of::<T>(),
        )
    }
}

/// Convert a slice of bytes to a vector of `f32`.
pub fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    let chunks: Vec<&[u8]> = data.chunks(4).collect();
    let v: Vec<f32> = chunks
        .into_iter()
        .map(|c| {
            let mut rdr = Cursor::new(c);
            rdr.read_f32::<LittleEndian>().expect("failed to read")
        })
        .collect();

    v.into_iter().collect()
}

/// Convert a slice of bytes to a vector of `i32`.
pub fn bytes_to_i32_vec(data: &[u8]) -> Vec<i32> {
    let chunks: Vec<&[u8]> = data.chunks(4).collect();
    let v: Vec<i32> = chunks
        .into_iter()
        .map(|c| {
            let mut rdr = Cursor::new(c);
            rdr.read_i32::<LittleEndian>().expect("failed to read")
        })
        .collect();

    v.into_iter().collect()
}

/// Convert a slice of bytes to a vector of `u32`.
pub fn bytes_to_u32_vec(data: &[u8]) -> Vec<u32> {
    let chunks: Vec<&[u8]> = data.chunks(4).collect();
    let v: Vec<u32> = chunks
        .into_iter()
        .map(|c| {
            let mut rdr = Cursor::new(c);
            rdr.read_u32::<LittleEndian>().expect("failed to read")
        })
        .collect();

    v.into_iter().collect()
}

/// Convert a slice of bytes to a vector of `i64`.
pub fn bytes_to_i64_vec(data: &[u8]) -> Vec<i64> {
    let chunks: Vec<&[u8]> = data.chunks(8).collect();
    let v: Vec<i64> = chunks
        .into_iter()
        .map(|c| {
            let mut rdr = Cursor::new(c);
            rdr.read_i64::<LittleEndian>().expect(
                format!(
                    "plugin: protocol: failed to read. input data size: {}",
                    data.len()
                )
                .as_str(),
            )
        })
        .collect();

    v.into_iter().collect()
}

#[derive(Debug, PartialEq, FromPrimitive, ToPrimitive)]
pub enum Optimizer {
    Adam,
    RmsProp,
    Sgd,
}

#[derive(Debug, PartialEq, FromPrimitive, ToPrimitive)]
pub enum LossFunction {
    CrossEntropy,
    CrossEntropyForLogits,
}

pub enum Device {
    Cpu,
    Cuda(usize),
    Mps,
}

pub const SIZE_OF_TENSOR: u32 = 20;
pub const SIZE_OF_TENSOR_ELEMENT: u32 = 4;
pub const SIZE_OF_TENSOR_ARRAY: u32 = 8;

pub type GraphBuilder<'a> = &'a [u8];
pub type GraphBuilderArray<'a> = &'a [GraphBuilder<'a>];

// size: 4 bytes
pub type TensorElement<'a> = &'a Tensor<'a>;
// size: 8 bytes
pub type TensorArray<'a> = &'a [TensorElement<'a>];

// size: 20 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Tensor<'a> {
    pub data: TensorData<'a>,             // 8 bytes
    pub dimensions: TensorDimensions<'a>, // 8 bytes
    pub dtype: Dtype,                     // 1 bytes
}

pub type TensorData<'a> = &'a [u8];
pub type TensorDimensions<'a> = &'a [u8];

#[derive(Debug, Clone, Copy, PartialEq, FromPrimitive, ToPrimitive)]
pub enum Dtype {
    F16,
    F32,
    U8,
    I32,
    I64,
}
