// use mnist::*;

use ndarray::{Array2, Array3};
use std::convert::From;
use std::fs::File;
use std::io::{self, BufReader, Read, Result, Write};
use std::path::Path;
use std::vec;

use crate::protocol::{GraphBuilder, GraphBuilderArray};

extern crate num as num_renamed;
#[macro_use]
extern crate num_derive;

mod plugin {
    #[link(wasm_import_module = "wasmedge-nn-training")]
    extern "C" {
        pub fn train(
            train_images_offset: i32,
            train_images_size: i32,
            labels: i64,
            device: i32,
            lr: f64,
            epochs: i32,
            batch_size: i64,
            optimizer: i32,
            loss_fn: i32,
            model_arr: i32,
            model_arr_len: i32,
        );
    }
}

fn main() {
    // download_mnist_images();

    let args: Vec<String> = std::env::args().collect();
    // model file
    let model_file = args[1].as_bytes();

    let mut dataset: Vec<&protocol::Tensor> = vec![];

    // training images

    print!("[Wasm] Preparing training images ... ");
    io::stdout().flush().unwrap();

    let trn_img_filename = Path::new("data/train-images-idx3-ubyte");
    let train_images = read_images_new(trn_img_filename).expect("failed to load training images");

    // shape = (batch, channel, row, cols)
    let train_images_shape = [60_000_u32, 1, 28, 28];
    let train_images_dims = protocol::to_bytes(&train_images_shape);
    let train_images_tensor = protocol::Tensor {
        dimensions: train_images_dims,
        dtype: protocol::TENSOR_TYPE_F32,
        data: train_images.as_slice(),
    };

    dataset.push(&train_images_tensor);

    println!("[Done]");

    // training labels

    print!("[Wasm] Preparing training labels ... ");
    io::stdout().flush().unwrap();

    let trn_lbl_filename = Path::new("data/train-labels-idx1-ubyte");
    let train_labels = read_labels_new(trn_lbl_filename).expect("failed to load training lables");

    let train_labels_shape = [60_000_u32];
    let train_labels_dims = protocol::to_bytes(&train_labels_shape);

    let train_labels_tensor = protocol::Tensor {
        data: train_labels.as_slice(),
        dimensions: train_labels_dims,
        dtype: protocol::TENSOR_TYPE_I64,
    };

    dataset.push(&train_labels_tensor);

    println!("[Done]");

    // test images

    print!("[Wasm] Preparing test images ... ");
    io::stdout().flush().unwrap();

    let tst_img_filename = Path::new("data/t10k-images-idx3-ubyte");
    let test_images = read_images_new(tst_img_filename).expect("failed to load test images");

    let test_images_shape = [10_000_u32, 1, 28, 28];
    let test_images_dims = protocol::to_bytes(test_images_shape.as_slice());

    let test_images_tensor = protocol::Tensor {
        dimensions: test_images_dims,
        dtype: protocol::TENSOR_TYPE_F32,
        data: test_images.as_slice(),
    };
    dataset.push(&test_images_tensor);

    println!("[Done]");

    // test labels

    print!("[Wasm] Preparing test lables ... ");
    io::stdout().flush().unwrap();

    let tst_lbl_filename = Path::new("data/t10k-labels-idx1-ubyte");
    let test_labels = read_labels_new(tst_lbl_filename).expect("failed to load test labels");

    let test_labels_shape = [10_000_u32];
    let test_labels_dims = protocol::to_bytes(test_labels_shape.as_slice());

    let test_labels_tensor = protocol::Tensor {
        dimensions: &test_labels_dims,
        dtype: protocol::TENSOR_TYPE_I64,
        data: test_labels.as_slice(),
    };

    dataset.push(&test_labels_tensor);

    println!("[Done]");

    let offset_dataset = dataset.as_ptr() as *const _ as usize as i32;
    let len_dataset = dataset.len() as i32;

    // optimizer
    let optimizer = num_renamed::ToPrimitive::to_i32(&protocol::Optimizer::Adam)
        .expect("[Wasm] Failed to convert optimizer to i32");

    // loss function
    let loss_fn = num_renamed::ToPrimitive::to_i32(&protocol::LossFunction::CrossEntropyForLogits)
        .expect("[Wasm] Failed to convert loss function to i32");

    unsafe {
        plugin::train(
            offset_dataset,
            len_dataset,
            10,
            0, // device: CPU
            1e-4,
            10,
            128,
            optimizer,
            loss_fn,
            model_file.as_ptr() as i32,
            model_file.len() as i32,
        )
    }
}

fn read_u32<T: Read>(reader: &mut T) -> Result<u32> {
    let mut b = vec![0u8; 4];
    reader.read_exact(&mut b)?;
    let (result, _) = b.iter().rev().fold((0u64, 1u64), |(s, basis), &x| {
        (s + basis * u64::from(x), basis * 256)
    });
    Ok(result as u32)
}

fn check_magic_number<T: Read>(reader: &mut T, expected: u32) -> Result<()> {
    let magic_number = read_u32(reader)?;
    if magic_number != expected {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("incorrect magic number {magic_number} != {expected}"),
        ));
    }
    Ok(())
}

fn _read_labels(filename: &std::path::Path) -> Result<ndarray::Array<i64, ndarray::Ix1>> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2049)?;
    let samples = read_u32(&mut buf_reader)?;
    let mut data = vec![0u8; samples as usize];
    buf_reader.read_exact(&mut data)?;

    Ok(ndarray::Array::from_vec(data).mapv(i64::from))

    // Ok(Tensor::of_slice(&data).to_kind(Kind::Int64))
}

fn read_labels_new(filename: &std::path::Path) -> Result<Vec<u8>> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2049)?;
    let samples = read_u32(&mut buf_reader)?;
    let mut data = vec![0u8; samples as usize];
    buf_reader.read_exact(&mut data)?;

    let data: Vec<i64> = data.into_iter().map(|c| i64::from(c)).collect();
    let data = protocol::to_bytes(&data).to_vec();

    Ok(data)

    // Ok(ndarray::Array::from_vec(data).mapv(i64::from))

    // Ok(Tensor::of_slice(&data).to_kind(Kind::Int64))
}

fn _read_images(filename: &std::path::Path) -> Result<ndarray::Array<f32, ndarray::Ix2>> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2051)?;
    let samples = read_u32(&mut buf_reader)?;
    let rows = read_u32(&mut buf_reader)?;
    let cols = read_u32(&mut buf_reader)?;
    let data_len = samples * rows * cols;
    let mut data = vec![0u8; data_len as usize];
    buf_reader.read_exact(&mut data)?;

    let shape = (samples as usize, (rows * cols) as usize);
    let arr = ndarray::Array::from_shape_vec(shape, data)
        .expect("failed to create ndarray for images")
        .mapv(f32::from)
        / 255.;

    Ok(arr)

    // let tensor = Tensor::of_slice(&data)
    //     .view((i64::from(samples), i64::from(rows * cols)))
    //     .to_kind(Kind::Float);
    // Ok(tensor / 255.)
}

fn read_images_new(filename: &std::path::Path) -> Result<Vec<u8>> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2051)?;
    let samples = read_u32(&mut buf_reader)?;
    let rows = read_u32(&mut buf_reader)?;
    let cols = read_u32(&mut buf_reader)?;
    let data_len = samples * rows * cols;
    let mut data = vec![0u8; data_len as usize];
    buf_reader.read_exact(&mut data)?;

    let data: Vec<f32> = data.into_iter().map(|c| f32::from(c) / 255.).collect();
    let data = protocol::to_bytes(&data).to_vec();

    Ok(data)

    // let shape = (samples as usize, (rows * cols) as usize);
    // let arr = ndarray::Array::from_shape_vec(shape, data)
    //     .expect("failed to create ndarray for images")
    //     .mapv(f32::from)
    //     / 255.;

    // Ok(arr)

    // let tensor = Tensor::of_slice(&data)
    //     .view((i64::from(samples), i64::from(rows * cols)))
    //     .to_kind(Kind::Float);
    // Ok(tensor / 255.)
}

fn _images_to_ndarray(
    data: Vec<u8>,
    dim1: usize,
    dim2: usize,
    dim3: usize,
) -> ndarray::Array3<f32> {
    Array3::from_shape_vec((dim1, dim2, dim3), data)
        .expect("Error converting data to 3D array")
        .map(|x| *x as f32)
}

fn _labels_to_ndarray(data: Vec<u8>, dim1: usize, dim2: usize) -> ndarray::Array2<i64> {
    Array2::from_shape_vec((dim1, dim2), data)
        .expect("Error converting data to 2D array")
        .map(|x| *x as i64)
}

fn load_model(filename: &str) -> Result<Vec<u8>> {
    dbg!(filename);

    let mut buf_reader = BufReader::new(File::open(filename)?);

    let mut data: Vec<u8> = vec![];
    buf_reader.read_to_end(&mut data)?;

    Ok(data)
}

// * interface protocol

pub mod protocol {

    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Cursor;

    pub fn to_bytes<'a, T>(data: &'a [T]) -> &'a [u8] {
        unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const _,
                data.len() * std::mem::size_of::<T>(),
            )
        }
    }

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

    #[repr(C)]
    #[derive(Clone, Debug)]
    pub struct Array {
        pub data: *const u8, // 4 bytes
        pub size: i32,       // 4 bytes
    }

    #[repr(C)]
    #[derive(Clone, Debug)]
    pub struct MyTensor {
        pub data: *const u8, // 4 bytes
        pub data_size: u32,  // 4 bytes
        pub dims: *const u8, // 4 bytes
        pub dims_size: u32,  // 4 bytes
        pub ty: u8,          // 1 byte
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
        pub dtype: TensorType,                // 1 bytes
    }

    pub type TensorData<'a> = &'a [u8];
    pub type TensorDimensions<'a> = &'a [u8];

    #[repr(transparent)]
    #[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
    pub struct TensorType(u8);
    pub const TENSOR_TYPE_F16: TensorType = TensorType(0);
    pub const TENSOR_TYPE_F32: TensorType = TensorType(1);
    pub const TENSOR_TYPE_U8: TensorType = TensorType(2);
    pub const TENSOR_TYPE_I32: TensorType = TensorType(3);
    pub const TENSOR_TYPE_I64: TensorType = TensorType(4);
    impl TensorType {
        pub const fn raw(&self) -> u8 {
            self.0
        }

        pub fn name(&self) -> &'static str {
            match self.0 {
                0 => "F16",
                1 => "F32",
                2 => "U8",
                3 => "I32",
                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }

        pub fn message(&self) -> &'static str {
            match self.0 {
                0 => "",
                1 => "",
                2 => "",
                3 => "",
                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }
    }
    impl std::fmt::Debug for TensorType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TensorType")
                .field("code", &self.0)
                .field("name", &self.name())
                .field("message", &self.message())
                .finish()
        }
    }
}
