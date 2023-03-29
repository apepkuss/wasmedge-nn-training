// use mnist::*;

extern crate num as num_renamed;

use ndarray::{Array2, Array3};
use std::convert::From;
use std::fs::File;
use std::io::{self, BufReader, Read, Result, Write};
use std::path::Path;
use std::vec;
use wasmedge_nn_common as common;

mod plugin {
    #[link(wasm_import_module = "wasmedge-nn-training")]
    extern "C" {
        pub fn train(
            inp_tensor_arr: i32,
            inp_tensor_arr_len: i32,
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
    if args.len() < 2 {
        println!("Usage: wasmedge --dir <host-dir>:<guest-dir> /host/path/to/resnet-pytorch.wasm /gust/path/to/<model_file>");
        return;
    }
    // model file
    let model_file = args[1].as_bytes();

    let mut dataset: Vec<&common::Tensor> = vec![];

    // training images

    print!("[Wasm] Preparing training images ... ");
    io::stdout().flush().unwrap();

    let trn_img_filename = Path::new("data/train-images-idx3-ubyte");
    let train_images = read_images(trn_img_filename).expect("failed to load training images");

    // shape = (batch, channel, row, cols)
    let train_images_shape = [60_000_u32, 1, 28, 28];
    let train_images_dims = common::to_bytes(&train_images_shape);
    let train_images_tensor = common::Tensor::new(
        train_images.as_slice(),
        train_images_dims,
        common::Dtype::F32,
        None,
    );

    dataset.push(&train_images_tensor);

    println!("[Done]");

    // training labels

    print!("[Wasm] Preparing training labels ... ");
    io::stdout().flush().unwrap();

    let trn_lbl_filename = Path::new("data/train-labels-idx1-ubyte");
    let train_labels = read_labels_new(trn_lbl_filename).expect("failed to load training lables");

    let train_labels_shape = [60_000_u32];
    let train_labels_dims = common::to_bytes(&train_labels_shape);

    let train_labels_tensor = common::Tensor::new(
        train_labels.as_slice(),
        train_labels_dims,
        common::Dtype::I64,
        None,
    );

    dataset.push(&train_labels_tensor);

    println!("[Done]");

    // test images

    print!("[Wasm] Preparing test images ... ");
    io::stdout().flush().unwrap();

    let tst_img_filename = Path::new("data/t10k-images-idx3-ubyte");
    let test_images = read_images(tst_img_filename).expect("failed to load test images");

    let test_images_shape = [10_000_u32, 1, 28, 28];
    let test_images_dims = common::to_bytes(test_images_shape.as_slice());

    let test_images_tensor = common::Tensor::new(
        test_images.as_slice(),
        test_images_dims,
        common::Dtype::F32,
        None,
    );
    dataset.push(&test_images_tensor);

    println!("[Done]");

    // test labels

    print!("[Wasm] Preparing test lables ... ");
    io::stdout().flush().unwrap();

    let tst_lbl_filename = Path::new("data/t10k-labels-idx1-ubyte");
    let test_labels = read_labels_new(tst_lbl_filename).expect("failed to load test labels");

    let test_labels_shape = [10_000_u32];
    let test_labels_dims = common::to_bytes(test_labels_shape.as_slice());

    let test_labels_tensor = common::Tensor::new(
        test_labels.as_slice(),
        test_labels_dims,
        common::Dtype::I64,
        None,
    );

    dataset.push(&test_labels_tensor);

    println!("[Done]");

    let offset_dataset = dataset.as_ptr() as *const _ as usize as i32;
    let len_dataset = dataset.len() as i32;

    // optimizer
    let optimizer = num_renamed::ToPrimitive::to_i32(&common::Optimizer::Adam)
        .expect("[Wasm] Failed to convert optimizer to i32");

    // loss function
    let loss_fn = num_renamed::ToPrimitive::to_i32(&common::LossFunction::CrossEntropyForLogits)
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
    let data = common::to_bytes(&data).to_vec();

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

fn read_images(filename: &std::path::Path) -> Result<Vec<u8>> {
    let mut buf_reader = BufReader::new(File::open(filename)?);
    check_magic_number(&mut buf_reader, 2051)?;
    let samples = read_u32(&mut buf_reader)?;
    let rows = read_u32(&mut buf_reader)?;
    let cols = read_u32(&mut buf_reader)?;
    let data_len = samples * rows * cols;
    let mut data = vec![0u8; data_len as usize];
    buf_reader.read_exact(&mut data)?;

    let data: Vec<f32> = data.into_iter().map(|c| f32::from(c) / 255.).collect();
    let data = common::to_bytes(&data).to_vec();

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

fn _load_model(filename: &str) -> Result<Vec<u8>> {
    dbg!(filename);

    let mut buf_reader = BufReader::new(File::open(filename)?);

    let mut data: Vec<u8> = vec![];
    buf_reader.read_to_end(&mut data)?;

    Ok(data)
}
