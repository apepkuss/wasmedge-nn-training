extern crate num as num_renamed;

use anyhow::Result;
use std::io::{self, Write};
use wasmedge_nn_common as common;
use wasmedge_sdk::{
    error::HostFuncError,
    host_function,
    plugin::{ffi, PluginDescriptor, PluginVersion},
    Caller, ImportObjectBuilder, ValType, WasmValue,
};

use tch::nn::{Adam, ModuleT, OptimizerConfig, RmsProp, Sgd, VarStore};
use tch::vision::dataset::Dataset;
use tch::TrainableCModule;
use tch::{Device, Tensor};

#[host_function]
fn train(caller: Caller, input: Vec<WasmValue>) -> Result<Vec<WasmValue>, HostFuncError> {
    println!("\n*** Welcome! This is `wasmedge-nn-training` plugin. ***\n");

    // check the number of inputs
    assert_eq!(input.len(), 11);

    // get the linear memory
    let memory = caller.memory(0).expect("failed to get memory at idex 0");

    let offset_tensors = if input[0].ty() == ValType::I32 {
        input[0].to_i32()
    } else {
        return Err(HostFuncError::User(1));
    };
    // println!("[plugin] offset_tensors: {offset_tensors}");

    let len_tensors = if input[1].ty() == ValType::I32 {
        input[1].to_i32()
    } else {
        return Err(HostFuncError::User(2));
    };
    // println!("[plugin] len_tensors: {len_tensors}");

    let ptr_tensors = memory
        .data_pointer(offset_tensors as u32, common::SIZE_OF_TENSOR_ARRAY)
        .expect("failed to get data from linear memory");
    // println!("[plugin] ptr_tensor: {:p}", ptr_tensors);
    let slice = unsafe {
        std::slice::from_raw_parts(
            ptr_tensors,
            common::SIZE_OF_TENSOR_ELEMENT as usize * len_tensors as usize,
        )
    };
    // println!("[Plugin] len of slice: {}", slice.len());

    // * extract train_images

    print!("[Plugin] Preparing train images ... ");
    io::stdout().flush().unwrap();
    let train_images: Tensor = {
        // * extract tenor1
        let offset1 = i32::from_le_bytes(slice[0..4].try_into().unwrap());
        let slice1 = memory.read(offset1 as u32, common::SIZE_OF_TENSOR).unwrap();

        // parse tensor1's data
        let offset_data1 = i32::from_le_bytes(slice1[0..4].try_into().unwrap());
        let len_data1 = i32::from_le_bytes(slice1[4..8].try_into().unwrap());
        let data1 = memory.read(offset_data1 as u32, len_data1 as u32).unwrap();

        // parse tensor1's dimensions
        let offset_dims1 = i32::from_le_bytes(slice1[8..12].try_into().unwrap());
        let len_dims1 = i32::from_le_bytes(slice1[12..16].try_into().unwrap());
        let dims1 = memory
            .read(offset_dims1 as u32, len_dims1 as u32)
            .expect("failed to read memory");
        let dims1: Vec<i64> = common::bytes_to_u32_vec(dims1.as_slice())
            .into_iter()
            .map(i64::from)
            .collect();

        // parse tensor1's type
        let dtype1 = slice1[16];

        // convert to tch::Tensor for train_images
        to_tch_tensor(dtype1, dims1.as_slice(), data1.as_slice())
    };
    println!(
        "[Done] (shape: {:?}, dtype: {:?})",
        train_images.size(),
        train_images.kind()
    );

    // * extract train_labels

    print!("[Plugin] Preparing train labels ... ");
    io::stdout().flush().unwrap();
    let train_labels: Tensor = {
        // * extract tenor2
        let offset1 = i32::from_le_bytes(slice[4..8].try_into().unwrap());
        let slice1 = memory.read(offset1 as u32, common::SIZE_OF_TENSOR).unwrap();

        // parse tensor2's data
        let offset_data1 = i32::from_le_bytes(slice1[0..4].try_into().unwrap());
        let len_data1 = i32::from_le_bytes(slice1[4..8].try_into().unwrap());
        let data1 = memory.read(offset_data1 as u32, len_data1 as u32).unwrap();

        // parse tensor2's dimensions
        let offset_dims1 = i32::from_le_bytes(slice1[8..12].try_into().unwrap());
        let len_dims1 = i32::from_le_bytes(slice1[12..16].try_into().unwrap());
        let dims1 = memory
            .read(offset_dims1 as u32, len_dims1 as u32)
            .expect("failed to read memory");
        let dims1: Vec<i64> = common::bytes_to_u32_vec(dims1.as_slice())
            .into_iter()
            .map(i64::from)
            .collect();

        // parse tensor2's type
        let dtype1 = slice1[16];

        // convert to tch::Tensor for train_labels
        to_tch_tensor(dtype1, dims1.as_slice(), data1.as_slice())
    };
    println!(
        "[Done] (shape: {:?}, dtype: {:?})",
        train_labels.size(),
        train_labels.kind()
    );

    // * extract test_images

    print!("[Plugin] Preparing test images ... ");
    io::stdout().flush().unwrap();
    let test_images: Tensor = {
        // * extract tenor3
        let offset1 = i32::from_le_bytes(slice[8..12].try_into().unwrap());
        let slice1 = memory.read(offset1 as u32, common::SIZE_OF_TENSOR).unwrap();

        // parse tensor3's data
        let offset_data1 = i32::from_le_bytes(slice1[0..4].try_into().unwrap());
        let len_data1 = i32::from_le_bytes(slice1[4..8].try_into().unwrap());
        let data1 = memory.read(offset_data1 as u32, len_data1 as u32).unwrap();

        // parse tensor3's dimensions
        let offset_dims1 = i32::from_le_bytes(slice1[8..12].try_into().unwrap());
        let len_dims1 = i32::from_le_bytes(slice1[12..16].try_into().unwrap());
        let dims1 = memory
            .read(offset_dims1 as u32, len_dims1 as u32)
            .expect("failed to read memory");
        let dims1: Vec<i64> = common::bytes_to_u32_vec(dims1.as_slice())
            .into_iter()
            .map(i64::from)
            .collect();

        // parse tensor3's type
        let dtype1 = slice1[16];

        // convert to tch::Tensor for test_images
        to_tch_tensor(dtype1, dims1.as_slice(), data1.as_slice())
    };
    println!(
        "[Done] (shape: {:?}, dtype: {:?})",
        test_images.size(),
        test_images.kind()
    );

    // extract test_labels

    print!("[Plugin] Preparing test labels ... ");
    io::stdout().flush().unwrap();
    let test_labels: Tensor = {
        // * extract tenor4
        let offset1 = i32::from_le_bytes(slice[12..16].try_into().unwrap());
        let slice1 = memory.read(offset1 as u32, common::SIZE_OF_TENSOR).unwrap();

        // parse tensor4's data
        let offset_data1 = i32::from_le_bytes(slice1[0..4].try_into().unwrap());
        let len_data1 = i32::from_le_bytes(slice1[4..8].try_into().unwrap());
        let data1 = memory.read(offset_data1 as u32, len_data1 as u32).unwrap();

        // parse tensor4's dimensions
        let offset_dims1 = i32::from_le_bytes(slice1[8..12].try_into().unwrap());
        let len_dims1 = i32::from_le_bytes(slice1[12..16].try_into().unwrap());
        let dims1 = memory
            .read(offset_dims1 as u32, len_dims1 as u32)
            .expect("plugin: test_labels: faied to extract tensor dimensions");
        let dims1: Vec<i64> = common::bytes_to_i32_vec(dims1.as_slice())
            .iter()
            .map(|&c| c as i64)
            .collect();

        // parse tensor4's type
        let dtype1 = slice1[16];

        // convert to tch::Tensor for test_labels
        to_tch_tensor(dtype1, dims1.as_slice(), data1.as_slice())
    };
    println!(
        "[Done] (shape: {:?}, dtype: {:?})",
        test_labels.size(),
        test_labels.kind()
    );

    let labels = if input[2].ty() == ValType::I64 {
        input[2].to_i64()
    } else {
        return Err(HostFuncError::User(3));
    };
    println!("[Plugin] Labels: {labels}");

    let ds = Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels,
    };

    // device
    let device_id = if input[3].ty() == ValType::I32 {
        input[3].to_i32()
    } else {
        return Err(HostFuncError::User(4));
    };
    let device = match device_id {
        0 => Device::Cpu,
        _ => panic!("unsupported device id: {device_id}"),
    };
    println!("[Plugin] Device: {:?}", device);

    // learning rate
    let lr = if input[4].ty() == ValType::F64 {
        input[4].to_f64()
    } else {
        return Err(HostFuncError::User(5));
    };
    println!("[Plugin] Learning rate: {lr}");

    // epoch
    let epochs = if input[5].ty() == ValType::I32 {
        input[5].to_i32()
    } else {
        return Err(HostFuncError::User(6));
    };
    println!("[Plugin] Epochs: {epochs}");

    // batch_size
    let batch_size = if input[6].ty() == ValType::I64 {
        input[6].to_i64()
    } else {
        return Err(HostFuncError::User(7));
    };
    println!("[Plugin] batch size: {batch_size}");

    // optimizer
    let optimizer = if input[7].ty() == ValType::I32 {
        input[7].to_i32()
    } else {
        return Err(HostFuncError::User(8));
    };
    let optimizer: common::Optimizer = num_renamed::FromPrimitive::from_i32(optimizer)
        .expect("[Plugin] failed to parse optimizer");
    println!("[Plugin] Optimizer: {:?}", optimizer);

    // loss function
    let loss_fn = if input[8].ty() == ValType::I32 {
        input[8].to_i32()
    } else {
        return Err(HostFuncError::User(9));
    };
    let loss_fn: common::LossFunction = num_renamed::FromPrimitive::from_i32(loss_fn)
        .expect("[Plugin] failed to parse loss function");
    println!("[Plugin] Loss function: {:?}", loss_fn);

    // model data
    let offset_model_file = if input[9].ty() == ValType::I32 {
        input[9].to_i32()
    } else {
        return Err(HostFuncError::User(10));
    };
    let len_model_file = if input[10].ty() == ValType::I32 {
        input[10].to_i32()
    } else {
        return Err(HostFuncError::User(11));
    };
    let ptr_model = memory
        .data_pointer(offset_model_file as u32, 4)
        .expect("[Plugin] Failed to parse model");
    let slice_model = unsafe { std::slice::from_raw_parts(ptr_model, len_model_file as usize) };
    let model_file = std::str::from_utf8(slice_model).expect("[Plugin] Failed to parse model file");
    let model_file = std::path::Path::new(model_file)
        .canonicalize()
        .expect("[Plugin] Failed to parse model filepath");
    let model_path = model_file.as_path();
    // println!("[Plugin] model path: {:?}", model_path);

    // start training
    train_torch_model(
        ds, device, lr, epochs, batch_size, optimizer, loss_fn, model_path,
    )
    .expect("failed to train model");

    Ok(vec![])
}

fn train_torch_model(
    dataset: Dataset,
    device: Device,
    lr: f64,
    epochs: i32,
    batch_size: i64,
    optimizer: common::Optimizer,
    loss_fn: common::LossFunction,
    model_path: &std::path::Path,
) -> Result<()> {
    let vs = VarStore::new(device);
    println!("[Plugin] Load model");
    let mut trainable = TrainableCModule::load(model_path, vs.root())?;
    trainable.set_train();

    let initial_acc = trainable.batch_accuracy_for_logits(
        &dataset.test_images,
        &dataset.test_labels,
        vs.device(),
        1024,
    );
    println!("[Plugin] Initial accuracy: {:5.2}%", 100. * initial_acc);

    let mut opt = match optimizer {
        common::Optimizer::Adam => Adam::default()
            .build(&vs, lr)
            .expect("[Train] failed to create tch::Adam optimizer"),
        common::Optimizer::RmsProp => RmsProp::default()
            .build(&vs, lr)
            .expect("[Train] failed to create tch::RmsProp optimizer"),
        common::Optimizer::Sgd => Sgd::default()
            .build(&vs, lr)
            .expect("[Train] failed to create tch::Sgd optimizer"),
    };

    println!("[Plugin] Start training ... ");
    // let mut opt = Adam::default().build(&vs, lr).expect("[Train] optimizer");
    for epoch in 1..=epochs {
        for (images, labels) in dataset
            .train_iter(batch_size)
            .shuffle()
            .to_device(vs.device())
            .take(50)
        {
            let loss = trainable
                .forward_t(&images, true)
                .cross_entropy_for_logits(&labels);
            opt.backward_step(&loss);
        }
        let test_accuracy = trainable.batch_accuracy_for_logits(
            &dataset.test_images,
            &dataset.test_labels,
            vs.device(),
            1024,
        );
        println!(
            "\tepoch: {:4} test acc: {:5.2}%",
            epoch,
            100. * test_accuracy,
        );
    }
    println!("[Plugin] Finished");

    let saved_model_filename =
        String::from("trained_") + model_path.file_name().unwrap().to_str().unwrap();
    let saved_model_file = model_path.with_file_name(saved_model_filename);
    trainable.save(&saved_model_file)?;
    println!(
        "[Plugin] The pre-trained model is dumped to {:?}",
        saved_model_file
    );

    Ok(())
}

/// Defines Plugin module instance
unsafe extern "C" fn create_test_module(
    _arg1: *const ffi::WasmEdge_ModuleDescriptor,
) -> *mut ffi::WasmEdge_ModuleInstanceContext {
    let module_name = "wasmedge-nn-training";
    let import = ImportObjectBuilder::new()
        // add a function
        .with_func::<(i32, i32, i64, i32, f64, i32, i64, i32, i32, i32, i32), ()>("train", train)
        .expect("failed to create set_dataset host function")
        .build(module_name)
        .expect("failed to create import object");

    let boxed_import = Box::new(import);
    let import = Box::leak(boxed_import);

    import.as_raw_ptr() as *mut _
}

/// Defines PluginDescriptor
#[export_name = "WasmEdge_Plugin_GetDescriptor"]
pub extern "C" fn plugin_hook() -> *const ffi::WasmEdge_PluginDescriptor {
    let name = "wasmedge-nn-training-plugin";
    let desc = "this is an experimental plugin for AI training";
    let version = PluginVersion::new(0, 0, 0, 0);
    let plugin_descriptor = PluginDescriptor::new(name, desc, version)
        .expect("Failed to create plugin descriptor")
        .add_module_descriptor(
            "wasmedge_nn_training_module",
            "this is a plugin module for AI training",
            Some(create_test_module),
        )
        .expect("Failed to add module descriptor");

    let boxed_plugin = Box::new(plugin_descriptor);
    let plugin = Box::leak(boxed_plugin);

    plugin.as_raw_ptr()
}

pub fn to_tch_tensor(dtype: u8, dims: &[i64], data: &[u8]) -> tch::Tensor {
    match dtype {
        0 => unimplemented!("F16"),
        1 => {
            let data = common::bytes_to_f32_vec(data);
            Tensor::of_slice(data.as_slice()).reshape(dims)
        }
        2 => Tensor::of_slice(data).reshape(dims),
        3 => {
            let data = common::bytes_to_i32_vec(data);
            Tensor::of_slice(data.as_slice()).reshape(dims)
        }
        4 => {
            let data = common::bytes_to_i64_vec(data);
            Tensor::of_slice(data.as_slice()).reshape(dims)
        }
        _ => panic!("plugin: train_images: unsupported dtype: {dtype}"),
    }
}
