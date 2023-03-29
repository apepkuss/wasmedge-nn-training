extern crate num as num_renamed;

use anyhow::Result;
use std::collections::HashMap;
use std::io::{self, Write};
use wasmedge_nn_common as common;
use wasmedge_sdk::{
    error::HostFuncError,
    host_function,
    plugin::{ffi, PluginDescriptor, PluginVersion},
    Caller, ImportObjectBuilder, ValType, WasmValue,
};

#[cfg(feature = "torch")]
use tch::{
    nn::{Adam, ModuleT, OptimizerConfig, RmsProp, Sgd, VarStore},
    vision::dataset::Dataset,
    Device, Tensor, TrainableCModule,
};

use rand;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[cfg(feature = "tensorflow")]
use tensorflow::{
    self as tf, Code, Graph, ImportGraphDefOptions, SavedModelBundle, Session, SessionOptions,
    SessionRunArgs, Status,
};

static ALLOCATOR: std::alloc::System = std::alloc::System;

#[cfg(feature = "torch")]
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
        let dtype1 = num_renamed::FromPrimitive::from_u8(slice1[16])
            .expect("[Plugin] failed to parse tensor's dtype: {slice1[16]}");

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
        let dtype1 = num_renamed::FromPrimitive::from_u8(slice1[16])
            .expect("[Plugin] failed to parse tensor's dtype: {slice1[16]}");

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
        let dtype1 = num_renamed::FromPrimitive::from_u8(slice1[16])
            .expect("[Plugin] failed to parse tensor's dtype: {slice1[16]}");

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
        let dtype1 = num_renamed::FromPrimitive::from_u8(slice1[16])
            .expect("[Plugin] failed to parse tensor's dtype: {slice1[16]}");

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

#[cfg(feature = "torch")]
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

#[cfg(feature = "tensorflow")]
#[host_function]
fn train(caller: Caller, input: Vec<WasmValue>) -> Result<Vec<WasmValue>, HostFuncError> {
    println!("\n*** Welcome! This is `wasmedge-nn-training` plugin. ***\n");

    // check the number of inputs
    assert_eq!(input.len(), 3);

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

    // * extract training tensor

    print!("[Plugin] Preparing training tensor ... ");
    io::stdout().flush().unwrap();

    let offset_train_input = i32::from_le_bytes(slice[0..4].try_into().unwrap());
    let slice_train_input = memory
        .read(offset_train_input as u32, common::SIZE_OF_TENSOR)
        .unwrap();

    // parse train_input data
    let offset_train_input_data = i32::from_le_bytes(slice_train_input[0..4].try_into().unwrap());
    let len_train_input_data = i32::from_le_bytes(slice_train_input[4..8].try_into().unwrap());
    let bytes_train_input_data = memory
        .read(offset_train_input_data as u32, len_train_input_data as u32)
        .unwrap();

    // parse train_input dimensions
    let offset_train_input_dims = i32::from_le_bytes(slice_train_input[8..12].try_into().unwrap());
    let len_train_input_dims = i32::from_le_bytes(slice_train_input[12..16].try_into().unwrap());
    let bytes_train_input_dims = memory
        .read(offset_train_input_dims as u32, len_train_input_dims as u32)
        .expect("failed to read memory");
    let train_input_dims: Vec<u64> = common::bytes_to_u32_vec(bytes_train_input_dims.as_slice())
        .into_iter()
        .map(u64::from)
        .collect();

    // parse train_input type
    let train_input_dtype: common::Dtype =
        num_renamed::FromPrimitive::from_u8(slice_train_input[16])
            .expect("[Plugin] failed to parse tensor's dtype: {slice_train_input[16]}");

    // parse train_input name
    let offset_train_input_name = i32::from_le_bytes(slice_train_input[20..24].try_into().unwrap());
    let len_train_input_name = i32::from_le_bytes(slice_train_input[24..28].try_into().unwrap());
    let bytes_train_input_name = memory
        .read(offset_train_input_name as u32, len_train_input_name as u32)
        .expect("failed to read memory");
    let train_input_name = std::str::from_utf8(bytes_train_input_name.as_slice())
        .expect("[Plugin] failed to convert to string");

    // create TFTensor
    let train_input_data = common::bytes_to_f32_vec(bytes_train_input_data.as_slice());
    // println!("train_input_data: {:?}", &train_input_data);
    let train_input_tensor = TFTensor {
        tensor: tf::Tensor::new(train_input_dims.as_slice())
            .with_values(train_input_data.as_slice())
            .unwrap(),
        name: train_input_name,
    };

    println!("[Done]");

    // * extract target tensor

    print!("[Plugin] Preparing target tensor ... ");
    io::stdout().flush().unwrap();

    let offset_train_target = i32::from_le_bytes(slice[4..8].try_into().unwrap());
    let slice_train_target = memory
        .read(offset_train_target as u32, common::SIZE_OF_TENSOR)
        .unwrap();

    // parse train_target data
    let offset_train_target_data = i32::from_le_bytes(slice_train_target[0..4].try_into().unwrap());
    let len_train_target_data = i32::from_le_bytes(slice_train_target[4..8].try_into().unwrap());
    let bytes_train_target_data = memory
        .read(
            offset_train_target_data as u32,
            len_train_target_data as u32,
        )
        .unwrap();

    // parse train_target dimensions
    let offset_train_target_dims =
        i32::from_le_bytes(slice_train_target[8..12].try_into().unwrap());
    let len_train_target_dims = i32::from_le_bytes(slice_train_target[12..16].try_into().unwrap());
    let bytes_train_target_dims = memory
        .read(
            offset_train_target_dims as u32,
            len_train_target_dims as u32,
        )
        .expect("failed to read memory");
    let train_target_dims: Vec<u64> = common::bytes_to_u32_vec(bytes_train_target_dims.as_slice())
        .into_iter()
        .map(u64::from)
        .collect();

    // parse train_target type
    let train_target_dtype: common::Dtype =
        num_renamed::FromPrimitive::from_u8(slice_train_target[16])
            .expect("[Plugin] failed to parse tensor's dtype: {slice1[16]}");

    // parse train_target name
    let offset_train_target_name =
        i32::from_le_bytes(slice_train_target[20..24].try_into().unwrap());
    let len_train_target_name = i32::from_le_bytes(slice_train_target[24..28].try_into().unwrap());
    let bytes_train_target_name = memory
        .read(
            offset_train_target_name as u32,
            len_train_target_name as u32,
        )
        .expect("failed to read memory");
    let train_target_name = std::str::from_utf8(bytes_train_target_name.as_slice())
        .expect("[Plugin] failed to convert to string");

    // create TFTensor
    let train_target_data = common::bytes_to_f32_vec(bytes_train_target_data.as_slice());
    // println!("train_target_data: {:?}", &train_target_data);
    let train_target_tensor = TFTensor {
        tensor: tf::Tensor::new(train_target_dims.as_slice())
            .with_values(train_target_data.as_slice())
            .unwrap(),
        name: train_target_name,
    };

    println!("[Done]");

    // * extract output tensor

    print!("[Plugin] Preparing output tensor ... ");
    io::stdout().flush().unwrap();

    let offset_train_output = i32::from_le_bytes(slice[8..12].try_into().unwrap());
    let slice_train_output = memory
        .read(offset_train_output as u32, common::SIZE_OF_TENSOR)
        .unwrap();

    // parse train_output name
    let offset_train_output_name =
        i32::from_le_bytes(slice_train_output[20..24].try_into().unwrap());
    let len_train_output_name = i32::from_le_bytes(slice_train_output[24..28].try_into().unwrap());
    let bytes_train_output_name = memory
        .read(
            offset_train_output_name as u32,
            len_train_output_name as u32,
        )
        .expect("failed to read memory");
    let train_output_name = std::str::from_utf8(bytes_train_output_name.as_slice())
        .expect("[Plugin] failed to convert to string");

    println!("[Done]");

    // * extract epochs
    let epochs = if input[2].ty() == ValType::I32 {
        input[2].to_i32()
    } else {
        return Err(HostFuncError::User(3));
    };
    println!("[Plugin] Epochs: {epochs}");

    train_custom_model(
        train_input_tensor,
        train_target_tensor,
        &[train_output_name],
        epochs,
    );

    Ok(vec![])
}

fn train_custom_model<S: tf::TensorType, T: tf::TensorType>(
    train_input_tensor: TFTensor<S>,
    train_target_tensor: TFTensor<T>,
    train_output_node_names: &[&str],
    epochs: i32,
) {
    println!("*** In train_custom_model ***\n");

    //Signatures declared when we saved the model
    // let train_input_parameter_input_name = "training_input";
    // let train_input_parameter_target_name = "training_target";
    // let pred_input_parameter_name = "inputs";

    //Names of output nodes of the graph, retrieved with the saved_model_cli command
    // let train_output_parameter_name = "output_0";
    // let pred_output_parameter_name = "output_0";

    //Create some tensors to feed to the model for training, one as input and one as the target value
    //Note: All tensors must be declared before args!

    //Path of the saved model
    let save_dir = "examples/tensorflow/custom-model/custom_model";

    //Create a graph
    let mut graph = Graph::new();

    //Load save model as graph
    let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, save_dir)
        .expect("Can't load saved model");

    //Initiate a session
    let session = &bundle.session;

    //The values will be fed to and retrieved from the model with this
    let mut args = SessionRunArgs::new();

    // ! Alternative to saved_model_cli. This will list all signatures in the console when run
    // let sigs = bundle.meta_graph_def().signatures();
    // println!("*** signatures: {:?}", sigs);

    //Retrieve the train functions signature
    let signature_train = bundle.meta_graph_def().get_signature("train").unwrap();

    // Input information
    let input_info_train = signature_train.get_input(train_input_tensor.name).unwrap();
    let input_op_train = graph
        .operation_by_name_required(&input_info_train.name().name)
        .unwrap();
    // Feed the input tensor into the graph
    args.add_feed(&input_op_train, 0, &train_input_tensor.tensor);

    // Target information
    let target_info_train = signature_train.get_input(train_target_tensor.name).unwrap();
    let target_op_train = graph
        .operation_by_name_required(&target_info_train.name().name)
        .unwrap();
    // Feed the target tensor into the graph
    args.add_feed(&target_op_train, 0, &train_target_tensor.tensor);

    // Output information
    let mut output_fetch_tokens = Vec::new();
    for train_output_node_name in train_output_node_names.into_iter() {
        let output_info_train = signature_train.get_output(train_output_node_name).unwrap();
        // Output operation
        let output_op_train = graph
            .operation_by_name_required(&output_info_train.name().name)
            .unwrap();
        // Fetch result from graph
        let token = args.request_fetch(&output_op_train, 0);

        output_fetch_tokens.push(token);
    }

    println!("[Plugin] Training model...");
    for i in 0..=epochs {
        print!("\tEpoch[{:?}]: ", i);
        io::stdout().flush().unwrap();

        session
            .run(&mut args)
            .expect("Error occurred during training");

        //Retrieve the result of the operation
        for (i, output_fetch_token) in output_fetch_tokens.iter().enumerate() {
            let loss: f32 = args.fetch(*output_fetch_token).unwrap()[0];
            println!("Loss of output[{i}]: {:?} ", loss);
        }
    }
    println!("[Done]");

    // //Retrieve the pred functions signature
    // let signature_train = bundle.meta_graph_def().get_signature("pred").unwrap();

    // //
    // let input_info_pred = signature_train
    //     .get_input(pred_input_parameter_name)
    //     .unwrap();

    // //
    // let output_info_pred = signature_train
    //     .get_output(pred_output_parameter_name)
    //     .unwrap();

    // //
    // let input_op_pred = graph
    //     .operation_by_name_required(&input_info_pred.name().name)
    //     .unwrap();

    // //
    // let output_op_pred = graph
    //     .operation_by_name_required(&output_info_pred.name().name)
    //     .unwrap();

    // args.add_feed(&input_op_pred, 0, &train_input_tensor.tensor);

    // let mut out = args.request_fetch(&output_op_pred, 0);

    // //Run the session
    // session
    //     .run(&mut args)
    //     .expect("Error occurred during calculations");

    // let prediction: f32 = args.fetch(out).unwrap()[0];

    // println!("Prediction: {:?}\nActual: 2.0", prediction);
}

#[cfg(feature = "tensorflow")]
#[host_function]
fn train_r(caller: Caller, input: Vec<WasmValue>) -> Result<Vec<WasmValue>, HostFuncError> {
    println!("\n*** Welcome! This is `wasmedge-nn-training` plugin. ***\n");

    // check the number of inputs
    assert_eq!(input.len(), 3);

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

    // * extract training tensor

    print!("[Plugin] Preparing training tensor ... ");
    io::stdout().flush().unwrap();
    let tensor_x = {
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
        let dims1: Vec<u64> = common::bytes_to_u32_vec(dims1.as_slice())
            .into_iter()
            .map(u64::from)
            .collect();

        // parse tensor1's type
        let dtype1: common::Dtype = num_renamed::FromPrimitive::from_u8(slice1[16])
            .expect("[Plugin] failed to parse tensor's dtype: {slice1[16]}");

        // create tf::Tensor
        let data = common::bytes_to_f32_vec(data1.as_slice());
        tf::Tensor::new(dims1.as_slice())
            .with_values(data.as_slice())
            .unwrap()
    };
    println!("[Done] (dims: {:?})", tensor_x.dims());

    // * extract target tensor

    print!("[Plugin] Preparing target tensor ... ");
    io::stdout().flush().unwrap();
    let tensor_y = {
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
        let dims1: Vec<u64> = common::bytes_to_u32_vec(dims1.as_slice())
            .into_iter()
            .map(u64::from)
            .collect();

        // parse tensor2's type
        let dtype1: common::Dtype = num_renamed::FromPrimitive::from_u8(slice1[16])
            .expect("[Plugin] failed to parse tensor's dtype: {slice1[16]}");

        // create tf::Tensor
        let data = common::bytes_to_f32_vec(data1.as_slice());
        tf::Tensor::new(dims1.as_slice())
            .with_values(data.as_slice())
            .unwrap()
    };
    println!("[Done] (dims: {:?})", tensor_y.dims());

    // * extract epochs
    let epochs = if input[2].ty() == ValType::I32 {
        input[2].to_i32()
    } else {
        return Err(HostFuncError::User(3));
    };
    println!("[Plugin] Epochs: {epochs}");

    train_regression(tensor_x, tensor_y, epochs).unwrap();

    Ok(vec![])
}

fn train_regression<T: tf::TensorType>(
    x: tf::Tensor<T>,
    y: tf::Tensor<T>,
    epochs: i32,
) -> Result<()> {
    println!("\n*** In train_regression ***\n");

    let filename = "examples/tensorflow/regression/model/model.pb"; // y = w * x + b

    // // Generate some test data.

    // let num_points = 100;
    // let num_epoch = 201;
    // let mut x = tf::Tensor::new(&[num_points as u64]);
    // let mut y = tf::Tensor::new(&[num_points as u64]);
    // for i in 0..num_points {
    //     x[i] = (2.0 * rand::random::<f64>() - 1.0) as f32;
    //     y[i] = w * x[i] + b;
    // }

    // Load the computation graph defined by regression.py.
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    let session = Session::new(&SessionOptions::new(), &graph)?;
    let op_x = graph.operation_by_name_required("x")?;
    let op_y = graph.operation_by_name_required("y")?;
    let op_init = graph.operation_by_name_required("init")?;
    let op_train = graph.operation_by_name_required("train")?;
    let op_w = graph.operation_by_name_required("w")?;
    let op_b = graph.operation_by_name_required("b")?;

    // Load the test data into the session.
    let mut init_step = SessionRunArgs::new();
    init_step.add_target(&op_init);
    session.run(&mut init_step)?;

    // Train the model.
    let mut train_step = SessionRunArgs::new();
    train_step.add_feed(&op_x, 0, &x);
    train_step.add_feed(&op_y, 0, &y);
    train_step.add_target(&op_train);
    for _ in 0..=epochs {
        session.run(&mut train_step)?;
    }

    // Grab the data out of the session.
    let mut output_step = SessionRunArgs::new();
    let w_ix = output_step.request_fetch(&op_w, 0);
    let b_ix = output_step.request_fetch(&op_b, 0);
    session.run(&mut output_step)?;

    // Check our results.
    let w_hat: f32 = output_step.fetch(w_ix)?[0];
    let b_hat: f32 = output_step.fetch(b_ix)?[0];
    let w = 0.1;
    let b = 0.3;
    println!(
        "Checking w: expected {}, got {}. {}",
        w,
        w_hat,
        if (w - w_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        }
    );
    println!(
        "Checking b: expected {}, got {}. {}",
        b,
        b_hat,
        if (b - b_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        }
    );

    Ok(())
}

/// Defines Plugin module instance
unsafe extern "C" fn create_test_module(
    _arg1: *const ffi::WasmEdge_ModuleDescriptor,
) -> *mut ffi::WasmEdge_ModuleInstanceContext {
    let module_name = "wasmedge-nn-training";

    // create ImportObject builder
    let io_builder = ImportObjectBuilder::new();

    #[cfg(feature = "torch")]
    let io_builder = io_builder
        .with_func::<(i32, i32, i64, i32, f64, i32, i64, i32, i32, i32, i32), ()>("train", train)
        .expect("failed to create set_dataset host function");

    #[cfg(feature = "tensorflow")]
    let io_builder = io_builder
        .with_func::<(i32, i32, i32), ()>("train", train)
        .expect("failed to create set_dataset host function");

    let import = io_builder
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

#[cfg(feature = "torch")]
pub fn to_tch_tensor(dtype: common::Dtype, dims: &[i64], data: &[u8]) -> tch::Tensor {
    match dtype {
        common::Dtype::F16 => unimplemented!("F16"),
        common::Dtype::F32 => {
            let data = common::bytes_to_f32_vec(data);
            tch::Tensor::of_slice(data.as_slice()).reshape(dims)
        }
        common::Dtype::U8 => Tensor::of_slice(data).reshape(dims),
        common::Dtype::I32 => {
            let data = common::bytes_to_i32_vec(data);
            tch::Tensor::of_slice(data.as_slice()).reshape(dims)
        }
        common::Dtype::I64 => {
            let data = common::bytes_to_i64_vec(data);
            tch::Tensor::of_slice(data.as_slice()).reshape(dims)
        }
    }
}

#[derive(Debug)]
pub struct TFDataset<D: tf::TensorType, T: tf::TensorType> {
    pub train_input_tensors: std::collections::HashMap<String, tf::Tensor<D>>,
    pub train_target_tensors: std::collections::HashMap<String, tf::Tensor<T>>,
}

#[derive(Debug)]
pub struct TFTensor<'a, T: tf::TensorType> {
    pub tensor: tf::Tensor<T>,
    pub name: &'a str,
}
