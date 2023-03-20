use anyhow::Result;
use std::io::{self, Write};

extern crate num as num_renamed;
#[macro_use]
extern crate num_derive;

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
    assert_eq!(input.len(), 9);

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
        .data_pointer(offset_tensors as u32, protocol::SIZE_OF_TENSOR_ARRAY)
        .expect("failed to get data from linear memory");
    // println!("[plugin] ptr_tensor: {:p}", ptr_tensors);
    let slice = unsafe {
        std::slice::from_raw_parts(
            ptr_tensors,
            protocol::SIZE_OF_TENSOR_ELEMENT as usize * len_tensors as usize,
        )
    };
    // println!("[Plugin] len of slice: {}", slice.len());

    // * extract train_images

    print!("[Plugin] Preparing train images ... ");
    io::stdout().flush().unwrap();
    let train_images: Tensor = {
        // * extract tenor1
        let offset1 = i32::from_le_bytes(slice[0..4].try_into().unwrap());
        let slice1 = memory
            .read(offset1 as u32, protocol::SIZE_OF_TENSOR)
            .unwrap();

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
        let dims1: Vec<i64> = protocol::bytes_to_u32_vec(dims1.as_slice())
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
        let slice1 = memory
            .read(offset1 as u32, protocol::SIZE_OF_TENSOR)
            .unwrap();

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
        let dims1: Vec<i64> = protocol::bytes_to_u32_vec(dims1.as_slice())
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
        let slice1 = memory
            .read(offset1 as u32, protocol::SIZE_OF_TENSOR)
            .unwrap();

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
        let dims1: Vec<i64> = protocol::bytes_to_u32_vec(dims1.as_slice())
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
        let slice1 = memory
            .read(offset1 as u32, protocol::SIZE_OF_TENSOR)
            .unwrap();

        // parse tensor4's data
        let offset_data1 = i32::from_le_bytes(slice1[0..4].try_into().unwrap());
        let len_data1 = i32::from_le_bytes(slice1[4..8].try_into().unwrap());
        let data1 = memory.read(offset_data1 as u32, len_data1 as u32).unwrap();

        // parse tensor4's dimensions
        let offset_dims1 = i32::from_le_bytes(slice1[8..12].try_into().unwrap());
        let len_dims1 = i32::from_le_bytes(slice1[12..16].try_into().unwrap());
        // let dims1 = memory
        //     .read(offset_dims1 as u32, len_dims1 as u32)
        //     .expect("failed to read memory");
        // let dims1: Vec<i64> = protocol::bytes_to_u32_vec(dims1.as_slice())
        //     .into_iter()
        //     .map(i64::from)
        //     .collect();
        let dims1 = memory
            .read(offset_dims1 as u32, len_dims1 as u32)
            .expect("plugin: test_labels: faied to extract tensor dimensions");
        let dims1: Vec<i64> = protocol::bytes_to_i32_vec(dims1.as_slice())
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
    let optimizer: protocol::Optimizer = num_renamed::FromPrimitive::from_i32(optimizer)
        .expect("[Plugin] failed to parse optimizer");
    println!("[Plugin] Optimizer: {:?}", optimizer);

    // loss function
    let loss_fn = if input[8].ty() == ValType::I32 {
        input[8].to_i32()
    } else {
        return Err(HostFuncError::User(9));
    };
    let loss_fn: protocol::LossFunction = num_renamed::FromPrimitive::from_i32(loss_fn)
        .expect("[Plugin] failed to parse loss function");
    println!("[Plugin] Loss function: {:?}", loss_fn);

    // start training
    train_model(ds, device, lr, epochs, batch_size, optimizer, loss_fn)
        .expect("failed to train model");

    Ok(vec![])
}

fn train_model(
    dataset: Dataset,
    device: Device,
    lr: f64,
    epochs: i32,
    batch_size: i64,
    optimizer: protocol::Optimizer,
    loss_fn: protocol::LossFunction,
) -> Result<()> {
    let module_path = "/root/workspace/wasi-nn-training/model.pt";

    let vs = VarStore::new(device);
    let mut trainable = TrainableCModule::load(module_path, vs.root())?;
    trainable.set_train();

    let initial_acc = trainable.batch_accuracy_for_logits(
        &dataset.test_images,
        &dataset.test_labels,
        vs.device(),
        1024,
    );
    println!("[Plugin] Initial accuracy: {:5.2}%", 100. * initial_acc);

    let mut opt = match optimizer {
        protocol::Optimizer::Adam => Adam::default()
            .build(&vs, lr)
            .expect("[Train] failed to create tch::Adam optimizer"),
        protocol::Optimizer::RmsProp => RmsProp::default()
            .build(&vs, lr)
            .expect("[Train] failed to create tch::RmsProp optimizer"),
        protocol::Optimizer::Sgd => Sgd::default()
            .build(&vs, lr)
            .expect("[Train] failed to create tch::Sgd optimizer"),
    };

    println!("[Plugin] Start training ... ");
    // let mut opt = Adam::default().build(&vs, lr).expect("[Train] optimizer");
    for epoch in 1..epochs {
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

    trainable.save("trained_model.pt")?;
    println!("[Plugin] The pre-trained model is dumped to `trained_model.pt`");

    Ok(())
}

// A native function to be wrapped as a host function
#[host_function]
fn real_add(_: Caller, input: Vec<WasmValue>) -> Result<Vec<WasmValue>, HostFuncError> {
    println!("Welcome! This is NaiveMath plugin.");

    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();

    if input.len() != 2 {
        return Err(HostFuncError::User(1));
    }

    let a = if input[0].ty() == ValType::I32 {
        input[0].to_i32()
    } else {
        return Err(HostFuncError::User(2));
    };

    let b = if input[1].ty() == ValType::I32 {
        input[1].to_i32()
    } else {
        return Err(HostFuncError::User(3));
    };

    let c = a + b;
    Ok(vec![WasmValue::from_i32(c)])
}

#[host_function]
fn set_input_tensor(
    caller: Caller,
    input: Vec<WasmValue>,
) -> Result<Vec<WasmValue>, HostFuncError> {
    println!("*** This is `set_input_tensor` host function ****");

    let memory = caller.memory(0).expect("failed to get memory at idex 0");

    if input.len() != 5 {
        return Err(HostFuncError::User(1));
    }

    {
        println!(
            "[plugin] size of GraphBuilder: {}",
            std::mem::size_of::<protocol::GraphBuilder>()
        );
        println!(
            "[plugin] size of GraphBuilderArray: {}",
            std::mem::size_of::<protocol::GraphBuilderArray>()
        );

        let data1_offset = if input[0].ty() == ValType::I32 {
            input[0].to_i32()
        } else {
            return Err(HostFuncError::User(2));
        };
        println!("data1_offset: {data1_offset}");

        let data1_size = if input[1].ty() == ValType::I32 {
            input[1].to_i32()
        } else {
            return Err(HostFuncError::User(2));
        };
        println!("data1_size: {data1_size}");

        let data = memory
            .data_pointer(data1_offset as u32, 8)
            .expect("failed to get data from linear memory");
        println!(
            "[plugin] size of *const: {}",
            std::mem::size_of::<*const u8>()
        );
        println!("[plugin] data ptr: {:p}", data);

        let slice = unsafe { std::slice::from_raw_parts(data, 8 * 2) };
        println!("slice1: {:?}", slice);

        // extract the first (offset, size) from the linear memory
        let offset1 = i32::from_le_bytes(slice[0..4].try_into().unwrap());
        let size1 = i32::from_le_bytes(slice[4..8].try_into().unwrap());
        println!("offset1: {}, size1: {}", offset1, size1);
        // extract the first sequence of numbers from the linear memory by (offset, size)
        let num1 = memory
            .read(offset1 as u32, size1 as u32)
            .expect("failed to read numbers");
        println!("num1: {:?}", num1);

        let offset2 = i32::from_le_bytes(slice[8..12].try_into().unwrap());
        let size2 = i32::from_le_bytes(slice[12..16].try_into().unwrap());
        println!("offset2: {offset2}, size2: {size2}");
        let num2 = memory
            .read(offset2 as u32, size2 as u32)
            .expect("failed to read numbers");
        println!("num2: {:?}", num2);
    }

    // ======= Single Tensor

    {
        // let offset_tensor = if input[2].ty() == ValType::I32 {
        //     input[2].to_i32()
        // } else {
        //     return Err(HostFuncError::User(3));
        // };
        // println!("[plugin] offset_tensor: {offset_tensor}");

        // let len_tensor = if input[3].ty() == ValType::I32 {
        //     input[3].to_i32()
        // } else {
        //     return Err(HostFuncError::User(4));
        // };
        // println!("[plugin] len_tensor: {len_tensor}");

        // let ptr_tensor = memory
        //     .data_pointer(offset_tensor as u32, 8)
        //     .expect("failed to get data from linear memory");
        // println!("[plugin] ptr_tensor: {:p}", ptr_tensor); // 0x7f12de413f20

        // let slice = unsafe { std::slice::from_raw_parts(ptr_tensor, 20) };

        // let offset_dims = i32::from_le_bytes(slice[0..4].try_into().unwrap());
        // let size_dims = i32::from_le_bytes(slice[4..8].try_into().unwrap());
        // let dims = memory
        //     .read(offset_dims as u32, size_dims as u32)
        //     .expect("failed to read tensor dims");
        // let dims = protocol::bytes_to_i32_vec(dims.as_slice());
        // println!("[plugin] tensor dims: {:?}", dims);

        // let offset_data = i32::from_le_bytes(slice[8..12].try_into().unwrap());
        // let size_data = i32::from_le_bytes(slice[12..16].try_into().unwrap());
        // println!("({offset_data}, {size_data})");
        // let nums = memory
        //     .read(offset_data as u32, size_data as u32)
        //     .expect("failed to read data");
        // println!("[plugin] nums: {:?}", nums);

        // let ty = slice[16];
        // println!("dtype: {ty}");

        // println!("tensor ty: {:?}", slice[17..20].as_ref());
    }

    println!(
        "[plugin] size of TensorElement: {}",
        std::mem::size_of::<protocol::TensorElement>()
    );
    println!(
        "[plugin] size of TensorArray: {}",
        std::mem::size_of::<protocol::TensorArray>()
    );

    let offset_tensors = if input[2].ty() == ValType::I32 {
        input[2].to_i32()
    } else {
        return Err(HostFuncError::User(3));
    };
    println!("[plugin] offset_tensors: {offset_tensors}");

    let len_tensors = if input[3].ty() == ValType::I32 {
        input[3].to_i32()
    } else {
        return Err(HostFuncError::User(4));
    };
    println!("[plugin] len_tensors: {len_tensors}");

    // ==== Tensor

    let ptr_tensors = memory
        .data_pointer(offset_tensors as u32, 4)
        .expect("failed to get data from linear memory");
    println!("[plugin] ptr_tensor: {:p}", ptr_tensors);

    let slice = unsafe {
        std::slice::from_raw_parts(
            ptr_tensors,
            protocol::SIZE_OF_TENSOR_ELEMENT as usize * len_tensors as usize,
        )
    };
    println!("[plugin] slice: {:?}", slice);

    // * extract tenor1
    let offset1 = i32::from_le_bytes(slice[0..4].try_into().unwrap());
    println!("[plugin] offset1: {offset1}");
    let slice1 = memory
        .read(offset1 as u32, protocol::SIZE_OF_TENSOR)
        .unwrap();
    println!("slice1: {:?}", slice1);

    // parse tensor1's dimensions
    let offset_dims1 = i32::from_le_bytes(slice1[0..4].try_into().unwrap());
    let len_dims1 = i32::from_le_bytes(slice1[4..8].try_into().unwrap());
    let dims1 = memory
        .read(offset_dims1 as u32, len_dims1 as u32)
        .expect("failed to read memory");
    let dims1 = protocol::bytes_to_i32_vec(dims1.as_slice());
    println!("[plugin] dims1: {:?}", dims1);

    // parse tensor1's data
    let offset_data1 = i32::from_le_bytes(slice1[8..12].try_into().unwrap());
    let len_data1 = i32::from_le_bytes(slice1[12..16].try_into().unwrap());
    let data1 = memory.read(offset_data1 as u32, len_data1 as u32).unwrap();
    println!("[plugin] data1: {:?}", data1);

    // parse tensor1's type
    let dtype1 = slice1[16];
    println!("[plugin] dtype1: {dtype1}");

    // * extract tensor2
    let offset2 = i32::from_le_bytes(slice[4..8].try_into().unwrap());
    println!("[plugin] offset2: {offset2}");
    let slice2 = memory
        .read(offset2 as u32, protocol::SIZE_OF_TENSOR)
        .unwrap();
    println!("slice2: {:?}", slice2);

    // parse tensor2's dimensions
    let offset_dims2 = i32::from_le_bytes(slice2[0..4].try_into().unwrap());
    let len_dims2 = i32::from_le_bytes(slice2[4..8].try_into().unwrap());
    let dims2 = memory
        .read(offset_dims2 as u32, len_dims2 as u32)
        .expect("failed to read memory");
    let dims2 = protocol::bytes_to_i32_vec(dims2.as_slice());
    println!("[plugin] dims2: {:?}", dims2);

    // parse tensor2's data
    let offset_data2 = i32::from_le_bytes(slice2[8..12].try_into().unwrap());
    let len_data2 = i32::from_le_bytes(slice2[12..16].try_into().unwrap());
    let data2 = memory.read(offset_data2 as u32, len_data2 as u32).unwrap();
    println!("[plugin] data2: {:?}", data2);

    // parse tensor2's type
    let dtype2 = slice2[16];
    println!("[plugin] dtype2: {dtype2}");

    Ok(vec![])
}

/// Defines Plugin module instance
unsafe extern "C" fn create_test_module(
    _arg1: *const ffi::WasmEdge_ModuleDescriptor,
) -> *mut ffi::WasmEdge_ModuleInstanceContext {
    let module_name = "naive-math";
    let import = ImportObjectBuilder::new()
        // add a function
        .with_func::<(i32, i32), i32>("add", real_add)
        .expect("failed to create host function: add")
        .with_func::<(i32, i32, i32, i32, i32), ()>("set_input_tensor", set_input_tensor)
        .expect("failed to create host function: set_input_tensor")
        .with_func::<(i32, i32, i64, i32, f64, i32, i64, i32, i32), ()>("train", train)
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
    let name = "naive_math_plugin";
    let desc = "this is naive math plugin";
    let version = PluginVersion::new(0, 0, 0, 0);
    let plugin_descriptor = PluginDescriptor::new(name, desc, version)
        .expect("Failed to create plugin descriptor")
        .add_module_descriptor(
            "naive_math_module",
            "this is naive math module",
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
            let data = protocol::bytes_to_f32_vec(data);
            Tensor::of_slice(data.as_slice()).reshape(dims)
        }
        2 => Tensor::of_slice(data).reshape(dims),
        3 => {
            let data = protocol::bytes_to_i32_vec(data);
            Tensor::of_slice(data.as_slice()).reshape(dims)
        }
        4 => {
            let data = protocol::bytes_to_i64_vec(data);
            Tensor::of_slice(data.as_slice()).reshape(dims)
        }
        _ => panic!("plugin: train_images: unsupported dtype: {dtype}"),
    }
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
    #[derive(Copy, Clone, Debug)]
    pub struct Array {
        pub data: *const u8,
        pub size: i32,
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

    pub const SIZE_OF_TENSOR: u32 = 20;
    pub const SIZE_OF_TENSOR_ELEMENT: u32 = 4;
    pub const SIZE_OF_TENSOR_ARRAY: u32 = 8;

    pub type TensorElement<'a> = &'a Tensor<'a>;
    pub type TensorArray<'a> = &'a [TensorElement<'a>];

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

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct Tensor<'a> {
        pub dimensions: TensorDimensions<'a>,
        pub type_: TensorType,
        pub data: TensorData<'a>,
    }

    pub type TensorData<'a> = &'a [u8];
    pub type TensorDimensions<'a> = &'a [u32];

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

    pub type GraphBuilder<'a> = &'a [u8];
    pub type GraphBuilderArray<'a> = &'a [GraphBuilder<'a>];
}
