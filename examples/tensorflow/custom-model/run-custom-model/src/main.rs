mod plugin {
    #[link(wasm_import_module = "wasmedge-nn-training")]
    extern "C" {
        pub fn train(inp_tensor_arr: i32, inp_tensor_arr_len: i32, epochs: i32);
    }
}

use wasmedge_nn_common as common;

fn main() {
    println!("*** run custom_model ***");

    let mut dataset: Vec<&common::Tensor> = vec![];

    println!(
        "size of common::Tensor: {}",
        std::mem::size_of::<common::Tensor>()
    );
    println!(
        "size of common::TensorName: {}",
        std::mem::size_of::<common::TensorName>()
    );
    println!(
        "size of common::TensorElement: {}",
        std::mem::size_of::<common::TensorElement>()
    );

    // train data
    let train_input_data = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let train_input_data_bytes = common::to_bytes(&train_input_data);
    let train_input_dims = [4_u32, 2];
    let train_input_dims_bytes = common::to_bytes(&train_input_dims);
    let train_input_name = "training_input";
    let train_input_tensor = common::Tensor::new(
        train_input_data_bytes,
        train_input_dims_bytes,
        common::Dtype::F32,
        Some(train_input_name),
    );
    dataset.push(&train_input_tensor);

    // target data
    let train_target_data = vec![1.0_f32, 0.0, 1.0, 2.0];
    let train_target_data_bytes = common::to_bytes(&train_target_data);
    let train_target_dims = [4_u32, 1];
    let train_target_dims_bytes = common::to_bytes(&train_target_dims);
    let train_target_name = "training_target";
    let train_target_tensor = common::Tensor::new(
        train_target_data_bytes,
        train_target_dims_bytes,
        common::Dtype::F32,
        Some(train_target_name),
    );
    dataset.push(&train_target_tensor);

    let offset_dataset = dataset.as_ptr() as *const _ as usize as i32;
    let len_dataset = dataset.len() as i32;

    unsafe {
        plugin::train(offset_dataset, len_dataset, 10);
    }
}
