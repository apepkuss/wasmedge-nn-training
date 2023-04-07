mod plugin {
    #[link(wasm_import_module = "wasmedge-nn-training")]
    extern "C" {
        pub fn train(
            builder_arr: i32,
            builder_arr_len: i32,
            inp_tensor_arr: i32,
            inp_tensor_arr_len: i32,
            epochs: i32,
        );
    }
}

use wasmedge_nn_common as common;

fn main() {
    println!("Demo: train a custom model");

    let mut dataset: Vec<&common::Tensor> = vec![];

    // specify the dir the saved model is located in
    let saved_model_dir = "examples/tensorflow/custom-model/custom_model";
    let builder_arr: common::GraphBuilderArray = &[saved_model_dir.as_bytes()];

    // training input tensor
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

    // training target tensor
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

    // training output tensor
    // names of output nodes of the graph can be retrieved with the saved_model_cli command
    let train_output_tensor = common::Tensor::new(&[], &[], common::Dtype::F32, Some("loss"));
    dataset.push(&train_output_tensor);

    let offset_dataset = dataset.as_ptr() as *const _ as usize as i32;
    let len_dataset = dataset.len() as i32;

    unsafe {
        plugin::train(
            builder_arr.as_ptr() as i32,
            builder_arr.len() as i32,
            offset_dataset,
            len_dataset,
            10,
        );
    }
}
