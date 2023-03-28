use wasmedge_nn_common as common;

mod plugin {
    #[link(wasm_import_module = "wasmedge-nn-training")]
    extern "C" {
        pub fn train(inp_tensor_arr: i32, inp_tensor_arr_len: i32, epochs: i32);
    }
}

const NUM_POINTS: usize = 100;

fn main() {
    println!("Demo: train a linear regression model");

    // Generate some test data.
    let w = 0.1;
    let b = 0.3;
    let mut x = vec![0f32; NUM_POINTS];
    let mut y = vec![0f32; NUM_POINTS];
    for i in 0..NUM_POINTS {
        x[i] = (2.0 * rand::random::<f64>() - 1.0) as f32;
        y[i] = w * x[i] + b;
    }

    let mut dataset: Vec<&common::Tensor> = vec![];

    let data_x = common::to_bytes(&x);
    let dims_x = [100_u32];
    let dims_x_bytes = common::to_bytes(&dims_x);
    let tensor_x = common::Tensor {
        data: data_x,
        dimensions: dims_x_bytes,
        dtype: common::Dtype::F32,
    };
    dataset.push(&tensor_x);

    let data_y = common::to_bytes(&y);
    let dims_y = [100_u32];
    let dims_y_bytes = common::to_bytes(&dims_y);
    let tensor_y = common::Tensor {
        data: data_y,
        dimensions: dims_y_bytes,
        dtype: common::Dtype::F32,
    };
    dataset.push(&tensor_y);

    let offset_dataset = dataset.as_ptr() as *const _ as usize as i32;
    let len_dataset = dataset.len() as i32;

    unsafe {
        plugin::train(offset_dataset, len_dataset, 200);
    }
}
