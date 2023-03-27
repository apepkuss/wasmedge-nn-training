mod plugin {
    #[link(wasm_import_module = "wasmedge-nn-training")]
    extern "C" {
        pub fn train();
    }
}

fn main() {
    println!("*** run custom_model ***");

    unsafe {
        plugin::train();
    }
}
