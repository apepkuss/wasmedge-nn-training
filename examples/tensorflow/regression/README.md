# Training a Linear Regression TensorFlow model on WasmEdge Runtime

In this example, we define a model with TensorFlow Python API, and then train it on WasmEdge Runtime.

## Requirements

- OS: Ubuntu 20.04+ (x86_64)

- Environment for defining TensorFlow model

  - [Install Anaconda/Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

- Install `rustup` and `Rust`

  ```bash
  # install rustup
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

  # add wasm32-wasi target
  rustup target add wasm32-wasi
  ```

- Rust Tensorflow

  [Rust Tensorflow crate](https://crates.io/crates/tensorflow) requires Tensorflow C library. You can refer to the [webpage](https://www.tensorflow.org/install/lang_c) to download and deploy the required version of Tensorflow C library.

- Install WasmEdge Runtime

  This example requires `WasmEdge-0.12.0-alpha.2`. You can follow the steps in this [webpage](https://wasmedge.org/book/en/contribute/build_from_src/linux.html).

## Steps for training

### Step 1: Download the example

```bash
git clone https://github.com/apepkuss/wasmedge-nn-training.git

cd wasmedge-nn-training
```

### Step 2: Run the model script to save the model in the SavedModel format

To run this step, you should create a conda environment and install tensorflow in this environment first:

```bash
# create a conda environment with python support
conda create -n tf2 python=3.10

# install tensorflow-cpu in this environment
conda install -c conda-forge tensorflow-cpu

# activate the environment
conda activate tf2
```

Now you can run the model script:

```bash
cd examples/tensorflow/regression

# run the model script
python3 regression.py
```

If the script runs sucessfully, a new directory named `regression_savedmodel` can be found.


### Step 3: Build and deploy `wasmedge-nn-training` plugin

In the root directory of the repo, run the following command to build `wasmedge-nn-training` plugin:

```bash
cargo build -p wasmedge-nn-training --release --features tensorflow
```

If the command runs successfully, `libwasmedge_nn_training.so` can be found in the directory `target/release/`.

Then, the plugin library file should be copied to the default plugin directory of WasmeEdge Runtime:

```bash
# Assume that WasmEdge library is installed in `/usr/local/lib`
cp target/release/libwasmedge_nn_training.so /usr/local/lib/wasmedge
```

### Step 4: Build `regression-tf` wasm app

```bash
cargo build -p regression-tf --target wasm32-wasi --release
```

If the command runs successfully, `regression-tf.wasm` can be found in the directory `target/wasm32-wasi/release/`.

### Final step: Train the linear regression model on WasmEdge Runtime

```bash
wasmedge --dir .:. target/wasm32-wasi/release/regression-tf.wasm
```

If the command runs successfully, the following info can be seen:

```bash
Demo: train a linear regression model

*** Welcome! This is `wasmedge-nn-training` plugin. ***

[Plugin] Preparing model ... [Done]
[Plugin] Preparing training tensor ... [Done]
[Plugin] Preparing target tensor ... [Done]
[Plugin] Preparing output tensor ... [Done]
[Plugin] Epochs: 20

*** In train_regression ***

[Plugin] Training model...
        Epoch[0]: loss: 0.15144789  
        Epoch[1]: loss: 0.02349366  
        Epoch[2]: loss: 0.01296371  
        Epoch[3]: loss: 0.00715378  
        Epoch[4]: loss: 0.00394767  
        Epoch[5]: loss: 0.00217845  
        Epoch[6]: loss: 0.00120213  
        Epoch[7]: loss: 0.00066337  
        Epoch[8]: loss: 0.00036607  
        Epoch[9]: loss: 0.00020201  
        Epoch[10]: loss: 0.00011147  
        Epoch[11]: loss: 0.00006152  
        Epoch[12]: loss: 0.00003395  
        Epoch[13]: loss: 0.00001873  
        Epoch[14]: loss: 0.00001034  
        Epoch[15]: loss: 0.00000570  
        Epoch[16]: loss: 0.00000315  
        Epoch[17]: loss: 0.00000174  
        Epoch[18]: loss: 0.00000096  
        Epoch[19]: loss: 0.00000053  
        Epoch[20]: loss: 0.00000029  
[Done]
```
