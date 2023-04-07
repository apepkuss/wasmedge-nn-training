# Training Resnet Model (PyTorch)

In this example, we define Resnet model with PyTorch Python API, and then train it on WasmEdge Runtime.

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

- tch

  [tch crate](https://crates.io/crates/tensorflow) requires `libtorch`. You can refer to the [webpage](https://crates.io/crates/tch) to install it.

- Install WasmEdge Runtime

  This example requires `WasmEdge-0.12.0-alpha.2`. You can follow the steps in this [webpage](https://wasmedge.org/book/en/contribute/build_from_src/linux.html).


## Steps for training

### Step 1: Download the example

```bash
git clone https://github.com/apepkuss/wasmedge-nn-training.git

cd wasmedge-nn-training
```

### Step 2: Run the model script to save the model in the SavedModel format

To run this step, you should create a conda environment and install PyTorch in this environment first:

```bash
# create a conda environment with python support
conda create -n torch python=3.10

# install pytorch-cpu in this environment
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# activate the environment
conda activate torch
```

Now you can run the model script:

```bash
cd examples/pytorch/resnet-pytorch

# run the model script
python3 resnet.py
```

If the script runs sucessfully, `trained_model.py` can be found in the directory `examples/pytorch/resnet-pytorch`.


### Step 3: Build and deploy `wasmedge-nn-training` plugin

In the root directory of the repo, run the following command to build `wasmedge-nn-training` plugin:

```bash
cargo build -p wasmedge-nn-training --release --features torch
```

If the command runs successfully, `libwasmedge_nn_training.so` can be found in the directory `target/release/`.

Then, the plugin library file should be copied to the default plugin directory of WasmeEdge Runtime:

```bash
# Assume that WasmEdge library is installed in `/usr/local/lib`
cp target/release/libwasmedge_nn_training.so /usr/local/lib/wasmedge
```

### Step 4: Build `resnet-pytorch` wasm app

```bash
cargo build -p resnet-pytorch --target wasm32-wasi --release
```

If the command runs successfully, `resnet-pytorch.wasm` can be found in the directory `target/wasm32-wasi/release/`.

### Final step: Train the custom model on WasmEdge Runtime

```bash
wasmedge --dir .:. target/wasm32-wasi/release/resnet-pytorch.wasm examples/pytorch/resnet-pytorch/trained_model.pt
```

If the command runs successfully, the following info can be seen:

```bash
root@665815af3427:~/workspace/wasmedge-nn-training# wasmedge --dir .:. target/wasm32-wasi/release/resnet-pytorch.wasm examples/pytorch/resnet-pytorch/trained_model.pt 
[Wasm] Preparing training images ... [Done]
[Wasm] Preparing training labels ... [Done]
[Wasm] Preparing test images ... [Done]
[Wasm] Preparing test lables ... [Done]

*** Welcome! This is `wasmedge-nn-training` plugin. ***

[Plugin] Preparing train images ... [Done] (shape: [60000, 1, 28, 28], dtype: Float)
[Plugin] Preparing train labels ... [Done] (shape: [60000], dtype: Int64)
[Plugin] Preparing test images ... [Done] (shape: [10000, 1, 28, 28], dtype: Float)
[Plugin] Preparing test labels ... [Done] (shape: [10000], dtype: Int64)
[Plugin] Labels: 10
[Plugin] Device: Cpu
[Plugin] Learning rate: 0.0001
[Plugin] Epochs: 10
[Plugin] batch size: 128
[Plugin] Optimizer: Adam
[Plugin] Loss function: CrossEntropyForLogits
[Plugin] Load model
[Plugin] Initial accuracy: 91.40%
[Plugin] Start training ... 
        epoch:    1 test acc: 91.56%
        epoch:    2 test acc: 91.68%
        epoch:    3 test acc: 91.43%
        epoch:    4 test acc: 91.36%
        epoch:    5 test acc: 91.43%
        epoch:    6 test acc: 91.37%
        epoch:    7 test acc: 91.59%
        epoch:    8 test acc: 91.62%
        epoch:    9 test acc: 91.70%
        epoch:   10 test acc: 91.62%
[Plugin] Finished
[Plugin] The pre-trained model is dumped to "/root/workspace/wasmedge-nn-training/examples/pytorch/resnet-pytorch/trained_trained_model.pt"
```



## Define a PyTorch model

The following Python script shows the content of `resnet.py`, which defines a `Resnet` model with PyTorch Python API. 

```python
import torch
from torch.nn import Module


class DemoModule(Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(5, 5), padding=(2, 2))
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(16 * 28 * 28, 100)
        self.linear2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return self.linear2(x)


traced_script_module = torch.jit.script(DemoModule())
# dump the model for training.
traced_script_module.save("model.pt")
```

Run the following command to dump the model to the `model.pt` file:

```bash
python3 resnet.py
```

## Build and deploy the `wasmedge-nn-training` plugin

> We assume you have already installed WasmEdge runtime in your local environment by following the steps in [WasmEdge Installation](https://wasmedge.org/book/en/quick_start/install.html#wasmedge-installation-and-uninstallation). And the installation directory is `/usr/local/`.

Follow the command lines below to build and deploy the `wasmedge-nn-training` plugin:
```bash
cargo build -p wasmedge-nn-training --release

// copy the plugin library into the wasmedge plugin directory
cp ./target/release/libwasmedge_nn_training.so /usr/local/lib/wasmedge
```

## Define and build wasm app

The wasm app defined in `resnet-pytorch` is responsible for loading images from a specified location, preprocessing the image data, and splitting the data into three parts for training, testing and validation. Finally, the wasm app calls the `train` interface exported by the external module *plugin* which is powered by WasmEdge wasi-nn-training plugin. 

Run the following command in the terminal program to build the wasm app:

```bash
cargo build -p resnet-pytorch --target wasm32-wasi --release
```

## Train

```bash
wasmedge --dir .:. target/wasm32-wasi/release/test-tch-backend.wasm
```

- The complete training process

  ```bash
  [Wasm] Preparing training images ... [Done]
  [Wasm] Preparing training labels ... [Done]
  [Wasm] Preparing test images ... [Done]
  [Wasm] Preparing test lables ... [Done]
  
  *** Welcome! This is `wasmedge-nn-training` plugin. ***
  
  [Plugin] Preparing train images ... [Done] (shape: [60000, 1, 28, 28], dtype: Float)
  [Plugin] Preparing train labels ... [Done] (shape: [60000], dtype: Int64)
  [Plugin] Preparing test images ... [Done] (shape: [10000, 1, 28, 28], dtype: Float)
  [Plugin] Preparing test labels ... [Done] (shape: [10000], dtype: Int64)
  [Plugin] Labels: 10
  [Plugin] Device: Cpu
  [Plugin] Learning rate: 0.0001
  [Plugin] Epochs: 10
  [Plugin] batch size: 128
  [Plugin] Optimizer: Adam
  [Plugin] Loss function: CrossEntropyForLogits
  [Plugin] Load model
  [Plugin] Initial accuracy:  9.47%
  [Plugin] Start training ... 
          epoch:    1 test acc: 87.18%
          epoch:    2 test acc: 89.36%
          epoch:    3 test acc: 90.07%
          epoch:    4 test acc: 90.55%
          epoch:    5 test acc: 90.68%
          epoch:    6 test acc: 91.28%
          epoch:    7 test acc: 91.00%
          epoch:    8 test acc: 91.39%
          epoch:    9 test acc: 91.47%
          epoch:   10 test acc: 91.61%
  [Plugin] Finished
  [Plugin] The pre-trained model is dumped to "/root/workspace/wasmedge-nn-training/examples/pytorch/resnet-pytorch/trained_model.pt"
  ```


