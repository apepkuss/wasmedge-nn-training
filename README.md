# wasmedge-nn-training

> This is an experimental project, and it is still in the active development. 

The goal of this project is to explore the feasibility of providing AI training capability based on WasmEdge Plugin mechanism. This project consists of three parts:

- `resnet.py` is a Python script defining a `Resnet` model with PyTorch Python API.

- `wasmedge-nn-training` constructs a plugin prototype integrated with PyTorch.

- `resnet-pytorch` is a wasm app that is responsible for preparing data and calling the `train` interface to trigger a training task.

## Requirements

- OS: Ubuntu 20.04+

- `Rust`

  Go to the [official Rust webpage](https://www.rust-lang.org/tools/install) and follow the instructions to install `rustup` and `Rust`.

  > It is recommended to use Rust 1.63 or above in the stable channel.

  Then, add `wasm32-wasi` target to the Rustup toolchain:

  ```bash
  rustup target add wasm32-wasi
  ```

- WasmEdge Runtime

  Refer to the [Quick Install](https://wasmedge.org/book/en/quick_start/install.html#quick-install) section of WasmEdge Runtime Book to install `libwasmedge`.

- Mnist image data

    The Mnist image data is located in the `data` directory of this repo.

- Install libtorch

    Reference [Libtorch Manual Install](https://github.com/LaurentMazare/tch-rs#libtorch-manual-install)

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
  [Plugin] Initial accuracy:  9.31%
  [Plugin] Start training ... 
          epoch:    1 test acc: 87.31%
          epoch:    2 test acc: 89.51%
          epoch:    3 test acc: 90.43%
          epoch:    4 test acc: 90.65%
          epoch:    5 test acc: 90.84%
          epoch:    6 test acc: 91.18%
          epoch:    7 test acc: 91.32%
          epoch:    8 test acc: 91.11%
          epoch:    9 test acc: 91.45%
          epoch:   10 test acc: 91.29%
  [Plugin] Finished
  [Plugin] The pre-trained model is dumped to `trained_model.pt`
  ```


