# LeNet 300 100 with Custom CUDA Kernel - Assignment 2

## Usage

To train LeNet with CUDA kernel run: `ptrhon main.py --kernel Custom` command. The --kernel flag is Custom for cuda or PyTorch for torch layers.
 
For compiling the kernel, enter the following command in the `longformer_util/deep-codegen` directory:
```bash
mkdir build && cd build
cmake ..
make -j
cp graphpy.cpython-38-x86_64-linux-gnu.so ../
```
