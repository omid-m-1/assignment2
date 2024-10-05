# LeNet 300 100 with Custom CUDA Kernel - Assignment 2

## Usage

Train LeNet with CUDA kernel: `ptrhon main.py --kernel Custom`
 
For compiling the kernel, enter the following command in the `longformer_util/deep-codegen` directory:
```bash
mkdir build && cd build
cmake ..
make -j
cp graphpy.cpython-38-x86_64-linux-gnu.so ../
```
