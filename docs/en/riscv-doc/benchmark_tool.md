## RISCV Benchmark tool

Note:
* Openppl.riscv currently only supports fp32 and fp16. The ONNX model does not need to be manually converted, and it will be done in inference process.
* Currently only the compilation tool chain of Allwinner D1 is supported.
* For the riscv vector instructions, currently PPLNN only supports the 0.71 version of rvv and 128bit vector length. In the future, it will gradually be compatible with rvv1.0 and support any vector length.

### 1. Preparation tools
You need to download c906 toolchain package from [https://occ.t-head.cn/community/download?id=3913221581316624384](https://occ.t-head.cn/community/download?id=3913221581316624384).
``` bash
tar -xf riscv64-linux-x86_64-20210512.tar.gz
export RISCV_ROOT_PATH=/path/to/riscv64-linux-x86_64-20210512
```

### 2. Compile

```
./build.sh -DPPLNN_TOOLCHAIN_DIR=$RISCV_ROOT_PATH -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-linux-gnu.cmake -DPPLNN_USE_RISCV64=ON -DPPLNN_ENABLE_KERNEL_PROFILING=ON -DPPLNN_ENABLE_PYTHON_API=OFF -DPPLNN_ENABLE_LUA_API=OFF -DCMAKE_INSTALL_PREFIX=pplnn-build/install
```
pplnn will be generated to: ./pplnn-build/tools/pplnn

### 3. Run

fp32 test
``` bash
./pplnn --use-riscv --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w [--warmup-iterations m] --enable-profiling
```
`n_c_h_w` indicates the actual input image size, and uses NCHW layout.

`--warmup-iterations` represents the number of warm-ups.

fp16 test
``` bash
./pplnn --use-riscv --use-fp16 --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w [--warmup-iterations m] --enable-profiling
```
`--use-fp16` means to use fp16 to execute, and the default is fp32 if it is not enabled.

## OpenPPL performance test on Allwinner D1

Platform Information: AllWinner D1

### 1. 1-batch RVV
|  Model Name  |  OpenPPL(fp16)  |  OpenPPL(fp32)  |
| :----------- | :------------ | :------------ |
|  alexnet  |  576.39  |    |
|  resnet18  |  748.20  |  1642.88  |
|  resnet34  |  1283.38  |  2713.80  |
|  resnet50  |  2672.72  |  5919.94  |
|  resnet101  |  4724.11  |  10061.33  |
|  resnet152  |  6783.49  |    |
|  resnext50_32x4d  |  3861.48  |  7447.43  |
|  shufflenet_v2_x0_5  |  55.81  |  116.40  |
|  shufflenet_v2_x1_0  |  144.84  |  319.82  |
|  shufflenet_v2_x1_5  |  271.91  |  590.50  |
|  shufflenet_v2_x2_0  |  548.06  |  1094.72  |
|  squeezenet1_0  |  584.26  |  905.23  |
|  squeezenet1_1  |  256.84  |  472.68  |
|  mobilenet_v2  |  283.18  |  667.04  |
|  mnasnet0_5  |  206.19  |  354.56  |
|  mnasnet0_75  |    |  609.69  |
|  mnasnet1_0  |  467.42  |  816.04  |
|  mnasnet1_3  |    |  1235.82  |
