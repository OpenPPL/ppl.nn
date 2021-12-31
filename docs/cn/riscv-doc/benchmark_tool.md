## RISCV Benchmark工具

注：
* 支持fp32和fp16，ONNX模型无需手动转换，框架解析过程会自动转换到fp16精度。
* 暂时仅支持全志D1的编译工具链。
* 暂时仅支持rvv0.71和128bit的向量长度，后续会逐步兼容rvv1.0并支持任意向量长度，欢迎pr~

### 1. 准备工具
需要先从[https://occ.t-head.cn/community/download?id=3913221581316624384](https://occ.t-head.cn/community/download?id=3913221581316624384)下载编译工具链并解压，为后续使用方便，可以将解压后的文件夹路径添加至环境变量。
``` bash
tar -xf riscv64-linux-x86_64-20210512.tar.gz
export RISCV_ROOT_PATH=/path/to/riscv64-linux-x86_64-20210512
```

### 2. 编译
编译pplnn riscv
```
./build.sh -DHPCC_TOOLCHAIN_DIR=$RISCV_ROOT_PATH -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-linux-gnu.cmake -DHPCC_USE_RISCV=ON -DPPLNN_ENABLE_KERNEL_PROFILING=ON -DPPLNN_ENABLE_PYTHON_API=OFF -DPPLNN_ENABLE_LUA_API=OFF -DCMAKE_INSTALL_PREFIX=pplnn-build/install
```
编译后pplnn工具的生成路径为：./pplnn-build/tools/pplnn

### 3. 运行

fp32测试
``` bash
./pplnn --use-riscv --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w [--warmup-iterations m] --enable-profiling
```
`n_c_h_w` 表示实际推理图片大小，使用NCHW排布。

`--warmup-iterations` 代表预热次数。

fp16测试
``` bash
./pplnn --use-riscv --use-fp16 --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w [--warmup-iterations m] --enable-profiling
```
`--use-fp16`表示使用fp16执行，不开启默认使用fp32执行。

## OpenPPL在全志D1上的性能测试

平台信息： 全志 D1

### 1. 单批 RVV 指令集
|  模型名称  |  OpenPPL(fp16)  |  OpenPPL(fp32)  |
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