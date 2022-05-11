## ARM Benchmark 工具

ARM架构性能测试使用pplnn工具。

本章仅介绍ARM架构下使用pplnn测速的方法，其他架构请参考对应目录下的文档。

### 1. 编译

pplnn工具编译请参考[building-from-source.md](../../en/building-from-source.md)。

ARM架构使用openmp作为线程池。若需要测试多线程性能，编译时请指定`-DPPLNN_USE_OPENMP=ON`。

如要测试 FP16 精度推理的性能，编译时需指定`-DPPLNN_USE_ARMV8_2=ON`（编译器需支持`armv8.2-a`指令集）。

```bash
./build.sh -DPPLNN_USE_AARCH64=ON -DPPLNN_USE_OPENMP=ON -DPPLNN_USE_ARMV8_2=ON
```

如果测试的机器有多个 NUMA 节点，建议编译时指定`-DPPLNN_USE_NUMA=ON`（需确保系统中已安装`libnuma`库）。

编译后pplnn工具的生成路径为：./pplnn-build/tools/pplnn

### 2. 准备测试数据

pplnn既可以生成随机数据，也可以从外部读入数据作为网络输入：

* 对于分类/语义分割等网络，网络的执行速度与输入数据的数值无关，测速时指定生成随机数据即可。
* 对于检测/实例分割等网络，网络的执行速度有可能会与输入数据的数值有关，因此建议从外部读入真实数据测试。

#### 2.1. 外部数据格式要求

当需要从外部读入测速数据时，推荐使用`--reshaped-inputs`选项来指定外部数据。

在此选项下，pplnn要求测试数据文件为二进制格式(可以用numpy的tofile函数将数据存为二进制格式)，网络中每个输入tensor需要单独存一个数据文件，文件命名方式为：

```
<tensor_name>-<input_shape>-<data_type>.dat
```

* \<tensor_name\>：对应onnx模型中输入tensor的名称，如：input
* \<input_shape\>：模型输入tensor的shape，以'\_'为分隔符，如：1_3_224_224
* \<data_type\>：文件的数据类型，目前支持fp64|fp32|fp16|int32|int64|bool

例如onnx模型中输入tensor的名称为input，shape为(1,3,224,224)，数据类型为float32，则数据文件命名应当为：

```
input-1_3_224_224-fp32.dat
```

### 3. 运行pplnn

#### 3.1. pplnn运行选项

pplnn中，与ARM架构测速相关的运行选项有：

* `--use-arm`：指定使用 ARM
* `--onnx-model`：指定onnx模型文件
* `--reshaped-inputs`：指定外部数据，格式要求上文已阐述
* `--mm-policy`：内存管理策略，mem代表更少的内存使用，perf代表更激进的内存优化，默认为mem
* `--enable-profiling`：使能测速，默认为不使能
* `--min-profiling-seconds`：指定测速的最少持续时间，单位为秒，默认为1s
* `--warmup-iterations`：指定warm up的次数，默认为0
* `--perf-with-io`：测速时将数据的输入输出耗时一并计入
* `--use-fp16`：启用fp16推理，模型中所有fp32精度的推理会用fp16代替
* `--wg-level`：winograd优化等级，0代表禁用winograd，1代表自动选择winograd block大小，2代表尽可能使用winograd block 2，3代表尽可能使用winograd block 4
* `--tuning-level`：算法选择模式，0代表静态算法选择，1代表动态算法选择（性能更好，初始化时间更长）
* `--numa-node-id`：arm engine运行时绑定的NUMA节点id，-1代表不绑定，只有在机器本身存在多个NUMA节点，且编译时开启了NUMA才生效

更多选项说明，请使用`--help`命令查看

#### 3.2. 环境变量设置

当编译指定了`-DPPLNN_USE_OPENMP=ON`时，可使用环境变量`OMP_NUM_THREADS`来指定线程数：

```bash
export OMP_NUM_THREADS=8    # 指定8线程
```

#### 3.3. 使用随机数据测速

使用随机数据测速时，示例如下：

```bash
./pplnn --use-arm                       \   # 指定使用 ARM
        --onnx-model <onnx_model>       \   # 指定onnx模型
        --mm-policy mem                 \   # 使用mem内存策略
        --enable-profiling              \   # 使能测速
        --min-profiling-seconds 10      \   # 测速时最少持续10s
        --warmup-iterations 5           \   # warm up 5次
        --perf-with-io                  \   # 测速时包含输入输出耗时
        --wg-level 3                    \   # 优先使用 winograd block 4
        --tuning-level 1                \   # 使用动态算法选择
        --use-fp16                      \   # 使用 fp16 推理
```

pplnn会自动根据模型输入的shape生成随机测试数据。

#### 3.4. 使用外部数据测速

外部数据格式要求见2.1节描述。

测速时，可使用如下命令测速：

```bash
./pplnn --use-arm                                       \   # 指定使用 ARM
        --onnx-model <onnx_model>                       \   # 指定onnx模型
        --reshaped-inputs input-1_3_224_224-fp32.dat    \   # 指定输入数据文件
        --mm-policy perf                                \   # 使用perf内存策略
        --enable-profiling                              \   # 使能测速
        --min-profiling-seconds 10                      \   # 测速时最少持续10s
        --warmup-iterations 5                           \   # warm up 5次
        --perf-with-io                                  \   # 测速时包含输入输出耗时
        --wg-level 3                                    \   # 优先使用 winograd block 4
        --tuning-level 1                                \   # 使用动态算法选择
        --numa-node-id 0                                \   # 运行时绑定到0号NUMA节点（非NUMA机器无效）
```

当有多个输入时，`--reshaped-inputs`使用逗号','分割。

### 附录1. OpenPPL 在 鲲鹏920-6426 上的性能测试

所用编译命令：

```bash
./build.sh -DPPLNN_USE_AARCH64=ON -DCMAKE_BUILD_TYPE=Release -DPPLNN_USE_OPENMP=ON -DPPLNN_USE_ARMV8_2=ON -DPPLNN_ENABLE_KERNEL_PROFILING=ON -DPPLNN_USE_NUMA=ON
```

所用测试命令：

```bash
<path_to_pplnn> --onnx-model <path_to_onnx_model> --reshaped-inputs <path_to_input_files> --save-outputs --use-arm --mm-policy=<mem|perf> --tuning-level=1 --wg-level=3 --numa-node-id=0 [--use-fp16] --enable-profiling --min-profiling-seconds=10 --perf-with-io
```

性能数据更新截至2022-01-21

#### 1. 单batch/单线程 FP16 性能对比

|  模型名称  |  OpenPPL  |  推理框架1  |  推理框架2  |
| :------------ | :------------ | :------------ | :------------ |
| mobilenet_v2 | 11.04 | 14.18 | 16.94 |
| resnet18 | 34.07 | 41.04 | 39.98 |
| resnet34 | 62.03 | 74.22 | 69.06 |
| resnet50 | 104.95 | 130.71 | 118.26 |
| resnet101 | 187.61 | 232.39 | 203.48 |
| resnet152 | 268.09 | 334.59 | 290.90 |
| seresnet50 | 110.95 | 134.91 | 124.12 |
| densenet121 | 74.04 | 89.31 | 87.35 |
| densenet161 | 203.85 | 239.94 | 223.32 |
| densenet169 | 94.63 | 111.32 | 108.05 |
| densenet201 | 126.59 | 149.96 | 142.24 |
| mnasnet0_5 | 4.67 | 6.35 | 7.97 |
| mnasnet0_75 | 8.52 | 11.40 | 13.18 |
| mnasnet1_0 | 11.78 | 18.01 | 16.81 |
| mnasnet1_3 | 18.73 | 24.90 | 24.83 |
| resnext50_32x4d | 139.45 | 174.70 | 149.43 |
| squeezenet1_0 | 20.79 | 22.60 | 22.88 |
| squeezenet1_1 | 8.93 | 10.74 | 12.79 |
| vgg11 | 113.19 | 136.80 | 113.68 |
| vgg13 | 147.91 | 178.05 | 160.30 |
| vgg16 | 186.66 | 234.31 | 199.83 |
| vgg19 | 226.06 | 290.08 | 240.97 |
| shufflenet_v2_x0_5 | 2.13 | 2.96 | 4.87 |
| shufflenet_v2_x1_0 | 5.67 | 7.89 | 11.65 |
| shufflenet_v2_x1_5 | 10.11 | 14.18 | 14.24 |
| shufflenet_v2_x2_0 | 19.97 | 26.99 | 31.95 |
| deeplabv3_resnet101 | 1517.48 | 2348.66 | 2201.26 |
| deeplabv3_resnet50 | 943.20 | 1610.93 | 1551.53 |
| esrgan | 3321.58 | 3768.09 | 4512.89 |

#### 2. 单batch/单线程 FP32 性能对比

|  模型名称  |  OpenPPL  |  推理框架1  |  推理框架2  |
| :------------ | :------------ | :------------ | :------------ |
| mobilenet_v2 | 24.25 | 28.82 | 31.56 |
| resnet18 | 69.22 | 85.46 | 77.44 |
| resnet34 | 128.54 | 154.48 | 137.27 |
| resnet50 | 221.66 | 270.50 | 242.39 |
| resnet101 | 399.39 | 480.64 | 425.39 |
| resnet152 | 577.29 | 685.98 | 605.80 |
| seresnet | 232.61 | 280.87 | 250.58 |
| densenet121 | 158.62 | 182.94 | 181.61 |
| densenet161 | 439.22 | 500.23 | 467.49 |
| densenet169 | 198.67 | 229.23 | 223.14 |
| densenet201 | 270.61 | 307.84 | 295.85 |
| mnasnet0_5 | 9.62 | 11.67 | 13.17 |
| mnasnet0_75 | 17.86 | 22.06 | 23.06 |
| mnasnet1_0 | 25.04 | 30.92 | 30.81 |
| mnasnet1_3 | 40.60 | 50.03 | 49.06 |
| resnext50_32x4d | 288.09 | 344.06 | 307.98 |
| squeezenet1_0 | 41.24 | 46.57 | 45.56 |
| squeezenet1_1 | 19.42 | 21.99 | 22.88 |
| vgg11 | 213.25 | 287.58 | 227.52 |
| vgg13 | 288.65 | 376.75 | 330.68 |
| vgg16 | 390.07 | 463.19 | 418.20 |
| vgg19 | 474.20 | 609.95 | 502.78 |
| shufflenet_v2_x0_5 | 4.07 | 4.97 | 6.23 |
| shufflenet_v2_x1_0 | 11.21 | 14.84 | 16.02 |
| shufflenet_v2_x1_5 | 21.09 | 27.84 | 25.60 |
| shufflenet_v2_x2_0 | 41.64 | 54.57 | 49.83 |
| deeplabv3_resnet101 | 3620.61 | 4979.19 | 4726.22 |
| deeplabv3_resnet50 | 2272.55 | 3428.60 | 3308.76 |
| esrgan | 6914.73 | 8090.13 | 9753.16 |

#### 3. 32 batch/32 线程 FP16 性能对比

|  模型名称  |  OpenPPL  |  推理框架1  |
| :------------ | :------------ | :------------ |
| mobilenet_v2 | 28.50 | 70.55 |
| resnet18 | 41.54 | 79.39 |
| resnet34 | 70.30 | 142.72 |
| resnet50 | 123.32 | 211.41 |
| resnet101 | 211.56 | 355.42 |
| resnet152 | 299.42 | 509.68 |
| seresnet | 151.97 | 272.42 |
| densenet121 | 136.11 | 251.89 |
| densenet161 | 305.02 | 608.83 |
| densenet169 | 188.38 | 337.41 |
| densenet201 | 228.78 | 436.69 |
| mnasnet0_5 | 14.61 | 43.93 |
| mnasnet0_75 | 19.17 | 63.23 |
| mnasnet1_0 | 25.03 | 64.87 |
| mnasnet1_3 | 33.59 | 88.35 |
| resnext50_32x4d | 200.97 | 395.51 |
| squeezenet1_0 | 35.35 | 80.14 |
| squeezenet1_1 | 17.49 | 54.37 |
| vgg11 | 111.52 | 216.46 |
| vgg13 | 185.61 | 282.92 |
| vgg16 | 227.80 | 378.37 |
| vgg19 | 269.83 | 479.27 |
| shufflenet_v2_x0_5 | 9.12 | 35.56 |
| shufflenet_v2_x1_0 | 14.42 | 48.60 |
| shufflenet_v2_x1_5 | 19.50 | 59.32 |
| shufflenet_v2_x2_0 | 34.25 | 74.56 |

#### 4. 32 batch/32 线程 FP32 性能对比

|  模型名称  |  OpenPPL  |  推理框架1  |
| :------------ | :------------ | :------------ |
| mobilenet_v2 | 48.02 | 86.25 |
| resnet18 | 96.76 | 141.43 |
| resnet34 | 154.56 | 255.14 |
| resnet50 | 281.25 | 463.27 |
| resnet101 | 475.47 | 800.87 |
| resnet152 | 694.47 | 1160.18 |
| seresnet | 350.96 | 588.65 |
| densenet121 | 349.71 | 534.66 |
| densenet161 | 723.02 | 1342.02 |
| densenet169 | 387.06 | 709.37 |
| densenet201 | 491.16 | 963.17 |
| mnasnet0_5 | 19.68 | 34.79 |
| mnasnet0_75 | 32.69 | 59.02 |
| mnasnet1_0 | 44.19 | 85.47 |
| mnasnet1_3 | 65.57 | 148.40 |
| resnext50_32x4d | 357.97 | 738.51 |
| squeezenet1_0 | 77.31 | 123.56 |
| squeezenet1_1 | 39.24 | 67.62 |
| vgg11 | 262.62 | 454.87 |
| vgg13 | 465.77 | 606.60 |
| vgg16 | 529.62 | 823.10 |
| vgg19 | 673.47 | 1033.65 |
| shufflenet_v2_x0_5 | 11.58 | 22.12 |
| shufflenet_v2_x1_0 | 19.34 | 50.48 |
| shufflenet_v2_x1_5 | 31.03 | 113.87 |
| shufflenet_v2_x2_0 | 55.77 | 245.24 |

### 附录2. OpenPPL 在 AWS Graviton2 上的性能测试

所用编译命令：

```bash
./build.sh -DPPLNN_USE_AARCH64=ON -DCMAKE_BUILD_TYPE=Release -DPPLNN_USE_OPENMP=ON -DPPLNN_USE_ARMV8_2=ON -DPPLNN_ENABLE_KERNEL_PROFILING=ON
```

所用测试命令：

```bash
<path_to_pplnn> --onnx-model <path_to_onnx_model> --reshaped-inputs <path_to_input_files> --save-outputs --use-arm --mm-policy=<mem|perf> --tuning-level=1 --wg-level=3 [--use-fp16] --enable-profiling --min-profiling-seconds=10 --perf-with-io
```

性能数据更新截至2022-01-21

#### 1. 单batch/单线程 FP16 性能对比

|  模型名称  |  OpenPPL  |  推理框架1  |  推理框架2  |
| :------------ | :------------ | :------------ | :------------ |
| mobilenet_v2 | 9.65 | 10.67 | 12.60 |
| resnet18 | 24.39 | 27.46 | 25.90 |
| resnet34 | 42.05 | 47.40 | 44.84 |
| resnet50 | 87.51 | 92.91 | 90.55 |
| resnet101 | 155.13 | 160.00 | 158.81 |
| resnet152 | 221.74 | 228.72 | 227.03 |
| seresnet | 91.93 | 94.17 | 94.52 |
| densenet121 | 64.24 | 67.96 | 68.51 |
| densenet161 | 171.77 | 178.98 | 175.27 |
| densenet169 | 79.88 | 84.53 | 83.81 |
| densenet201 | 106.26 | 111.40 | 110.52 |
| mnasnet0_5 | 4.08 | 4.89 | 5.83 |
| mnasnet0_75 | 7.52 | 8.57 | 9.57 |
| mnasnet1_0 | 10.45 | 11.63 | 12.62 |
| mnasnet1_3 | 16.86 | 18.11 | 18.98 |
| resnext50_32x4d | 126.80 | 132.25 | 123.60 |
| squeezenet1_0 | 17.91 | 18.36 | 19.51 |
| squeezenet1_1 | 8.10 | 8.30 | 9.92 |
| vgg11 | 74.09 | 75.17 | 77.28 |
| vgg13 | 106.67 | 106.12 | 112.89 |
| vgg16 | 138.44 | 138.91 | 143.50 |
| vgg19 | 168.72 | 167.54 | 174.25 |
| shufflenet_v2_x0_5 | 1.73 | 2.44 | 3.25 |
| shufflenet_v2_x1_0 | 5.02 | 6.40 | 8.83 |
| shufflenet_v2_x1_5 | 9.12 | 11.16 | 11.29 |
| shufflenet_v2_x2_0 | 18.28 | 20.59 | 25.97 |
| deeplabv3_resnet101 | 1357.72 | 1586.87 | 1701.54 |
| deeplabv3_resnet50 | 833.43 | 1062.04 | 1155.97 |
| esrgan | 2898.04 | 2922.67 | 3009.15 |

#### 2. 单batch/单线程 FP32 性能对比

|  模型名称  |  OpenPPL  |  推理框架1  |  推理框架2  |
| :------------ | :------------ | :------------ | :------------ |
| mobilenet_v2 | 19.48 | 20.82 | 21.91 |
| resnet18 | 50.16 | 55.66 | 50.46 |
| resnet34 | 86.54 | 95.95 | 88.08 |
| resnet50 | 178.64 | 186.52 | 177.26 |
| resnet101 | 317.71 | 326.12 | 312.29 |
| resnet152 | 453.92 | 468.64 | 446.26 |
| seresnet | 186.53 | 191.51 | 181.13 |
| densenet121 | 132.00 | 138.07 | 132.33 |
| densenet161 | 348.36 | 351.42 | 343.26 |
| densenet169 | 162.40 | 170.55 | 162.23 |
| densenet201 | 217.10 | 222.45 | 216.19 |
| mnasnet0_5 | 8.02 | 9.26 | 8.68 |
| mnasnet0_75 | 15.09 | 16.68 | 16.08 |
| mnasnet1_0 | 21.04 | 22.84 | 21.66 |
| mnasnet1_3 | 34.06 | 36.48 | 34.42 |
| resnext50_32x4d | 238.20 | 247.89 | 238.59 |
| squeezenet1_0 | 37.38 | 34.54 | 35.00 |
| squeezenet1_1 | 16.76 | 16.80 | 16.77 |
| vgg11 | 152.03 | 152.35 | 158.69 |
| vgg13 | 217.75 | 219.25 | 230.19 |
| vgg16 | 279.39 | 284.90 | 291.58 |
| vgg19 | 341.48 | 341.80 | 352.71 |
| shufflenet_v2_x0_5 | 3.23 | 3.89 | 3.90 |
| shufflenet_v2_x1_0 | 9.33 | 11.52 | 11.61 |
| shufflenet_v2_x1_5 | 17.86 | 20.49 | 19.44 |
| shufflenet_v2_x2_0 | 34.45 | 38.51 | 38.10 |
| deeplabv3_resnet101 | 3061.82 | 3223.65 | 3394.90 |
| deeplabv3_resnet50 | 1901.08 | 2185.68 | 2319.97 |
| esrgan | 5537.44 | 5817.98 | 6015.26 |

#### 3. 32 batch/32 线程 FP16 性能对比

|  模型名称  |  OpenPPL  |  推理框架1  |
| :------------ | :------------ | :------------ |
| mobilenet_v2 | 16.57 | 63.03 |
| resnet18 | 27.56 | 67.52 |
| resnet34 | 47.01 | 106.63 |
| resnet50 | 93.72 | 147.10 |
| resnet101 | 163.59 | 232.30 |
| resnet152 | 231.65 | 322.92 |
| seresnet | 107.05 | 173.51 |
| densenet121 | 92.22 | 159.10 |
| densenet161 | 217.35 | 361.27 |
| densenet169 | 125.00 | 218.23 |
| densenet201 | 146.42 | 275.35 |
| mnasnet0_5 | 6.76 | 46.06 |
| mnasnet0_75 | 10.90 | 59.53 |
| mnasnet1_0 | 14.16 | 60.96 |
| mnasnet1_3 | 21.54 | 73.56 |
| resnext50_32x4d | 140.04 | 238.01 |
| squeezenet1_0 | 22.12 | 64.08 |
| squeezenet1_1 | 11.11 | 50.04 |
| vgg11 | 88.12 | 140.64 |
| vgg13 | 129.32 | 178.23 |
| vgg16 | 162.62 | 231.77 |
| vgg19 | 197.24 | 296.81 |
| shufflenet_v2_x0_5 | 4.23 | 39.75 |
| shufflenet_v2_x1_0 | 8.89 | 47.26 |
| shufflenet_v2_x1_5 | 14.16 | 55.06 |
| shufflenet_v2_x2_0 | 25.36 | 60.41 |

#### 4. 32 batch/32 线程 FP32 性能对比

|  模型名称  |  OpenPPL  |  推理框架1  |
| :------------ | :------------ | :------------ |
| mobilenet_v2 | 26.57 | 67.61 |
| resnet18 | 65.81 | 103.50 |
| resnet34 | 116.87 | 168.21 |
| resnet50 | 202.48 | 274.60 |
| resnet101 | 354.01 | 456.21 |
| resnet152 | 509.25 | 644.41 |
| seresnet | 230.05 | 329.57 |
| densenet121 | 201.31 | 296.27 |
| densenet161 | 454.66 | 697.22 |
| densenet169 | 226.28 | 398.02 |
| densenet201 | 296.43 | 518.78 |
| mnasnet0_5 | 10.18 | 40.06 |
| mnasnet0_75 | 18.85 | 39.56 |
| mnasnet1_0 | 25.76 | 55.21 |
| mnasnet1_3 | 39.36 | 112.89 |
| resnext50_32x4d | 253.21 | 413.42 |
| squeezenet1_0 | 50.19 | 84.31 |
| squeezenet1_1 | 24.61 | 53.09 |
| vgg11 | 195.66 | 251.31 |
| vgg13 | 315.64 | 330.01 |
| vgg16 | 391.11 | 435.62 |
| vgg19 | 467.06 | 552.04 |
| shufflenet_v2_x0_5 | 5.60 | 32.18 |
| shufflenet_v2_x1_0 | 12.09 | 45.83 |
| shufflenet_v2_x1_5 | 21.15 | 87.37 |
| shufflenet_v2_x2_0 | 38.53 | 162.03 |
