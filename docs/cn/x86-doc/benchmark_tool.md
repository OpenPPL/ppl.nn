## X86 Benchmark 工具

x86架构性能测试使用pplnn工具。

本章仅介绍x86架构下使用pplnn测速的方法，cuda架构测速请参考：[Cuda Benchmark 工具](../cuda-doc/benchmark_tool.md)。

### 1. 编译

pplnn工具编译请参考[building-from-source.md](../../en/building-from-source.md)。

x86架构使用openmp作为线程池。若需要测试多线程性能，编译时请指定`-DHPCC_USE_OPENMP=ON`：

```bash
./build.sh -DHPCC_USE_OPENMP=ON
```

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

pplnn中，与x86架构测速相关的运行选项有：

* `--onnx-model`：指定onnx模型文件
* `--reshaped-inputs`：指定外部数据，格式要求上文已阐述
* `--mm-policy`：内存管理策略，mem代表更少的内存使用，perf代表更激进的内存优化，默认为mem
* `--enable-profiling`：使能测速，默认为不使能
* `--min-profiling-time`：指定测速的最少持续时间，单位为秒，默认为1s
* `--warmuptimes`：指定warm up的次数，默认为0
* `--disable-avx512`：指定禁用avx512指令集，默认为启用
* `--core-binding`：启用绑核，默认不启用

#### 3.2. 环境变量设置

当编译指定了`-DHPCC_USE_OPENMP=ON`时，可使用环境变量`OMP_NUM_THREADS`来指定线程数：

```bash
export OMP_NUM_THREADS=8    # 指定8线程
```

#### 3.3. 使用随机数据测速

使用随机数据测速时，示例如下：

```bash
./pplnn --onnx-model <onnx_model> \   # 指定onnx模型
        --mm-policy mem           \   # 使用mem内存策略
        --enable-profiling        \   # 使能测速
        --min-profiling-time 10   \   # 测速时最少持续10s
        --warmuptimes 5           \   # warm up 5次
        --core-binding            \   # 启用绑核
        --disable-avx512              # 禁用avx512指令集
```

pplnn会自动根据模型输入的shape生成随机测试数据。

#### 3.4. 使用外部数据测速

外部数据格式要求见2.1节描述。

测速时，可使用如下命令测速：

```bash
./pplnn --onnx-model <onnx_model>                       \   # 指定onnx模型
        --reshaped-inputs input-1_3_224_224-fp32.dat    \   # 指定输入数据文件
        --mm-policy mem                                 \   # 使用mem内存策略
        --enable-profiling                              \   # 使能测速
        --min-profiling-time 10                         \   # 测速时最少持续10s
        --warmuptimes 5                                     # warm up 5次
```

当有多个输入时，`--reshaped-inputs`使用逗号','分割。
