## X86 Benchmark Tool

We recommend to use a test tool named "pplnn" to benchmark x86 architecture.

This chapter only introduces the method of using pplnn to benchmark x86 architecture. If you want to benchmark cuda architecture, please refer to: [Cuda Benchmark Tool](../cuda-doc/benchmark_tool.md).

### 1. Compile

For compilation method of pplnn, please refer to: [building-from-source.md](../../en/building-from-source.md).

X86 architecture uses openmp as the thread pool, so if you need to test multi-thread performance, please compile with `-DHPCC_USE_OPENMP=ON` option as below:

```bash
./build.sh -DHPCC_USE_OPENMP=ON
```

pplnn will be generated to: ./pplnn-build/tools/pplnn

### 2. Prepare Test Data

pplnn can either generate random data, or read data from outside as network input:

* For networks such as classification/semantic segmentation, the execution speed of the network has nothing to do with the input data value, so you can generate random data for benchmark.
* For networks such as detection/instance segmentation, the execution speed of the network may be related to the input data value, so it is recommended to read in real data externally.

#### 2.1. External Data Format Requirements

When pplnn reads data externally, it is recommended to use `--reshaped-inputs` option to specify external test data.

Under this option, pplnn requires the test data file to be in binary format (you can use numpy's `tofile` function to store data in binary format). Each input tensor of the model needs to store a separate test data file. The test datafile naming method is:

```
<tensor_name>-<input_shape>-<data_type>.dat
```

* \<tensor_name\>: corresponded to the name of the input tensor in the onnx model, such as: "input"
* \<input_shape\>: shape of the model input tensor, with '_' as the separator, such as: 1_3_224_224
* \<data_type\>: data type of the test file, now supports fp64|fp32|fp16|int32|int64|bool

For example, if the name of the input tensor is "input" in the onnx model, the shape is (1,3,224,224), and the data type is float32, then the test data file name should be:

```
input-1_3_224_224-fp32.dat
```

### 3. Run pplnn

#### 3.1. Run Options

pplnn's run options related to the x86 architecture benchmark are:

* `--onnx-model`: Specify the tested onnx model file
* `--in-shapes`:  Specify the input tensor shape
* `--mm-policy`: Memory management strategy, "mem" means less memory usage, and "perf" means more radical memory optimization. Default is mem
* `--enable-profiling`: Enable profiling. Default is false
* `--min-profiling-time`: Specify the minimum time duration of benchmark in seconds. Default is 1s
* `--warmuptimes`: Specify the warm up times. Default is 0
* `--disable-avx512`: Disable avx512 instruction set. Default is false
* `--core-binding`: Enable core binding. Default is false.

#### 3.2. Environment Variable Settings

When the compilation specifies `-DHPCC_USE_OPENMP=ON`, the environment variable `OMP_NUM_THREADS` can be used to specify the number of threads:

```bash
export OMP_NUM_THREADS=8    # use 8 threads
```

#### 3.3. Use Random Test Data to Benchmark

Here is an example to use random test data for benchmark:

```bash
./pplnn --onnx-model <onnx_model> \   # specify onnx model
        --mm-policy mem           \   # use "mem" memory management policy
        --enable-profiling        \   # enable profiling
        --min-profiling-time 10   \   # benchmark lasts at least 10s
        --warmuptimes 5           \   # warm up 5 times
        --core-binding            \   # enable core binding
        --disable-avx512              # disable avx512 instruction set
```

pplnn will automatically generate random test data by the input tensor shape of the model.

#### 3.4. Use External Test Data to Benchmark

The external test data format requirements are described in section 2.1.

You can use the following command for benchmark:

```bash
./pplnn --onnx-model <onnx_model>                       \   # specify onnx model
        --reshaped-inputs input-1_3_224_224-fp32.dat    \   # specify input test data file
        --mm-policy mem                                 \   # use "mem" memory management policy
        --enable-profiling                              \   # enable profiling
        --min-profiling-time 10                         \   # benchmark lasts at least 10s
        --warmuptimes 5                                     # warm up 5 times
```

When there are multiple inputs, `--reshaped-inputs` is separated by commas ','.
