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

### Appendix 1. OpenPPL Bechmark on 10980XE

Platform Information:
 - CPU: [Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz](https://ark.intel.com/content/www/us/en/ark/products/198017/intel-core-i9-10980xe-extreme-edition-processor-24-75m-cache-3-00-ghz.html "Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz")
 - Memory: 4-channel DDR4 @ 3000MHz
 - OS: Ubuntu 18.04 LTS
 - Toolchains: GCC 7.5.0

#### 1. 1-batch/1-thread with AVX512F

|  Model Name  |  OpenPPL  |  onnxruntime v1.8  |  OpenVINO r2021.2  |
| :------------ | :------------ | :------------ | :------------ |
|  deeplabv3plus  |  394.12  |  453.81  |  476.82  |
|  esrgan  |  1800.47  |  3216.80  |  3298.32  |
|  mobilenet_v2  |  4.45  |  5.41  |  5.04  |
|  resnet18  |  15.42  |  20.38  |  20.95  |
|  se_resnet50  |  45.68  |  56.87  |  53.08  |
|  densenet121  |  31.84  |  35.77  |  39.03  |
|  googlenet  |  12.64  |  16.32  |  16.91  |
|  inception_v3  |  28.84  |  36.31  |  36.01  |
|  mnasnet1_0  |  4.58  |  4.77  |  5.39  |
|  resnet34  |  30.70  |  40.98  |  41.80  |
|  resnet50  |  40.78  |  47.00  |  48.34  |
|  squeezenet1_1  |  3.83  |  4.31  |  4.27  |
|  vgg16  |  106.04  |  182.99  |  192.52  |
|  wide_resnet101_2  |  197.82  |  247.34  |  258.57  |

#### 2. 16-batch/16-thread with AVX512F

|  Model Name  |  OpenPPL  |  onnxruntime v1.8  |  OpenVINO r2021.2  |
| :------------ | :------------ | :------------ | :------------ |
|  deeplabv3plus\*  |  423.09|  518.72  |  520.29  |
|  esrgan  |  unavailable  |  unavailable  |  unavailable  |
|  mobilenet_v2  |  12.85  |  18.29  |  13.07  |
|  resnet18  |  20.85  |  28.14  |  26.05  |
|  se_resnet50  |  99.93  |  198.59  |  89.83  |
|  densenet121  |  89.49  |  168.49  |  93.11  |
|  googlenet  |  23.35  |  37.28  |  25.48  |
|  inception_v3  |  48.48  |  88.36  |  44.65  |
|  mnasnet1_0  |  12.12  |  11.13  |  11.24  |
|  resnet34  |  36.00  |  54.16  |  50.19  |
|  resnet50  |  65.37  |  72.69  |  67.27  |
|  squeezenet1_1  |  10.38  |  19.83  |  7.89  |
|  vgg16  |  107.52  |  236.27  |  233.77  |
|  wide_resnet101_2  |  259.18  |  344.04  |  321.00  |

\* deeplabv3plus is benchmarked with 4-batch/4-thread

#### 3. 1-batch/1-thread with FMA3

|  Model Name  |  OpenPPL  |  onnxruntime v1.8  |  OpenVINO r2021.2  |  MindSpore Lite r1.3  |
| :------------ | :------------ | :------------ | :------------ | :------------ |
|  deeplabv3plus  |  630.45  |  919.51  |  836.47  |  938.73  |
|  esrgan  |  2428.81  |  5118.05  |  4686.33  |  3636.94  |
|  mobilenet_v2  |  6.84  |  7.56  |  7.06  |  7.54  |
|  resnet18  |  25.11  |  32.77  |  34.04  |  35.08  |
|  se_resnet50  |  72.38  |  87.49  |  82.97  |  91.65  |
|  densenet121  |  47.57  |  55.27  |  58.17  |  65.40  |
|  googlenet  |  19.99  |  26.95  |  28.01  |  25.58  |
|  inception_v3  |  48.89  |  55.19  |  56.08  |  unsupported  |
|  mnasnet1_0  |  7.09  |  7.29  |  7.56  |  7.57  |
|  resnet34  |  46.33  |  65.74  |  69.38  |  67.63  |
|  resnet50  |  67.26  |  76.53  |  77.94  |  87.58  |
|  squeezenet1_1  |  5.91  |  6.64  |  6.76  |  6.60  |
|  vgg16  |  139.41  |  292.68  |  298.62  |  221.88  |
|  wide_resnet101_2  |  312.80  |  407.25  |  425.77  |  449.09  |

#### 4. 16-batch/16-thread with FMA3

|  Model Name  |  OpenPPL  |  onnxruntime v1.8  |  OpenVINO r2021.2  |  MindSpore Lite r1.3  |
| :------------ | :------------ | :------------ | :------------ | :------------ |
|  deeplabv3plus\*  |  685.18  |  972.02  |  884.11  |  1129.83  |
|  esrgan  |  unavailable  |  unavailable  |  unavailable  |  unavailable  |
|  mobilenet_v2  |  14.87  |  18.33  |  9.29  |  24.57  |
|  resnet18  |  26.49  |  39.14  |  43.07  |  133.39  |
|  se_resnet50  |  118.81  |  230.49  |  142.70  |  323.55  |
|  densenet121  |  98.40  |  183.81  |  112.32  |  249.91  |
|  googlenet  |  27.82  |  45.97  |  37.03  |  73.40  |
|  inception_v3  |  61.70  |  104.37  |  69.58  |  unsupported  |
|  mnasnet1_0  |  13.60  |  12.24  |  9.69  |  22.18  |
|  resnet34  |  44.43  |  77.27  |  83.74  |  273.40  |
|  resnet50  |  83.37  |  104.05  |  98.72  |  266.43  |
|  squeezenet1_1  |  12.44  |  20.02  |  10.30  |  27.87  |
|  vgg16  |  132.64  |  354.90  |  339.40  |  443.71  |
|  wide_resnet101_2  |  337.46  |  499.50  |  509.89  |  1527.74  |

\* deeplabv3plus is benchmarked with 4-batch/4-thread

### Appendix 2. OpenPPL 3700X vs. 10980XE

Platform Information:
 - CPU1: [AMD Ryzen 7 3700X 8-Core Processor](https://www.amd.com/zh-hans/products/cpu/amd-ryzen-7-3700x)
 - Memory1: 2-channel DDR4 @ 3200MHz
 - CPU2: [Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz](https://ark.intel.com/content/www/us/en/ark/products/198017/intel-core-i9-10980xe-extreme-edition-processor-24-75m-cache-3-00-ghz.html "Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz")
 - Memory2: 4-channel DDR4 @ 3000MHz
 - OS: Ubuntu 18.04 LTS
 - Toolchains: GCC 7.5.0
 - Instruction Set: FMA3

#### 1. 1-batch with 1 to 8 threads

|  Model Name  |  3700X 1-thread  |  10980XE 1-thread  |  3700X 2-thread  |  10980XE 2-thread  |  3700X 4-thread  |  10980XE 4-thread  |  3700X 8-thread  |  10980XE 8-thread  |
| :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ |
|  deeplabv3plus  |  610.78  |  674.02  |  280.42  |  321.09  |  148.20  |  174.57  |  90.13  |  94.43  |
|  esrgan  |  1791.85  |  2459.53  |  982.46  |  1289.04  |  643.06  |  812.01  |  655.05  |  473.52  |
|  mobilenet_v2  |  6.14  |  6.77  |  3.79  |  4.26  |  2.25  |  2.51  |  2.16  |  2.30  |
|  resnet18  |  18.17  |  25.59  |  10.01  |  13.83  |  6.98  |  7.80  |  6.99  |  5.04  |
|  se_resnet50  |  57.75  |  72.36  |  30.53  |  38.20  |  18.39  |  21.39  |  15.84  |  14.56  |
|  densenet121  |  38.48  |  47.57  |  21.98  |  28.44  |  13.14  |  16.49  |  13.97  |  13.82  |
|  googlenet  |  16.54  |  19.85  |  9.43  |  11.72  |  5.64  |  6.71  |  5.67  |  4.92  |
|  inception_v3  |  38.78  |  48.32  |  22.55  |  28.53  |  13.02  |  16.59  |  11.69  |  12.18  |
|  mnasnet1_0  |  6.31  |  7.05  |  3.78  |  4.24  |  2.25  |  2.55  |  2.10  |  2.28  |
|  resnet34  |  31.33  |  47.61  |  17.41  |  25.74  |  11.89  |  14.37  |  12.44  |  8.98  |
|  resnet50  |  55.73  |  67.17  |  29.27  |  35.58  |  17.32  |  20.06  |  14.38  |  13.07  |
|  squeezenet1_1  |  5.29  |  5.96  |  2.87  |  3.41  |  1.55  |  2.01  |  1.60  |  1.62  |
|  vgg16  |  97.64  |  143.35  |  53.94  |  75.22  |  35.10  |  42.01  |  31.22  |  24.61  |
|  wide_resnet101_2  |  247.82  |  313.41  |  131.10  |  166.45  |  77.52  |  89.30  |  64.17  |  54.73  |

#### 2. 16-batch with 1 to 8 threads

|  Model Name  |  3700X 1-thread  |  10980XE 1-thread  |  3700X 2-thread  |  10980XE 2-thread  |  3700X 4-thread  |  10980XE 4-thread  |  3700X 8-thread  |  10980XE 8-thread  |
| :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ |
|  deeplabv3plus\*  |  2314.52  |  2646.51  |  1132.35  |  1299.84  |  604.27  |  683.62  |  337.72  |  357.54  |
|  esrgan  |  unavailable  |  unavailable  |  unavailable  |  unavailable  |  unavailable  |  unavailable  |  unavailable  |  unavailable  |
|  mobilenet_v2  |  99.76  |  115.57  |  53.04  |  61.05  |  34.24  |  33.71  |  30.91  |  20.53  |
|  resnet18  |  240.29  |  279.60  |  122.91  |  142.06  |  66.82  |  75.48  |  42.14  |  40.27  |
|  se_resnet50  |  938.73  |  1057.31  |  492.15  |  549.39  |  297.04  |  297.67  |  211.17  |  169.26  |
|  densenet121  |  690.09  |  789.15  |  388.93  |  415.81  |  258.43  |  232.00  |  188.48  |  136.29  |
|  googlenet  |  256.14  |  288.49  |  135.52  |  150.42  |  79.86  |  81.30  |  56.85  |  45.29  |
|  inception_v3  |  581.54  |  667.63  |  307.68  |  346.93  |  172.80  |  186.82  |  106.53  |  102.00  |
|  mnasnet1_0  |  101.10  |  118.79  |  54.10  |  62.30  |  33.40  |  34.60  |  28.15  |  20.84  |
|  resnet34  |  405.41  |  495.02  |  208.81  |  250.61  |  113.08  |  133.37  |  72.73  |  71.14  |
|  resnet50  |  839.27  |  946.68  |  539.82  |  482.61  |  289.67  |  255.34  |  137.18  |  136.48  |
|  squeezenet1_1  |  92.39  |  105.33  |  49.00  |  55.29  |  32.16  |  30.60  |  27.57  |  18.38  |
|  vgg16  |  1288.17  |  1654.04  |  665.49  |  845.69  |  362.35  |  421.77  |  231.20  |  229.89  |
|  wide_resnet101_2  |  3604.05  |  4171.32  |  1841.64  |  2107.49  |  966.50  |  1099.83  |  547.85  |  575.81  |

\* deeplabv3plus is benchmarked with 4-batch
