## X86 Benchmark Tool

We recommend to use a test tool named "pplnn" to benchmark x86 architecture.

This chapter only introduces the method of using pplnn to benchmark x86 architecture. If you want to benchmark cuda architecture, please refer to: [Cuda Benchmark Tool](../cuda-doc/benchmark_tool.md).

### 1. Compile

For compilation method of pplnn, please refer to: [building-from-source.md](../../en/building-from-source.md).

X86 architecture uses openmp as the thread pool, so if you need to test multi-thread performance, please compile with `-DPPLNN_USE_OPENMP=ON` option as below:

```bash
./build.sh -DPPLNN_USE_X86_64=ON -DPPLNN_USE_OPENMP=ON
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

* `--use-x86`: use x86 engine
* `--onnx-model`: Specify the tested onnx model file
* `--in-shapes`:  Specify the input tensor shape
* `--mm-policy`: Memory management strategy, "mem" means less memory usage, and "perf" means more radical memory optimization. Default is mem
* `--enable-profiling`: Enable profiling. Default is false
* `--min-profiling-seconds`: Specify the minimum time duration of benchmark in seconds. Default is 1s
* `--warmup-iterations`: Specify the warm up times. Default is 0
* `--disable-avx512`: Disable avx512 instruction set. Default is false
* `--disable-avx-fma3`: Disable avx, fma3 and avx512 instruction sets. Default is false
* `--core-binding`: Enable core binding. Default is false.

#### 3.2. Environment Variable Settings

When the compilation specifies `-DPPLNN_USE_OPENMP=ON`, the environment variable `OMP_NUM_THREADS` can be used to specify the number of threads:

```bash
export OMP_NUM_THREADS=8    # use 8 threads
```

#### 3.3. Use Random Test Data to Benchmark

Here is an example to use random test data for benchmark:

```bash
./pplnn --use-x86                 \   # use x86 engine
        --onnx-model <onnx_model> \   # specify onnx model
        --mm-policy mem           \   # use "mem" memory management policy
        --enable-profiling        \   # enable profiling
        --min-profiling-seconds 10   \   # benchmark lasts at least 10s
        --warmup-iterations 5           \   # warm up 5 times
        --core-binding            \   # enable core binding
        --disable-avx512              # disable avx512 instruction set
```

pplnn will automatically generate random test data by the input tensor shape of the model.

#### 3.4. Use External Test Data to Benchmark

The external test data format requirements are described in section 2.1.

You can use the following command for benchmark:

```bash
./pplnn --use-x86                                       \   # use x86 engine
        --onnx-model <onnx_model>                       \   # specify onnx model
        --reshaped-inputs input-1_3_224_224-fp32.dat    \   # specify input test data file
        --mm-policy mem                                 \   # use "mem" memory management policy
        --enable-profiling                              \   # enable profiling
        --min-profiling-seconds 10                         \   # benchmark lasts at least 10s
        --warmup-iterations 5                                     # warm up 5 times
```

When there are multiple inputs, `--reshaped-inputs` is separated by commas ','.

### Appendix 1. OpenPPL Bechmark on 10980XE

Platform Information:
 - CPU: [Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz](https://ark.intel.com/content/www/us/en/ark/products/198017/intel-core-i9-10980xe-extreme-edition-processor-24-75m-cache-3-00-ghz.html "Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz")
 - Memory: 4-channel DDR4 @ 3000MHz
 - OS: Ubuntu 18.04 LTS
 - Toolchains: GCC 7.5.0

#### 1. 1-batch/1-thread with AVX512F

|  Model Name  |  OpenPPL  |  OpenPPL v0.1  |  onnxruntime v1.8  |  OpenVINO r2021.2  |
| :------------ | :------------ | :------------ | :------------ | :------------ |
|  faster_rcnn  |  1866.25  |  1983.34  |    |    |
|  fsaf  |  1241.93  |  1243.11  |  1902.21  |    |
|  mask_rcnn  |  2968.83  |  3062.78  |    |    |
|  retinanet  |  1453.06  |  1471.61  |  2210.59  |    |
|  deeplabv3  |  455.88  |  475.85  |  556.01  |  758.68  |
|  deeplabv3plus  |  371.58  |  394.12  |  471.25  |  476.82  |
|  fcn  |  333.43  |  353.93  |  501.88  |  502.76  |
|  pspnet  |  318.44  |  332.70  |  473.52  |  462.06  |
|  esrgan  |  1805.96  |  1800.48  |  3235.00  |  3298.32  |
|  srcnn  |  87.93  |  92.24  |  130.33  |  129.54  |
|  alexnet  |  23.90  |  23.57  |  23.87  |  24.55  |
|  densenet121  |  31.12  |  31.84  |  36.60  |  39.03  |
|  densenet161  |  82.61  |  83.89  |  96.66  |  105.82  |
|  densenet169  |  39.51  |  40.86  |  46.17  |  50.80  |
|  densenet201  |  52.95  |  53.75  |  59.37  |  72.85  |
|  googlenet  |  12.73  |  12.64  |  16.34  |  16.91  |
|  inception_v3  |  29.26  |  28.84  |  36.41  |  36.01  |
|  mnasnet0_5  |  1.73  |  1.88  |  2.12  |  2.30  |
|  mnasnet0_75  |  3.11  |  3.41  |  3.62  |  4.02  |
|  mnasnet1_0  |  4.23  |  4.58  |  4.84  |  5.39  |
|  mnasnet1_3  |  6.91  |  7.46  |  7.94  |  8.51  |
|  mobilenet_v2  |  3.96  |  4.47  |  5.46  |  5.03  |
|  resnet101  |  75.65  |  76.84  |  86.85  |  89.85  |
|  resnet152  |  108.88  |  110.36  |  126.66  |  128.98  |
|  resnet18  |  15.12  |  15.43  |  20.49  |  21.23  |
|  resnet34  |  29.89  |  30.70  |  41.32  |  41.80  |
|  resnet50  |  40.65  |  40.78  |  47.17  |  48.34  |
|  resnext101_32x8d  |  185.45  |  182.06  |  193.68  |  196.25  |
|  resnext50_32x4d  |  55.38  |  54.47  |  57.72  |  60.22  |
|  shufflenet_v2_x0_5  |  1.05  |  1.14  |  1.41  |  1.83  |
|  shufflenet_v2_x1_0  |  2.45  |  2.72  |  3.63  |  4.08  |
|  shufflenet_v2_x1_5  |  4.22  |  4.62  |  5.13  |  6.17  |
|  shufflenet_v2_x2_0  |  8.17  |  8.89  |  11.31  |  11.65  |
|  squeezenet1_0  |  7.44  |  7.78  |  9.31  |  9.18  |
|  squeezenet1_1  |  3.75  |  3.83  |  4.38  |  4.27  |
|  vgg11  |  70.77  |  71.22  |  107.92  |  114.09  |
|  vgg13  |  85.98  |  87.88  |  143.50  |  150.38  |
|  vgg16  |  102.67  |  106.04  |  182.47  |  192.52  |
|  vgg19  |  120.11  |  121.82  |  226.63  |  234.54  |
|  wide_resnet101_2  |  197.17  |  197.82  |  250.64  |  258.57  |
|  wide_resnet50_2  |  101.38  |  104.74  |  128.78  |  133.75  |

#### 2. 16-batch/16-thread with AVX512F

|  Model Name  |  OpenPPL  |  OpenPPL v0.1  |  onnxruntime v1.8  |  OpenVINO r2021.2  |
| :------------ | :------------ | :------------ | :------------ | :------------ |
|  faster_rcnn\*  |  2233.64  |  2338.54  |    |    |
|  fsaf\*  |  1376.48  |  1395.33  |  2117.94  |    |
|  mask_rcnn\*  |  3983.99  |  4130.19  |    |    |
|  retinanet\*  |  1655.03  |  1689.89  |  2491.72  |    |
|  deeplabv3\*  |  501.63  |  505.97  |  601.63  |  774.24  |
|  deeplabv3plus\*  |  417.75  |  423.09  |  533.76  |  520.29  |
|  fcn\*  |  368.90  |  370.03  |  521.95  |  535.95  |
|  pspnet\*  |  357.67  |  360.79  |  531.11  |  497.72  |
|  alexnet  |  11.08  |  11.44  |  14.02  |  17.04  |
|  densenet121  |  86.11  |  89.49  |  163.30  |  93.11  |
|  densenet161  |  203.39  |  212.76  |  370.08  |  224.44  |
|  densenet169  |  101.72  |  106.46  |  199.01  |  120.42  |
|  densenet201  |  136.55  |  142.96  |  267.61  |  166.03  |
|  googlenet  |  22.36  |  23.35  |  37.12  |  25.48  |
|  inception_v3  |  46.51  |  48.48  |  88.19  |  44.65  |
|  mnasnet0_5  |  3.04  |  4.67  |  5.53  |  4.12  |
|  mnasnet0_75  |  5.86  |  10.56  |  9.72  |  8.87  |
|  mnasnet1_0  |  7.35  |  12.12  |  10.92  |  11.24  |
|  mnasnet1_3  |  13.42  |  18.37  |  20.03  |  18.16  |
|  mobilenet_v2  |  7.11  |  12.72  |  18.53  |  12.96  |
|  resnet101  |  109.12  |  109.20  |  125.05  |  118.79  |
|  resnet152  |  149.18  |  153.85  |  175.19  |  167.98  |
|  resnet18  |  20.52  |  20.93  |  28.02  |  25.89  |
|  resnet34  |  36.78  |  36.00  |  54.02  |  50.19  |
|  resnet50  |  64.82  |  65.37  |  72.59  |  67.27  |
|  resnext101_32x8d  |  281.04  |  303.44  |  333.42  |  271.06  |
|  resnext50_32x4d  |  95.48  |  102.60  |  149.57  |  95.19  |
|  shufflenet_v2_x0_5  |  2.25  |  3.41  |  12.51  |  3.59  |
|  shufflenet_v2_x1_0  |  4.49  |  8.76  |  38.76  |  7.18  |
|  shufflenet_v2_x1_5  |  7.19  |  9.49  |  39.82  |  10.20  |
|  shufflenet_v2_x2_0  |  12.42  |  21.34  |  85.45  |  17.55  |
|  squeezenet1_0  |  20.84  |  21.62  |  36.76  |  15.49  |
|  squeezenet1_1  |  11.24  |  10.38  |  19.71  |  7.89  |
|  vgg11  |  62.11  |  62.40  |  121.21  |  128.60  |
|  vgg13  |  84.68  |  85.22  |  183.13  |  178.91  |
|  vgg16  |  106.05  |  107.52  |  236.45  |  233.77  |
|  vgg19  |  128.52  |  128.37  |  287.83  |  287.35  |
|  wide_resnet101_2  |  251.44  |  259.18  |  342.38  |  321.00  |
|  wide_resnet50_2  |  137.96  |  139.25  |  183.35  |  171.06  |

\* benchmarked with 4-batch/4-thread

#### 3. 1-batch/1-thread with FMA3

|  Model Name  |  OpenPPL  |  OpenPPL v0.1  |  onnxruntime v1.8  |  OpenVINO r2021.2  |
| :------------ | :------------ | :------------ | :------------ | :------------ |
|  faster_rcnn  |  2461.32  |  2659.48  |    |    |
|  fsaf  |  1848.48  |  1880.65  |  3327.35  |    |
|  mask_rcnn  |  3836.06  |  4041.42  |    |    |
|  retinanet  |  2108.36  |  2113.40  |  3848.44  |    |
|  deeplabv3  |  811.83  |  861.88  |  1096.70  |  1324.94   |
|  deeplabv3plus  |  623.15  |  674.02  |  887.20  |  836.47   |
|  fcn  |  557.23  |  616.76  |  1011.60  |  892.30   |
|  pspnet  |  535.49  |  572.07  |  950.06  |  823.12   |
|  esrgan  |  2365.24  |  2459.53  |  5118.21  |  4686.33   |
|  srcnn  |  120.21  |  120.03  |  184.23  |  233.47   |
|  alexnet  |  26.68  |  27.20  |  28.83  |  30.61   |
|  densenet121  |  45.73  |  47.57  |  55.27  |  58.17   |
|  densenet161  |  123.38  |  127.31  |  156.72  |  166.85   |
|  densenet169  |  58.11  |  60.55  |  68.18  |  73.39   |
|  densenet201  |  77.58  |  80.68  |  88.02  |  100.34   |
|  googlenet  |  19.34  |  19.85  |  26.96  |  28.01   |
|  inception_v3  |  45.26  |  48.32  |  55.19  |  56.08   |
|  mnasnet0_5  |  2.54  |  2.62  |  2.84  |  3.04   |
|  mnasnet0_75  |  4.87  |  5.01  |  5.14  |  5.36   |
|  mnasnet1_0  |  6.89  |  7.05  |  7.24  |  7.56   |
|  mnasnet1_3  |  11.34  |  11.66  |  11.54  |  12.03   |
|  mobilenet_v2  |  6.44  |  6.81  |  7.53  |  7.21   |
|  resnet101  |  118.19  |  120.78  |  142.02  |  147.23   |
|  resnet152  |  168.48  |  171.71  |  206.37  |  215.60   |
|  resnet18  |  24.21  |  25.51  |  32.70  |  34.35   |
|  resnet34  |  44.92  |  47.61  |  65.69  |  69.38   |
|  resnet50  |  65.27  |  67.17  |  76.53  |  77.94   |
|  resnext101_32x8d  |  296.33  |  299.04  |  302.93  |  314.55   |
|  resnext50_32x4d  |  84.45  |  85.53  |  84.72  |  88.92   |
|  shufflenet_v2_x0_5  |  1.39  |  1.50  |  1.36  |  1.87   |
|  shufflenet_v2_x1_0  |  3.62  |  3.89  |  4.32  |  4.82   |
|  shufflenet_v2_x1_5  |  6.51  |  6.93  |  6.93  |  7.86   |
|  shufflenet_v2_x2_0  |  12.74  |  13.81  |  14.65  |  15.39   |
|  squeezenet1_0  |  11.70  |  12.07  |  15.05  |  15.33   |
|  squeezenet1_1  |  5.72  |  5.96  |  6.67  |  6.76   |
|  vgg11  |  87.00  |  90.56  |  164.98  |  166.76   |
|  vgg13  |  109.84  |  114.87  |  224.01  |  226.64   |
|  vgg16  |  135.97  |  143.35  |  292.71  |  298.62   |
|  vgg19  |  162.22  |  168.27  |  361.65  |  368.07   |
|  wide_resnet101_2  |  306.34  |  313.41  |  407.54  |  425.77   |
|  wide_resnet50_2  |  164.98  |  170.87  |  208.45  |  217.03   |

#### 4. 16-batch/16-thread with FMA3

|  Model Name  |  OpenPPL  |  OpenPPL v0.1  |  onnxruntime v1.8  |  OpenVINO r2021.2  |
| :------------ | :------------ | :------------ | :------------ | :------------ |
|  faster_rcnn\*  |  2857.25  |  2958.97  |    |    |
|  fsaf \* |  2020.64  |  2057.20  |  3708.59  |    |
|  mask_rcnn\*  |  4893.28  |  5018.35  |    |    |
|  retinanet\*  |  2332.15  |  2356.20  |  4334.09  |    |
|  deeplabv3\*  |  875.86  |  874.97  |  1054.09  |  1370.56   |
|  deeplabv3plus\*  |  684.21  |  683.62  |  973.05  |  884.11   |
|  fcn\*  |  604.05  |  603.18  |  930.05  |  937.06   |
|  pspnet\*  |  585.70  |  584.44  |  987.59  |  866.69   |
|  alexnet  |  14.25  |  14.46  |  17.53  |  22.56   |
|  densenet121  |  95.72  |  98.40  |  183.81  |  112.32   |
|  densenet161  |  237.12  |  246.96  |  438.04  |  287.10   |
|  densenet169  |  114.37  |  119.02  |  219.08  |  143.25   |
|  densenet201  |  154.54  |  162.34  |  291.88  |  195.45   |
|  googlenet  |  27.77  |  29.50  |  46.43  |  37.03   |
|  inception_v3  |  60.60  |  63.41  |  104.37  |  69.58   |
|  mnasnet0_5  |  3.61  |  5.29  |  4.32  |  3.72   |
|  mnasnet0_75  |  7.20  |  11.92  |  9.19  |  6.83   |
|  mnasnet1_0  |  9.78  |  14.42  |  12.43  |  9.69   |
|  mnasnet1_3  |  16.42  |  22.65  |  21.01  |  15.64   |
|  mobilenet_v2  |  9.23  |  14.73  |  18.67  |  9.08   |
|  resnet101  |  144.00  |  144.87  |  180.13  |  178.05   |
|  resnet152  |  200.32  |  204.74  |  257.08  |  257.34   |
|  resnet18  |  26.04  |  26.91  |  39.28  |  42.99   |
|  resnet34  |  45.60  |  46.41  |  77.56  |  83.74   |
|  resnet50  |  83.42  |  84.24  |  104.05  |  98.72   |
|  resnext101_32x8d  |  399.43  |  421.71  |  412.82  |  376.64   |
|  resnext50_32x4d  |  119.95  |  125.72  |  148.01  |  118.41   |
|  shufflenet_v2_x0_5  |  2.36  |  3.52  |  8.99  |  2.75   |
|  shufflenet_v2_x1_0  |  5.45  |  9.15  |  35.01  |  6.57   |
|  shufflenet_v2_x1_5  |  9.17  |  11.54  |  36.72  |  10.21   |
|  shufflenet_v2_x2_0  |  16.52  |  25.50  |  84.18  |  19.06   |
|  squeezenet1_0  |  24.50  |  25.35  |  39.81  |  20.79   |
|  squeezenet1_1  |  12.86  |  12.31  |  19.94  |  10.30   |
|  vgg11  |  74.79  |  77.03  |  172.38  |  181.85   |
|  vgg13  |  102.87  |  105.48  |  272.66  |  253.94   |
|  vgg16  |  132.07  |  135.28  |  354.90  |  339.40   |
|  vgg19  |  161.67  |  166.55  |  437.07  |  423.91   |
|  wide_resnet101_2  |  333.56  |  343.12  |  500.60  |  509.89   |
|  wide_resnet50_2  |  183.80  |  188.58  |  265.90  |  263.79   |

\* benchmarked with 4-batch/4-thread

### Appendix 2. OpenPPL Bechmark on 3700X

Platform Information:
 - CPU: [AMD Ryzen 7 3700X 8-Core Processor](https://www.amd.com/zh-hans/products/cpu/amd-ryzen-7-3700x)
 - Memory: 2-channel DDR4 @ 3200MHz
 - OS: Ubuntu 18.04 LTS
 - Toolchains: GCC 7.5.0

#### 1. 1-batch/1-thread

|  Model Name  |  OpenPPL  |  onnxruntime v1.8  |  OpenVINO r2021.2  |
| :------------ | :------------ | :------------ | :------------ |
|  faster_rcnn  |  2142.90  |    |    |
|  fsaf  |  1601.32  |  3101.56  |    |
|  mask_rcnn  |  3249.24  |    |    |
|  retinanet  |  1859.40  |  3624.42  |    |
|  deeplabv3  |  732.12  |  972.95  |  1111.05   |
|  deeplabv3plus  |  549.88  |  812.81  |  693.50   |
|  fcn  |  491.65  |  899.78  |  759.60   |
|  pspnet  |  478.39  |  829.70  |  695.12   |
|  esrgan  |  1815.03  |  4612.41  |  4128.80   |
|  srcnn  |  127.11  |  183.75  |  943.90   |
|  alexnet  |  16.45  |  20.29  |  18.04   |
|  densenet121  |  39.23  |  48.56  |  50.02   |
|  densenet161  |  100.05  |  140.40  |  141.88   |
|  densenet169  |  47.55  |  58.41  |  62.55   |
|  densenet201  |  62.83  |  75.07  |  82.64   |
|  googlenet  |  16.75  |  25.42  |  27.53   |
|  inception_v3  |  37.83  |  49.68  |  48.55   |
|  mnasnet0_5  |  2.21  |  2.50  |  2.48   |
|  mnasnet0_75  |  4.39  |  4.66  |  4.47   |
|  mnasnet1_0  |  6.29  |  6.64  |  6.39   |
|  mnasnet1_3  |  10.15  |  10.60  |  10.33   |
|  mobilenet_v2  |  5.84  |  6.77  |  6.05   |
|  resnet101  |  99.17  |  127.17  |  123.25   |
|  resnet152  |  137.47  |  184.05  |  180.55   |
|  resnet18  |  21.30  |  30.63  |  30.29   |
|  resnet34  |  37.45  |  61.64  |  59.56   |
|  resnet50  |  57.87  |  68.68  |  65.71   |
|  resnext101_32x8d  |  260.01  |  272.62  |  260.30   |
|  resnext50_32x4d  |  75.44  |  75.52  |  76.07   |
|  shufflenet_v2_x0_5  |  1.12  |  1.11  |  1.38   |
|  shufflenet_v2_x1_0  |  3.19  |  3.60  |  3.91   |
|  shufflenet_v2_x1_5  |  5.82  |  5.81  |  6.58   |
|  shufflenet_v2_x2_0  |  11.68  |  12.59  |  12.81   |
|  squeezenet1_0  |  10.94  |  14.24  |  13.93   |
|  squeezenet1_1  |  5.07  |  6.05  |  6.34   |
|  vgg11  |  58.35  |  135.29  |  133.81   |
|  vgg13  |  78.03  |  191.09  |  186.28   |
|  vgg16  |  98.70  |  258.40  |  248.63   |
|  vgg19  |  118.64  |  322.35  |  310.53   |
|  wide_resnet101_2  |  258.71  |  370.21  |  359.08   |
|  wide_resnet50_2  |  143.61  |  188.75  |  183.49   |

#### 2. 8-batch/8-thread

|  Model Name  |  OpenPPL  |  onnxruntime v1.8  |  OpenVINO r2021.2  |
| :------------ | :------------ | :------------ | :------------ |
|  faster_rcnn\*  |  2724.80  |    |    |
|  fsaf\*  |  1828.90  |  3521.65  |    |
|  mask_rcnn\*  |  5001.09  |    |    |
|  retinanet\*  |  2141.40  |  4064.99  |    |
|  deeplabv3\*  |  789.79  |  938.03  |  1210.12   |
|  deeplabv3plus\*  |  606.59  |  777.55  |  802.60   |
|  fcn\*  |  539.08  |  812.54  |  865.37   |
|  pspnet\*  |  523.69  |  873.55  |  795.23   |
|  alexnet  |  25.75  |  30.01  |  35.25   |
|  densenet121  |  189.81  |  231.90  |  205.37   |
|  densenet161  |  471.41  |  576.64  |  511.63   |
|  densenet169  |  223.96  |  280.82  |  252.81   |
|  densenet201  |  316.42  |  376.46  |  343.40   |
|  googlenet  |  54.45  |  72.61  |  69.43   |
|  inception_v3  |  108.85  |  153.29  |  111.21   |
|  mnasnet0_5  |  8.25  |  8.57  |  6.90   |
|  mnasnet0_75  |  15.22  |  19.10  |  12.28   |
|  mnasnet1_0  |  18.71  |  25.52  |  17.05   |
|  mnasnet1_3  |  31.46  |  42.32  |  28.45   |
|  mobilenet_v2  |  19.49  |  35.55  |  17.44   |
|  resnet101  |  246.19  |  289.77  |  283.49   |
|  resnet152  |  349.16  |  414.17  |  407.07   |
|  resnet18  |  44.36  |  67.10  |  67.00   |
|  resnet34  |  77.39  |  129.67  |  128.53   |
|  resnet50  |  143.80  |  167.50  |  160.32   |
|  resnext101_32x8d  |  652.26  |  616.42  |  590.32   |
|  resnext50_32x4d  |  202.90  |  209.55  |  207.75   |
|  shufflenet_v2_x0_5  |  5.26  |  6.77  |  5.32   |
|  shufflenet_v2_x1_0  |  10.89  |  31.01  |  13.54   |
|  shufflenet_v2_x1_5  |  16.74  |  37.72  |  19.70   |
|  shufflenet_v2_x2_0  |  29.31  |  87.77  |  42.48   |
|  squeezenet1_0  |  52.79  |  57.51  |  40.94   |
|  squeezenet1_1  |  27.24  |  30.80  |  21.17   |
|  vgg11  |  132.00  |  290.26  |  296.41   |
|  vgg13  |  176.84  |  494.38  |  413.66   |
|  vgg16  |  229.45  |  634.61  |  546.63   |
|  vgg19  |  281.66  |  771.16  |  680.21   |
|  wide_resnet101_2  |  571.07  |  805.13  |  787.45   |
|  wide_resnet50_2  |  313.85  |  427.07  |  408.02   |

\* benchmarked with 4-batch

### Appendix 3. OpenPPL Bechmark on ZHAOXIN KaiXian KX-6640MA

Platform Information:
 - CPU: ZHAOXIN KaiXian KX-6640MA
 - Memory: 1-channel DDR4 @ 2666MHz
 - OS: Ubuntu 21.04 LTS
 - Toolchains: GCC 10.3.0

#### 1-batch/1-thread

|  Model Name  |  OpenPPL  |  onnxruntime v1.8  |  OpenVINO r2021.2  |
| :------------ | :------------ | :------------ | :------------ |
|  faster_rcnn  |  41266.46  |   |   |
|  fsaf  |  32911.77  |  40583.10  |    |
|  mask_rcnn  |  54911.34  |   |    |
|  retinanet  |  38023.71  |  46869.10  |    |
|  deeplabv3  |  11092.04  |  10449.10  |  17504.02  |
|  deeplabv3plus  |  7523.28  |  9336.28  |  10434.66  |
|  fcn  |  8796.98  |  10199.10  |  11102.00  |
|  pspnet  |  7540.82  |  9277.38  |  10135.17  |
|  esrgan  |  55204.37  |  55303.60  |  53251.06  |
|  srcnn  |  1336.26  |  1608.47  |  1774.05  |
|  alexnet  |  221.27  |  164.30  |  162.26  |
|  densenet121  |  576.86  |  606.78  |  638.60  |
|  densenet161  |  1444.43  |  1670.39  |  1746.00  |
|  densenet169  |  685.06  |  723.69  |  766.02  |
|  densenet201  |  857.31  |  930.46  |  994.28  |
|  googlenet  |  328.26  |  305.16  |  313.00  |
|  inception_v3  |  592.46  |  575.86  |  594.00  |
|  mnasnet0_5  |  25.27  |  26.39  |  26.16  |
|  mnasnet0_75  |  46.77  |  51.49  |  52.16  |
|  mnasnet1_0  |  64.85  |  72.98  |  75.01  |
|  mnasnet1_3  |  104.34  |  120.05  |  125.47  |
|  mobilenet_v2  |  65.01  |  72.37  |  63.12  |
|  resnet101  |  1345.77  |  1590.25  |  1656.17  |
|  resnet152  |  1961.40  |  2335.19  |  2433.50  |
|  resnet18  |  343.18  |  356.18  |  356.55  |
|  resnet34  |  667.49  |  710.05  |  722.06  |
|  resnet50  |  730.50  |  846.86  |  880.33  |
|  resnext101_32x8d  |  2862.74  |  3470.37  |  3744.12  |
|  resnext50_32x4d  |  774.73  |  936.71  |  1007.18  |
|  shufflenet_v2_x0_5  |  15.86  |  11.80  |  12.24  |
|  shufflenet_v2_x1_0  |  35.12  |  39.13  |  52.56  |
|  shufflenet_v2_x1_5  |  61.41  |  69.55  |  74.47  |
|  shufflenet_v2_x2_0  |  114.47  |  141.20  |  173.45  |
|  squeezenet1_0  |  185.14  |  172.62  |  173.39  |
|  squeezenet1_1  |  90.32  |  74.29  |  75.89  |
|  vgg11  |  1451.70  |  1547.74  |  1509.99  |
|  vgg13  |  2098.08  |  2300.49  |  2250.96  |
|  vgg16  |  2762.70  |  3111.86  |  3060.71  |
|  vgg19  |  3423.60  |  3922.71  |  3868.17  |
|  wide_resnet101_2  |  3804.07  |  4534.11  |  4674.70  |
|  wide_resnet50_2  |  1946.25  |  2302.55  |  2372.67  |

### Appendix 4. OpenPPL Bechmark on Hygon C86 7185

Platform Information:
 - CPU: Hygon C86 7185
 - Memory: 4-channel DDR4 @ 2666MHz
 - OS: Ubuntu 16.04 LTS
 - Toolchains: GCC 5.4.0

#### 1-batch/1-thread

|  Model Name  |  OpenPPL  |  onnxruntime v1.8  |  OpenVINO r2021.2  |
| :------------ | :------------ | :------------ | :------------ |
|  faster_rcnn  |  14487.04  |    |    |
|  fsaf  |  12093.44  |  23463.60  |    |
|  mask_rcnn  |  19336.49  |    |    |
|  retinanet  |  13351.42  |  27274.00  |    |
|  deeplabv3  |  5251.41  |  5818.40  |  8992.17  |
|  deeplabv3plus  |  4310.80  |  5395.37  |  5441.46  |
|  fcn  |  3812.95  |  5969.52  |  5984.72  |
|  pspnet  |  3697.87  |  5444.76  |  5452.76  |
|  esrgan  |  11727.20  |  33567.80  |  32000.89  |
|  srcnn  |  782.28  |  849.77  |  947.87  |
|  alexnet  |  71.11  |  87.93  |  90.94  |
|  densenet121  |  267.63  |  343.74  |  348.94  |
|  densenet161  |  761.14  |  946.83  |  945.15  |
|  densenet169  |  334.38  |  410.59  |  418.24  |
|  densenet201  |  453.46  |  523.03  |  538.40  |
|  googlenet  |  110.02  |  176.90  |  176.65  |
|  inception_v3  |  274.76  |  333.47  |  333.04  |
|  mnasnet0_5  |  14.23  |  13.83  |  14.19  |
|  mnasnet0_75  |  29.01  |  27.63  |  28.02  |
|  mnasnet1_0  |  41.45  |  39.39  |  40.10  |
|  mnasnet1_3  |  68.72  |  65.17  |  66.32  |
|  mobilenet_v2  |  40.38  |  40.23  |  38.95  |
|  resnet101  |  723.07  |  917.33  |  910.78  |
|  resnet152  |  1030.11  |  1345.56  |  1338.91  |
|  resnet18  |  125.90  |  208.22  |  206.17  |
|  resnet34  |  219.87  |  417.33  |  416.24  |
|  resnet50  |  411.84  |  480.38  |  481.30  |
|  resnext101_32x8d  |  2056.74  |  1963.46  |  1980.55  |
|  resnext50_32x4d  |  556.95  |  519.52  |  538.74  |
|  shufflenet_v2_x0_5  |  6.28  |  5.94  |  6.44  |
|  shufflenet_v2_x1_0  |  20.80  |  22.79  |  21.83  |
|  shufflenet_v2_x1_5  |  39.90  |  37.80  |  39.43  |
|  shufflenet_v2_x2_0  |  78.44  |  82.77  |  78.14  |
|  squeezenet1_0  |  73.02  |  98.51  |  97.35  |
|  squeezenet1_1  |  35.29  |  42.16  |  42.02  |
|  vgg11  |  309.22  |  913.58  |  897.11  |
|  vgg13  |  445.47  |  1353.19  |  1331.30  |
|  vgg16  |  588.90  |  1844.19  |  1811.62  |
|  vgg19  |  732.26  |  2332.24  |  2288.74  |
|  wide_resnet101_2  |  1770.98  |  2634.87  |  2628.51  |
|  wide_resnet50_2  |  989.37  |  1334.92  |  1321.77  |
