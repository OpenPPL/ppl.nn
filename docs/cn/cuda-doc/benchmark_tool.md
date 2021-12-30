## Cuda Benchmark 工具

**注：Openppl.cuda目前仅支持Turing fp16 和 int8。ONNX模型无需手动转换，框架解析过程会自动转换到fp16精度。**

### Fp16 测试

执行语句:

```
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w [--warmup-iterations m] --enable-profiling
```

`n_c_h_w` 表示实际推理图片大小，使用NCHW排布。

`--warmup-iterations` 代表预热次数，测试性能推荐设置预热参数值大于400。

### Int8 测试

测试int8精度下网络性能需要指定量化文件或者指定数据类型。

1. 指定量化文件

```C++
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w --quant-file quant.json [--warmup-iterations m] --enable-profiling
```

`--quantizaiton` 指定量化文件，标准量化文件格式参见 [json_format](../../../tests/testdata/quant_test.json)

2. 指定数据类型

```C++
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w --kernel-type DATATYPE_INT8 [--warmup-iterations m] --enable-profiling
```

同时，我们提供不依赖量化文件的测试性能。通过`--kernel-type DATATYPE_INT8`将所有的算子类型指定为int8，框架会优先调用int8精度下的算子。

### 测试样例

```
// fp16
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin --in-shapes 1_3_224_224 --warmup-iterations 100 --enable-profiling

./pplnn --use-cuda --onnx-model model.onnx --inputs input1.bin,input2.bin --in-shapes 4_3_1200_1200,4_8 --warmup-iterations 400 --enable-profiling

// int8
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin --in-shapes 1_3_224_224 --quant-file quant.json --warmup-iterations 100 --enable-profiling

./pplnn --use-cuda --onnx-model model.onnx --inputs input1.bin,input2.bin --in-shapes 4_3_1200_1200,4_8 --kernel-type DATATYPE_INT8 --warmup-iterations 400 --enable-profiling
```

运行时间以如下格式打印:

```
Average run cost: *** ms.
```

## PPL 及 TRT 在T4显卡上单batch Banchmark

测试模型均为pytorch model zoo导出的动态模型。测试指令如下：

```
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin -–in-shapes 1_3_224_224 --warmup-iterations 400 --enable-profiling
```

测试结果对比如下：

| 模型名称               | PPL(fp16)  | TRT-8(fp16)  | PPL(int8)  | TRT-8(int8)  |
|--------------------------|----------|----------|----------|-----------|
| alexnet                  | 0.60907  | 0.66893  | 0.37125  | 0.463265  |
| densenet121_opt          | 2.64249  | 3.13724  | 2.33323  | 2.385010  |
| densenet161_opt          | 5.23693  | 6.60783  | 4.51682  | 5.670840  |
| densenet169_opt          | 4.28589  | 5.14267  | 4.12065  | 4.559300  |
| densenet201_opt          | 6.00780  | 7.25393  | 5.67984  | 5.752000  |
| googlenet                | 0.59959  | 0.59827  | 0.50674  | 0.475418  |
| inception_v3_opt         | 1.18229  | 1.30643  | 0.94391  | 1.020050  |
| mnasnet0_5               | 0.30234  | 0.42881  | 0.27158  | 0.289278  |
| mnasnet0_75              | 0.36110  | 0.48128  | 0.31308  | 0.325904  |
| mnasnet1_0               | 0.40067  | 0.51282  | 0.34513  | 0.354618  |
| mnasnet1_3               | 0.49457  | 0.58434  | 0.38310  | 0.392936  |
| mobilenet_v2_opt         | 0.41884  | 0.41421  | 0.33321  | 0.323009  |
| resnet101                | 1.93669  | 1.98859  | 1.37663  | 1.430260  |
| resnet152                | 2.70742  | 2.85473  | 1.99847  | 2.063570  |
| resnet18                 | 0.46188  | 0.51945  | 0.32491  | 0.385676  |
| resnet34                 | 0.80389  | 0.93401  | 0.57163  | 0.700629  |
| resnet50                 | 1.00594  | 1.07656  | 0.72971  | 0.753079  |
| resnext101_32x8d         | 3.18114  | 4.05643  | 2.32215  | 2.249280  |
| resnext50_32x4d          | 1.31330  | 1.05280  | 1.14167  | 0.663751  |
| shufflenet_v2_x0_5_opt   | 0.37315  | 0.42875  | 0.32093  | 0.406154  |
| shufflenet_v2_x1_0_opt   | 0.43867  | 0.60099  | 0.35906  | 0.544227  |
| shufflenet_v2_x1_5_opt   | 0.49457  | 0.70718  | 0.39081  | 0.637523  |
| shufflenet_v2_x2_0_opt   | 0.62771  | 0.69287  | 0.44558  | 0.678042  |
| squeezenet1_0            | 0.32559  | 0.32315  | 0.25710  | 0.236034  |
| squeezenet1_1            | 0.21896  | 0.22982  | 0.20898  | 0.182581  |
| vgg11_bn                 | 2.05248  | 1.77883  | 1.02692  | 1.165940  |
| vgg11                    | 2.05087  | 1.76463  | 1.03267  | 1.156350  |
| vgg13_bn                 | 2.32206  | 2.08602  | 1.26304  | 1.311380  |
| vgg13                    | 2.32517  | 2.08724  | 1.26259  | 1.331050  |
| vgg16_bn                 | 2.71657  | 2.47089  | 1.50753  | 1.538240  |
| vgg16                    | 2.71951  | 2.43448  | 1.50485  | 1.563360  |
| vgg19_bn                 | 3.15396  | 2.79457  | 1.75063  | 1.782030  |
| vgg19                    | 3.24230  | 2.81195  | 1.74979  | 1.768750  |
| wide_resnet101_2         | 4.14432  | 3.95147  | 2.42690  | 3.070870  |
| wide_resnet50_2          | 2.18955  | 2.01969  | 1.27713  | 1.475030  |
