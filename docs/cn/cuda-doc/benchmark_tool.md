## Cuda Benchmark 工具

**注：Openppl.cuda目前仅支持Turing/Ampere fp16 和 int8。ONNX模型无需手动转换，框架解析过程会自动转换到fp16精度。**

### cuda独有参数解析

`--use-cuda` 使用CUDA的必选项

`--kernel-type`  指定一个数据类型，强制所有op按此类型执行算子，目前可选项包括 int8,float16.

`--quick-select` 跳过Conv/Gemm/Matmul的常规选算法，直接指定一个通用算法执行算子。可以节约预处理时间，但是性能会变差。建议在算子非常多的模型中使用这个参数。

`--export-algo-file` 导出选算法的结果到一个json文件，用于下一次执行该模型时导入

`--import-algo-file` 导入之前保存的json文件，OpenPPL会根据文件选择算法而不用重新花费时间做选算法的过程，可以在第二次执行模型时减少预处理时间，同时保证最优的执行性能。

`--quant-file` 适用于执行量化模型，需要添加一个量化文件，一个简单的量化文件参见[样例](../../../tests/testdata/quant_test.json).

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
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w --kernel-type int8 [--warmup-iterations m] --enable-profiling
```

同时，我们提供不依赖量化文件的测试性能。通过`--kernel-type int8`将所有的算子类型指定为int8，框架会优先调用int8精度下的算子。

### 测试样例

```
// fp16
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin --in-shapes 1_3_224_224 --warmup-iterations 100 --enable-profiling

./pplnn --use-cuda --onnx-model model.onnx --inputs input1.bin,input2.bin --in-shapes 4_3_1200_1200,4_8 --warmup-iterations 400 --enable-profiling

// int8
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin --in-shapes 1_3_224_224 --quant-file quant.json --warmup-iterations 100 --enable-profiling

./pplnn --use-cuda --onnx-model model.onnx --inputs input1.bin,input2.bin --in-shapes 4_3_1200_1200,4_8 --kernel-type int8 --warmup-iterations 400 --enable-profiling
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


## PPL 及 TRT 在A100显卡上单batch Banchmark

测试模型均为pytorch model zoo导出的动态模型。测试指令如下：

```
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin -–in-shapes 1_3_224_224 --warmup-iterations 400 --enable-profiling
```

测试结果对比如下：

| 模型名称               | PPL(fp16)  | TRT-8(fp16)  | PPL(int8)  | TRT-8(int8)  |
|--------------------------|----------|----------|----------|-----------|
| alexnet                  | 0.69166  | 0.75580  | 0.57939  | 0.58799   |
| densenet121_opt          | 1.31893  | 1.47582  | 1.11157  | 1.13391   |
| densenet161_opt          | 0.77013  | 0.61255  | 0.48386  | 0.44009   |
| densenet169_opt          | 0.77038  | 0.61238  | 0.47408  | 0.43213   |
| densenet201_opt          | 0.68839  | 0.53785  | 0.42359  | 0.38924   |
| googlenet                | 0.68250  | 0.54045  | 0.42496  | 0.38812   |
| inception_v3_opt         | 0.59749  | 0.47030  | 0.36946  | 0.33871   |
| mnasnet0_5               | 0.59844  | 0.46495  | 0.37036  | 0.34008   |
| mnasnet0_75              | 0.53366  | 0.42194  | 0.31743  | 0.30304   |
| mnasnet1_0               | 0.53305  | 0.41373  | 0.31547  | 0.30280   |
| mnasnet1_3               | 0.27181  | 0.21609  | 0.18845  | 0.19090   |
| mobilenet_v2_opt         | 0.27216  | 0.24206  | 0.20574  | 0.21182   |
| resnet101                | 0.51198  | 0.68966  | 0.44752  | 0.66234   |
| resnet152                | 0.50150  | 0.65037  | 0.43025  | 0.60930   |
| resnet18                 | 0.48689  | 0.63296  | 0.41215  | 0.60837   |
| resnet34                 | 0.46391  | 0.47182  | 0.39389  | 0.47703   |
| resnet50                 | 0.55127  | 0.52909  | 0.55861  | 0.44232   |
| resnext101_32x8d         | 1.32752  | 1.16566  | 1.02201  | 0.91733   |
| resnext50_32x4d          | 0.52467  | 0.60301  | 0.49741  | 0.47155   |
| shufflenet_v2_x0_5_opt   | 0.42961  | 0.53858  | 0.41207  | 0.39974   |
| shufflenet_v2_x1_0_opt   | 0.24522  | 0.29660  | 0.21845  | 0.21931   |
| shufflenet_v2_x1_5_opt   | 1.40943  | 1.67678  | 1.38994  | 1.25584   |
| shufflenet_v2_x2_0_opt   | 0.96760  | 1.13811  | 0.95007  | 0.86511   |
| squeezenet1_0            | 0.37607  | 0.38720  | 0.36516  | 0.32030   |
| squeezenet1_1            | 0.37983  | 0.47629  | 0.35943  | 0.36411   |
| vgg11_bn                 | 0.36366  | 0.45259  | 0.33725  | 0.34626   |
| vgg11                    | 0.35880  | 0.43913  | 0.35753  | 0.34363   |
| vgg13_bn                 | 0.35656  | 0.42420  | 0.33973  | 0.32771   |
| vgg13                    | 1.01372  | 0.94329  | 1.05876  | 0.75047   |
| vgg16_bn                 | 0.54162  | 0.47882  | 0.45099  | 0.41977   |
| vgg16                    | 7.91039  | 9.86797  | 8.06547  | 8.41490   |
| vgg19_bn                 | 5.59416  | 6.54615  | 5.50678  | 5.57383   |
| vgg19                    | 5.69565  | 6.28967  | 5.16531  | 6.08624   |
| wide_resnet101_2         | 3.20323  | 3.51518  | 2.97470  | 2.96758   |
| wide_resnet50_2          | 0.27644  | 0.24743  | 0.17007  | 0.20691   |

