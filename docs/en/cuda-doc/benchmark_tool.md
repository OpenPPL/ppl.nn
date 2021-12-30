## Cuda Benchmark Tool

**Note: Openppl.cuda only supports Turing fp16 and int8. ONNX model does not need to convert manually, it will be done in inference process. **

### Fp16 test

Execute the command:

```C++
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w [--warmup-iterations m] --enable-profiling
```

`n_c_h_w` represents the actual input image size

`--warmup-iterations` indicates warm up times. The number of m is optional, but a suitable m is over than 400.

### Int8 test

There are two methods testing the network performance under int8 accuracy. You can specify the quantization file or specify the kernel type.

1. Use quantization file

```C++
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w --quantization quant.json [--warmup-iterations m] --enable-profiling
```

`--quant-file` indicates your quantization json file. We show a sample for the standard format at [json_format](../../../tests/testdata/quant_test.json)

2. Use default kernel type

```C++
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w --kernel-type DATATYPE_INT8 [--warmup-iterations m] --enable-profiling
```

We also support testing benchmark without quantitation file. You can specify all kernel types as int8 through `--kernel-type DATATYPE_INT8`, and our framework will preferentially running kernels with int8 precision.


###Example:

```
// fp16
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin --in-shapes 1_3_224_224 --warmup-iterations 100 --enable-profiling

./pplnn --use-cuda --onnx-model model.onnx --inputs input1.bin,input2.bin --in-shapes 4_3_1200_1200,4_8 --warmup-iterations 400 --enable-profiling

// int8
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin --in-shapes 1_3_224_224 --quant-file quant.json --warmup-iterations 100 --enable-profiling

./pplnn --use-cuda --onnx-model model.onnx --inputs input1.bin,input2.bin --in-shapes 4_3_1200_1200,4_8 --kernel-type DATATYPE_INT8 --warmup-iterations 400 --enable-profiling
```

The running cost is shown in log as following:

```
Average run cost: *** ms.
```

## PPL and TRT Bechmark for single batch on T4:

All tested model are dynamic model created by pytorch model zoo.

```
./pplnn --use-cuda --onnx-model model.onnx --inputs input.bin -–in-shapes 1_3_224_224 --warmup-iterations 400 --enable-profiling
```


| model name                  | PPL(fp16)  | TRT-8(fp16)  | PPL(int8)  | TRT-8(int8)  |
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
