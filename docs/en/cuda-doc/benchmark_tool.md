## Cuda Benchmark Tool

**Note: Openppl.cuda only supports Turing fp16. ONNX model does not need to convert manually, it will be done in inference process. **

Execute the command:

```C++
./pplnn --onnx-model model.onnx --inputs input.bin -–in-shapes n_c_h_w [--dims d] [--warmuptimes m] --enable-profiling
```

`n_c_h_w` represents the actual input image size

`--dims` sets the training image shapes, which is only used for the algorithm selection. It is optional but we recommend setting it to be the same as the input image. d has two formats.
First format is same as `--in-shapes`, several numbers splited by '_', which means setting all inputs into same input shapes. 
Second format is several pairs representing for node names and their shapes. different nodes splitted by ';'. Node and its shape splitted by ','.

`--warmuptimes` indicates warm up times. The number of m is optional, but a suitable m is over than 400.

Example:

```
./pplnn --onnx-model model.onnx --inputs input.bin --in-shapes 1_3_224_224 --dims 1_3_224_224 --warmuptimes 400 --enable-profiling

./pplnn --onnx-model model.onnx --inputs input1.bin,input2.bin --in-shapes 4_3_1200_1200,4_8 --dims input1,4_3_800_1216;input2,4_8 --warmuptimes 400 --enable-profiling
```

The running cost is shown in log as following:

```
Average run cost: *** ms.
```

## PPL and TRT Bechmark for single batch on T4:

All tested model are dynamic model created by pytorch model zoo.

```
./pplnn --onnx-model model.onnx --inputs input.bin -–in-shapes 1_3_224_224 --dims 1_3_224_224 --warmuptimes 400 --enable-profiling
```


| model name                     | PPL      | TRT-8    |
|--------------------------|----------|----------|
| alexnet                  | 0.60907  | 0.66893  |
| densenet121_opt          | 2.64249  | 3.13724  |
| densenet161_opt          | 5.23693  | 6.60783  |
| densenet169_opt          | 4.28589  | 5.14267  |
| densenet201_opt          | 6.00780  | 7.25393  |
| googlenet                | 0.59959  | 0.59827  |
| inception_v3_opt         | 1.18229  | 1.30643  |
| mnasnet0_5               | 0.30234  | 0.42881  |
| mnasnet0_75              | 0.36110  | 0.48128  |
| mnasnet1_0               | 0.40067  | 0.51282  |
| mnasnet1_3               | 0.49457  | 0.58434  |
| mobilenet_v2_opt         | 0.41884  | 0.41421  |
| resnet101                | 1.93669  | 1.98859  |
| resnet152                | 2.70742  | 2.85473  |
| resnet18                 | 0.46188  | 0.51945  |
| resnet34                 | 0.80389  | 0.93401  |
| resnet50                 | 1.00594  | 1.07656  |
| resnext101_32x8d         | 3.18114  | 4.05643  |
| resnext50_32x4d          | 1.31330  | 1.05280  |
| shufflenet_v2_x0_5_opt   | 0.37315  | 0.42875  |
| shufflenet_v2_x1_0_opt   | 0.43867  | 0.60099  |
| shufflenet_v2_x1_5_opt   | 0.49457  | 0.70718  |
| shufflenet_v2_x2_0_opt   | 0.62771  | 0.69287  |
| squeezenet1_0            | 0.32559  | 0.32315  |
| squeezenet1_1            | 0.21896  | 0.22982  |
| vgg11_bn                 | 2.05248  | 1.77883  |
| vgg11                    | 2.05087  | 1.76463  |
| vgg13_bn                 | 2.32206  | 2.08602  |
| vgg13                    | 2.32517  | 2.08724  |
| vgg16_bn                 | 2.71657  | 2.47089  |
| vgg16                    | 2.71951  | 2.43448  |
| vgg19_bn                 | 3.15396  | 2.79457  |
| vgg19                    | 3.24230  | 2.81195  |
| wide_resnet101_2         | 4.14432  | 3.95147  |
| wide_resnet50_2          | 2.18955  | 2.01969  |