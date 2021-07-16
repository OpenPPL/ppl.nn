## Cuda Benchmark 工具

**注：Openppl.cuda目前仅支持Turing fp16。ONNX模型无需手动转换，框架解析过程会自动转换到fp16精度。**

执行语句:

```C++
./pplnn --onnx-model model.onnx --inputs input.bin –in-shapes n_c_h_w [--dims d] [--warmuptimes m] --enable-profiling
```

`n_c_h_w` 表示实际推理图片大小，使用NCHW排布。

`--dims` 表示选算法时输入的图片大小，推荐和推理图片大小一致。它有两种输入格式。一种只包含输入形状，以'_'分割，所有输入形状都会设置为该大小。另一种是多组数据，每组数据都是边的名字加输入形状的格式，每组数据间用';'分割，名字和输入形状用','分割，此时，选算法过程会根据指定的名称设置相应的形状。

`--warmuptimes` 代表预热次数，测试性能推荐设置预热参数值大于400。

测试样例:

```
./pplnn --onnx-model model.onnx --inputs input.bin --in-shapes 1_3_224_224 --dims 1_3_224_224 --warmuptimes 100 --enable-profiling

./pplnn --onnx-model model.onnx --inputs input1.bin,input2.bin --in-shapes 4_3_1200_1200,4_8 --dims input1,4_3_800_1216;input2,4_8 --warmuptimes 400 --enable-profiling
```

运行时间以如下格式打印:

```
Average run cost: *** ms.
```