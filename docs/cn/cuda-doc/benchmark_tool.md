## Cuda Benchmark 工具

执行语句:

```C++
./pplnn --onnx-model model.onnx --inputs input.bin –in-shapes n_c_h_w [--dims d] [--warmuptimes m] [--runningtimes n]
```

`n_c_h_w` 表示实际推理图片大小，使用NCHW排布。

`--dims` 表示选算法时输入的图片大小，推荐和推理图片大小一致。它有两种输入格式。一种只包含输入形状，以'_'分割，所有输入形状都会设置为该大小。另一种是多组数据，每组数据都是边的名字加输入形状的格式，每组数据间用';'分割，名字和输入形状用','分割，此时，选算法过程会根据指定的名称设置相应的形状。

`--warmuptimes` 代表预热次数，`--runningtimes` 代表执行次数，两者默认状态是预热0次，执行1次。测试性能推荐设置两个参数值大于100。

测试样例:

```
./pplnn --onnx-model model.onnx --inputs input.bin --in-shapes 1_3_224_224 --dims 1_3_224_224 --warmuptimes 100 --runningtimes 100

./pplnn --onnx-model model.onnx --inputs input1.bin,input2.bin --in-shapes 4_3_1200_1200,4_8 --dims input1,4_3_800_1216;input2,4_8 --warmuptimes 400 --runningtimes 200
```

运行时间以如下格式打印:

```
Run() costs: *** ms.
```

模型的执行效率就是运行时间除以运行次数。

