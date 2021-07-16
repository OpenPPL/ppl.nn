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

