## Cuda Benchmark Tool

Execute the command:

```C++
./pplnn --onnx-model model.onnx --inputs input.bin –in-shapes n_c_h_w [--dims d] [--warmuptimes m] [--runningtimes n]
```

`n_c_h_w` represents the actual input image size

`--dims` sets the training image shapes, which is only used for the algorithm selection. It is optional but we recommend setting it to be the same as the input image. d has two formats.
First format is same as `--in-shapes`, several numbers splited by '_', which means setting all inputs into same input shapes. 
Second format is several pairs representing for node names and their shapes. different nodes splitted by ';'. Node and its shape splitted by ','.

`--warmuptimes` indicates warm up times, and `--runningtimes` indicates running times. The number of m and n are optional, but a suitable number is over than 100.

Example:

```
./pplnn --onnx-model model.onnx --inputs input.bin --in-shapes 1_3_224_224 --dims 1_3_224_224 --warmuptimes 100 --runningtimes 100

./pplnn --onnx-model model.onnx --inputs input1.bin,input2.bin --in-shapes 4_3_1200_1200,4_8 --dims input1,4_3_800_1216;input2,4_8 --warmuptimes 400 --runningtimes 200
```

The running cost is shown in log as following:

```
Run() costs: *** ms.
```


The average running time for once reasoning is running cost divided by the number of running times you set.
