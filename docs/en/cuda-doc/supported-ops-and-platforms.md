## Supported precision

CUDA only supports FP16 precision and int8 precision on Turing Devices. 
If you want to use int8 percision, you should maintain a json file like [sample quant file](../../../tests/testdata/quant_test.json)

## Supported operators and opsets

* ONNX

| Op Type              | Op Set | Linux/Windows CUDA |
|:--------------------:|:------:|:------------------:|
| Add                  | 7~12   | &check;            |
| And                  | 7~16   | &check;            |
| ArgMax               | 1~11   | &check;            |
| AveragePool          | 1~16   | &check;            |
| BatchNormalization   | 9~13   | &check;            |
| Cast                 | 9~12   | &check;            |
| Ceil                 | 6~12   | &check;            |
| Clip                 | 6~13   | &check;            |
| Concat               | 4~12   | &check;            |
| ConstantOfShape      | 9~16   | &check;            |
| Conv                 | 1~16   | &check;            |
| ConvTranspose        | 1~16   | &check;            |
| Cos                  | 7~16   | &check;            |
| CumSum               | 11~16  | &check;            |
| DepthToSpace         | 1~12   | &check;            |
| Div                  | 7~12   | &check;            |
| Equal                | 7~16   | &check;            |
| Erf                  | 9~16   | &check;            |
| Exp                  | 6~12   | &check;            |
| Expand               | 8~12   | &check;            |
| Flatten              | 1~12   | &check;            |
| Floor                | 6~16   | &check;            |
| Gather               | 1~16   | &check;            |
| GatherND             | 11     | &check;            |
| Gemm                 | 11~12  | &check;            |
| GlobalAveragePool    | 1~16   | &check;            |
| GlobalMaxPool        | 1~16   | &check;            |
| Greater              | 9~16   | &check;            |
| GreaterOrEqual       | 9~16   | &check;            |
| Identity             | 1~12   | &check;            |
| If                   | 1~12   | &check;            |
| InstanceNormalization| 6~13   | &check;            |
| LeakyRelu            | 6~16   | &check;            |
| Less                 | 9~16   | &check;            |
| Log                  | 6~12   | &check;            |
| Loop                 | 1~12   | &check;            |
| LSTM                 | 7~13   | &check;            |
| MatMul               | 9~12   | &check;            |
| Max                  | 8~11   | &check;            |
| MaxPool              | 1~16   | &check;            |
| MaxUnpool            | 9~16   | &check;            |
| Min                  | 8~11   | &check;            |
| Mul                  | 7~12   | &check;            |
| NonMaxSuppression    | 10~16  | &check;            |
| NonZero              | 9~12   | &check;            |
| Not                  | 1~16   | &check;            |
| Pad                  | 2~12   | &check;            |
| Pow                  | 7~11   | &check;            |
| Range                | 11~16  | &check;            |
| ReduceL2             | 1~16   | &check;            |
| ReduceMax            | 1~16   | &check;            |
| ReduceMean           | 1~16   | &check;            |
| ReduceMin            | 1~16   | &check;            |
| ReduceProd           | 1~16   | &check;            |
| ReduceSum            | 1~16   | &check;            |
| Relu                 | 6~12   | &check;            |
| Reshape              | 5~12   | &check;            |
| Resize               | 11~12  | &check;            |
| RoiAlign             | 10~15  | &check;            |
| ScatterElements      | 11~12  | &check;            |
| ScatterND            | 11~12  | &check;            |
| SequenceAt           | 11~16  | &check;            |
| Shape                | 1~12   | &check;            |
| Sigmoid              | 6~12   | &check;            |
| Sin                  | 1~16   | &check;            |
| Slice                | 1~12   | &check;            |
| Softmax              | 1~12   | &check;            |
| Split                | 2~12   | &check;            |
| SplitToSequence      | 11~16  | &check;            |
| Sqrt                 | 6~12   | &check;            |
| Squeeze              | 1~12   | &check;            |
| Sub                  | 7~12   | &check;            |
| Tanh                 | 6~12   | &check;            |
| Tile                 | 6~12   | &check;            |
| TopK                 | 11~16  | &check;            |
| Transpose            | 1~12   | &check;            |
| Unsqueeze            | 1~12   | &check;            |
| Where                | 9~15   | &check;            |

* MMCV

| Op Type                 | Op Set | Linux/Windows CUDA |
|:-----------------------:|:------:|:------------------:|
| grid_sample             | 1      | &check;            |
| ModulatedDeformConv2d   | 1      | &check;            |
| NonMaxSuppression       | 1      | &check;            |
| RoiAlign                | 1      | &check;            |

* PPL

| Op Type                              | Op Set | Linux/Windows CUDA |
|:------------------------------------:|:------:|:------------------:|
| ChannelShuffle                       | 1      | &check;            |
| Reduce                               | 1      | &check;            |
| [ShapeOperation](shape_operation.md) | 1      | &check;            |
