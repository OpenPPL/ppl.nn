## Supported precision

1. CUDA only supports FP16 precision on Turing Devices.
2. x86 only supports FP32 precision on AVX512/FMA.


## Supported operators and opsets

* ONNX

| Op Type            | Op Set | Linux/Windows/Darwin X86-64 | Linux/Windows CUDA |
|:------------------:|:------:|:---------------------------:|:------------------:|
| Add                | 7~12   | &check;                     | &check;            |
| And                | 7~16   | &check;                     | &check;            |
| ArgMax             | 11     | &check;                     | &check;            |
| AveragePool        | 11~16  | &check;                     | &check;            |
| BatchNormalization | 9~13   | &check;                     | &check;            |
| Cast               | 9~12   | &check;                     | &check;            |
| Ceil               | 6~12   | &check;                     | &check;            |
| Clip               | 11     | &check;                     | &check;            |
| Concat             | 11~12  | &check;                     | &check;            |
| Constant           | 9~16   | &check;                     |                    |
| ConstantOfShape    | 9~16   | &check;                     | &check;            |
| Conv               | 11~16  | &check;                     | &check;            |
| ConvTranspose      | 11~16  | &check;                     | &check;            |
| DepthToSpace       | 11~12  | &check;                     | &check;            |
| Div                | 7~12   | &check;                     | &check;            |
| Equal              | 11~12  | &check;                     | &check;            |
| Exp                | 6~12   | &check;                     | &check;            |
| Expand             | 8~12   | &check;                     | &check;            |
| Flatten            | 11~12  | &check;                     | &check;            |
| Floor              | 11~12  | &check;                     | &check;            |
| Gather             | 11~12  | &check;                     | &check;            |
| GatherND           | 11     | &check;                     | &check;            |
| Gemm               | 11~12  | &check;                     | &check;            |
| Greater            | 9~12   | &check;                     | &check;            |
| Identity           | 1~12   | &check;                     | &check;            |
| If                 | 11~12  | &check;                     | &check;            |
| LeakyRelu          | 6~16   | &check;                     | &check;            |
| Less               | 9~12   | &check;                     | &check;            |
| Log                | 6~12   | &check;                     | &check;            |
| Loop               | 11~12  | &check;                     | &check;            |
| LSTM               | 7~13   | &check;                     | &check;            |
| MatMul             | 9~12   | &check;                     | &check;            |
| Max                | 8~11   | &check;                     | &check;            |
| MaxPool            | 11     | &check;                     | &check;            |
| MaxUnpool          | 11~16  | &check;                     | &check;            |
| Min                | 8~11   | &check;                     | &check;            |
| Mul                | 7~12   | &check;                     | &check;            |
| NonMaxSuppression  | 11~16  | &check;                     | &check;            |
| NonZero            | 9~12   | &check;                     | &check;            |
| Not                | 1~16   | &check;                     | &check;            |
| Pad                | 11~12  | &check;                     | &check;            |
| Pow                | 7~11   | &check;                     | &check;            |
| Range              | 11~16  | &check;                     | &check;            |
| ReduceMax          | 11     | &check;                     | &check;            |
| ReduceMean         | 11~12  | &check;                     | &check;            |
| ReduceMin          | 11     | &check;                     | &check;            |
| ReduceProd         | 11~12  | &check;                     | &check;            |
| ReduceSum          | 11~12  | &check;                     | &check;            |
| Relu               | 6~12   | &check;                     | &check;            |
| Reshape            | 5~12   | &check;                     | &check;            |
| Resize             | 11~12  | &check;                     | &check;            |
| RoiAlign           | 10~15  | &check;                     | &check;            |
| ScatterElements    | 11~12  | &check;                     | &check;            |
| ScatterND          | 11~12  | &check;                     | &check;            |
| SequenceAt         | 11~16  | &check;                     | &check;            |
| Shape              | 1~12   | &check;                     | &check;            |
| Sigmoid            | 6~12   | &check;                     | &check;            |
| Slice              | 11~12  | &check;                     | &check;            |
| Softmax            | 11~12  | &check;                     | &check;            |
| Split              | 11~12  | &check;                     | &check;            |
| SplitToSequence    | 11~16  | &check;                     | &check;            |
| Sqrt               | 6~12   | &check;                     | &check;            |
| Squeeze            | 11~12  | &check;                     | &check;            |
| Sub                | 7~12   | &check;                     | &check;            |
| Sum                | 8~12   | &check;                     |                    |
| Tanh               | 6~12   | &check;                     | &check;            |
| Tile               | 6~12   | &check;                     | &check;            |
| TopK               | 11~16  | &check;                     | &check;            |
| Transpose          | 1~12   | &check;                     | &check;            |
| Unsqueeze          | 11~12  | &check;                     | &check;            |
| Where              | 9~15   | &check;                     | &check;            |

* MMCV

| Op Type           | Op Set | Linux/Windows/Darwin X86-64 | Linux/Windows CUDA |
|:-----------------:|:------:|:---------------------------:|:------------------:|
| NonMaxSuppression | 1      | &check;                     | &check;            |
| RoiAlign          | 1      | &check;                     | &check;            |
| grid_sample       | 1      | &check;                     | &check;            |

* PPL

| Op Type                              | Op Set | Linux/Windows/Darwin X86-64 | Linux/Windows CUDA |
|:------------------------------------:|:------:|:---------------------------:|:------------------:|
| ChannelShuffle                       | 1      | &check;                     | &check;            |
| [ShapeOperation](shape_operation.md) | 1      | &check;                     | &check;            |
| Swish                                | 1      | &check;                     |                    |
