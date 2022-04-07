## Supported precision

x86 only supports FP32 precision on AVX512/FMA.

## Supported operators and opsets

* ONNX

| Op Type            | Op Set | Linux/Windows/Darwin X86-64 |
|:------------------:|:------:|:---------------------------:|
| Add                | 7~16   | &check;                     |
| And                | 7~16   | &check;                     |
| ArgMax             | 1~11   | &check;                     |
| AveragePool        | 1~16   | &check;                     |
| BatchNormalization | 9~13   | &check;                     |
| Cast               | 9~16   | &check;                     |
| Ceil               | 6~16   | &check;                     |
| Clip               | 6~16   | &check;                     |
| Concat             | 4~16   | &check;                     |
| Constant           | 9~16   | &check;                     |
| ConstantOfShape    | 9~16   | &check;                     |
| Conv               | 1~16   | &check;                     |
| ConvTranspose      | 1~16   | &check;                     |
| Cos                | 7~16   | &check;                     |
| CumSum             | 11~16  | &check;                     |
| DepthToSpace       | 1~16   | &check;                     |
| Dropout            | 1~16   | &check;                     |
| Div                | 7~16   | &check;                     |
| Equal              | 7~16   | &check;                     |
| Erf                | 9~16   | &check;                     |
| Exp                | 6~16   | &check;                     |
| Expand             | 8~16   | &check;                     |
| Flatten            | 1~16   | &check;                     |
| Floor              | 6~16   | &check;                     |
| Gather             | 1~16   | &check;                     |
| GatherND           | 11     | &check;                     |
| Gemm               | 9~16   | &check;                     |
| GlobalAveragePool  | 1~16   | &check;                     |
| Greater            | 7~16   | &check;                     |
| Identity           | 1~13   | &check;                     |
| If                 | 1~12   | &check;                     |
| LeakyRelu          | 6~16   | &check;                     |
| Less               | 7~16   | &check;                     |
| Log                | 6~16   | &check;                     |
| Loop               | 1~12   | &check;                     |
| LSTM               | 7~13   | &check;                     |
| MatMul             | 1~16   | &check;                     |
| Max                | 6~16   | &check;                     |
| MaxPool            | 1~16   | &check;                     |
| MaxUnpool          | 9~16   | &check;                     |
| Min                | 6~16   | &check;                     |
| Mul                | 7~16   | &check;                     |
| NonMaxSuppression  | 10~16  | &check;                     |
| NonZero            | 9~16   | &check;                     |
| Not                | 1~16   | &check;                     |
| Pad                | 2~16   | &check;                     |
| Pow                | 7~16   | &check;                     |
| PRelu              | 6~16   | &check;                     |
| Range              | 11~16  | &check;                     |
| ReduceMax          | 1~16   | &check;                     |
| ReduceMean         | 1~16   | &check;                     |
| ReduceMin          | 1~16   | &check;                     |
| ReduceProd         | 1~16   | &check;                     |
| ReduceSum          | 1~16   | &check;                     |
| Relu               | 6~16   | &check;                     |
| Reshape            | 5~13   | &check;                     |
| Resize             | 11~16  | &check;                     |
| RoiAlign           | 10~15  | &check;                     |
| ScatterElements    | 11~15  | &check;                     |
| ScatterND          | 11~15  | &check;                     |
| SequenceAt         | 11~16  | &check;                     |
| Shape              | 1~14   | &check;                     |
| Sigmoid            | 6~16   | &check;                     |
| Sin                | 7~16   | &check;                     |
| Slice              | 1~16   | &check;                     |
| Softmax            | 1~12   | &check;                     |
| Split              | 2~12   | &check;                     |
| SplitToSequence    | 11~16  | &check;                     |
| Sqrt               | 6~16   | &check;                     |
| Squeeze            | 1~12   | &check;                     |
| Sub                | 7~16   | &check;                     |
| Sum                | 6~16   | &check;                     |
| Tanh               | 6~16   | &check;                     |
| Tile               | 6~16   | &check;                     |
| TopK               | 1~16   | &check;                     |
| Transpose          | 1~16   | &check;                     |
| Unsqueeze          | 1~12   | &check;                     |
| Where              | 9~16   | &check;                     |

* MMCV

| Op Type           | Op Set | Linux/Windows/Darwin X86-64 |
|:-----------------:|:------:|:---------------------------:|
| NonMaxSuppression | 1      | &check;                     |
| RoiAlign          | 1      | &check;                     |
| grid_sample       | 1      | &check;                     |

* PPL

| Op Type                              | Op Set | Linux/Windows/Darwin X86-64 |
|:------------------------------------:|:------:|:---------------------------:|
| ChannelShuffle                       | 1      | &check;                     |
| [ShapeOperation](shape_operation.md) | 1      | &check;                     |
| Swish                                | 1      | &check;                     |
