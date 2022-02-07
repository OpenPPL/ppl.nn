## Supported precision

1. CUDA only supports FP16 precision on Turing Devices.
2. x86 only supports FP32 precision on AVX512/FMA.
3. riscv only supports FP16 and FP32 precision on Allwinner D1 device.

## Supported operators and opsets

* ONNX

| Op Type            | Op Set | Linux/Windows/Darwin X86-64 | Linux/Windows CUDA | Linux RISCV |
|:------------------:|:------:|:---------------------------:|:------------------:|:-----------:|
| Add                | 7~12   | &check;                     | &check;            | &check;     |
| And                | 7~16   | &check;                     | &check;            | &check;     |
| ArgMax             | 11     | &check;                     | &check;            | &check;     |
| AveragePool        | 11~16  | &check;                     | &check;            | &check;     |
| BatchNormalization | 9~13   | &check;                     | &check;            |             |
| Cast               | 9~12   | &check;                     | &check;            |             |
| Ceil               | 6~12   | &check;                     | &check;            |             |
| Clip               | 11     | &check;                     | &check;            | &check;     |
| Concat             | 11~12  | &check;                     | &check;            | &check;     |
| Constant           | 9~16   | &check;                     |                    |             |
| ConstantOfShape    | 9~16   | &check;                     | &check;            | &check;     |
| Conv               | 11~16  | &check;                     | &check;            | &check;     |
| ConvTranspose      | 11~16  | &check;                     | &check;            | &check;     |
| DepthToSpace       | 11~12  | &check;                     | &check;            |             |
| Div                | 7~12   | &check;                     | &check;            | &check;     |
| Equal              | 11~12  | &check;                     | &check;            | &check;     |
| Exp                | 6~12   | &check;                     | &check;            |
| Expand             | 8~12   | &check;                     | &check;            | &check;     |
| Flatten            | 11~12  | &check;                     | &check;            | &check;     |
| Floor              | 11~12  | &check;                     | &check;            |             |
| Gather             | 11~12  | &check;                     | &check;            | &check;     |
| GatherND           | 11     | &check;                     | &check;            |             |
| Gemm               | 11~12  | &check;                     | &check;            | &check;     |
| Greater            | 9~12   | &check;                     | &check;            |             |
| Identity           | 1~12   | &check;                     | &check;            |             |
| If                 | 11~12  | &check;                     | &check;            |             |
| LeakyRelu          | 6~16   | &check;                     | &check;            | &check;     |
| Less               | 9~12   | &check;                     | &check;            | &check;     |
| Log                | 6~12   | &check;                     | &check;            |             |
| Loop               | 11~12  | &check;                     | &check;            |             |
| LSTM               | 7~13   | &check;                     | &check;            |             |
| MatMul             | 9~12   | &check;                     | &check;            |             |
| Max                | 8~11   | &check;                     | &check;            |             |
| MaxPool            | 11     | &check;                     | &check;            | &check;     |
| MaxUnpool          | 11~16  | &check;                     | &check;            |             |
| Min                | 8~11   | &check;                     | &check;            |             |
| Mul                | 7~12   | &check;                     | &check;            | &check;     |
| NonMaxSuppression  | 11~16  | &check;                     | &check;            | &check;     |
| NonZero            | 9~12   | &check;                     | &check;            |             |
| Not                | 1~16   | &check;                     | &check;            |             |
| Pad                | 11~12  | &check;                     | &check;            |             |
| Pow                | 7~11   | &check;                     | &check;            |             |
| PRelu              | 9~16   | &check;                     |                    |             |
| Range              | 11~16  | &check;                     | &check;            | &check;     |
| ReduceMax          | 11     | &check;                     | &check;            | &check;     |
| ReduceMean         | 11~12  | &check;                     | &check;            | &check;     |
| ReduceMin          | 11     | &check;                     | &check;            | &check;     |
| ReduceProd         | 11~12  | &check;                     | &check;            |             |
| ReduceSum          | 11~12  | &check;                     | &check;            | &check;     |
| Relu               | 6~12   | &check;                     | &check;            | &check;     |
| Reshape            | 5~12   | &check;                     | &check;            | &check;     |
| Resize             | 11~12  | &check;                     | &check;            | &check;     |
| RoiAlign           | 10~15  | &check;                     | &check;            |             |
| ScatterElements    | 11~12  | &check;                     | &check;            |             |
| ScatterND          | 11~12  | &check;                     | &check;            | &check;     |
| SequenceAt         | 11~16  | &check;                     | &check;            |             |
| Shape              | 1~12   | &check;                     | &check;            | &check;     |
| Sigmoid            | 6~12   | &check;                     | &check;            | &check;     |
| Slice              | 11~12  | &check;                     | &check;            | &check;     |
| Softmax            | 11~12  | &check;                     | &check;            | &check;     |
| Split              | 11~12  | &check;                     | &check;            | &check;     |
| SplitToSequence    | 11~16  | &check;                     | &check;            |             |
| Sqrt               | 6~12   | &check;                     | &check;            |             |
| Squeeze            | 11~12  | &check;                     | &check;            | &check;     |
| Sub                | 7~12   | &check;                     | &check;            | &check;     |
| Sum                | 8~12   | &check;                     |                    |             |
| Tanh               | 6~12   | &check;                     | &check;            |             |
| Tile               | 6~12   | &check;                     | &check;            | &check;     |
| TopK               | 11~16  | &check;                     | &check;            | &check;     |
| Transpose          | 1~12   | &check;                     | &check;            | &check;     |
| Unsqueeze          | 11~12  | &check;                     | &check;            | &check;     |
| Where              | 9~15   | &check;                     | &check;            | &check;     |

* MMCV

| Op Type           | Op Set | Linux/Windows/Darwin X86-64 | Linux/Windows CUDA | Linux RISCV  |
|:-----------------:|:------:|:---------------------------:|:------------------:|:------------:|
| NonMaxSuppression | 1      | &check;                     | &check;            |              |
| RoiAlign          | 1      | &check;                     | &check;            | &check;      |
| grid_sample       | 1      | &check;                     | &check;            |              |

* PPL

| Op Type                              | Op Set | Linux/Windows/Darwin X86-64 | Linux/Windows CUDA | Linux RISCV |
|:------------------------------------:|:------:|:---------------------------:|:------------------:|:-----------:|
| ChannelShuffle                       | 1      | &check;                     | &check;            | &check;     |
| [ShapeOperation](shape_operation.md) | 1      | &check;                     | &check;            | &check;     |
| Swish                                | 1      | &check;                     |                    |             |
