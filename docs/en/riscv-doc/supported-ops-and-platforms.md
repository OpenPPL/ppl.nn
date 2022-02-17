## Supported precision

riscv only supports FP16 and FP32 precision on Allwinner D1 device.

## Supported operators and opsets

* ONNX

| Op Type            | Op Set | Linux RISCV |
|:------------------:|:------:|:-----------:|
| Add                | 7~12   | &check;     |
| And                | 7~16   | &check;     |
| ArgMax             | 11     | &check;     |
| AveragePool        | 11~16  | &check;     |
| Clip               | 11     | &check;     |
| Concat             | 11~12  | &check;     |
| ConstantOfShape    | 9~16   | &check;     |
| Conv               | 11~16  | &check;     |
| ConvTranspose      | 11~16  | &check;     |
| Div                | 7~12   | &check;     |
| Equal              | 11~12  | &check;     |
| Exp                | 6~12   | &check;     |
| Expand             | 8~12   | &check;     |
| Flatten            | 11~12  | &check;     |
| Floor              | 11~12  | &check;     |
| Gather             | 11~12  | &check;     |
| Gemm               | 11~12  | &check;     |
| LeakyRelu          | 6~16   | &check;     |
| Less               | 9~12   | &check;     |
| Log                | 6~12   | &check;     |
| MaxPool            | 11     | &check;     |
| Mul                | 7~12   | &check;     |
| NonMaxSuppression  | 11~16  | &check;     |
| Not                | 1~16   | &check;     |
| Range              | 11~16  | &check;     |
| ReduceMax          | 11     | &check;     |
| ReduceMean         | 11~12  | &check;     |
| ReduceMin          | 11     | &check;     |
| ReduceSum          | 11~12  | &check;     |
| Relu               | 6~12   | &check;     |
| Reshape            | 5~12   | &check;     |
| Resize             | 11~12  | &check;     |
| ScatterND          | 11~12  | &check;     |
| Shape              | 1~12   | &check;     |
| Sigmoid            | 6~12   | &check;     |
| Slice              | 11~12  | &check;     |
| Softmax            | 11~12  | &check;     |
| Split              | 11~12  | &check;     |
| Sqrt               | 6~12   | &check;     |
| Squeeze            | 11~12  | &check;     |
| Sub                | 7~12   | &check;     |
| Tile               | 6~12   | &check;     |
| TopK               | 11~16  | &check;     |
| Transpose          | 1~12   | &check;     |
| Unsqueeze          | 11~12  | &check;     |
| Where              | 9~15   | &check;     |

* MMCV

| Op Type           | Op Set | Linux RISCV  |
|:-----------------:|:------:|:------------:|
| NonMaxSuppression | 1      | &check;      |
| GridSample        | 1      | &check;      |
| RoiAlign          | 1      | &check;      |

* PPL

| Op Type                              | Op Set | Linux RISCV |
|:------------------------------------:|:------:|:-----------:|
| ChannelShuffle                       | 1      | &check;     |
| [ShapeOperation](shape_operation.md) | 1      | &check;     |
