## Supported precision

Arm server only supports FP16 and FP32 precision on Aarch64 device.

## Supported operators and opsets

* ONNX

| Op Type            | Op Set | Linux Aarch64 |
|:------------------:|:------:|:-------------:|
| Add                | 7~12   | &check;       |
| ArgMax             | 11     | &check;       |
| AveragePool        | 11~16  | &check;       |
| BatchNormalization | 9~13   | &check;       |
| Cast               | 9~16   | &check;       |
| Clip               | 6~16   | &check;       |
| Concat             | 4~16   | &check;       |
| ConstantOfShape    | 9~16   | &check;       |
| Conv               | 1~16   | &check;       |
| Div                | 7~16   | &check;       |
| Equal              | 7~16   | &check;       |
| Exp                | 6~16   | &check;       |
| Expand             | 8~16   | &check;       |
| Flatten            | 1~16   | &check;       |
| Gather             | 1~16   | &check;       |
| Gemm               | 9~16   | &check;       |
| LeakyRelu          | 6~16   | &check;       |
| Less               | 7~16   | &check;       |
| Log                | 6~16   | &check;       |
| MaxPool            | 1~16   | &check;       |
| Mul                | 7~16   | &check;       |
| Not                | 1~16   | &check;       |
| Range              | 11~16  | &check;       |
| ReduceMax          | 1~16   | &check;       |
| ReduceMean         | 1~16   | &check;       |
| ReduceMin          | 1~16   | &check;       |
| ReduceSum          | 1~16   | &check;       |
| Relu               | 6~16   | &check;       |
| Reshape            | 5~13   | &check;       |
| Resize             | 11~16  | &check;       |
| ScatterND          | 11~15  | &check;       |
| Shape              | 1~14   | &check;       |
| Sigmoid            | 6~16   | &check;       |
| Slice              | 1~16   | &check;       |
| Softmax            | 1~12   | &check;       |
| Split              | 2~12   | &check;       |
| Sqrt               | 6~16   | &check;       |
| Squeeze            | 1~12   | &check;       |
| Sub                | 7~16   | &check;       |
| Tile               | 6~16   | &check;       |
| TopK               | 1~16   | &check;       |
| Transpose          | 1~16   | &check;       |
| Unsqueeze          | 1~12   | &check;       |
| Where              | 9~16   | &check;       |

* PPL

| Op Type                              | Op Set | Linux Aarch64 |
|:------------------------------------:|:------:|:-------------:|
| ChannelShuffle                       | 1      | &check;       |
| Reorder                              | 1      | &check;       |
| [ShapeOperation](shape_operation.md) | 1      | &check;       |
