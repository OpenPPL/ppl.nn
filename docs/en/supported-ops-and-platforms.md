## OpenPPL supported operators

* ONNX

| Op Type            | Op Set | Linux X86-64 | Linux CUDA |
|:------------------:|:------:|:------------:|:----------:|
| Add                | 11     | &check;      | &check;    |
| And                | 11     | &check;      | &check;    |
| ArgMax             | 11     | &check;      | &check;    |
| AveragePool        | 11     | &check;      | &check;    |
| BatchNormalization | 11     | &check;      | &check;    |
| Cast               | 11     | &check;      | &check;    |
| Ceil               | 11     | &check;      | &check;    |
| Clip               | 11     | &check;      | &check;    |
| Concat             | 11     | &check;      | &check;    |
| Constant           | 11     | &check;      |            |
| ConstantOfShape    | 11     | &check;      | &check;    |
| Conv               | 11     | &check;      | &check;    |
| ConvTranspose      | 11     | &check;      | &check;    |
| DepthToSpace       | 11     | &check;      | &check;    |
| Div                | 11     | &check;      | &check;    |
| Equal              | 11     | &check;      | &check;    |
| Exp                | 11     | &check;      | &check;    |
| Expand             | 11     | &check;      | &check;    |
| Flatten            | 11     | &check;      | &check;    |
| Floor              | 11     | &check;      | &check;    |
| Gather             | 11     | &check;      | &check;    |
| GatherND           | 11     | &check;      | &check;    |
| Gemm               | 11     | &check;      | &check;    |
| Greater            | 11     | &check;      | &check;    |
| Identity           | 11     | &check;      | &check;    |
| If                 | 13     | &check;      | &check;    |
| LeakyRelu          | 11     | &check;      | &check;    |
| Less               | 11     | &check;      | &check;    |
| Log                | 11     | &check;      | &check;    |
| Loop               | 13     | &check;      | &check;    |
| MatMul             | 11     | &check;      |            |
| Max                | 11     | &check;      | &check;    |
| MaxPool            | 11     | &check;      | &check;    |
| MaxUnpool          | 11     | &check;      | &check;    |
| Min                | 11     | &check;      | &check;    |
| Mul                | 11     | &check;      | &check;    |
| NonMaxSuppression  | 11     | &check;      | &check;    |
| NonZero            | 11     | &check;      | &check;    |
| Not                | 11     | &check;      | &check;    |
| Pad                | 11     | &check;      | &check;    |
| Pow                | 11     | &check;      | &check;    |
| ReduceMax          | 11     | &check;      | &check;    |
| ReduceMean         | 11     | &check;      | &check;    |
| ReduceMin          | 11     | &check;      | &check;    |
| ReduceProd         | 11     | &check;      | &check;    |
| ReduceSum          | 11     | &check;      | &check;    |
| Relu               | 11     | &check;      | &check;    |
| Reshape            | 11     | &check;      | &check;    |
| Resize             | 11     | &check;      | &check;    |
| RoiAlign           | 11     | &check;      | &check;    |
| ScatterElements    | 11     | &check;      | &check;    |
| ScatterND          | 11     | &check;      | &check;    |
| SequenceAt         | 13     | &check;      | &check;    |
| Shape              | 11     | &check;      | &check;    |
| Sigmoid            | 11     | &check;      | &check;    |
| Size               | 11     | &check;      | &check;    |
| Slice              | 11     | &check;      | &check;    |
| Softmax            | 11     | &check;      | &check;    |
| Split              | 11     | &check;      | &check;    |
| SplitToSequence    | 13     | &check;      | &check;    |
| Sqrt               | 11     | &check;      | &check;    |
| Squeeze            | 11     | &check;      | &check;    |
| Sub                | 11     | &check;      | &check;    |
| Sum                | 11     | &check;      | &check;    |
| Tanh               | 11     | &check;      | &check;    |
| TopK               | 11     | &check;      | &check;    |
| Transpose          | 11     | &check;      | &check;    |
| Unsqueeze          | 11     | &check;      | &check;    |
| Where              | 11     | &check;      | &check;    |

* MMCV

| Op Type           | Op Set | Linux X86-64 | Linux CUDA |
|:-----------------:|:------:|:------------:|:----------:|
| NonMaxSuppression | 1      | &check;      | &check;    |
| RoiAlign          | 1      | &check;      | &check;    |
| grid_sample       | 1      | &check;      | &check;    |

* PPL

| Op Type        | Op Set | Linux X86-64 | Linux CUDA |
|:--------------:|:------:|:------------:|:----------:|
| ChannelShuffle | 1      | &check;      | &check;    |  

## OpenPPL supported precision 

1. CUDA only supports FP16 precision on Turing Devices.
2. x86 only supports FP32 precision on AVX512/FMA.
