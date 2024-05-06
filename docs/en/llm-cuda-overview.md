# PPLNN for LLM

`ppl.nn.llm` is a part of `PPL.LLM` system.

![SYSTEM_OVERVIEW](../images/llm-system-overview.png)

**We recommend users who are new to this project to read the [Overview of system](llm-system-overview.md).**

## Features

- Flash Attention
- Split-k Attention(Similar with Flash Decoding)
- Group-query Attention
- Dynamic Batching(Also called Continous Batching or In-flight Batching)
- Tensor Parallelism
- Graph Optimization
- INT8 groupwise KV Cache(Numerical accuracy is very close to FP16🚀)
- INT8 per token per channel Quantization(W8A8)

## News

> TBC

## Model Zoo

- [LLaMA 1&2&3](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/llama)
- [ChatGLM 2&3](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/chatglm2)
- [Baichuan-7b](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/baichuan/huggingface)
- [InternLM](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/internlm/huggingface)

## Known Issues

 - NCCL issue on some Device: Currently reported that L40S and H800 may encounter illegal memory access on NCCL AllReduce. We suggest trying to turn NCCL protocol `Simple` off by setting environment `NCCL_PROTO=^Simple` to fix this issue.

## Prerequisites

* Linux running on x86_64 or arm64 CPUs
* GCC >= 9.4.0
* [CMake](https://cmake.org/download/) >= 3.18
* [Git](https://git-scm.com/downloads) >= 2.7.0
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) >= 11.4. 11.6 recommended. (for CUDA)
* mpich >= 4.1.2 (when `PPLNN_CUDA_ENABLE_NCCL=ON`)

## Quick Start

* Installing prerequisites(on Debian 12 or Ubuntu 20.04 for example):

Cuda toolkit and mpich are recommended installing manually:

```bash
apt-get install build-essential cmake git
```

* Cloning source code:

```bash
git clone https://github.com/openppl-public/ppl.nn.git
```

* Building from source:

For nvidia ampere architecture we should use sm80, sm86 and sm87.

```bash
cd ppl.nn
./build.sh -DPPLNN_USE_LLM_CUDA=ON -DPPLNN_CUDA_ENABLE_NCCL=ON -DPPLNN_ENABLE_CUDA_JIT=OFF -DPPLNN_CUDA_ARCHITECTURES="'80;86;87'" -DPPLCOMMON_CUDA_ARCHITECTURES="'80;86;87'"
```

* Export a model(such as LLaMA) and dump some tensors for testing:

Refer to [ppl.pmx/model_zoo/llama/facebook/README.md](https://github.com/openppl-public/ppl.pmx/blob/master/model_zoo/llama/facebook/README.md)

* Build a bash script `benchmark.sh` for testing as below(this example is only for dynamic batching model):

```bash
if [ ! -n "$MPI_LOCALRANKID" ]; then
    echo "[WARNING] MPI_LOCALRANKID not found, set to 0"
    MPI_LOCALRANKID=0
fi
DEVICE_ID=$MPI_LOCALRANKID

STEP=$1
if [ ! -n "$STEP" ]; then
    STEP=0
fi

MODEL_PATH="/path/to/exported/model/model_slice_${MPI_LOCALRANKID}/model.onnx"
OUTPUT_DIR="/path/to/output_dir/rank_${MPI_LOCALRANKID}" # we should make the rank_* directories first
TEST_DATA_DIR="/path/to/dumped/tensor/data/rank_${MPI_LOCALRANKID}"

# we should rearrange the input tensors if the model exporting parameters has been changed.
TOKEN_IDS=`ls ${TEST_DATA_DIR}/step${STEP}_token_ids-*`
ATTN_MASK=`ls ${TEST_DATA_DIR}/step${STEP}_attn_mask-*`
SEQSTARTS=`ls ${TEST_DATA_DIR}/step${STEP}_seqstarts-*`
KVSTARTS=`ls ${TEST_DATA_DIR}/step${STEP}_kvstarts-*`
CACHESTARTS=`ls ${TEST_DATA_DIR}/step${STEP}_cachestarts-*`
DECODING_BATCHES=`ls ${TEST_DATA_DIR}/step${STEP}_decoding_batches-*`
START_POS=`ls ${TEST_DATA_DIR}/step${STEP}_start_pos-*`
MAX_SEQLEN=`ls ${TEST_DATA_DIR}/step${STEP}_max_seqlen-*`
MAX_KVLEN=`ls ${TEST_DATA_DIR}/step${STEP}_max_kvlen-*`
KV_CAHCE=`ls ${TEST_DATA_DIR}/step${STEP}_kv_cache-*`
KV_SCALE=`ls ${TEST_DATA_DIR}/step${STEP}_kv_scale-*`

TEST_INPUTS="$TOKEN_IDS,$ATTN_MASK,$SEQSTARTS,$KVSTARTS,$CACHESTARTS,$DECODING_BATCHES,$START_POS,$MAX_SEQLEN,$MAX_KVLEN,$KV_CAHCE,$KV_SCALE"
INPUT_DEVICES="device,device,device,device,device,host,device,host,host,device,device"

CMD="~/path/to/pplnn-build/tools/pplnn_llm --use-llm-cuda \
--onnx-model $MODEL_PATH \
--shaped-input-files $TEST_INPUTS \
--save-outputs \
--device-id $DEVICE_ID \
--save-data-dir $OUTPUT_DIR \
--in-devices $INPUT_DEVICES \
--enable-profiling \
--min-profiling-seconds 3 \
--warmup-iterations 10"

echo "RUN RANK${MPI_LOCALRANKID} STEP${STEP} -> $CMD"

eval "$CMD"
```

* Run `benchmark.sh` with `mpirun`

```
mpirun -np <MP> benchmark.sh <STEP>
```

The `MP` value for LLaMA can be found [Here](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/llama/facebook#export).

## Python Support


build with `-DPPLNN_ENABLE_PYTHON_API=ON` (use sm80, sm86 and sm87 in this example):

```bash
cd ppl.nn
./build.sh -DPPLNN_USE_LLM_CUDA=ON -DPPLNN_CUDA_ENABLE_NCCL=ON -DPPLNN_ENABLE_CUDA_JIT=OFF -DPPLNN_CUDA_ARCHITECTURES="'80;86;87'" -DPPLCOMMON_CUDA_ARCHITECTURES="'80;86;87'" -DPPLNN_ENABLE_PYTHON_API=ON
```

and test:

```bash
PYTHONPATH=./pplnn-build/install/lib python3 tools/pplnn.py --use-llm-cuda --onnx-model </path/to/model.onnx>
```

## Benchmark LLaMA

Building a benchmark test for LLaMA model with full generation steps. The difference with previous benchmark, is that previous one benchmark with one step, while here benchmark for whole generation steps. Refer to [tools/benchmark.sh](tools/benchmark.sh) for more detail.
```
~/ppl.nn/pplnn-build/tools/benchmark_llama \
    --model-dir $MODEL_DIR \
    --model-param_path $MODEL_PARAM_PATH \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --warmup-loops $WARMUP_LOOPS \
    --generation_len $GENERATION_LEN \
    --benchmark-loops $BENCHMARK_LOOPS \
    --input-len $INPUT_LEN \
    --batch-size $BATCH_SIZE \
    --quant-method $QUANT_METHOD
```

Input parameter explains:
- `model-type`: basic type of model, including diffenent weight. For example, model type `llama` including llama-7b, llama-13b, llama-30b and llama-65b. For other llm model, we will support soon.

- `model-dir`: path to the model.

- `model-param-path`: path to model params.

- `tensor-parallel-size`: size of tensor parallel. For llama_7b, the value is 1.

- `warmup-loops`: loops to warm up GPU for accurate performance.

- `generation-len`: length of generated tokens.

- `benchmark-loops`: benchmark loops.

- `input-len`: length of input tokens, default is zero.

- `batch-size`: batch size.

- `quant-method`: only accept two value: {`none`, `online_i8i8`}. `none` means not quantize, and `online_i8i8` (also called `w8a8`) means weight and tensor are both quantized with int8.

Notice: we found that nsys profiler do not trace cuda kernel statisic in nccl multi-thread communication mode. If you want to profile with nsys, pleace compile with `PPLNN_CUDA_ENABLE_NCCL=OFF`, and it could only trace the performance in one card model.
```bash
./build.sh -DPPLNN_USE_LLM_CUDA=ON -DPPLNN_CUDA_ENABLE_NCCL=OFF -DPPLNN_ENABLE_CUDA_JIT=OFF -DPPLNN_CUDA_ARCHITECTURES="'80;86;87'" -DPPLCOMMON_CUDA_ARCHITECTURES="'80;86;87'"
```

## Performance

### A100 40G(FP16, INT8 KV Cache)

> Tested under base clock in 12/2023

| Model      | Batch | TP | Input Length | Output Length | Throughput (out tok/s) |
| :--------- | :---- | :- | :----------- | :------------ | ---------------------: |
| LLaMA 7B   | 1     | 1  | 8            | 256           | 85.5                   |
| LLaMA 7B   | 256   | 1  | 8            | 256           | 8933.3                 |
| LLaMA 7B   | 1     | 1  | 1024         | 1024          | 82.4                   |
| LLaMA 7B   | 32    | 1  | 1024         | 1024          | 1247.9                 |
|            |       |    |              |               |                        |
| LLaMA 13B  | 1     | 2  | 8            | 256           | 76.1                   |
| LLaMA 13B  | 384   | 2  | 8            | 256           | 8393.0                 |
| LLaMA 13B  | 1     | 2  | 1024         | 1024          | 73.8                   |
| LLaMA 13B  | 32    | 2  | 1024         | 1024          | 1199.7                 |
|            |       |    |              |               |                        |
| LLaMA 70B  | 1     | 8  | 8            | 256           | 42.5                   |
| LLaMA 70B  | 1024  | 8  | 8            | 256           | 6872.7                 |
| LLaMA 70B  | 1     | 8  | 1024         | 1024          | 41.3                   |
| LLaMA 70B  | 128   | 8  | 1024         | 1024          | 1930.7                 |

## Documents

Refer to [ppl.pmx](https://github.com/openppl-public/ppl.pmx) for how to export onnx models. Refer to APIs in [documents](https://github.com/openppl-public/ppl.nn#documents) of ppl.nn.

### License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
