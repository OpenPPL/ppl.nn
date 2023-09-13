# PPLNN for LLM

`ppl.nn.llm` is a part of `PPL.LLM` system.

![SYSTEM_OVERVIEW](docs/system_overview.png)

**We recommend users who are new to this project to read the [Overview of system](docs/system_overview.md).**

`ppl.nn.llm` is a collection of Large Language Models(LLM) inferencing engines based on [ppl.nn](https://github.com/openppl-public/ppl.nn).

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
git clone https://github.com/openppl-public/ppl.nn.llm.git
```

* Building from source:

For nvidia ampere architecture we should use sm80, sm86 and sm87.

```bash
cd ppl.nn.llm
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

## Documents

Refer to [ppl.pmx](https://github.com/openppl-public/ppl.pmx) for how to export onnx models. Refer to APIs in [documents](https://github.com/openppl-public/ppl.nn#documents) of ppl.nn.

### License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
