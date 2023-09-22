#!/bin/bash

MODEL_TYPE="llama"
MODEL_DIR="/mnt/hpc/shengyunrui/model_card/llama_7b_ppl"
MODEL_PARAM_PATH="/mnt/hpc/shengyunrui/model_card/llama_7b_ppl/params.json"
TENSOR_PARALLEL_SIZE=1
TOP_P=0.0
TOP_K=1
TEMPERATURE=1.0
WARMUP_LOOPS=2
BENCHMARK_LOOPS=2
INPUT_FILE_BASE="/home/shengyunrui/ppl.nn.llm/tools/tokens_input"


INPUT_LEN=8
GENERATION_LEN=256
BATCH_SIZE_LIST=(1 2 4 8 16 32 64 128 256)

for BATCH_SIZE in ${BATCH_SIZE_LIST[@]}; do
    INPUT_FILE=${INPUT_FILE_BASE}_${INPUT_LEN}

    ~/ppl.nn.llm/pplnn-build/tools/benchmark_llama \
        --model_type $MODEL_TYPE \
        --model_dir $MODEL_DIR \
        --model_param_path $MODEL_PARAM_PATH \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --top_p $TOP_P \
        --top_k $TOP_K \
        --temperature $TEMPERATURE \
        --warmup_loops $WARMUP_LOOPS \
        --generation_len $GENERATION_LEN \
        --benchmark_loops $BENCHMARK_LOOPS \
        --input_file $INPUT_FILE \
        --batch_size $BATCH_SIZE

done

# input token: 1, 306, 4658, 278, 6593, 310, 2834, 338
# output token ground truth: 304, 1284, 596, 19797, 29889, 450, 6437, 310