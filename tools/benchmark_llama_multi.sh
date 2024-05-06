SCRIPT=$(realpath -s "$0")
BASE_SCRIPT_PATH=$(dirname "$SCRIPT")

# Used by benchmark_llama_single.sh
# Where contains opmx_models. Set this by yourself.
BASE_MODEL_PATH=

# Used by benchmark_llama_single.sh
if [ ! -n "$BENCHMARK_LLAMA" ]; then
    echo "[ERROR] please set env BENCHMARK_LLM to the benchmark_llama executable"
    exit 1
fi

_I8O256_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256 384 512 768 1024 1280 1536 1792 2048)
_I128O128_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256 384 512 768 1024 1280 1536 1792 2048)
_I2048O2048_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256)
_I128O2048_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256)
_I2048O128_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256)
_I1024O1024_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256)
_I512O512_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256 384 512)
_I256O512_BATCH_SIZE_LIST=(1 2 4 8 16 32 48 64 80 96 112 128 160 192 224 256 384 512 768 1024)
_I32KO128_BATCH_SIZE_LIST=(1 2 4 8 12 16)
_I16KO128_BATCH_SIZE_LIST=(1 2 4 8 12 16 20 24 28 32)

function benchmark() {
    MODEL_SIZE=$1
    GPUS=$2
    BATCH=$3
    INLEN=$4
    OUTLEN=$5
    echo "[BENCHMARK ${MODEL_SIZE}B TP${GPUS} BATCH${BATCH} I${INLEN} O${OUTLEN}]"
    RES=`bash $BASE_SCRIPT_PATH/benchmark_llama_single.sh ${MODEL_SIZE} ${GPUS} ${BATCH} ${INLEN} ${OUTLEN} | grep "CSV format output"`
    RES=${RES##*:}
    if [ ! -n "$RES" ]; then
        echo "[FAILED]"
    else
        echo "[OK] $RES"
        echo "$MODEL_SIZE,$GPUS,$BATCH,$INLEN,$OUTLEN,$RES" >> $BASE_SCRIPT_PATH/benchmark_llama_result.csv
    fi
}

echo "model_size(B),tp,batch,inlen,outlen,generate(ms),prefill(ms),decode(ms),step(ms),prefill_tps,decode_tps,o_tps,io_tps,mem(gib)" > $BASE_SCRIPT_PATH/benchmark_llama_result.csv

for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    benchmark 7 1 $BATCH_SIZE 8 256
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    benchmark 7 1 $BATCH_SIZE 256 512
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    benchmark 7 1 $BATCH_SIZE 512 512
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    benchmark 7 1 $BATCH_SIZE 1024 1024
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    benchmark 7 1 $BATCH_SIZE 128 128
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 7 1 $BATCH_SIZE 128 2048
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    benchmark 7 1 $BATCH_SIZE 2048 128
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 7 1 $BATCH_SIZE 2048 2048
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 7 1 $BATCH_SIZE 16384 128
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 7 1 $BATCH_SIZE 32768 128
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    benchmark 13 1 $BATCH_SIZE 8 256
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    benchmark 13 1 $BATCH_SIZE 256 512
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    benchmark 13 1 $BATCH_SIZE 512 512
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    benchmark 13 1 $BATCH_SIZE 1024 1024
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    benchmark 13 1 $BATCH_SIZE 128 128
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 13 1 $BATCH_SIZE 128 2048
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    benchmark 13 1 $BATCH_SIZE 2048 128
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 13 1 $BATCH_SIZE 2048 2048
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 13 1 $BATCH_SIZE 16384 128
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 13 1 $BATCH_SIZE 32768 128
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    benchmark 13 2 $BATCH_SIZE 8 256
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    benchmark 13 2 $BATCH_SIZE 256 512
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    benchmark 13 2 $BATCH_SIZE 512 512
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    benchmark 13 2 $BATCH_SIZE 1024 1024
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    benchmark 13 2 $BATCH_SIZE 128 128
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 13 2 $BATCH_SIZE 128 2048
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    benchmark 13 2 $BATCH_SIZE 2048 128
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 13 2 $BATCH_SIZE 2048 2048
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 13 2 $BATCH_SIZE 16384 128
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 13 2 $BATCH_SIZE 32768 128
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    benchmark 65 8 $BATCH_SIZE 8 256
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    benchmark 65 8 $BATCH_SIZE 256 512
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    benchmark 65 8 $BATCH_SIZE 512 512
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    benchmark 65 8 $BATCH_SIZE 1024 1024
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    benchmark 65 8 $BATCH_SIZE 128 128
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 65 8 $BATCH_SIZE 128 2048
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    benchmark 65 8 $BATCH_SIZE 2048 128
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 65 8 $BATCH_SIZE 2048 2048
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 65 8 $BATCH_SIZE 16384 128
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 65 8 $BATCH_SIZE 32768 128
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    benchmark 70 4 $BATCH_SIZE 8 256
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    benchmark 70 4 $BATCH_SIZE 256 512
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    benchmark 70 4 $BATCH_SIZE 512 512
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    benchmark 70 4 $BATCH_SIZE 1024 1024
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    benchmark 70 4 $BATCH_SIZE 128 128
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 70 4 $BATCH_SIZE 128 2048
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    benchmark 70 4 $BATCH_SIZE 2048 128
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 70 4 $BATCH_SIZE 2048 2048
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 70 4 $BATCH_SIZE 16384 128
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 70 4 $BATCH_SIZE 32768 128
done


for BATCH_SIZE in ${_I8O256_BATCH_SIZE_LIST[@]}; do
    benchmark 70 8 $BATCH_SIZE 8 256
done
for BATCH_SIZE in ${_I256O512_BATCH_SIZE_LIST[@]}; do
    benchmark 70 8 $BATCH_SIZE 256 512
done
for BATCH_SIZE in ${_I512O512_BATCH_SIZE_LIST[@]}; do
    benchmark 70 8 $BATCH_SIZE 512 512
done
for BATCH_SIZE in ${_I1024O1024_BATCH_SIZE_LIST[@]}; do
    benchmark 70 8 $BATCH_SIZE 1024 1024
done
for BATCH_SIZE in ${_I128O128_BATCH_SIZE_LIST[@]}; do
    benchmark 70 8 $BATCH_SIZE 128 128
done
for BATCH_SIZE in ${_I128O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 70 8 $BATCH_SIZE 128 2048
done
for BATCH_SIZE in ${_I2048O128_BATCH_SIZE_LIST[@]}; do
    benchmark 70 8 $BATCH_SIZE 2048 128
done
for BATCH_SIZE in ${_I2048O2048_BATCH_SIZE_LIST[@]}; do
    benchmark 70 8 $BATCH_SIZE 2048 2048
done
for BATCH_SIZE in ${_I16KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 70 8 $BATCH_SIZE 16384 128
done
for BATCH_SIZE in ${_I32KO128_BATCH_SIZE_LIST[@]}; do
    benchmark 70 8 $BATCH_SIZE 32768 128
done


done
