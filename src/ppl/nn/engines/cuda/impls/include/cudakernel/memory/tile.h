#ifndef PPLCUDA_KERNEL_INCLUDE_TILE_TILE_H_
#define PPLCUDA_KERNEL_INCLUDE_TILE_TILE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include <stdint.h>
#define MAX_DIM_SIZE 7 // should acquired from ppl.common
struct TileParam {
    int64_t repeats[MAX_DIM_SIZE];
};

ppl::common::RetCode PPLCUDATileForwardImp(
    cudaStream_t stream,
    TileParam param,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output);

#endif //PPLCUDA_KERNEL_INCLUDE_TILE_TILE_H_