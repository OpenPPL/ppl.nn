#ifndef PPLCUDA_KERNEL_INCLUDE_RANGE_RANGE_H_
#define PPLCUDA_KERNEL_INCLUDE_RANGE_RANGE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include <stdint.h>

ppl::common::RetCode PPLCUDARangeForwardImp(
    cudaStream_t stream,
    const void* start,
    const void* delta,
    ppl::nn::TensorShape* output_shape,
    void* output);

#endif //PPLCUDA_KERNEL_INCLUDE_RANGE_RANGE_H_