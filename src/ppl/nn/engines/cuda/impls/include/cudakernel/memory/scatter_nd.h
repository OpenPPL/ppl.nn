#ifndef PPLCUDA_KERNEL_INCLUDE_SCATTERND_SCATTERND_H_
#define PPLCUDA_KERNEL_INCLUDE_SCATTERND_SCATTERND_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

int64_t PPLScatterNDGetTempBufferSize(
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* indices_shape,
    const void* indices);

ppl::common::RetCode PPLCUDAScatterNDForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* indices_shape,
    const void* indices,
    const ppl::nn::TensorShape* updates_shape,
    const void* updates,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    void* temp_buffer);

#endif //PPLCUDA_KERNEL_INCLUDE_SCATTERND_SCATTERND_H_
