#ifndef PPLCUDA_KERNEL_INCLUDE_SOFTMAX_SOFTMAX_H_
#define PPLCUDA_KERNEL_INCLUDE_SOFTMAX_SOFTMAX_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

uint64_t PPLSoftmaxGetTempBufferSize(const ppl::nn::TensorShape* input_shape, int axis);

ppl::common::RetCode PPLCUDASoftmaxForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    int axis);

#endif //PPLCUDA_KERNEL_INCLUDE_SOFTMAX_SOFTMAX_H_
