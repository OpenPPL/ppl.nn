#ifndef _PPLCUDA_KERNEL_INCLUDE_TOPK_H_
#define _PPLCUDA_KERNEL_INCLUDE_TOPK_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
// #include <cuda_runtime.h>

int64_t PPLTopKGetTempBufferSize(
    const ppl::nn::TensorShape* indices_shape, 
    const int K, 
    int dim_k, 
    bool sorted = true);

ppl::common::RetCode PPLCUDATopKForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* topk_shape,
    void* topk,
    ppl::nn::TensorShape* indices_shape,
    int* indices,
    void* temp_buffer,
    int64_t temp_buffer_bytes,
    int K,
    int dim_k,
    const bool largest = true,
    const bool sorted  = true);

#endif // _PPLCUDA_KERNEL_INCLUDE_TOPK_H_
