#ifndef PPLCUDA_KERNEL_INCLUDE_POOLING_POOLING_MAX_H_
#define PPLCUDA_KERNEL_INCLUDE_POOLING_POOLING_MAX_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDAMaxPoolingForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width);

ppl::common::RetCode PPLCUDAMaxPoolingForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output,
    ppl::nn::TensorShape* indices_shape,
    int64_t* indices,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width);

#endif //PPLCUDA_KERNEL_INCLUDE_POOLING_POOLING_MAX_H_
