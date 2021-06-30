#ifndef PPLCUDA_KERNEL_INCLUDE_POOLING_POOLING_AVE_H_
#define PPLCUDA_KERNEL_INCLUDE_POOLING_POOLING_AVE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDAAvePoolingForwardImp(
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
    int padding_width,
    int if_exclude_padding);

#endif //PPLCUDA_KERNEL_INCLUDE_POOLING_POOLING_AVE_H_
