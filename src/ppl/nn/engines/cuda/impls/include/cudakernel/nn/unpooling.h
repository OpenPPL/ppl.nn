#ifndef PPLCUDA_KERNEL_INCLUDE_UNPOOLING_UNPOOLING_MAX_H_
#define PPLCUDA_KERNEL_INCLUDE_UNPOOLING_UNPOOLING_MAX_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>

ppl::common::RetCode PPLCUDAMaxUnpoolForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output,
    bool use_bottom_mask,
    const int64_t* bottom_mask,
    int unpool,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int kernel_h,
    int kernel_w);

#endif //PPLCUDA_KERNEL_INCLUDE_UNPOOLING_UNPOOLING_MAX_H_
