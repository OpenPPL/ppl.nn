#ifndef PPLCUDA_KERNEL_INCLUDE_SUBPIXEL_SUBPIXEL_H_
#define PPLCUDA_KERNEL_INCLUDE_SUBPIXEL_SUBPIXEL_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDASubpixelDownForwardImp(
    cudaStream_t stream,
    int down_ratio,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDASubpixelUpForwardImp(
    cudaStream_t stream,
    int up_ratio,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output);

#endif //PPLCUDA_KERNEL_INCLUDE_SUBPIXEL_SUBPIXEL_H_
