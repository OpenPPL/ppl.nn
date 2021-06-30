#ifndef PPLCUDA_KERNEL_INCLUDE_RESIZE_RESIZE_H_
#define PPLCUDA_KERNEL_INCLUDE_RESIZE_RESIZE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>

ppl::common::RetCode PPLCUDAResizeForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* outData,
    bool scale_pre_set,
    float h_scale,
    float w_scale,
    int transform_mode,
    int inter_mode,
    float cubic_coeff);

#endif //PPLCUDA_KERNEL_INCLUDE_RESIZE_RESIZE_H_
