#ifndef PPLCUDA_KERNEL_INCLUDE_CLIP_CLIP_H_
#define PPLCUDA_KERNEL_INCLUDE_CLIP_CLIP_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDAClipForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    float _min,
    float _max);

#endif //PPLCUDA_KERNEL_INCLUDE_CLIP_CLIP_H_
