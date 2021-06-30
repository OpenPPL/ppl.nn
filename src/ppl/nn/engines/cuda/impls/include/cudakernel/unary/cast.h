#ifndef PPLCUDA_KERNEL_INCLUDE_CAST_CAST_H_
#define PPLCUDA_KERNEL_INCLUDE_CAST_CAST_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDACastForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    int to_);

#endif //PPLCUDA_KERNEL_INCLUDE_CAST_CAST_H_
