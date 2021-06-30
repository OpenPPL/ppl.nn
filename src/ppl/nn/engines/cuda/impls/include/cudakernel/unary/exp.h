#ifndef PPLCUDA_KERNEL_INCLUDE_EXP_EXP_H_
#define PPLCUDA_KERNEL_INCLUDE_EXP_EXP_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDAExpForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output);

#endif //PPLCUDA_KERNEL_INCLUDE_EXP_EXP_H_
