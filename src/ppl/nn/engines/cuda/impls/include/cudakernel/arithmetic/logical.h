#ifndef PPLCUDA_KERNEL_INCLUDE_LOGICAL_LOGICAL_H_
#define PPLCUDA_KERNEL_INCLUDE_LOGICAL_LOGICAL_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDALogicalAndForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const bool* input0,
    const ppl::nn::TensorShape* input_shape1,
    const bool* input1,
    const ppl::nn::TensorShape* output_shape,
    bool* output);

#endif //PPLCUDA_KERNEL_INCLUDE_LOGICAL_LOGICAL_H_
