#ifndef PPLCUDA_KERNEL_INCLUDE_NOT_NOT_H_
#define PPLCUDA_KERNEL_INCLUDE_NOT_NOT_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDANotForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const bool* input,
    const ppl::nn::TensorShape* output_shape,
    bool* output);

#endif //PPLCUDA_KERNEL_INCLUDE_NOT_NOT_H_
