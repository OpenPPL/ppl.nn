#ifndef PPLCUDA_KERNEL_INCLUDE_WHERE_WHERE_H_
#define PPLCUDA_KERNEL_INCLUDE_WHERE_WHERE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDAWhereForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* condition_shape,
    const bool* condition,
    const ppl::nn::TensorShape* input_x_shape,
    const void* input_x,
    const ppl::nn::TensorShape* input_y_shape,
    const void* input_y,
    const ppl::nn::TensorShape* output_shape,
    void* output);

#endif //PPLCUDA_KERNEL_INCLUDE_WHERE_WHERE_H_
