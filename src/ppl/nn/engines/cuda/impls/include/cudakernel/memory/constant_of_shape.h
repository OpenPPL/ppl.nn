#ifndef PPLCUDA_KERNEL_INCLUDE_CONSTANT_OF_SHAPE_CONSTANT_OF_SHAPE_H_
#define PPLCUDA_KERNEL_INCLUDE_CONSTANT_OF_SHAPE_CONSTANT_OF_SHAPE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDAConstantOfShapeForwardImp(
    cudaStream_t stream,
    const void* pre_set_value,
    const ppl::nn::TensorShape* output_shape,
    void* output);

#endif //PPLCUDA_KERNEL_INCLUDE_CONSTANT_OF_SHAPE_CONSTANT_OF_SHAPE_H_
