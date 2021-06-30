#ifndef PPLCUDA_KERNEL_INCLUDE_SCATTER_ELEMENTS_SCATTER_ELEMENTS_H_
#define PPLCUDA_KERNEL_INCLUDE_SCATTER_ELEMENTS_SCATTER_ELEMENTS_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDAScatterElementsForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* indices_shape,
    const void* indices,
    const ppl::nn::TensorShape* updates_shape,
    const void* updates,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    int axis);

#endif //PPLCUDA_KERNEL_INCLUDE_SCATTER_ELEMENTS_SCATTER_ELEMENTS_H_
