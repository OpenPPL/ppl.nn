#ifndef PPLCUDA_KERNEL_INCLUDE_GATHER_GATHER_H_
#define PPLCUDA_KERNEL_INCLUDE_GATHER_GATHER_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDAGatherForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* indices_shape,
    const void* indices,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    int axis);

#endif //PPLCUDA_KERNEL_INCLUDE_GATHER_GATHER_H_
