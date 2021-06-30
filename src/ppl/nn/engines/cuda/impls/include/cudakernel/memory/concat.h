#ifndef PPLCUDA_KERNEL_INCLUDE_CONCAT_CONCAT_H_
#define PPLCUDA_KERNEL_INCLUDE_CONCAT_CONCAT_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDAConcatForwardImp(
    cudaStream_t stream,
    int axis,
    int num_inputs,
    int* input_dims[],
    int* input_padded_dims[],
    const void* inputs[],
    ppl::nn::TensorShape* output_shape,
    void* output,
    int mask = 0);

#endif //PPLCUDA_KERNEL_INCLUDE_CONCAT_CONCAT_H_
