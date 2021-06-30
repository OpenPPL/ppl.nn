#ifndef PPLCUDA_KERNEL_INCLUDE_SPLIT_SPLIT_H_
#define PPLCUDA_KERNEL_INCLUDE_SPLIT_SPLIT_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDASplitForwardImp(
    cudaStream_t stream,
    int split_axis,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    int num_outputs,
    const int64_t* out_dims[],
    void* output[]);

#endif //PPLCUDA_KERNEL_INCLUDE_SPLIT_SPLIT_H_
