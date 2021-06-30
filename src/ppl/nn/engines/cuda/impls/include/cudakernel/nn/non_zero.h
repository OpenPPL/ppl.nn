#ifndef PPLCUDA_KERNEL_INCLUDE_NONZERO_NONZERO_MAX_H_
#define PPLCUDA_KERNEL_INCLUDE_NONZERO_NONZERO_MAX_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>

int64_t PPLNonZeroGetTempBufferSize(ppl::nn::TensorShape* input_shape);

ppl::common::RetCode PPLCUDANonZeroForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    int64_t* output,
    int32_t* tempbuffer);

#endif //PPLCUDA_KERNEL_INCLUDE_NONZERO_NONZERO_MAX_H_
