#ifndef PPLCUDA_KERNEL_INCLUDE_TRANSPOSE_TRANSPOSE_H_
#define PPLCUDA_KERNEL_INCLUDE_TRANSPOSE_TRANSPOSE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/params/onnx/transpose_param.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDATransposeForwardImp(
    cudaStream_t stream,
    ppl::nn::common::TransposeParam param,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output);

#endif //PPLCUDA_KERNEL_INCLUDE_TRANSPOSE_TRANSPOSE_H_
