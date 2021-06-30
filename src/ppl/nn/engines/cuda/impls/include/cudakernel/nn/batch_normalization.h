#ifndef PPLCUDA_KERNEL_INCLUDE_BATCH_NORMALIZATION_BATCH_NORMALIZATION_H_
#define PPLCUDA_KERNEL_INCLUDE_BATCH_NORMALIZATION_BATCH_NORMALIZATION_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/nn/params/onnx/batch_normalization_param.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDABatchNormalizationForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* scale_shape,
    const void* scale,
    // share scale shape
    const void* B,
    const void* mean,
    const void* var,
    ppl::nn::TensorShape* output_shape,
    void* output,
    float epsilon);
#endif //PPLCUDA_KERNEL_INCLUDE_BATCH_NORMALIZATION_BATCH_NORMALIZATION_H_
