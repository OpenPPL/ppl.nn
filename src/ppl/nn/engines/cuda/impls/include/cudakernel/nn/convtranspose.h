#ifndef PPLCUDA_KERNEL_INCLUDE_CONVTRANSPOSE_CONVTRANSPOSE_H_
#define PPLCUDA_KERNEL_INCLUDE_CONVTRANSPOSE_CONVTRANSPOSE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/nn/params/onnx/convtranspose_param.h"
#include <cuda_runtime.h>

uint64_t PPLConvTransposeGetBufSizeCuda(
    ppl::nn::TensorShape* input_shape,
    ppl::nn::TensorShape* output_shape,
    const ppl::nn::common::ConvTransposeParam* param);

ppl::common::RetCode PPLCUDAConvTransposeForward(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    const void* filter,
    const void* bias,
    const ppl::nn::common::ConvTransposeParam* param,
    void* temp_buffer,
    ppl::nn::TensorShape* output_shape,
    void* output);

#endif //PPLCUDA_KERNEL_INCLUDE_CONVTRANSPOSE_CONVTRANSPOSE_H_
