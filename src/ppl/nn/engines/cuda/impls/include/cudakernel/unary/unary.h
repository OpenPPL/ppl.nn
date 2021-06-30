#ifndef PPLCUDA_KERNEL_INCLUDE_UNARY_UNARY_H_
#define PPLCUDA_KERNEL_INCLUDE_UNARY_UNARY_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDAUnaryAbsForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAUnaryReluForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAUnarySigmoidForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAUnarySqrtForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAUnaryTanHForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAUnaryFloorForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAUnaryCeilForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output);

#endif //PPLCUDA_KERNEL_INCLUDE_UNARY_UNARY_H_
