#ifndef PPLCUDA_KERNEL_INCLUDE_ARITHMETIC_ARITHMETIC_H_
#define PPLCUDA_KERNEL_INCLUDE_ARITHMETIC_ARITHMETIC_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDAArithMeticAddForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAArithMeticSubForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAArithMeticMulForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAArithMeticDivForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAArithMeticMaxForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAArithMeticMinForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAArithMeticPowForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output);

ppl::common::RetCode PPLCUDAArithMeticPReluForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output);
#endif //PPLCUDA_KERNEL_INCLUDE_ARITHMETIC_ARITHMETIC_H_
