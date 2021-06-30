#ifndef PPLCUDA_KERNEL_INCLUDE_RELATION_RELATION_H_
#define PPLCUDA_KERNEL_INCLUDE_RELATION_RELATION_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDARelationEqualForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    bool* output);

ppl::common::RetCode PPLCUDARelationGreaterForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    bool* output);

ppl::common::RetCode PPLCUDARelationLessForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const void* input0,
    const ppl::nn::TensorShape* input_shape1,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    bool* output);

#endif //PPLCUDA_KERNEL_INCLUDE_RELATION_RELATION_H_