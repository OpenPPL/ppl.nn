#ifndef PPLCUDA_KERNEL_INCLUDE_GEMM_GEMM_H_
#define PPLCUDA_KERNEL_INCLUDE_GEMM_GEMM_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/params/onnx/gemm_param.h"
#include "ppl/common/retcode.h"
#include "cudakernel/nn/conv/conv_fp16.h"

uint64_t PPLGemmCUDAGetBufSize(
    const ppl::nn::TensorShape* input_shape,
    int transA);

unsigned int PPLCUDAGemmGetBiasSize(
    ppl::common::datatype_t type,
    int K,
    bool is_scalar);

int PPLCUDAGemmSelectKernel(
    const cudaStream_t &stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* weight_shape,
    const void* weight,
    const void* bias,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    const ppl::nn::common::GemmParam &param,
    void* temp_buffer, 
    const fuse_param_t &fuse_param);

ppl::common::RetCode PPLCUDAGemmForwardImp(
    const cudaStream_t &stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* weight_shape,
    const void* weight,
    const void* bias,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    const ppl::nn::common::GemmParam &param,
    void* temp_buffer,
    const fuse_param_t &fuse_param,
    int kid);

ppl::common::RetCode PPLCUDAGemmModifyWeights(
    const cudaStream_t &stream,
    ppl::nn::TensorShape* weight_shape,
    void* weight,
    void* out_weight,
    const ppl::nn::common::GemmParam *param);

ppl::common::RetCode PPLCUDAGemmModifyBias(
    const cudaStream_t &stream,
    const ppl::nn::TensorShape* bias_shape,
    void* bias,
    const ppl::nn::common::GemmParam *param);
 

#endif //PPLCUDA_KERNEL_INCLUDE_GEMM_GEMM_H_
