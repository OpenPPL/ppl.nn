#ifndef PPLCUDA_KERNEL_INCLUDE_CONV_DEPTHWISE_H_
#define PPLCUDA_KERNEL_INCLUDE_CONV_DEPTHWISE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "cudakernel/nn/conv_fuse_type.h"
#include "cuda_fp16.h"
// #include <cuda_runtime.h>

void ConvDirectDepthwiseGetBufSizeFP16(
    ppl::nn::TensorShape* input_shape,
    int inHeight,
    int inWidth,
    int channels,
    int batch,
    int group,
    int filterHeight,
    int filterWidth,
    int numFilters,
    int paddingHeight,
    int paddingWidth,
    int strideHeight,
    int strideWidth,
    int holeHeight,
    int holeWidth,
    int outHeight,
    int outWidth,
    uint64_t* filterBufSize,
    uint64_t* tempBufSize);

void ConvDirectDepthwiseConvertFilterCudaFP16(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const half* filter,
    int channels,
    int group,
    int filterHeight,
    int filterWidth,
    int numFilters,
    half* tempBuf,
    half* filterBuf);

void ConvDirectDepthwiseForwardCudaFP16(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const half* input,
    const half* cvtFilter,
    const half* bias,
    half* tempBuf,
    int inHeight,
    int inWidth,
    int channels,
    int batch,
    int group,
    int fltHeight,
    int fltWidth,
    int numFlt,
    int padHeight,
    int padWidth,
    int strideHeight,
    int strideWidth,
    int holeHeight,
    int holeWidth,
    int outHeight,
    int outWidth,
    half* output,
    ConvFuse fuse_params);

#endif //PPLCUDA_KERNEL_INCLUDE_CONV_DEPTHWISE_H_