#ifndef PPLCUDA_KERNEL_INCLUDE_CONV_GEMM_H_
#define PPLCUDA_KERNEL_INCLUDE_CONV_GEMM_H_
#include "cudakernel/nn/conv_fuse_type.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

void ConvImpgemmPrecomputeConvertFilterCudaCoreFp16(
    cudaStream_t stream,
    int kernelIndex,
    const half* filter,
    int channels,
    int group,
    int filterHeight,
    int filterWidth,
    int numFilters,
    half* tempBuf,
    half* filterBuf);

void ConvImpgemmPrecomputeConvertFilterCudaCoreFp16In2C(
    cudaStream_t stream,
    int kernelIndex,
    const half* filter,
    int channels,
    int group,
    int filterHeight,
    int filterWidth,
    int numFilters,
    half* tempBuf,
    half* filterBuf);

void ConvImpgemmPrecomputeCudaCoreFp16(
    cudaStream_t stream,
    int kernelIndex,
    const half* input,
    const half* cvtFilter,
    const half* bias,
    void *tempBuf,
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
    half* output,
    ConvFuse fuse_params);

#endif //PPLCUDA_KERNEL_INCLUDE_CONV_GEMM_H_