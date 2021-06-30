#ifndef __PPLCUDA_DEPTHWISE_CONV_H_
#define __PPLCUDA_DEPTHWISE_CONV_H_
#include <cuda_runtime.h>
#include "conv_fp16.h"


int PPLCUDADepthwiseSelectKernel(
    cudaStream_t &stream,
    void* input,              
    void* filter,
    void* bias,
    int times,
	conv_param_t &conv_param, 
	fuse_param_t &fuse_param,
    void* output);

void PPLCUDADepthwiseForwardCudaImp(
    cudaStream_t &stream, 
    int kernel_id,
    void* input,              
    void* filter,
    void* bias,
    conv_param_t &conv_param, 
    fuse_param_t &fuse_param,
    void* output);

void PPLCUDADepthwiseConvertFilter(
    cudaStream_t &stream,
    void* filter,
    void* cvt_filter,
    struct conv_param_t &conv_param);

#endif// __PPLCUDA_DEPTHWISE_CONV_H_
