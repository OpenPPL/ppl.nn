#ifndef __PPL_CUDA_GROUP_PADDING_H__
#define __PPL_CUDA_GROUP_PADDING_H__
#include "ppl/common/types.h"
#include "conv_fp16.h"

void PPLCUDAConvolutionCvtFlt(
	cudaStream_t &stream, 
	void* output, 
	const void* input, 
	ppl::common::datatype_t type, 
	conv_param_t &conv_param);

void PPLCUDAConvolutionCvtBias(
	cudaStream_t &stream,
	void* output,
	const void* input, 
	ppl::common::datatype_t type, 
	conv_param_t &conv_param);

void PPLCUDAConvolutionCvtInput(
	cudaStream_t &stream, 
	void* output, 
	const void* input, 
	ppl::common::datatype_t type, 
	conv_param_t &conv_param);

void PPLCUDAConvolutionCvtOutput(
	cudaStream_t &stream, 
	void* output, 
	const void* input, 
	ppl::common::datatype_t type, 
	conv_param_t &conv_param);
#endif
