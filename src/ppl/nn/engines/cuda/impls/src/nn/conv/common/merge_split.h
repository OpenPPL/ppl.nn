#ifndef __PPLCUDA_MERGE_SPLIT_H__
#define __PPLCUDA_MERGE_SPLIT_H__

#include <cuda.h>
#include <cuda_fp16.h>

//////////////////////////////////////////////////
// merge kernel
//////////////////////////////////////////////////

__global__ void MergeConvSplitResults(
        int4* input,             int4* output, 
	    int split_height_v1,     int split_width_v8, 
	    int out_hw,              int split, 
        int has_bias,            const int4* bias,
        int has_relu,            const __half2 clip_min,
	    bool has_clip,           const __half2 clip_max,
        int has_prelu,           const void* prelu,
        bool has_elt,            const int4* pre_data,
        int has_elt_relu,        const __half2 elt_clip_min,
	    bool has_elt_clip,       const __half2 elt_clip_max,
        int has_elt_prelu,       const void* elt_prelu,
        const __half leaky,      const __half elt_leaky,
        bool has_concat,         int concat_offset_v8,
        int concat_stride_v8);

#endif // __PPLCUDA_MERGE_SPLIT_H__
