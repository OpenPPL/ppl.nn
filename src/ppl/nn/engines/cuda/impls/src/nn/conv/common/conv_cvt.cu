// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <cuda.h>
#include <cuda_fp16.h>

#include "ppl/common/types.h"
#include "cudakernel/nn/conv/group_padding.h"
#include "cudakernel/common/divmod_fast.h"
#include "conv_common.h"

template <typename T>
__global__ void group_padding(T *output, T *input, uint64_t out_size, const int num_grp, const int num_chl_per_grp, const int num_chl_pad, int num_chl_per_grp_pad)
{
    uint64_t out_off  = blockIdx.x * blockDim.x + threadIdx.x;
    // in this case, num_chl_per_grp is naturally not aligned with padding size,
    // so we just use T to access memory.
    T value           = 0;
    int chl_id_in_grp = out_off % (num_chl_per_grp_pad); // FIXME magic
    uint64_t nhw_id   = out_off / (num_chl_per_grp_pad * num_grp); // FIXME magic
    int total_chl_id  = out_off - nhw_id * num_chl_per_grp_pad * num_grp;
    int grp_id        = total_chl_id / num_chl_per_grp_pad;
    uint64_t in_off   = nhw_id * num_chl_pad + grp_id * num_chl_per_grp + chl_id_in_grp;

    if (out_off < out_size) {
        if (chl_id_in_grp < num_chl_per_grp)
            value = input[in_off];

        output[out_off] = value;
    }
}

template <typename T>
__global__ void split_group(
    T *output,
    T *input,
    DivModFast fast_div_channel,
    uint64_t out_size,
    const int num_grp,
    const int num_chl_per_grp,
    const int num_chl,
    int num_chl_per_grp_pad)
{
    int32_t out_off = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_off >= out_size)
        return;
    int32_t channel = fast_div_channel.mod(out_off);
    bool in_range   = channel < num_chl_per_grp;
    int32_t nhw_id  = out_off / (num_chl_per_grp_pad * num_grp);
    int32_t grp_id  = (fast_div_channel.div(out_off)) % num_grp;
    int32_t in_off  = nhw_id * num_chl + grp_id * num_chl_per_grp + channel;
    T value         = in_range ? input[in_off] : T(0);
    output[out_off] = value;
}

template <typename T>
__global__ void merge_group(
    T *output,
    T *input,
    DivModFast fast_div_channel_pad,
    DivModFast fast_div_channel_per_grp,
    uint64_t out_size,
    const int num_grp,
    const int num_chl_per_grp,
    const int num_chl,
    int num_chl_per_grp_pad,
    int flt_align)
{
    int32_t out_off = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_off >= out_size) return;
    //int32_t channel = fast_div_channel_per_grp.mod(out_off);
    //int32_t nhw_id = out_off / (flt_align);
    //int chl_id = out_off % flt_align;
    //int32_t grp_id = (fast_div_channel_per_grp.div(out_off)) % num_grp;
    int32_t chl_id = fast_div_channel_pad.mod(out_off);
    int32_t nhw_id = fast_div_channel_pad.div(out_off);
    int32_t grp_id = (fast_div_channel_per_grp.div(chl_id));
    int32_t channel = fast_div_channel_per_grp.mod(chl_id);

    int32_t in_off  = nhw_id * num_grp * num_chl_per_grp_pad + grp_id * num_chl_per_grp_pad + channel;
    output[out_off] = chl_id < num_chl ? input[in_off] : T(0);
}

template <typename T>
__global__ void flt_group_padding(T *output, T *input, unsigned int in_size_per_grp, const int num_grp, int num_chl_per_grp_pad, unsigned int out_size_per_grp)
{
    unsigned int in_off  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int grp_id  = blockIdx.y;
    bool in_range        = (in_off < in_size_per_grp);
    T value              = in_range ? input[in_off + grp_id * in_size_per_grp] : (T)0;
    unsigned int c_id    = in_off % num_chl_per_grp_pad;
    unsigned int nhw_id  = in_off / num_chl_per_grp_pad;
    unsigned int out_off = nhw_id * num_chl_per_grp_pad + grp_id * out_size_per_grp + c_id;
    if (in_range)
        output[out_off] = value;
}

void PPLCUDAConvolutionCvtFlt(
    cudaStream_t &stream,
    void *output,
    const void *input,
    ppl::common::datatype_t type,
    conv_param_t &conv_param)
{
    const int flt_num    = conv_param.num_flt;
    const int num_chl    = conv_param.num_chl;
    const int flt_height = conv_param.flt_height;
    const int flt_width  = conv_param.flt_width;
    const int num_grp    = conv_param.num_grp;

    int align_size = GetPadSize(type);

    int num_chl_per_grp     = num_chl / num_grp;
    int num_chl_per_grp_pad = Align(num_chl_per_grp, align_size);
    int num_flt_per_grp     = flt_num / num_grp;
    int num_flt_per_grp_pad = Align(num_flt_per_grp, align_size);

    const int cta_size = 512;
    dim3 grid;
    int in_size_per_grp  = flt_num / num_grp * flt_height * flt_width * num_chl_per_grp_pad;
    int out_size_per_grp = num_flt_per_grp_pad * flt_height * flt_width * num_chl_per_grp_pad;
    grid.x               = DivUp(in_size_per_grp, cta_size);
    grid.y               = num_grp;
    grid.z               = 1;
    if (type == ppl::common::DATATYPE_FLOAT32) {
        cudaMemset(output, 0, sizeof(float) * num_grp * out_size_per_grp);
        flt_group_padding<float><<<grid, cta_size, 0, stream>>>((float *)output, (float *)input, in_size_per_grp, num_grp, num_chl_per_grp_pad, out_size_per_grp);

    } else if (type == ppl::common::DATATYPE_FLOAT16) {
        cudaMemset(output, 0, sizeof(half) * num_grp * out_size_per_grp);
        flt_group_padding<__half><<<grid, cta_size, 0, stream>>>((__half*)output, (__half*)input, in_size_per_grp, num_grp, num_chl_per_grp_pad,
        out_size_per_grp);
    } else if (type == ppl::common::DATATYPE_INT8) {
        //FIXME different from fp16: flt num not paded
        out_size_per_grp = num_flt_per_grp * flt_height * flt_width * num_chl_per_grp_pad;
        cudaMemset(output, 0, sizeof(int8_t) * num_grp * out_size_per_grp);
        flt_group_padding<int8_t><<<grid, cta_size, 0, stream>>>((int8_t*)output, (int8_t*)input, in_size_per_grp, num_grp, num_chl_per_grp_pad,
        out_size_per_grp);
    }

}

void PPLCUDAConvolutionCvtInput(
    cudaStream_t &stream,
    void *output,
    const void *input,
    ppl::common::datatype_t type,
    conv_param_t &conv_param)
{
    const int in_num    = conv_param.in_num;   
    const int num_chl   = conv_param.num_chl;   
    const int num_chl_pad = conv_param.num_chl_pad;   
    const int in_height = conv_param.in_height;
    const int in_width  = conv_param.in_width;
    const int num_grp   = conv_param.num_grp;

    int align_size = GetPadSize(type);

    int num_chl_per_grp     = num_chl / num_grp;
    int num_chl_per_grp_pad = Align(num_chl_per_grp, align_size);
    const int cta_size      = 512;
    uint64_t out_size       = in_num * in_height * in_width * num_chl_per_grp_pad * num_grp;
    DivModFast fast_div_channel(num_chl_per_grp_pad);
    dim3 grid(DivUp(out_size, cta_size), 1, 1);
    if (type == ppl::common::DATATYPE_FLOAT32) {
        split_group<float><<<grid, cta_size, 0, stream>>>((float*)output, (float*)input, fast_div_channel,
        out_size, num_grp, num_chl_per_grp, num_chl_pad, num_chl_per_grp_pad);

    } else if (type == ppl::common::DATATYPE_FLOAT16) {
        split_group<__half><<<grid, cta_size, 0, stream>>>((__half*)output, (__half*)input, fast_div_channel,
        out_size, num_grp, num_chl_per_grp, num_chl_pad, num_chl_per_grp_pad);
    } else if (type == ppl::common::DATATYPE_INT8) {
        split_group<int8_t><<<grid, cta_size, 0, stream>>>((int8_t*)output, (int8_t*)input, fast_div_channel,
        out_size, num_grp, num_chl_per_grp, num_chl_pad, num_chl_per_grp_pad);
    }

}

void PPLCUDAConvolutionCvtOutput(
    cudaStream_t &stream,
    void *output,
    const void *input,
    ppl::common::datatype_t type,
    conv_param_t &conv_param)
{
    const int in_num     = conv_param.in_num;
    const int num_flt    = conv_param.num_flt;
    const int out_height = conv_param.out_height;
    const int out_width  = conv_param.out_width;
    const int num_grp    = conv_param.num_grp;

    int align_size = GetPadSize(type);

    int num_flt_per_grp     = num_flt / num_grp; // FIXME magic
    int num_flt_per_grp_pad = Align(num_flt_per_grp, align_size);

    int flt_align = Align(num_flt, align_size);

    const int cta_size = 512;

    uint64_t out_size = in_num * out_height * out_width * flt_align;
    DivModFast fast_div_channel_per_grp(num_flt_per_grp);
    DivModFast fast_div_channel_pad(flt_align);

    dim3 grid(DivUp(out_size, cta_size), 1, 1);
    if (type == ppl::common::DATATYPE_FLOAT32) {
        merge_group<float><<<grid, cta_size, 0, stream>>>((float*)output, (float*)input, fast_div_channel_pad,
        fast_div_channel_per_grp, out_size, num_grp, num_flt_per_grp, num_flt, num_flt_per_grp_pad, flt_align);

    } else if (type == ppl::common::DATATYPE_FLOAT16) {
        merge_group<__half><<<grid, cta_size, 0, stream>>>((__half*)output, (__half*)input, fast_div_channel_pad,
        fast_div_channel_per_grp, out_size, num_grp, num_flt_per_grp, num_flt, num_flt_per_grp_pad, flt_align);
    } else if (type == ppl::common::DATATYPE_INT8) {
        merge_group<int8_t><<<grid, cta_size, 0, stream>>>((int8_t*)output, (int8_t*)input, fast_div_channel_pad,
        fast_div_channel_per_grp, out_size, num_grp, num_flt_per_grp, num_flt, num_flt_per_grp_pad, flt_align);
    }
}

void PPLCUDAConvolutionCvtBias(
    cudaStream_t &stream,
    void *output,
    const void *input,
    ppl::common::datatype_t type,
    conv_param_t &conv_param)
{
    const int flt_num = conv_param.num_flt;
    const int num_grp = conv_param.num_grp;

    int align_size          = GetPadSize(type);
    int num_flt_per_grp     = flt_num / num_grp;
    int num_flt_per_grp_pad = Align(num_flt_per_grp, align_size);

    const int cta_size = 256;
    dim3 grid;
    int out_size = num_flt_per_grp_pad * num_grp;
    // int in_size = conv_param.num_flt_pad;
    grid.x       = DivUp(out_size, cta_size);
    grid.y       = 1;
    grid.z       = 1;
    if (type == ppl::common::DATATYPE_FLOAT32) {
        group_padding<float><<<grid, cta_size, 0, stream>>>(
            (float *)output, (float *)input, out_size, num_grp, num_flt_per_grp, conv_param.num_flt_pad, num_flt_per_grp_pad);

    } else if (type == ppl::common::DATATYPE_FLOAT16) {
        group_padding<__half><<<grid, cta_size, 0, stream>>>(
			(__half*)output, (__half*)input, 
			out_size, num_grp, 
			num_flt_per_grp, conv_param.num_flt_pad, num_flt_per_grp_pad);
    } else if (type == ppl::common::DATATYPE_INT8) {
        group_padding<float><<<grid, cta_size, 0, stream>>>(
			(float*)output, (float*)input, 
			out_size, num_grp, 
			num_flt_per_grp, conv_param.num_flt_pad, num_flt_per_grp_pad);
    }
}
