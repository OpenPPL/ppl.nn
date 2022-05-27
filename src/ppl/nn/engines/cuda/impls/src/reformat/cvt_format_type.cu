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

#include <float.h>
#include <iostream>
#include "cudakernel/reformat/reformat.h"
#include "cudakernel/common/common.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/macro.h"
#include "cvt_type_per_elems.cuh"

#include "cuda_fp16.h"
using namespace PPLCUDA;
using namespace ppl::nn;
using namespace ppl::common;

#define DIM 32
#define LEASTCHANNEL 16

#define cvtTYPEALL(cvt_format)  \
    cvt_format(FLOAT32_INT8)    \
    cvt_format(INT8_FLOAT32)    \
    cvt_format(FLOAT32_FLOAT16) \
    cvt_format(FLOAT16_FLOAT32) \
    cvt_format(FLOAT32_INT4B)   \
    cvt_format(INT4B_FLOAT32)   \
    cvt_format(INT8_FLOAT16)    \
    cvt_format(FLOAT16_INT8)    \
    cvt_format(INT8_INT4B)      \
    cvt_format(INT8_INT8)       \
    cvt_format(INT4B_INT4B)     \
    cvt_format(INT32_INT64)     \
    cvt_format(INT64_INT32)     \
    cvt_format(INT64_FLOAT32)   \
    cvt_format(FLOAT32_INT64)

template <CVTTypeMode t_mode, CVTFormatMode mode>
__global__ void cuda_kernel_cvtformat_type(
    const void* input,
    void* output,
    ReFormatParam param)
{
}

#define cvtC16TOC8(type_mode)                                                                           \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat_type<type_mode, NHWC16_NHWC8>(                                         \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
                                                                                                        \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n+= blockDim.z) {                                              \
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        int64_t idx_h = blockIdx.y * blockDim.y + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.dst_pad && idx_h < param.n_inner) {                                           \
            int64_t dst_offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;     \
            int64_t src_offset = n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;     \
            cuda_kernel_cvt_per_elems<type_mode>(input, src_offset, output, dst_offset, param);         \
        }                                                                                               \
    }                                                                                                   \
}                                                                                                       

cvtTYPEALL(cvtC16TOC8)

#define cvtC8TOC16(type_mode)                                                                           \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat_type<type_mode, NHWC8_NHWC16>(                                         \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n+= blockDim.z) {                                              \
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        int64_t idx_h = blockIdx.y * blockDim.y + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.dst_pad && idx_h < param.n_inner) {                                           \
            int64_t dst_offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;     \
            int64_t src_offset = n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;     \
            if (idx_w < param.src_pad) {                                                                \
              cuda_kernel_cvt_per_elems<type_mode>(input, src_offset, output, dst_offset, param);       \
            } else {                                                                                    \
              cuda_kernel_set_zero_per_elems<type_mode>(output, dst_offset);                            \
            }                                                                                           \
        }                                                                                               \
    }                                                                                                   \
}                                                                                                       
cvtTYPEALL(cvtC8TOC16)

#define cvtNCTONHWC(type, type_mode)                                                             \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat_type<type_mode, NDARRAY_NHWC>(                                         \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    __shared__ type share_val[DIM][DIM + 1];                                                            \
                                                                                                        \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n+= blockDim.z) {                                              \
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        int64_t idx_h = blockIdx.y * blockDim.y + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.n_inner && idx_h < param.src_pad) {                                           \
            int64_t offset = n * param.src_pad * param.n_inner + idx_h * param.n_inner + idx_w;         \
            share_val[threadIdx.y][threadIdx.x] = ((const type*)input)[offset];                         \
        } else {                                                                                        \
            share_val[threadIdx.y][threadIdx.x] = type(0);                                              \
        }                                                                                               \
        __syncthreads();                                                                                \
                                                                                                        \
        idx_w = blockIdx.y * blockDim.y + threadIdx.x;                                                  \
        idx_h = blockIdx.x * blockDim.x + threadIdx.y;                                                  \
                                                                                                        \
        if (idx_w < param.dst_pad && idx_h < param.n_inner) {                                           \
            int64_t offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;         \
            int in_offset = threadIdx.x * (DIM + 1) + threadIdx.y;                                      \
            cuda_kernel_cvt_per_elems<type_mode>((const void*)share_val, in_offset, output, offset, param);  \
        }                                                                                               \
    }                                                                                                   \
}
cvtNCTONHWC(float, FLOAT32_INT8)
cvtNCTONHWC(char, INT8_FLOAT32)
cvtNCTONHWC(float, FLOAT32_FLOAT16)
cvtNCTONHWC(half, FLOAT16_FLOAT32)
cvtNCTONHWC(float, FLOAT32_INT4B)
cvtNCTONHWC(char, INT4B_FLOAT32)
cvtNCTONHWC(char, INT8_FLOAT16)
cvtNCTONHWC(half, FLOAT16_INT8)
cvtNCTONHWC(char, INT8_INT4B)
cvtNCTONHWC(char, INT8_INT8)
cvtNCTONHWC(char, INT4B_INT4B)
cvtNCTONHWC(float, INT32_INT64)
cvtNCTONHWC(double, INT64_INT32)
cvtNCTONHWC(double, INT64_FLOAT32)
cvtNCTONHWC(float, FLOAT32_INT64)

#define cvtNHWC8TONC(type, type_mode)                                                                   \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat_type<type_mode, NHWC_NDARRAY>(                                         \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    __shared__ type share_val[DIM][DIM + 1];                                                            \
                                                                                                        \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n += blockDim.z) {                                             \
        for (int t = blockIdx.y; t < DivUp(param.n_inner, 32) ; t+= gridDim.y) { \
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        int64_t idx_h = t * blockDim.y + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.src_pad && idx_h < param.n_inner) {                                           \
            int64_t offset = n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;         \
            share_val[threadIdx.y][threadIdx.x] = ((const type*)input)[offset];                         \
        } else {                                                                                        \
            share_val[threadIdx.y][threadIdx.x] = (type)0;                                              \
        }                                                                                               \
        __syncthreads();                                                                                \
                                                                                                        \
        idx_w = t * blockDim.y + threadIdx.x;                                                  \
        idx_h = blockIdx.x * blockDim.x + threadIdx.y;                                                  \
                                                                                                        \
        if (idx_w < param.n_inner && idx_h < param.dst_pad) {                                           \
            int64_t offset = n * param.dst_pad * param.n_inner + idx_h * param.n_inner + idx_w;         \
            int in_offset = threadIdx.x * (DIM + 1) + threadIdx.y;                                      \
            cuda_kernel_cvt_per_elems<type_mode>((const void*)share_val, in_offset, output, offset, param);  \
        }                                                                                               \
        }\
    }                                                                                                   \
}

cvtNHWC8TONC(float, FLOAT32_INT8)
cvtNHWC8TONC(char, INT8_FLOAT32)
cvtNHWC8TONC(float, FLOAT32_FLOAT16)
cvtNHWC8TONC(half, FLOAT16_FLOAT32)
cvtNHWC8TONC(float, FLOAT32_INT4B)
cvtNHWC8TONC(char, INT4B_FLOAT32)
cvtNHWC8TONC(char, INT8_FLOAT16)
cvtNHWC8TONC(half, FLOAT16_INT8)
cvtNHWC8TONC(char, INT8_INT4B)
cvtNHWC8TONC(char, INT8_INT8)
cvtNHWC8TONC(char, INT4B_INT4B)
cvtNHWC8TONC(float, INT32_INT64)
cvtNHWC8TONC(double, INT64_INT32)
cvtNHWC8TONC(double, INT64_FLOAT32)
cvtNHWC8TONC(float, FLOAT32_INT64)

#define cvtN4CXTONC(type_mode)                                                                                         \
template <>                                                                                                            \
__global__ void cuda_kernel_cvtformat_type<type_mode, N4CX_NDARRAY>(                                                        \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                               \
{                                                                                                                      \
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;                                                        \
    if (tid >= param.n_inner)                                                                                          \
        return;                                                                                                        \
    const uint64_t inner_idx = tid;                                                                                    \
    const uint64_t num_inner = blockIdx.z;                                                                             \
    const uint64_t c4_idx    = blockIdx.y;                                                                             \
    _Pragma("unroll 4") for (int c_in_c4_idx = 0; c_in_c4_idx < 4; c_in_c4_idx++)                                      \
    {                                                                                                                  \
        const uint64_t c_idx       = c4_idx * 4 + c_in_c4_idx;                                                         \
        const uint64_t size        = param.n_inner;                                                                    \
        const uint64_t padChannels = gridDim.y * 4;                                                                    \
        const uint64_t numChannels = param.channel;                                                                    \
        if (c_idx < numChannels) {                                                                                     \
            const uint64_t offset    = num_inner * padChannels * size + (c4_idx * size + inner_idx) * 4 + c_in_c4_idx; \
            const uint64_t outOffset = num_inner * numChannels * size + c_idx * size + inner_idx;                      \
            cuda_kernel_cvt_per_elems<type_mode>(input, offset, output, outOffset, param);                             \
        }                                                                                                              \
    }                                                                                                                  \
}

cvtTYPEALL(cvtN4CXTONC)

#define cvtNCTON4CX(type_mode)                                                                                        \
template <>                                                                                                           \
__global__ void cuda_kernel_cvtformat_type<type_mode, NDARRAY_N4CX>(                                                       \
    const void* input,                                                                                  \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                              \
{                                                                                                                     \
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;                                                       \
    if (tid >= param.n_inner)                                                                                         \
        return;                                                                                                       \
    const uint64_t inner_idx = tid;                                                                                   \
    const uint64_t num_inner = blockIdx.z;                                                                            \
    const uint64_t c4_idx    = blockIdx.y;                                                                            \
    _Pragma("unroll 4") for (int c_in_c4_idx = 0; c_in_c4_idx < 4; c_in_c4_idx++)                                     \
    {                                                                                                                 \
        const uint64_t c_idx       = c4_idx * 4 + c_in_c4_idx;                                                        \
        const uint64_t size        = param.n_inner;                                                                   \
        const uint64_t padChannels = gridDim.y * 4;                                                                   \
        const uint64_t numChannels = param.channel;                                                                   \
        if (c_idx < numChannels) {                                                                                    \
            const uint64_t offset   = num_inner * padChannels * size + (c4_idx * size + inner_idx) * 4 + c_in_c4_idx; \
            const uint64_t inOffset = num_inner * numChannels * size + c_idx * size + inner_idx;                      \
            cuda_kernel_cvt_per_elems<type_mode>(input, inOffset, output, offset, param);                             \
        }                                                                                                             \
    }                                                                                                                 \
}

cvtTYPEALL(cvtNCTON4CX)

template <CVTTypeMode t_mode, CVTFormatMode mode>
__global__ void cuda_kernel_small_channel_cvtformat_type(
    const void* input,
    int num_elems,
    DivModFast inner_fast,
    DivModFast src_pad_fast,
    DivModFast dst_pad_fast,
    void* output,
    ReFormatParam param)
{
}

#define cvtSMCHANNELNCTONHWC8(type_mode)                                                                \
template<>                                                                                              \
__global__ void cuda_kernel_small_channel_cvtformat_type<type_mode, NDARRAY_NHWC>(                      \
    const void* input,                                                                                  \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    dst_pad_fast.divmod(tid, num_inner, c_idx);                                                         \
    inner_idx = inner_fast.mod(num_inner);                                                              \
    int outer_idx = inner_fast.div(num_inner);                                                          \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx * param.n_inner + inner_idx;         \
    if (c_idx < param.src_pad) {                                                                        \
        cuda_kernel_cvt_per_elems<type_mode>(input, offset, output, tid, param);                        \
    } else {                                                                                            \
        cuda_kernel_set_zero_per_elems<type_mode>(output, tid);                                         \
    }                                                                                                   \
}
cvtTYPEALL(cvtSMCHANNELNCTONHWC8)

#define cvtSMCHANNELNHWC8TONC(type_mode)                                                                \
template<>                                                                                              \
__global__ void cuda_kernel_small_channel_cvtformat_type<type_mode, NHWC_NDARRAY>(                      \
    const void* input,                                                                                  \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    void* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    inner_fast.divmod(tid, num_inner, inner_idx);                                                       \
    c_idx = dst_pad_fast.mod(num_inner);                                                                \
    int outer_idx = tid / (param.dst_pad * param.n_inner);                                              \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx + inner_idx * param.src_pad;         \
    cuda_kernel_cvt_per_elems<type_mode>(input, offset, output, tid, param);                            \
}
cvtTYPEALL(cvtSMCHANNELNHWC8TONC)

#define cvtSMCHANNELN4CXTONC(type_mode)                                                                          \
template <>                                                                                                      \
__global__ void cuda_kernel_small_channel_cvtformat_type<type_mode, N4CX_NDARRAY>(                               \
    const void* input,                                                                                           \
    int num_elems,                                                                                               \
    DivModFast inner_fast,                                                                                       \
    DivModFast src_pad_fast,                                                                                     \
    DivModFast dst_pad_fast,                                                                                     \
    void* output,                                                                                                \
    ReFormatParam param)                                                                                         \
{                                                                                                                \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                       \
    if (tid >= num_elems)                                                                                        \
        return;                                                                                                  \
    int inner_idx, num_inner, c_idx;                                                                             \
    inner_fast.divmod(tid, num_inner, inner_idx);                                                                \
    src_pad_fast.divmod(num_inner, num_inner, c_idx);                                                            \
    const int c4_idx           = c_idx / 4;                                                                      \
    const int c_in_c4_idx      = c_idx % 4;                                                                      \
    const uint64_t size        = param.n_inner;                                                                  \
    const uint64_t padChannels = param.src_pad;                                                                  \
    const uint64_t numChannels = param.channel;                                                                  \
    const uint64_t offset      = num_inner * padChannels * size + (c4_idx * size + inner_idx) * 4 + c_in_c4_idx; \
    const uint64_t outOffset   = num_inner * numChannels * size + c_idx * size + inner_idx;                      \
    cuda_kernel_cvt_per_elems<type_mode>(input, offset, output, outOffset, param);                               \
}
cvtTYPEALL(cvtSMCHANNELN4CXTONC)

#define cvtSMCHANNELNCTON4CX(type_mode)                                                                          \
template <>                                                                                                      \
__global__ void cuda_kernel_small_channel_cvtformat_type<type_mode, NDARRAY_N4CX>(                                    \
    const void * input,                                                                                          \
    int num_elems,                                                                                               \
    DivModFast inner_fast,                                                                                       \
    DivModFast src_pad_fast,                                                                                     \
    DivModFast dst_pad_fast,                                                                                     \
    void* output,                                                                                                \
    ReFormatParam param)                                                                                         \
{                                                                                                                \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                       \
    if (tid >= num_elems)                                                                                        \
        return;                                                                                                  \
    int inner_idx, num_inner, c_idx;                                                                             \
    inner_fast.divmod(tid, num_inner, inner_idx);                                                                \
    src_pad_fast.divmod(num_inner, num_inner, c_idx);                                                            \
    const int c4_idx           = c_idx / 4;                                                                      \
    const int c_in_c4_idx      = c_idx % 4;                                                                      \
    const uint64_t size        = param.n_inner;                                                                  \
    const uint64_t padChannels = param.dst_pad;                                                                  \
    const uint64_t numChannels = param.channel;                                                                  \
    const uint64_t offset      = num_inner * padChannels * size + (c4_idx * size + inner_idx) * 4 + c_in_c4_idx; \
    const uint64_t inOffset    = num_inner * numChannels * size + c_idx * size + inner_idx;                      \
    cuda_kernel_cvt_per_elems<type_mode>(input, inOffset, output, offset, param);                                \
}
cvtTYPEALL(cvtSMCHANNELNCTON4CX)

#define MAX_DIM 65533
template<CVTFormatMode mode>
void GenDimParam(
    ReFormatParam param,
    dim3& dimBlock,
    dim3& dimGrid)
{
    dimGrid.z = param.n_outer >= MAX_DIM ? MAX_DIM : param.n_outer;
    if (mode == NHWC_NDARRAY) {
        dimBlock.x = DIM;
        dimBlock.y = DIM;
        dimGrid.x  = DivUp(param.src_pad, DIM);
        dimGrid.y  = DivUp(param.n_inner, DIM) > MAX_DIM? MAX_DIM : DivUp(param.n_inner, DIM);
    } else if (mode == NDARRAY_NHWC) {
        dimBlock.x = DIM;
        dimBlock.y = DIM;
        dimGrid.x  = DivUp(param.n_inner, DIM);
        dimGrid.y  = DivUp(param.dst_pad, DIM);
    } else if (mode == N4CX_NDARRAY) {
        dimBlock.x = DIM;
        dimBlock.y = 1;
        dimGrid.x  = DivUp(param.n_inner, DIM);
        dimGrid.y  = param.src_pad / 4;
    } else if (mode == NDARRAY_N4CX) {
        dimBlock.x = DIM;
        dimBlock.y = 1;
        dimGrid.x  = DivUp(param.n_inner, DIM);
        dimGrid.y  = param.dst_pad / 4;
    } else if (mode == NHWC8_NHWC16){
        dimBlock.x = DIM;
        dimBlock.y = DIM;
        dimGrid.x  = DivUp(param.dst_pad, DIM);
        dimGrid.y  = DivUp(param.n_inner, DIM);
    } else if (mode == NHWC16_NHWC8){
        dimBlock.x = DIM;
        dimBlock.y = DIM;
        dimGrid.x  = DivUp(param.dst_pad, DIM);
        dimGrid.y  = DivUp(param.n_inner, DIM);
    } 
}
#define RFC8C16              \
    case NHWC8_NHWC16:         \
        RUN(NHWC8_NHWC16);     \
    case NHWC16_NHWC8:         \
        RUN(NHWC16_NHWC8);

#define RFNHWC                 \
    case NDARRAY_NHWC:         \
        RUN(NDARRAY_NHWC);     \
    case NHWC_NDARRAY:         \
        RUN(NHWC_NDARRAY);

#define RFN4CX             \
    case NDARRAY_N4CX:     \
        RUN(NDARRAY_N4CX); \
    case N4CX_NDARRAY:     \
        RUN(N4CX_NDARRAY);


template <CVTTypeMode mode>
__global__ void cuda_kernel_packed_cvtformat_type(
    const void *input,
    void *output,
    DivModFast inner_fast,
    int num_elems,
    ReFormatParam param) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elems) return;
    char val[16];
    _Pragma("unroll")
    for (int i = 0; i < 16; i++) {
        val[i] = 0;
    }
    int b = 0, hw_idx = 0;
    inner_fast.divmod(tid, b, hw_idx);
    int offset = b * param.n_inner * param.src_pad + hw_idx;
    for (int i = 0; i < param.src_pad; i++) {
        cuda_kernel_cvt_per_elems<mode>(input, offset, val, i, param);
        offset += param.n_inner;
    }
    float4* dst = (float4*)val;
    float4* dst_out = (float4*)output;
    dst_out[tid] = dst[0];
}
void PPLCUDASmallChannelCVTPackedFormatType(cudaStream_t stream, const void *input, void *output, ReFormatParam param)
{
    dim3 dimBlock(256, 1, 1);
    int num_elems = param.out_elems / param.dst_pad;
    dim3 dimGrid(DivUp(num_elems, 256), 1, 1);
    DivModFast inner_fast(param.n_inner);
    switch (GetCVTTypeMode(param)) {
        case FLOAT32_INT8:
            cuda_kernel_packed_cvtformat_type<FLOAT32_INT8><<<dimGrid, dimBlock, 0, stream>>>(input, output, inner_fast, num_elems, param);
            break;
        case FLOAT16_INT8:
            cuda_kernel_packed_cvtformat_type<FLOAT16_INT8><<<dimGrid, dimBlock, 0, stream>>>(input, output, inner_fast, num_elems, param);
            break;
        case INT8_INT8:
            cuda_kernel_packed_cvtformat_type<INT8_INT8><<<dimGrid, dimBlock, 0, stream>>>(input, output, inner_fast, num_elems, param);
            break;
        default:
            break;
    }
}
void PPLCUDASmallChannelCVTFormatType(cudaStream_t stream, const void *input, void *output, ReFormatParam param)
{
    if (param.out_type == ppl::common::DATATYPE_INT8 && param.out_format == ppl::common::DATAFORMAT_NHWC16
        && param.in_format == ppl::common::DATAFORMAT_NDARRAY) {
            PPLCUDASmallChannelCVTPackedFormatType(stream, input, output, param);
            return; 
        }
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 block_size(256, 1, 1);                                                   \
        int num_elems = param.out_elems;                                              \
        dim3 grid_size(DivUp(num_elems, 256), 1, 1);                                  \
        DivModFast inner_fast(param.n_inner);                                         \
        DivModFast src_pad_fast(param.src_pad);                                       \
        DivModFast dst_pad_fast(param.dst_pad);                                       \
        switch (GetCVTTypeMode(param)) {                                              \
            case FLOAT32_INT8:                                                                  \
                cuda_kernel_small_channel_cvtformat_type<FLOAT32_INT8, mode>                    \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT8_FLOAT32:                                                                  \
                cuda_kernel_small_channel_cvtformat_type<INT8_FLOAT32, mode>                    \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT32_FLOAT16:                                                               \
                cuda_kernel_small_channel_cvtformat_type<FLOAT32_FLOAT16, mode>                 \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT16_FLOAT32:                                                               \
                cuda_kernel_small_channel_cvtformat_type<FLOAT16_FLOAT32, mode>                 \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT32_INT4B:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<FLOAT32_INT4B, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT4B_FLOAT32:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<INT4B_FLOAT32, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT8_FLOAT16:                                                                  \
                cuda_kernel_small_channel_cvtformat_type<INT8_FLOAT16, mode>                    \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT16_INT8:                                                                  \
                cuda_kernel_small_channel_cvtformat_type<FLOAT16_INT8, mode>                    \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT8_INT4B:                                                                    \
                cuda_kernel_small_channel_cvtformat_type<INT8_INT4B, mode>                      \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT8_INT8:                                                                     \
                cuda_kernel_small_channel_cvtformat_type<INT8_INT8, mode>                       \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT4B_INT4B:                                                                   \
                cuda_kernel_small_channel_cvtformat_type<INT4B_INT4B, mode>                     \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT32_INT64:                                                                   \
                cuda_kernel_small_channel_cvtformat_type<INT32_INT64, mode>                     \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT64_INT32:                                                                   \
                cuda_kernel_small_channel_cvtformat_type<INT64_INT32, mode>                     \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case INT64_FLOAT32:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<INT64_FLOAT32, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            case FLOAT32_INT64:                                                                 \
                cuda_kernel_small_channel_cvtformat_type<FLOAT32_INT64, mode>                   \
                    <<<grid_size, block_size, 0, stream>>>(input, num_elems, inner_fast,        \
                    src_pad_fast, dst_pad_fast, output, param);                                 \
                break;                                                                          \
            default:                                                                            \
                break;                                                                          \
        }                                                                                       \
        return;                                                                                 \
    } while (0)

    switch (GetCVTFormatMode(param)) {
        RFNHWC
        RFN4CX
        default:
            return;
    }
#undef RUN
}

void PPLCUDANormalCVTFormatType(cudaStream_t stream, const void *input, void *output, ReFormatParam param)
{
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 block_size(32, 1, 1);                                                    \
        dim3 grid_size(32, 1, 1);                                                     \
        GenDimParam<mode>(param, block_size, grid_size);                              \
        switch (GetCVTTypeMode(param)) {                                              \
            case FLOAT32_INT8:                                                                  \
                cuda_kernel_cvtformat_type<FLOAT32_INT8, mode>                                  \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT8_FLOAT32:                                                                  \
                cuda_kernel_cvtformat_type<INT8_FLOAT32, mode>                                  \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT32_FLOAT16:                                                               \
                cuda_kernel_cvtformat_type<FLOAT32_FLOAT16, mode>                               \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT16_FLOAT32:                                                               \
                cuda_kernel_cvtformat_type<FLOAT16_FLOAT32, mode>                               \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT32_INT4B:                                                                 \
                cuda_kernel_cvtformat_type<FLOAT32_INT4B, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT4B_FLOAT32:                                                                 \
                cuda_kernel_cvtformat_type<INT4B_FLOAT32, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT8_FLOAT16:                                                                  \
                cuda_kernel_cvtformat_type<INT8_FLOAT16, mode>                                  \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT16_INT8:                                                                  \
                cuda_kernel_cvtformat_type<FLOAT16_INT8, mode>                                  \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT8_INT4B:                                                                    \
                cuda_kernel_cvtformat_type<INT8_INT4B, mode>                                    \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT8_INT8:                                                                     \
                cuda_kernel_cvtformat_type<INT8_INT8, mode>                                     \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT4B_INT4B:                                                                   \
                cuda_kernel_cvtformat_type<INT4B_INT4B, mode>                                   \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT32_INT64:                                                                   \
                cuda_kernel_cvtformat_type<INT32_INT64, mode>                                   \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT64_INT32:                                                                   \
                cuda_kernel_cvtformat_type<INT64_INT32, mode>                                   \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case INT64_FLOAT32:                                                                 \
                cuda_kernel_cvtformat_type<INT64_FLOAT32, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            case FLOAT32_INT64:                                                                 \
                cuda_kernel_cvtformat_type<FLOAT32_INT64, mode>                                 \
                    <<<grid_size, block_size, 0, stream>>>(input, output, param);               \
                break;                                                                          \
            default:                                                                            \
                break;                                                                          \
        }                                                                                       \
        return;                                                                                 \
    } while (0)

    switch (GetCVTFormatMode(param)) {
        RFC8C16
        RFNHWC
        RFN4CX
        default:
            return;
    }
#undef RUN
}

void PPLCUDACVTFormatType(
    cudaStream_t stream,
    const void* input,
    void* output,
    ReFormatParam param)
{
    if (param.channel < LEASTCHANNEL) {
        PPLCUDASmallChannelCVTFormatType(stream, input, output, param);
    } else {
        PPLCUDANormalCVTFormatType(stream, input, output, param);
    }
}