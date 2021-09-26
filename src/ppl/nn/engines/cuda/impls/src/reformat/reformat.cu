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

#include "cudakernel/reformat/reformat.h"
#include "cudakernel/common/common.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/macro.h"

#include "cuda_fp16.h"
using namespace PPLCUDA;
using namespace ppl::nn;
using namespace ppl::common;

#define DIM 32
#define LEASTCHANNEL 16
template <typename T, CVTFormatMode mode>
__global__ void cuda_kernel_cvtformat(
    T* input,
    T* output,
    ReFormatParam param)
{
}

#define cvtNCTONHWC8(type)                                                                               \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat<type, NDARRAY_NHWC8>(                                              \
    type* input,                                                                                        \
    type* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    __shared__ type share_val[DIM][DIM + 1];                                                            \
                                                                                                        \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n+= blockDim.x) {                                              \
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        int64_t idx_h = blockIdx.y * blockDim.y + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.n_inner && idx_h < param.src_pad) {                                           \
            int64_t offset = n * param.src_pad * param.n_inner + idx_h * param.n_inner + idx_w;         \
            share_val[threadIdx.y][threadIdx.x] = input[offset];                                        \
        } else {                                                                                        \
            share_val[threadIdx.y][threadIdx.x] = (type)0;                                              \
        }                                                                                               \
        __syncthreads();                                                                                \
                                                                                                        \
        idx_w = blockIdx.y * blockDim.y + threadIdx.x;                                                  \
        idx_h = blockIdx.x * blockDim.x + threadIdx.y;                                                  \
                                                                                                        \
        if (idx_w < param.dst_pad && idx_h < param.n_inner) {                                           \
            int64_t offset = n * param.dst_pad * param.n_inner + idx_h * param.dst_pad + idx_w;         \
            output[offset] = share_val[threadIdx.x][threadIdx.y];                                       \
        }                                                                                               \
    }                                                                                                   \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtNCTONHWC8(half)
#endif
    cvtNCTONHWC8(float)
    cvtNCTONHWC8(char)
    cvtNCTONHWC8(double)



#define cvtNHWC8TONC(type)                                                                               \
template<>                                                                                              \
__global__ void cuda_kernel_cvtformat<type, NHWC8_NDARRAY>(                                              \
    type* input,                                                                                        \
    type* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    __shared__ type share_val[DIM][DIM + 1];                                                            \
                                                                                                        \
    int64_t num = blockIdx.z;                                                                           \
    for (int n = num; n < param.n_outer; n += blockDim.x) {                                              \
        int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        int64_t idx_h = blockIdx.y * blockDim.y + threadIdx.y;                                          \
                                                                                                        \
        if (idx_w < param.src_pad && idx_h < param.n_inner) {                                           \
            int64_t offset = n * param.src_pad * param.n_inner + idx_h * param.src_pad + idx_w;         \
            share_val[threadIdx.y][threadIdx.x] = input[offset];                                        \
        } else {                                                                                        \
            share_val[threadIdx.y][threadIdx.x] = (type)0;                                              \
        }                                                                                               \
        __syncthreads();                                                                                \
                                                                                                        \
        idx_w = blockIdx.y * blockDim.y + threadIdx.x;                                                  \
        idx_h = blockIdx.x * blockDim.x + threadIdx.y;                                                  \
                                                                                                        \
        if (idx_w < param.n_inner && idx_h < param.dst_pad) {                                           \
            int64_t offset = n * param.dst_pad * param.n_inner + idx_h * param.n_inner + idx_w;         \
            output[offset] = share_val[threadIdx.x][threadIdx.y];                                       \
        }                                                                                               \
    }                                                                                                   \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtNHWC8TONC(half)
#endif
    cvtNHWC8TONC(float)
    cvtNHWC8TONC(char)
    cvtNHWC8TONC(double)



#define cvtN4CXTONC(type)                                                                                              \
template <>                                                                                                            \
__global__ void cuda_kernel_cvtformat<type, N4CX_NDARRAY>(                                                             \
    type * input,                                                                                                      \
    type * output,                                                                                                     \
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
            output[outOffset]        = input[offset];                                                                  \
        }                                                                                                              \
    }                                                                                                                  \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtN4CXTONC(half)
#endif
    cvtN4CXTONC(float)
    cvtN4CXTONC(char)
    cvtN4CXTONC(double)



#define cvtNCTON4CX(type)                                                                                             \
template <>                                                                                                           \
__global__ void cuda_kernel_cvtformat<type, NDARRAY_N4CX>(                                                            \
    type * input,                                                                                                     \
    type * output,                                                                                                    \
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
            output[offset]          = input[inOffset];                                                                \
        }                                                                                                             \
    }                                                                                                                 \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtNCTON4CX(half)
#endif
    cvtNCTON4CX(float)
    cvtNCTON4CX(char)
    cvtNCTON4CX(double)



template <typename T, CVTFormatMode mode>
__global__ void cuda_kernel_small_channel_cvtformat(
    T* input,
    int num_elems,
    DivModFast inner_fast,
    DivModFast src_pad_fast,
    DivModFast dst_pad_fast,
    T* output,
    ReFormatParam param)
{
}
/*
// #define cvtSMCHANNELNCTONHWC8(type)                                                                      \
// template<>                                                                                              \
// __global__ void cuda_kernel_small_channel_cvtformat<type, NDARRAY_NHWC8>(                                \
//     type* input,                                                                                        \
//     int64_t num_elems,                                                                                  \
//     type* output,                                                                                       \
//     ReFormatParam param)                                                                                \
// {                                                                                                       \
//     int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;                                                \
//     if (tid >= num_elems) return;                                                                       \
//     int c_idx = tid % param.dst_pad;                                                                    \
//     int inner_idx = (tid / param.dst_pad) % param.n_inner;                                              \
//     int outer_idx = tid / (param.dst_pad * param.n_inner);                                              \
//     int64_t offset = outer_idx * param.src_pad * param.n_inner + c_idx * param.n_inner + inner_idx;     \
//     output[tid] = c_idx > param.channel ? input[offset] : (type)0;                                      \
// }
*/
#define cvtSMCHANNELNCTONHWC8(type)                                                                      \
template<>                                                                                              \
__global__ void cuda_kernel_small_channel_cvtformat<type, NDARRAY_NHWC8>(                                \
    type* input,                                                                                        \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    type* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    dst_pad_fast.divmod(tid, num_inner, c_idx);                                                         \
    inner_idx = inner_fast.mod(num_inner);                                                              \
    int outer_idx = inner_fast.div(num_inner);                                                              \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx * param.n_inner + inner_idx;         \
    output[tid] =  c_idx < param.src_pad ? input[offset] : (type)0;                                     \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtSMCHANNELNCTONHWC8(half)
#endif
    cvtSMCHANNELNCTONHWC8(float)
    cvtSMCHANNELNCTONHWC8(char)
    cvtSMCHANNELNCTONHWC8(double)



#define cvtSMCHANNELNHWC8TONC(type)                                                                      \
template<>                                                                                              \
__global__ void cuda_kernel_small_channel_cvtformat<type, NHWC8_NDARRAY>(                                \
    type* input,                                                                                        \
    int num_elems,                                                                                      \
    DivModFast inner_fast,                                                                              \
    DivModFast src_pad_fast,                                                                            \
    DivModFast dst_pad_fast,                                                                            \
    type* output,                                                                                       \
    ReFormatParam param)                                                                                \
{                                                                                                       \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                                                    \
    if (tid >= num_elems) return;                                                                       \
    int inner_idx = 0, num_inner = 0, c_idx = 0;                                                        \
    inner_fast.divmod(tid, num_inner, inner_idx);                                                       \
    c_idx = dst_pad_fast.mod(num_inner);                                                                \
    int outer_idx = tid / (param.dst_pad * param.n_inner);                                              \
    int offset = outer_idx * param.src_pad * param.n_inner + c_idx + inner_idx * param.src_pad;         \
    output[tid] = input[offset];                                                                        \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtSMCHANNELNHWC8TONC(half)
#endif
    cvtSMCHANNELNHWC8TONC(float)
    cvtSMCHANNELNHWC8TONC(char)
    cvtSMCHANNELNHWC8TONC(double)



#define cvtSMCHANNELN4CXTONC(type)                                                                               \
template <>                                                                                                      \
__global__ void cuda_kernel_small_channel_cvtformat<type, N4CX_NDARRAY>(                                         \
    type * input,                                                                                                \
    int num_elems,                                                                                               \
    DivModFast inner_fast,                                                                                       \
    DivModFast src_pad_fast,                                                                                     \
    DivModFast dst_pad_fast,                                                                                     \
    type* output,                                                                                                \
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
    output[outOffset]          = input[offset];                                                                  \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtSMCHANNELN4CXTONC(half)
#endif
    cvtSMCHANNELN4CXTONC(float)
    cvtSMCHANNELN4CXTONC(char)
    cvtSMCHANNELN4CXTONC(double)



#define cvtSMCHANNELNCTON4CX(type)                                                                               \
template <>                                                                                                      \
__global__ void cuda_kernel_small_channel_cvtformat<type, NDARRAY_N4CX>(                                         \
    type * input,                                                                                                \
    int num_elems,                                                                                               \
    DivModFast inner_fast,                                                                                       \
    DivModFast src_pad_fast,                                                                                     \
    DivModFast dst_pad_fast,                                                                                     \
    type* output,                                                                                                \
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
    output[offset]             = input[inOffset];                                                                \
}

#if __CUDACC_VER_MAJOR__ >= 9
    cvtSMCHANNELNCTON4CX(half)
#endif
    cvtSMCHANNELNCTON4CX(float)
    cvtSMCHANNELNCTON4CX(char)
    cvtSMCHANNELNCTON4CX(double)



#define MAX_DIM 65533
template<CVTFormatMode mode>
void GenDimParam(
    ReFormatParam param,
    dim3& dimBlock,
    dim3& dimGrid)
{
    dimGrid.z = param.n_outer >= MAX_DIM ? MAX_DIM : param.n_outer;
    if (mode == NHWC8_NDARRAY) {
        dimBlock.x = DIM;
        dimBlock.y = DIM;
        dimGrid.x  = DivUp(param.src_pad, DIM);
        dimGrid.y  = DivUp(param.n_inner, DIM);
    } else if (mode == NDARRAY_NHWC8) {
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
    } else {

    }
}

#define RFNHWC8                 \
    case NDARRAY_NHWC8:         \
        RUN(NDARRAY_NHWC8);     \
    case NHWC8_NDARRAY:         \
        RUN(NHWC8_NDARRAY);

#define RFN4CX             \
    case NDARRAY_N4CX:     \
        RUN(NDARRAY_N4CX); \
    case N4CX_NDARRAY:     \
        RUN(N4CX_NDARRAY);

void PPLCUDANormalCVTFormat(cudaStream_t stream, const void *input, void *output, ReFormatParam param)
{
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 dimBlock(32, 1, 1);                                                      \
        dim3 dimGrid(32, 1, 1);                                                       \
        GenDimParam<mode>(param, dimBlock, dimGrid);                                  \
        switch (GetSizeOfDataType(param.out_type)) {                                    \
            case 1:                                                                   \
                cuda_kernel_cvtformat<char, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (char *)input, (char *)output, param);                            \
                break;                                                                \
            case 2:                                                                   \
                cuda_kernel_cvtformat<half, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (half *)input, (half *)output, param);                            \
                break;                                                                \
            case 4:                                                                   \
                cuda_kernel_cvtformat<float, mode><<<dimGrid, dimBlock, 0, stream>>>( \
                    (float *)input, (float *)output, param);                          \
                break;                                                                \
            case 8:                                                                   \
                cuda_kernel_cvtformat<double, mode><<<dimGrid, dimBlock, 0, stream>>>(\
                    (double *)input, (double *)output, param);                        \
                break;                                                                \
            default:                                                                  \
                break;                                                                \
        }                                                                             \
        return;                                                                       \
    } while (0)

    switch (GetCVTFormatMode(param)) {
        RFNHWC8
        RFN4CX
        default:
            return;
    }
#undef RUN
}

void PPLCUDASmallChannelCVTFormat(cudaStream_t stream, const void *input, void *output, ReFormatParam param)
{
#define RUN(mode)                                                                     \
    do {                                                                              \
        dim3 dimBlock(256, 1, 1);                                                     \
        int num_elems = param.out_elems;                                              \
        dim3 dimGrid(DivUp(num_elems, 256), 1, 1);                                    \
        DivModFast inner_fast(param.n_inner);                                         \
        DivModFast src_pad_fast(param.src_pad);                                       \
        DivModFast dst_pad_fast(param.dst_pad);                                       \
        switch (GetSizeOfDataType(param.out_type)) {                                    \
            case 1:                                                                   \
                cuda_kernel_small_channel_cvtformat<char, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (char *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast, \
                                    (char *)output, param);                           \
                break;                                                                \
            case 2:                                                                   \
                cuda_kernel_small_channel_cvtformat<half, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (half *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast, \
                                (half *)output, param);                               \
                break;                                                                \
            case 4:                                                                   \
                cuda_kernel_small_channel_cvtformat<float, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (float *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast, \
                                (float *)output, param);                               \
                break;                                                                \
            case 8:                                                                   \
                cuda_kernel_small_channel_cvtformat<double, mode><<<dimGrid, dimBlock, 0, stream>>>(  \
                    (double *)input, num_elems, inner_fast, src_pad_fast, dst_pad_fast, \
                                (double *)output, param);                               \
                break;                                                                \
            default:                                                                  \
                break;                                                                \
        }                                                                             \
        return;                                                                       \
    } while (0)

    switch (GetCVTFormatMode(param)) {
        RFNHWC8
        RFN4CX
        default:
            return;
    }
#undef RUN
}

void PPLCUDACVTFormat(
    cudaStream_t stream,
    const void* input,
    void* output,
    ReFormatParam param)
{
    if (param.channel < LEASTCHANNEL) {
        if (param.out_type == DATATYPE_INT8) {
            auto host_in = new int8_t[param.in_elems];
            auto host_out = new int8_t[param.out_elems];
            cudaMemcpy(host_in, input, param.in_elems, cudaMemcpyDefault);
            PPLCUDASmallChannelCVTFormat(stream, input, output, param);
            cudaMemcpy(host_out, output, param.out_elems, cudaMemcpyDefault);
            delete[] host_in;
            delete[] host_out;
        } else if (param.out_type == DATATYPE_FLOAT32) {
            auto host_in = new float[param.in_elems];
            auto host_out = new float[param.out_elems];
            cudaMemcpy(host_in, input, 4 * param.in_elems, cudaMemcpyDefault);
            PPLCUDASmallChannelCVTFormat(stream, input, output, param);
            cudaMemcpy(host_out, output, 4 * param.out_elems, cudaMemcpyDefault);
            delete[] host_in;
            delete[] host_out;
        } else {
            PPLCUDASmallChannelCVTFormat(stream, input, output, param);
        }
    } else
    {
        PPLCUDANormalCVTFormat(stream, input, output, param);
    }
}
CVTFormatMode GetCVTFormatMode(ReFormatParam param)
{
    if (param.in_format == DATAFORMAT_NDARRAY) {
        switch (param.out_format) {
            case DATAFORMAT_NHWC8:
                return NDARRAY_NHWC8;
            case DATAFORMAT_N4CX:
                return NDARRAY_N4CX;
            default:
                return CVTFormatUnknown;
        }
    } else if (param.in_format == DATAFORMAT_N4CX) {
        switch (param.out_format) {
            case DATAFORMAT_NDARRAY:
                return N4CX_NDARRAY;
            default:
                return CVTFormatUnknown;
        }
    } else if (param.in_format == DATAFORMAT_NHWC8) {
        switch (param.out_format) {
            case DATAFORMAT_NDARRAY:
                return NHWC8_NDARRAY;
            default:
                return CVTFormatUnknown;
        }
    } else {
        return CVTFormatUnknown;
    }
}

CVTTypeMode GetCVTTypeMode(ReFormatParam param)
{
    if (param.in_type == DATATYPE_FLOAT32) {
        switch (param.out_type) {
            case DATATYPE_FLOAT16:
                return FLOAT32_FLOAT16;
            case DATATYPE_INT8:
                return FLOAT32_INT8;
            case DATATYPE_INT4B:
                return FLOAT32_INT4B;
            default:
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_FLOAT16) {
        switch (param.out_type) {
            case DATATYPE_FLOAT32:
                return FLOAT16_FLOAT32;
            case DATATYPE_INT8:
                return FLOAT16_INT8;
            case DATATYPE_INT4B:
                return FLOAT16_INT4B;
            default:
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_INT8) {
        switch (param.out_type) {
            case DATATYPE_FLOAT16:
                return INT8_FLOAT16;
            case DATATYPE_FLOAT32:
                return INT8_FLOAT32;
            case DATATYPE_INT4B:
                return INT8_INT4B;
            case DATATYPE_INT8:
                return INT8_INT8;
            default:
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_INT4B) {
        switch (param.out_type) {
            case DATATYPE_FLOAT16:
                return INT4B_FLOAT16;
            case DATATYPE_FLOAT32:
                return INT4B_FLOAT32;
            case DATATYPE_INT8:
                return INT4B_INT8;
            case DATATYPE_INT4B:
                return INT4B_INT4B;
            default:
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_INT32) {
        switch (param.out_type) {
            case DATATYPE_INT64:
                return INT32_INT64;
            default:
                return CVTTypeUnknown;
        }
    }
    if (param.in_type == DATATYPE_INT64) {
        switch (param.out_type) {
            case DATATYPE_INT32:
                return INT64_INT32;
            default:
                return CVTTypeUnknown;
        }
    }
    return CVTTypeUnknown;
}

bool IsFloatEqual(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (uint32_t i = 0; i < a.size(); i++) {
        if (fabs(a[0] - b[0]) > FLT_EPSILON) {
            return false;
        }
    }
    return true;
}

bool EqualQuant(const ppl::nn::cuda::CudaTensorQuant& quant_a, const ppl::nn::cuda::CudaTensorQuant& quant_b) {
    return quant_a.bit_width == quant_b.bit_width &&
           IsFloatEqual(quant_a.scale, quant_b.scale) &&
           IsFloatEqual(quant_a.zero_point, quant_b.zero_point);
}

ppl::common::RetCode SetReLayoutParam(
    ReFormatParam *param,
    const TensorShape& input,
    const TensorShape& output)
{
    // if (input.GetDimCount() <= 1 &&
    //     ((input.GetDataFormat() == DATAFORMAT_NHWC8) ||
    //     (output.GetDataFormat() == DATAFORMAT_NHWC8)))
    //     return RC_INVALID_VALUE;
    param->n_outer = input.GetDim(0);
    param->channel = input.GetDimCount() > 1 ? input.GetDim(1) : 1;
    param->n_inner = input.GetDimCount() > 2 ? input.GetElementsFromDimensionIncludingPadding(2) : 1;
    param->in_format = input.GetDataFormat();
    param->out_format = output.GetDataFormat();
    param->in_type = input.GetDataType();
    param->out_type = output.GetDataType();
    param->mix_type   = (param->in_type != param->out_type);
    param->mix_format = (param->in_format != param->out_format);

    param->src_pad = Align(param->channel, AlignDataFormat(param->in_format));
    param->dst_pad = Align(param->channel, AlignDataFormat(param->out_format));

    param->out_elems = output.GetElementsIncludingPadding();
    param->in_elems = input.GetElementsIncludingPadding();
    return RC_SUCCESS;

}

ppl::common::RetCode SetReLayoutParam(
    ReFormatParam *param,
    const TensorShape& input,
    const ppl::nn::cuda::CudaTensorQuant& input_quant,
    const TensorShape& output,
    const ppl::nn::cuda::CudaTensorQuant& output_quant)
{
    SetReLayoutParam(param, input, output);
    param->i_step = input_quant.scale[0];
    param->i_zero_point = input_quant.zero_point[0];
    param->o_step = output_quant.scale[0];
    param->o_zero_point = output_quant.zero_point[0];
    if (param->in_type == param->out_type) {
        param->mix_type = !EqualQuant(input_quant, output_quant);
    }
    return RC_SUCCESS;
}

void PPLCUDADataConvert(
    cudaStream_t stream,
    const void* input,
    void* output,
    void* tempBuf,
    ReFormatParam& param)
{
    if (param.in_format != param.out_format && param.in_type != param.out_type) {
        PPLCUDACVTTypePerTensor(stream, input, tempBuf, param);
        PPLCUDACVTFormat(stream, tempBuf, output, param);
        return;
    } else if (param.in_format != param.out_format) {
        PPLCUDACVTFormat(stream, input, output, param);
        return;
    } else if (param.in_type != param.out_type) {
        PPLCUDACVTTypePerTensor(stream, input, output, param);
        return;
    } else {
        return;
    }
}
