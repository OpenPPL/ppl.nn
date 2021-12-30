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

#include "cudakernel/reformat/reformat.h"
#include <cuda_fp16.h>

#define JUDGE(elems) uint64_t id = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x; \
                    if (id >= elems) return;
static __device__ inline signed char _float2int8(
    float data_in,
    float step,
    signed char zeroPoint)
{
    float tmp = (float)data_in / step + zeroPoint;

    return tmp > 127 ? 127 : tmp < -128 ? -128
                                        : (signed char)(__float2int_rn(tmp)); //saturate
}

static __device__ inline float _int82float(
    signed char data_in,
    float step,
    signed char zeroPoint)
{
    float tmp = (float)(data_in - zeroPoint) * step;

    return tmp;
}

static __device__ inline signed char _float2int4B(
    float data_in,
    float step,
    signed char zeroPoint)
{
    float tmp = (float)data_in / step + zeroPoint;

    return tmp > 7 ? 7 : tmp < -8 ? -8
                                  : (signed char)(__float2int_rn(tmp)); //saturate
}

static __device__ inline float _int4B2float(
    signed char data_in,
    float step,
    signed char zeroPoint)
{
    float tmp = (float)(data_in - zeroPoint) * step;
    return tmp;
}

template <CVTTypeMode mode>
__global__ void cuda_kernel_cvt(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
}

template <>
__global__ void cuda_kernel_cvt<INT8_FLOAT16>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    int c_id = (id / stride) % channels;
    float i_step = param.i_step_ptr[c_id];
    ((__half*)output)[id] = (__half)_int82float(((int8_t*)input)[id], i_step, param.i_zero_point);
}
template <>
__global__ void cuda_kernel_cvt<FLOAT16_INT8>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    int c_id = (id / stride) % channels;
    float o_step = param.o_step_ptr[c_id];
    *((int8_t*)output + id) = _float2int8((float)*((__half*)input + id), o_step, param.o_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<INT8_FLOAT32>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    int c_id = (id / stride) % channels;
    float i_step = param.i_step_ptr[c_id];
    ((float*)output)[id] = _int82float(((int8_t*)input)[id], i_step, param.i_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<INT8_INT8>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    int c_id = (id / stride) % channels;
    float i_step = param.i_step_ptr[c_id];
    float o_step = param.o_step_ptr[c_id];
    float tmp             = _int82float(((int8_t*)input)[id], i_step, param.i_zero_point);
    ((int8_t*)output)[id] = _float2int8(tmp, o_step, param.o_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<FLOAT32_INT8>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    int c_id = (id / stride) % channels;
    float o_step = param.o_step_ptr[c_id];
    *((int8_t*)output + id) = _float2int8(*((float*)input + id), o_step, param.o_zero_point);
}
template <>
__global__ void cuda_kernel_cvt<FLOAT32_INT4B>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    int c_id = (id / stride) % channels;
    float o_step = param.o_step_ptr[c_id];
    *((int8_t*)output + id) = _float2int4B(*((float*)input + id), o_step, param.o_zero_point);
}
template <>
__global__ void cuda_kernel_cvt<INT4B_FLOAT32>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    int c_id = (id / stride) % channels;
    float i_step = param.i_step_ptr[c_id];
    ((float*)output)[id] = _int4B2float(((int8_t*)input)[id], i_step, param.i_zero_point);
}
template <>
__global__ void cuda_kernel_cvt<INT4B_INT4B>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    int c_id = (id / stride) % channels;
    float i_step = param.i_step_ptr[c_id];
    float o_step = param.o_step_ptr[c_id];
    float tmp             = _int4B2float(((int8_t*)input)[id], i_step, param.i_zero_point);
    ((int8_t*)output)[id] = _float2int4B(tmp, o_step, param.o_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<FLOAT16_FLOAT32>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = __half2float(((__half*)input)[id]);
}

template <>
__global__ void cuda_kernel_cvt<FLOAT32_FLOAT16>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((half*)output)[id] = __float2half(((float*)input)[id]);
}

template <>
__global__ void cuda_kernel_cvt<INT8_INT4B>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    signed char tmp     = ((signed char*)input)[id];
    ((char*)output)[id] = tmp > 7 ? 7 : tmp < -8 ? -8 : tmp;
}

template <>
__global__ void cuda_kernel_cvt<INT32_INT64>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((int64_t*)output)[id] = ((int32_t*)input)[id];
}

template <>
__global__ void cuda_kernel_cvt<INT64_INT32>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((int32_t*)output)[id] = ((int64_t*)input)[id];
}


template <>
__global__ void cuda_kernel_cvt<INT64_FLOAT32>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = ((int64_t*)input)[id];
}

template <>
__global__ void cuda_kernel_cvt<FLOAT32_INT64>(size_t num_elems, int channels, int stride, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((int64_t*)output)[id] = ((float*)input)[id];
}

void PPLCUDACVTTypePerChannel(
    cudaStream_t stream,
    const void* input,
    void* output,
    ReFormatParam param)
{
    int block_size   = 256;
    uint64_t num_elems = param.n_outer * param.src_pad * param.n_inner;

    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    switch (GetCVTTypeMode(param)) {
        case FLOAT32_INT8:
            cuda_kernel_cvt<FLOAT32_INT8><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case INT8_FLOAT32:
            cuda_kernel_cvt<INT8_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case FLOAT32_FLOAT16:
            cuda_kernel_cvt<FLOAT32_FLOAT16><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case FLOAT16_FLOAT32:
            cuda_kernel_cvt<FLOAT16_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case FLOAT32_INT4B:
            cuda_kernel_cvt<FLOAT32_INT4B><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case INT4B_FLOAT32:
            cuda_kernel_cvt<INT4B_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case INT8_FLOAT16:
            cuda_kernel_cvt<INT8_FLOAT16><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case FLOAT16_INT8:
            cuda_kernel_cvt<FLOAT16_INT8><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case INT8_INT4B:
            cuda_kernel_cvt<INT8_INT4B><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case INT8_INT8:
            cuda_kernel_cvt<INT8_INT8><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case INT4B_INT4B:
            cuda_kernel_cvt<INT4B_INT4B><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case INT32_INT64:
            cuda_kernel_cvt<INT32_INT64><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case INT64_INT32:
            cuda_kernel_cvt<INT64_INT32><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case INT64_FLOAT32:
            cuda_kernel_cvt<INT64_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        case FLOAT32_INT64:
            cuda_kernel_cvt<FLOAT32_INT64><<<grid_size, block_size, 0, stream>>>(num_elems, param.quant_dim_size, param.quant_stride, input, param, output);
            break;
        default:
            break;
    }
}