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
__global__ void cuda_kernel_cvt(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
}

template <>
__global__ void cuda_kernel_cvt<INT8_FLOAT16>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((half*)output)[id] = __float2half(_int82float(((int8_t*)input)[id], param.i_step, param.i_zero_point));
}

template <>
__global__ void cuda_kernel_cvt<INT8_FLOAT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = _int82float(((int8_t*)input)[id], param.i_step, param.i_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<INT8_INT8>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    float tmp             = _int82float(((int8_t*)input)[id], param.i_step, param.i_zero_point);
    ((int8_t*)output)[id] = _float2int8(tmp, param.o_step, param.o_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<FLOAT32_INT8>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    *((int8_t*)output + id) = _float2int8(*((float*)input + id), param.o_step, param.o_zero_point);
}
template <>
__global__ void cuda_kernel_cvt<FLOAT32_INT4B>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    *((int8_t*)output + id) = _float2int4B(*((float*)input + id), param.o_step, param.o_zero_point);
}
template <>
__global__ void cuda_kernel_cvt<INT4B_FLOAT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = _int4B2float(((int8_t*)input)[id], param.i_step, param.i_zero_point);
}
template <>
__global__ void cuda_kernel_cvt<INT4B_INT4B>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    float tmp             = _int4B2float(((int8_t*)input)[id], param.i_step, param.i_zero_point);
    ((int8_t*)output)[id] = _float2int4B(tmp, param.o_step, param.o_zero_point);
}

template <>
__global__ void cuda_kernel_cvt<FLOAT16_FLOAT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = __half2float(((half*)input)[id]);
}

template <>
__global__ void cuda_kernel_cvt<FLOAT32_FLOAT16>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((half*)output)[id] = __float2half(((float*)input)[id]);
}

template <>
__global__ void cuda_kernel_cvt<INT8_INT4B>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    signed char tmp     = ((signed char*)input)[id];
    ((char*)output)[id] = tmp > 7 ? 7 : tmp < -8 ? -8 : tmp;
}

template <>
__global__ void cuda_kernel_cvt<INT32_INT64>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((int64_t*)output)[id] = ((int32_t*)input)[id];
}

template <>
__global__ void cuda_kernel_cvt<INT64_INT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((int32_t*)output)[id] = ((int64_t*)input)[id];
}


template <>
__global__ void cuda_kernel_cvt<INT64_FLOAT32>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((float*)output)[id] = ((int64_t*)input)[id];
}

template <>
__global__ void cuda_kernel_cvt<FLOAT32_INT64>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    ((int64_t*)output)[id] = ((float*)input)[id];
}

static __device__ inline char4 _float42int8(
    float4 data_in,
    float step,
    signed char zeroPoint)
{
    float4 tmp;
    tmp.x = data_in.x / step + zeroPoint;
    tmp.y = data_in.y / step + zeroPoint;
    tmp.z = data_in.z / step + zeroPoint;
    tmp.w = data_in.w / step + zeroPoint;
    char4 dst;
    dst.x = tmp.x > 127 ? 127 : tmp.x < -128 ? -128 : (signed char)((tmp.x));
    dst.y = tmp.y > 127 ? 127 : tmp.y < -128 ? -128 : (signed char)((tmp.y));
    dst.z = tmp.z > 127 ? 127 : tmp.z < -128 ? -128 : (signed char)((tmp.z));
    dst.w = tmp.w > 127 ? 127 : tmp.w < -128 ? -128 : (signed char)((tmp.w));
    return dst;
}

template <CVTTypeMode mode>
__global__ void cuda_kernel_cvt_packed(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
}
template <>
__global__ void cuda_kernel_cvt_packed<FLOAT32_INT8>(size_t num_elems, const void* input, ReFormatParam param, void* output)
{
    JUDGE(num_elems)
    *((char4*)output + id) = _float42int8(*((float4*)input + id), param.o_step, param.o_zero_point);
}

void PPLCUDACVTTypePerTensor(
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
            if(num_elems % 4 == 0) {
                block_size = 128;
                num_elems = num_elems >> 2;
                grid_size = (num_elems + block_size - 1) / block_size;
                cuda_kernel_cvt_packed<FLOAT32_INT8><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            } else{
                cuda_kernel_cvt<FLOAT32_INT8><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            }
            break;
        case INT8_FLOAT32:
            cuda_kernel_cvt<INT8_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case FLOAT32_FLOAT16:
            cuda_kernel_cvt<FLOAT32_FLOAT16><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case FLOAT16_FLOAT32:
            cuda_kernel_cvt<FLOAT16_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case FLOAT32_INT4B:
            cuda_kernel_cvt<FLOAT32_INT4B><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT4B_FLOAT32:
            cuda_kernel_cvt<INT4B_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT8_FLOAT16:
            cuda_kernel_cvt<INT8_FLOAT16><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case FLOAT16_INT8:
            cuda_kernel_cvt<FLOAT16_INT8><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT8_INT4B:
            cuda_kernel_cvt<INT8_INT4B><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT8_INT8:
            cuda_kernel_cvt<INT8_INT8><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT4B_INT4B:
            cuda_kernel_cvt<INT4B_INT4B><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT32_INT64:
            cuda_kernel_cvt<INT32_INT64><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT64_INT32:
            cuda_kernel_cvt<INT64_INT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case INT64_FLOAT32:
            cuda_kernel_cvt<INT64_FLOAT32><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        case FLOAT32_INT64:
            cuda_kernel_cvt<FLOAT32_INT64><<<grid_size, block_size, 0, stream>>>(num_elems, input, param, output);
            break;
        default:
            break;
    }
}