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
__device__ inline void cuda_kernel_cvt_per_elems(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
}

template <>
__device__ inline void cuda_kernel_cvt_per_elems<INT8_FLOAT16>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    float i_step = param.i_step;
    ((__half*)output)[out_offset] = (__half)_int82float(((int8_t*)input)[in_offset], i_step, param.i_zero_point);
}
template <>
__device__ inline void cuda_kernel_cvt_per_elems<FLOAT16_INT8>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    float o_step = param.o_step;
    *((int8_t*)output + out_offset) = _float2int8((float)*((__half*)input + in_offset), o_step, param.o_zero_point);
}

template <>
__device__ inline void cuda_kernel_cvt_per_elems<INT8_FLOAT32>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    float i_step = param.i_step;
    ((float*)output)[out_offset] = _int82float(((int8_t*)input)[in_offset], i_step, param.i_zero_point);
}

template <>
__device__ inline void cuda_kernel_cvt_per_elems<INT8_INT8>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    float i_step = param.i_step;
    float o_step = param.o_step;
    float tmp             = _int82float(((int8_t*)input)[in_offset], i_step, param.i_zero_point);
    ((int8_t*)output)[out_offset] = _float2int8(tmp, o_step, param.o_zero_point);
}

template <>
__device__ inline void cuda_kernel_cvt_per_elems<FLOAT32_INT8>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    float o_step = param.o_step;
    *((int8_t*)output + out_offset) = _float2int8(*((float*)input + in_offset), o_step, param.o_zero_point);
}
template <>
__device__ inline void cuda_kernel_cvt_per_elems<FLOAT32_INT4B>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    float o_step = param.o_step;
    *((int8_t*)output + out_offset) = _float2int4B(*((float*)input + in_offset), o_step, param.o_zero_point);
}
template <>
__device__ inline void cuda_kernel_cvt_per_elems<INT4B_FLOAT32>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    float i_step = param.i_step;
    ((float*)output)[out_offset] = _int4B2float(((int8_t*)input)[in_offset], i_step, param.i_zero_point);
}
template <>
__device__ inline void cuda_kernel_cvt_per_elems<INT4B_INT4B>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    float i_step = param.i_step;
    float o_step = param.o_step;
    float tmp             = _int4B2float(((int8_t*)input)[in_offset], i_step, param.i_zero_point);
    ((int8_t*)output)[out_offset] = _float2int4B(tmp, o_step, param.o_zero_point);
}

template <>
__device__ inline void cuda_kernel_cvt_per_elems<FLOAT16_FLOAT32>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    ((float*)output)[out_offset] = __half2float(((__half*)input)[in_offset]);
}

template <>
__device__ inline void cuda_kernel_cvt_per_elems<FLOAT32_FLOAT16>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    ((half*)output)[out_offset] = __float2half(((float*)input)[in_offset]);
}

template <>
__device__ inline void cuda_kernel_cvt_per_elems<INT8_INT4B>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    signed char tmp     = ((signed char*)input)[in_offset];
    ((char*)output)[out_offset] = tmp > 7 ? 7 : tmp < -8 ? -8 : tmp;
}

template <>
__device__ inline void cuda_kernel_cvt_per_elems<INT32_INT64>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    ((int64_t*)output)[out_offset] = ((int32_t*)input)[in_offset];
}

template <>
__device__ inline void cuda_kernel_cvt_per_elems<INT64_INT32>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    ((int32_t*)output)[out_offset] = ((int64_t*)input)[in_offset];
}


template <>
__device__ inline void cuda_kernel_cvt_per_elems<INT64_FLOAT32>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    ((float*)output)[out_offset] = ((int64_t*)input)[in_offset];
}

template <>
__device__ inline void cuda_kernel_cvt_per_elems<FLOAT32_INT64>(const void* input, int in_offset, void* output, int out_offset, ReFormatParam param)
{
    ((int64_t*)output)[out_offset] = ((float*)input)[in_offset];
}

template <CVTTypeMode mode>
__device__ inline void cuda_kernel_set_zero_per_elems(void* output, int out_offset)
{
}

#define INST_DEST_HALF(mode)                                                                \
template <>                                                                                 \
__device__ inline void cuda_kernel_set_zero_per_elems<mode>(void* output, int out_offset)   \
{                                                                                           \
    ((__half*)output)[out_offset] = (__half)(0);                                            \
}
INST_DEST_HALF(INT8_FLOAT16)
INST_DEST_HALF(FLOAT32_FLOAT16)

#define INST_DEST_INT8(mode)                                                                \
template <>                                                                                 \
__device__ inline void cuda_kernel_set_zero_per_elems<mode>(void* output, int out_offset)   \
{                                                                                           \
    *((int8_t*)output + out_offset) = 0;                                                    \
}
INST_DEST_INT8(FLOAT16_INT8)
INST_DEST_INT8(INT8_INT8)
INST_DEST_INT8(FLOAT32_INT8)
INST_DEST_INT8(FLOAT32_INT4B)
INST_DEST_INT8(INT4B_INT4B)

#define INST_DEST_FLOAT(mode)                                                               \
template <>                                                                                 \
__device__ inline void cuda_kernel_set_zero_per_elems<mode>(void* output, int out_offset)   \
{                                                                                           \
    ((float*)output)[out_offset] = 0.f;                                                     \
}
INST_DEST_FLOAT(INT8_FLOAT32)
INST_DEST_FLOAT(INT4B_FLOAT32)
INST_DEST_FLOAT(FLOAT16_FLOAT32)
INST_DEST_FLOAT(INT64_FLOAT32)

#undef INST_DEST_HALF
#undef INST_DEST_INT8
#undef INST_DEST_FLOAT

template <>
__device__ inline void cuda_kernel_set_zero_per_elems<INT8_INT4B>(void* output, int out_offset)
{
    ((char*)output)[out_offset] = 0;
}

template <>
__device__ inline void cuda_kernel_set_zero_per_elems<INT32_INT64>(void* output, int out_offset)
{
    ((int64_t*)output)[out_offset] = 0;
}

template <>
__device__ inline void cuda_kernel_set_zero_per_elems<INT64_INT32>(void* output, int out_offset)
{
    ((int32_t*)output)[out_offset] = 0;
}

template <>
__device__ inline void cuda_kernel_set_zero_per_elems<FLOAT32_INT64>(void* output, int out_offset)
{
    ((int64_t*)output)[out_offset] = 0;
}