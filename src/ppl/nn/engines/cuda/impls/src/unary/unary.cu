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

#include "cudakernel/unary/unary.h"
#include "ppl/nn/engines/cuda/impls/src/reformat/cvt_int8_float.cuh"
#include <cuda_fp16.h>

enum UnaryOpType {
    Unary_Unknown = 0,
    Unary_Abs,
    Unary_Relu,
    Unary_Sigmoid,
    Unary_Sqrt,
    Unary_Square,
    Unary_TanH,
    Unary_Floor,
    Unary_Ceil,
    Unary_OpNum,
    Unary_Erf,
    Unary_Sin,
    Unary_Cos,
    Unary_Round,
    Unary_ForceWord = INT_MAX,
};

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <UnaryOpType OpT, typename DataT>
__device__ __inline__ DataT ppl_scalar_unary(const DataT& in_val);

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Abs, float>(const float& in_val)
{
    return fabsf(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Abs, half>(const half& in_val)
{
    return __float2half(fabsf(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Abs, int8_t>(const int8_t& in_val)
{
    return in_val >= 0 ? in_val : -in_val; 
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Relu, float>(const float& in_val)
{
    float res;
    res = (in_val > 0) ? in_val : 0;
    return res;
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Relu, half>(const half& in_val)
{
    half res;
    res = __hgt(in_val, 0) ? in_val : half(0);
    return res;
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Relu, int8_t>(const int8_t& in_val)
{
    int8_t res;
    res = (in_val > 0) ? in_val : 0;
    return res;
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Sigmoid, float>(const float& in_val)
{
    return 1.f / (1.f + expf(-in_val));
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Sigmoid, half>(const half& in_val)
{
    float in_valf = __half2float(in_val);
    float resf    = 1.f / (1.f + expf(-in_valf));
    return __float2half(resf);
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Sigmoid, int8_t>(const int8_t& in_val)
{
    return 1 / (1 + int8_t(expf(float(-in_val))));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Sqrt, float>(const float& in_val)
{
    return sqrt(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Sqrt, half>(const half& in_val)
{
    return __float2half(sqrt(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Sqrt, int8_t>(const int8_t& in_val)
{
    return int8_t(sqrt(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Square, float>(const float& in_val)
{
    return in_val * in_val;
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Square, half>(const half& in_val)
{
    return in_val * in_val;
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Square, int8_t>(const int8_t& in_val)
{
    return in_val * in_val;
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_TanH, float>(const float& in_val)
{
    return tanh(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_TanH, half>(const half& in_val)
{
    return __float2half(tanh(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_TanH, int8_t>(const int8_t& in_val)
{
    return int8_t(tanh(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Floor, float>(const float& in_val)
{
    return floor(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Floor, half>(const half& in_val)
{
    return hfloor(in_val);
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Floor, int8_t>(const int8_t& in_val)
{
    return int8_t(floor(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Ceil, float>(const float& in_val)
{
    return ceil(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Ceil, half>(const half& in_val)
{
    return hceil(in_val);
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Ceil, int8_t>(const int8_t& in_val)
{
    return int8_t(ceil(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Erf, float>(const float& in_val)
{
    return erf(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Erf, half>(const half& in_val)
{
    return __float2half(erf(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Erf, int8_t>(const int8_t& in_val)
{
    return int8_t(erf(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Sin, float>(const float& in_val)
{
    return sin(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Sin, half>(const half& in_val)
{
    return __float2half(sin(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Sin, int8_t>(const int8_t& in_val)
{
    return int8_t(sin(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Cos, float>(const float& in_val)
{
    return cos(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Cos, half>(const half& in_val)
{
    return __float2half(cos(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Cos, int8_t>(const int8_t& in_val)
{
    return int8_t(cos(float(in_val)));
}

template <>
__device__ __inline__ float ppl_scalar_unary<Unary_Round, float>(const float& in_val)
{
    return roundf(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_unary<Unary_Round, half>(const half& in_val)
{
    return __float2half(roundf(__half2float(in_val)));
}

template <>
__device__ __inline__ int8_t ppl_scalar_unary<Unary_Round, int8_t>(const int8_t& in_val)
{
    return int8_t(roundf(float(in_val)));
}


#endif

template <UnaryOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_any(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    output[index] = ppl_scalar_unary<OpT, DataT>(in_val);
#endif
}

template <UnaryOpType OpT, typename DataT>
__global__ void ppl_cukernel_unary_any_int8(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output,
    ppl::nn::cuda::QuantParamCuda qparam)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    float in_val_f = _int82float(in_val, qparam.i_step, qparam.i_zero_point);
    float out_val_f = ppl_scalar_unary<OpT, float>(in_val_f);
    output[index] = _float2int8(out_val_f, qparam.o_step, qparam.o_zero_point);
#endif
}

#define UNARY_INSTANT(TYPE)                                                                                                                    \
    ppl::common::RetCode PPLCUDAUnary##TYPE##ForwardImp(                                                                                       \
        cudaStream_t stream,                                                                                                                   \
        const ppl::nn::TensorShape* input_shape,                                                                                               \
        const void* input,                                                                                                                     \
        const ppl::nn::TensorShape* output_shape,                                                                                              \
        void* output,                                                                                                                          \
        const ppl::nn::cuda::QuantParamCuda* qparam)                                                                                                 \
    {                                                                                                                                          \
        uint64_t num_elems = output_shape->CalcElementsIncludingPadding();                                                                     \
        int block_size     = 256;                                                                                                              \
        uint64_t grid_size = (num_elems + block_size - 1) / block_size;                                                                        \
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {                                                                    \
            ppl_cukernel_unary_any<Unary_##TYPE, float><<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (float*)output); \
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {                                                             \
            ppl_cukernel_unary_any<Unary_##TYPE, half><<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, (half*)output);    \
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {                                                                \
            ppl_cukernel_unary_any_int8<Unary_##TYPE, int8_t><<<grid_size, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, *qparam);    \
        } else {                                                                                                                               \
            return ppl::common::RC_UNSUPPORTED;                                                                                                \
        }                                                                                                                                      \
        return ppl::common::RC_SUCCESS;                                                                                                        \
    }

UNARY_INSTANT(Abs);
UNARY_INSTANT(Relu);
UNARY_INSTANT(TanH);
UNARY_INSTANT(Sigmoid);
UNARY_INSTANT(Sqrt);
UNARY_INSTANT(Square);
UNARY_INSTANT(Floor);
UNARY_INSTANT(Ceil);
UNARY_INSTANT(Erf);
UNARY_INSTANT(Sin);
UNARY_INSTANT(Cos);
UNARY_INSTANT(Round);

#undef UNARY_INSTANT
