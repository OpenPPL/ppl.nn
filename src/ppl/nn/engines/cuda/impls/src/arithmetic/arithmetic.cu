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

#include "cudakernel/arithmetic/arithmetic.h"
#include "ppl/common/types.h"
#include <cuda_fp16.h>
#include <algorithm>
enum ArithmeticOpType {
    Arithmetic_Unknown = 0,
    Arithmetic_Add,
    Arithmetic_Sub,
    Arithmetic_Mul,
    Arithmetic_Div,
    Arithmetic_Max,
    Arithmetic_Min,
    Arithmetic_Pow,
    Arithmetic_PRelu, // similar to arithmetic
    Arithmetic_OpNum,
    Arithmetic_ForceWord = INT_MAX,
};

struct half8_ {
    half x0;
    half y0;
    half z0;
    half w0;
    half x1;
    half y1;
    half z1;
    half w1;
};

template<ArithmeticOpType op_type, typename T>
__device__ inline T ppl_arithmetic_scalar(T a, T b);

template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Add, float>(float a, float b) {
    return a + b;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Sub, float>(float a, float b) {
    return a - b;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Mul, float>(float a, float b) {
    return a * b;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Div, float>(float a, float b) {
    return a / b;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Max, float>(float a, float b) {
    return (a > b) ? a : b;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Min, float>(float a, float b) {
    return (a > b) ? b : a;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Pow, float>(float a, float b) {
    return powf(a ,b);
}

template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_PRelu, float>(float a, float b) {
    float res = a;
    res = (a > 0) ? res : res * b;
    return res;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Add, int64_t>(int64_t a, int64_t b) {
    return a + b;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Sub, int64_t>(int64_t a, int64_t b) {
    return a - b;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Mul, int64_t>(int64_t a, int64_t b) {
    return a * b;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Div, int64_t>(int64_t a, int64_t b) {
    return a / b;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Max, int64_t>(int64_t a, int64_t b) {
    return (a > b) ? a : b;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Min, int64_t>(int64_t a, int64_t b) {
    return (a > b) ? b : a;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Pow, int64_t>(int64_t a, int64_t b) {
    return powf(a ,b);
}

template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_PRelu, int64_t>(int64_t a, int64_t b) {
    int64_t res = a;
    res = (a > 0) ? res : res * b;
    return res;
}

template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Add, int32_t>(int32_t a, int32_t b) {
    return a + b;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Sub, int32_t>(int32_t a, int32_t b) {
    return a - b;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Mul, int32_t>(int32_t a, int32_t b) {
    return a * b;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Div, int32_t>(int32_t a, int32_t b) {
    return a / b;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Max, int32_t>(int32_t a, int32_t b) {
    return (a > b) ? a : b;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Min, int32_t>(int32_t a, int32_t b) {
    return (a > b) ? b : a;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Pow, int32_t>(int32_t a, int32_t b) {
    return powf(a ,b);
}

template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_PRelu, int32_t>(int32_t a, int32_t b) {
    int32_t res = a;
    res = (a > 0) ? res : res * b;
    return res;
}

template<ArithmeticOpType op_type, typename T>
__device__ inline T ppl_arithmetic_scalar_int8(T a, T b, float in_scale0, float in_scale1, float out_scale);

template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Add, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = round((a * in_scale0 + b * in_scale1) / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Sub, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = round((a * in_scale0 - b * in_scale1) / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Mul, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = round(a * b * in_scale0 * in_scale1 / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Div, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = round((float(a) / b * in_scale0 / in_scale1) / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Max, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    return (a > b) ? a : b;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Min, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    return (a > b) ? b : a;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Pow, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = powf(a, b);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}

template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_PRelu, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = a;
    res = (a > 0) ? res : res * b;
    res = round(res * in_scale0 / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}


template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Add, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hadd(a, b);
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Sub, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hsub(a, b);
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Mul, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hmul(a, b);
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Div, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hdiv(a, b);
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Max, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hgt(a, b) ? a : b;
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Min, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hgt(a, b) ? b : a;
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Pow, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __float2half(powf(__half2float(a), __half2float(b)));
#else
    return 0;
#endif
}

template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_PRelu, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    half res = a;
    res      = __hgt(a, 0) ? res : __hmul(res, b);
    return res;
#else
    return 0;
#endif
}

template <ArithmeticOpType op_type>
static __device__ inline half ppl_arithmetic_vector_fp16(half a, half b)
{
    half res;
    res = ppl_arithmetic_scalar<op_type, half>(a, b);
    return res;
}


template <ArithmeticOpType op_type>
static __device__ inline half8_ ppl_arithmetic_vector_fp16(half8_ a, half8_ b)
{
    half8_ res;
    res.x0 = ppl_arithmetic_scalar<op_type, half>(a.x0, b.x0);
    res.y0 = ppl_arithmetic_scalar<op_type, half>(a.y0, b.y0);
    res.z0 = ppl_arithmetic_scalar<op_type, half>(a.z0, b.z0);
    res.w0 = ppl_arithmetic_scalar<op_type, half>(a.w0, b.w0);
    res.x1 = ppl_arithmetic_scalar<op_type, half>(a.x1, b.x1);
    res.y1 = ppl_arithmetic_scalar<op_type, half>(a.y1, b.y1);
    res.z1 = ppl_arithmetic_scalar<op_type, half>(a.z1, b.z1);
    res.w1 = ppl_arithmetic_scalar<op_type, half>(a.w1, b.w1);
    return res;
}

static void ppl_pad_tensor_shape(const ppl::nn::TensorShape *tensor_shape0,
                          const ppl::nn::TensorShape *tensor_shape1,
                          ppl::nn::TensorShape *pad_tensor_shape0,
                          ppl::nn::TensorShape *pad_tensor_shape1) {
    int max_dims = std::max(tensor_shape0->GetDimCount(), tensor_shape1->GetDimCount());
    if (pad_tensor_shape0->GetDimCount() < pad_tensor_shape1->GetDimCount()) {
        pad_tensor_shape0->SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape0->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape0->SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape0->SetDim(i, tensor_shape0->GetDim(i - offset));
        }
    } else {
        pad_tensor_shape1->SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape1->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape1->SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape1->SetDim(i, tensor_shape1->GetDim(i - offset));
        }
    }
}

static int ppl_get_num_broadcast_dims(const ppl::nn::TensorShape *tensor_shape0,
                            const ppl::nn::TensorShape *tensor_shape1,
                            int &aixs, bool &bidirectional) {
    ppl::nn::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::nn::TensorShape pad_tensor_shape1 = *tensor_shape1;
    ppl_pad_tensor_shape(tensor_shape0, tensor_shape1,
            &pad_tensor_shape0, &pad_tensor_shape1);
    int dim_count = pad_tensor_shape0.GetDimCount();
    int num_broadcast_dims = 0;
    int fisrt_broadcast = 0, second_broadcast = 0;
    for(int it = 0; it < dim_count; ++it) {
        if (pad_tensor_shape0.GetDim(it) < pad_tensor_shape1.GetDim(it)) {
            ++num_broadcast_dims; ++fisrt_broadcast;
        } else if (pad_tensor_shape0.GetDim(it) > pad_tensor_shape1.GetDim(it)) {
            ++num_broadcast_dims; ++second_broadcast;
        }
    }
    if (fisrt_broadcast > 0 && second_broadcast > 0) bidirectional = true;
    if (num_broadcast_dims == 1) {
        for(int it = 0; it < dim_count; ++it) {
            if (pad_tensor_shape0.GetDim(it) != pad_tensor_shape1.GetDim(it))
                aixs = it;
        }
    }
    return num_broadcast_dims;
}

bool ppl_feature_broadcast(
    const ppl::nn::TensorShape *tensor_shape0,
    const ppl::nn::TensorShape *tensor_shape1,
    int *axis) 
{
    bool bidirectional = false;
    ppl::nn::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::nn::TensorShape pad_tensor_shape1 = *tensor_shape1;
    ppl_pad_tensor_shape(tensor_shape0, tensor_shape1,
            &pad_tensor_shape0, &pad_tensor_shape1);
    int dim_count = pad_tensor_shape0.GetDimCount();
    int num_broadcast_dims = 0;
    int fisrt_broadcast = 0, second_broadcast = 0;
    for(int it = 0; it < dim_count; ++it) {
        if (pad_tensor_shape0.GetDim(it) < pad_tensor_shape1.GetDim(it)) {
            ++num_broadcast_dims; ++fisrt_broadcast;
        } else if (pad_tensor_shape0.GetDim(it) > pad_tensor_shape1.GetDim(it)) {
            ++num_broadcast_dims; ++second_broadcast;
        }
    }
    if (fisrt_broadcast > 0 && second_broadcast > 0) bidirectional = true;
    for(int it = 0; it < dim_count; ++it) {
        if (pad_tensor_shape0.GetDim(it) != pad_tensor_shape1.GetDim(it))
        {
            *axis = it;
            break;
        }
    }
    return !bidirectional && (num_broadcast_dims == dim_count - *axis) && (dim_count > 2) && (*axis == 2);
}
void ppl_arithmetic_prepare_strides(
    const ppl::nn::TensorShape *tensor_shape0,
    const ppl::nn::TensorShape *tensor_shape1,
    const ppl::nn::TensorShape *tensor_shape_out,
    const int packed_channel,
    uint32_t *stride_in0,
    uint32_t *stride_in1,
    uint32_t *stride_out)
{
    ppl::nn::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::nn::TensorShape pad_tensor_shape1 = *tensor_shape1;
    ppl_pad_tensor_shape(tensor_shape0, tensor_shape1,
            &pad_tensor_shape0, &pad_tensor_shape1);

    const int dimCount   = tensor_shape_out->GetDimCount();
    uint32_t stride0     = 1;
    uint32_t stride1     = 1;
    uint32_t stride_out0 = 1;

    for (int i = dimCount - 1; i >= 0; i--) {
        stride_in0[i] = pad_tensor_shape0.GetDim(i) == 1 ? 0 : stride0;
        stride_in1[i] = pad_tensor_shape1.GetDim(i) == 1 ? 0 : stride1;
        stride_out[i] = stride_out0;
        if (i == 1) { // for channel dim, div packed_channel
            stride0 *= (pad_tensor_shape0.GetDim(i) + packed_channel - 1) / packed_channel;
            stride1 *= (pad_tensor_shape1.GetDim(i) + packed_channel - 1) / packed_channel;
            stride_out0 *= (tensor_shape_out->GetDim(i) + packed_channel - 1) / packed_channel;
        } else {
            stride0 *= pad_tensor_shape0.GetDim(i);
            stride1 *= pad_tensor_shape1.GetDim(i);
            stride_out0 *= tensor_shape_out->GetDim(i);
        }
    }
}

void ppl_arithmetic_prepare_strides_nhwc(
    const ppl::nn::TensorShape *tensor_shape0,
    const ppl::nn::TensorShape *tensor_shape1,
    const ppl::nn::TensorShape *tensor_shape_out,
    const int packed_channel,
    uint32_t *stride_in0,
    uint32_t *stride_in1,
    uint32_t *stride_out,
    int suppled_channel = 1)
{
    if (tensor_shape0->GetDimCount() < 2 || tensor_shape1->GetDimCount() < 2) return;
    ppl::nn::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::nn::TensorShape pad_tensor_shape1 = *tensor_shape1;
    ppl_pad_tensor_shape(tensor_shape0, tensor_shape1,
            &pad_tensor_shape0, &pad_tensor_shape1);

    const int dimCount   = tensor_shape_out->GetDimCount();
    uint32_t stride0     = 1;
    uint32_t stride1     = 1;
    uint32_t stride_out0 = 1;

    for (int stride_pos = dimCount - 1; stride_pos >= 0; stride_pos--) {
        int i = stride_pos;
        if (stride_pos == dimCount - 1) i = 1;
        else if (stride_pos == 0) i = 0;
        else i = stride_pos + 1;
        stride_in0[stride_pos] = pad_tensor_shape0.GetDim(i) == 1 ? 0 : stride0;
        stride_in1[stride_pos] = pad_tensor_shape1.GetDim(i) == 1 ? 0 : stride1;
        stride_out[stride_pos] = stride_out0;
        if (i == 1) { // for channel dim, div packed_channel
            stride0 *= (pad_tensor_shape0.GetDim(i) + packed_channel - 1) / packed_channel * suppled_channel;
            stride1 *= (pad_tensor_shape1.GetDim(i) + packed_channel - 1) / packed_channel * suppled_channel;
            stride_out0 *= (tensor_shape_out->GetDim(i) + packed_channel - 1) / packed_channel * suppled_channel;
        } else {
            stride0 *= pad_tensor_shape0.GetDim(i);
            stride1 *= pad_tensor_shape1.GetDim(i);
            stride_out0 *= tensor_shape_out->GetDim(i);
        }
    }
}

#define MAXDIMENSIONS 7

struct ArithmeticParam {
    uint32_t stride_in0[MAXDIMENSIONS];
    uint32_t stride_in1[MAXDIMENSIONS];
    uint32_t stride_out[MAXDIMENSIONS];
};

template <ArithmeticOpType op_type, typename T1, typename T2>
__global__ void ppl_cukernel_arithmetic_fp16(
    const uint64_t num_elems,
    const int dim_count,
    ArithmeticParam param,
    const T1 *input0,
    const T1 *input1,
    T1 *output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int tid = threadIdx.x;
    __shared__ T2 transm[512];
    T1 *transm_half      = reinterpret_cast<T1 *>(transm);
    const T2 *input0_ptr = reinterpret_cast<const T2 *>(input0);
    const T2 *input1_ptr = reinterpret_cast<const T2 *>(input1);
    T2 *output_ptr       = reinterpret_cast<T2 *>(output);

    uint64_t out_index = index;
    uint64_t offset0   = 0;
    uint64_t offset1   = 0;
    for (int i = 0; i < dim_count; i++) {
        uint64_t dim_off = index / param.stride_out[i];
        offset0 += dim_off * param.stride_in0[i];
        offset1 += dim_off * param.stride_in1[i];
        index = index % param.stride_out[i];
    }

    transm[tid + 0]   = input0_ptr[offset0];
    transm[tid + 256] = input1_ptr[offset1];

    transm_half[tid] = ppl_arithmetic_vector_fp16<op_type>(transm_half[tid + 0], transm_half[tid + 256]);

    output_ptr[out_index] = transm[tid];
#endif
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic(
    const uint64_t num_elems,
    const int dim_count, 
    ArithmeticParam param,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;

    uint64_t out_index = index;
    uint64_t offset0 = 0;
    uint64_t offset1 = 0;
    for (int i = 0; i < dim_count; i++) {
        uint64_t dim_off = index / param.stride_out[i];
        offset0 += dim_off * param.stride_in0[i];
        offset1 += dim_off * param.stride_in1[i];
        index = index % param.stride_out[i]; 
    }
    
    output[out_index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_int8(
    const uint64_t num_elems,
    const int dim_count, 
    ArithmeticParam param,
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;

    uint64_t out_index = index;
    uint64_t offset0 = 0;
    uint64_t offset1 = 0;
    for (int i = 0; i < dim_count; i++) {
        uint64_t dim_off = index / param.stride_out[i];
        offset0 += dim_off * param.stride_in0[i];
        offset1 += dim_off * param.stride_in1[i];
        index = index % param.stride_out[i]; 
    }
    
    output[out_index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1], in_scale0, in_scale1, out_scale);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_nobroadcast(
    const uint64_t num_elems,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    output[index] = ppl_arithmetic_scalar<op_type, T>(input0[index], input1[index]);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_nobroadcast_int8(
    const uint64_t num_elems,
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    output[index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[index], input1[index], in_scale0, in_scale1, out_scale);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_scalar(
    const uint64_t num_elems,
    const bool first_shorter, 
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int calc_index = 0;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_scalar_int8(
    const uint64_t num_elems,
    const bool first_shorter, 
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int calc_index = 0;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1], in_scale0, in_scale1, out_scale);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_dimension(
    const uint64_t num_elems,
    const int32_t inner_dim,
    const bool first_shorter, 
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int calc_index = index % inner_dim;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_dimension_int8(
    const uint64_t num_elems,
    const int32_t inner_dim,
    const bool first_shorter, 
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int calc_index = index % inner_dim;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1], in_scale0, in_scale1, out_scale);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_broadcast(
    const uint64_t num_elems,
    const int outer_stride, 
    const int inner_dim, 
    const bool first_shorter, 
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int inner_idx = index % inner_dim;
    int outer_idx = index / outer_stride;
    uint64_t calc_index = outer_idx * inner_dim + inner_idx;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_broadcast_int8(
    const uint64_t num_elems,
    const int outer_stride, 
    const int inner_dim, 
    const bool first_shorter, 
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int inner_idx = index % inner_dim;
    int outer_idx = index / outer_stride;
    uint64_t calc_index = outer_idx * inner_dim + inner_idx;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1], in_scale0, in_scale1, out_scale);
}

template<ArithmeticOpType op_type, typename T>
ppl::common::RetCode PPLCUDAArithMeticForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const T *input0,
    const ppl::nn::TensorShape* input_shape1,
    const T *input1,
    const ppl::nn::TensorShape* output_shape,
    T *output) {
    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int dim_count = output_shape->GetDimCount();
    int block_size = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    int axis = 0; bool bidirectional = false;
    int num_broadcast_dims = ppl_get_num_broadcast_dims(input_shape0, input_shape1, axis, bidirectional);
    if (!bidirectional && ((input_shape0->GetDimCount() < 2) || (input_shape1->GetDimCount() < 2))) {
        bool first_shorter = false;
        if (input_shape0->GetElementsIncludingPadding() < input_shape1->GetElementsIncludingPadding()) {
            first_shorter = true;
        }
        if (input_shape0->GetElementsIncludingPadding() == 1 || input_shape1->GetElementsIncludingPadding() == 1) {
            ppl_cukernel_arithmetic_one_scalar<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
        } else {
            int inner_dim = first_shorter ? input_shape0->GetDim(0) : input_shape1->GetDim(0);
            ppl_cukernel_arithmetic_one_dimension<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
        }
    } else if (num_broadcast_dims == 0) {
        ppl_cukernel_arithmetic_nobroadcast<op_type, T><<<grid_size, block_size, 0,
            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output);
    } else {
        ArithmeticParam param;
        int packed_channel = 1;
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
            if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
                // one broadcast (or last dimensions broadcast)
                if (ppl_feature_broadcast(input_shape0, input_shape1, &axis)) {
                    int inner_dim = output_shape->GetElementsToDimensionIncludingPadding(axis) * 
                                    output_shape->GetElementsFromDimensionIncludingPadding(axis - 1) / 
                                    output_shape->GetElementsIncludingPadding();
                    int outer_stride =  output_shape->GetElementsFromDimensionIncludingPadding(1);
                    bool first_shorter = false;
                    if (input_shape0->GetRealDimCount() == input_shape1->GetRealDimCount() &&
                        input_shape0->GetDim(axis) < input_shape1->GetDim(axis)) {
                        first_shorter = true;
                    }
                    if (input_shape0->GetElementsExcludingPadding() < input_shape1->GetElementsExcludingPadding())  {
                        first_shorter = true;
                    }
                    ppl_cukernel_arithmetic_one_broadcast<op_type, half><<<grid_size, block_size, 0,
                        stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const half*)input0, (const half*)input1, (half*)output); 
                // normal case, deal one half once
                } else if ((input_shape0->GetDim(1) & 0x7) || (input_shape1->GetDim(1) & 0x7)) {
                    int channel_shift  = 0;
                    int packed_channel = 8;
                    int suppled_channel = 8;
                    uint64_t grid_size = ((num_elems >> channel_shift) + block_size - 1) / block_size;
                    ppl_arithmetic_prepare_strides_nhwc(input_shape0, input_shape1, output_shape, packed_channel, param.stride_in0, param.stride_in1, param.stride_out, suppled_channel);
                    ppl_cukernel_arithmetic_fp16<op_type, half, half><<<grid_size, block_size,0, stream>>>(num_elems >> channel_shift, dim_count, param, (const half*)input0, (const half*)input1, (half*)output);
                } else { // deal 8 half once
                    int channel_shift  = 3;
                    int packed_channel = 8;
                    ppl_arithmetic_prepare_strides_nhwc(input_shape0, input_shape1, output_shape, packed_channel, param.stride_in0, param.stride_in1, param.stride_out);
                    ppl_cukernel_arithmetic_fp16<op_type, half8_, float4><<<grid_size, block_size,0, stream>>>(num_elems >> channel_shift, dim_count, param, (const half8_ *)input0, (const half8_ *)input1, (half8_ *)output);
                }
                return ppl::common::RC_SUCCESS;
            }
            ppl_arithmetic_prepare_strides_nhwc(input_shape0, input_shape1, output_shape, packed_channel,
                param.stride_in0, param.stride_in1, param.stride_out);
            ppl_cukernel_arithmetic<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, dim_count, param, (const T*)input0, (const T*)input1, (T*)output);
         } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            if (num_broadcast_dims == 1) {
                int inner_dim = 1;
                for(int it = axis + 1; it < dim_count; inner_dim *= output_shape->GetDim(it), ++it);
                int outer_stride = inner_dim * output_shape->GetDim(axis);
                bool first_shorter = false;
                if (input_shape0->GetRealDimCount() == input_shape1->GetRealDimCount() &&
                    input_shape0->GetDim(axis) < input_shape1->GetDim(axis)) {
                    first_shorter = true;
                }
                if (input_shape0->GetElementsExcludingPadding() < input_shape1->GetElementsExcludingPadding())  {
                    first_shorter = true;
                }
                ppl_cukernel_arithmetic_one_broadcast<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
                return ppl::common::RC_SUCCESS;
            }
            ppl_arithmetic_prepare_strides(input_shape0, input_shape1,
                output_shape, packed_channel, param.stride_in0, param.stride_in1, param.stride_out);
            ppl_cukernel_arithmetic<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, dim_count, param, (const T*)input0, (const T*)input1, (T*)output);
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    return ppl::common::RC_SUCCESS;
}

template<ArithmeticOpType op_type, typename T>
ppl::common::RetCode PPLCUDAArithMeticForwardImpInt8(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const T *input0,
    const ppl::nn::TensorShape* input_shape1,
    const T *input1,
    const ppl::nn::TensorShape* output_shape,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int dim_count = output_shape->GetDimCount();
    int block_size = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    int axis = 0; bool bidirectional = false;
    int num_broadcast_dims = ppl_get_num_broadcast_dims(input_shape0, input_shape1, axis, bidirectional);
    if (!bidirectional && ((input_shape0->GetDimCount() < 2) || (input_shape1->GetDimCount() < 2))) {
        bool first_shorter = false;
        if (input_shape0->GetElementsIncludingPadding() < input_shape1->GetElementsIncludingPadding()) {
            first_shorter = true;
        }
        if (input_shape0->GetElementsIncludingPadding() == 1 || input_shape1->GetElementsIncludingPadding() == 1) {
            ppl_cukernel_arithmetic_one_scalar_int8<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
        } else {
            int inner_dim = first_shorter ? input_shape0->GetDim(0) : input_shape1->GetDim(0);
            ppl_cukernel_arithmetic_one_dimension_int8<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
        }
    } else if (num_broadcast_dims == 0) {
        ppl_cukernel_arithmetic_nobroadcast_int8<op_type, T><<<grid_size, block_size, 0,
            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
    } else {
        ArithmeticParam param;
        int packed_channel = 1;
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
            output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
                if(ppl_feature_broadcast(input_shape0, input_shape1, &axis)) {
                    int inner_dim = output_shape->GetElementsToDimensionIncludingPadding(axis) * 
                                    output_shape->GetElementsFromDimensionIncludingPadding(axis - 1) / 
                                    output_shape->GetElementsIncludingPadding();
                    int outer_stride =  output_shape->GetElementsFromDimensionIncludingPadding(1);
                    bool first_shorter = false;
                    if (input_shape0->GetRealDimCount() == input_shape1->GetRealDimCount() &&
                        input_shape0->GetDim(axis) < input_shape1->GetDim(axis)) {
                        first_shorter = true;
                    }
                    if (input_shape0->GetElementsExcludingPadding() < input_shape1->GetElementsExcludingPadding())  {
                        first_shorter = true;
                    }
                    ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
                } else {
                    ppl_arithmetic_prepare_strides_nhwc(input_shape0, input_shape1, output_shape, packed_channel,
                    param.stride_in0, param.stride_in1, param.stride_out);
                    ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
                }
        } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            if (num_broadcast_dims == 1) {
                int inner_dim = 1;
                for(int it = axis + 1; it < dim_count; inner_dim *= output_shape->GetDim(it), ++it);
                int outer_stride = inner_dim * output_shape->GetDim(axis);
                bool first_shorter = false;
                if (input_shape0->GetRealDimCount() == input_shape1->GetRealDimCount() &&
                    input_shape0->GetDim(axis) < input_shape1->GetDim(axis)) {
                    first_shorter = true;
                }
                if (input_shape0->GetElementsExcludingPadding() < input_shape1->GetElementsExcludingPadding())  {
                    first_shorter = true;
                }
                ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
                return ppl::common::RC_SUCCESS;
            }
            ppl_arithmetic_prepare_strides(input_shape0, input_shape1,
                output_shape, packed_channel, param.stride_in0, param.stride_in1, param.stride_out);
            ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    return ppl::common::RC_SUCCESS;
}


#define INSTANT(OPTYPE) \
ppl::common::RetCode PPLCUDAArithMetic##OPTYPE##ForwardImp( \
    cudaStream_t stream, \
    const ppl::nn::TensorShape* input_shape0, \
    const void *input0, \
    const ppl::nn::TensorShape* input_shape1, \
    const void *input1, \
    const ppl::nn::TensorShape* output_shape, \
    void *output, \
    float in_scale0 = 0, \
    float in_scale1 = 0, \
    float out_scale = 0) { \
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, half>(stream, \
            input_shape0, (const half*)input0, input_shape1, \
            (const half*)input1, output_shape, (half*)output); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, float>(stream, \
            input_shape0, (const float*)input0, input_shape1, \
            (const float*)input1, output_shape, (float*)output); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT64) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, int64_t>(stream, \
            input_shape0, (const int64_t*)input0, input_shape1, \
            (const int64_t*)input1, output_shape, (int64_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT32) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, int32_t>(stream, \
            input_shape0, (const int32_t*)input0, input_shape1, \
            (const int32_t*)input1, output_shape, (int32_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT8) { \
        return PPLCUDAArithMeticForwardImpInt8<Arithmetic_##OPTYPE, int8_t>(stream, \
            input_shape0, (const int8_t*)input0, input_shape1, \
            (const int8_t*)input1, output_shape, (int8_t*)output, in_scale0, in_scale1, out_scale); \
    } else { \
        return ppl::common::RC_UNSUPPORTED; \
    } \
}

#define INSTANT_LIMNHWC8(OPTYPE) \
ppl::common::RetCode PPLCUDAArithMetic##OPTYPE##ForwardImp( \
    cudaStream_t stream, \
    const ppl::nn::TensorShape* input_shape0, \
    const void *input0, \
    const ppl::nn::TensorShape* input_shape1, \
    const void *input1, \
    const ppl::nn::TensorShape* output_shape, \
    void *output, \
    float in_scale0 = 0, \
    float in_scale1 = 0, \
    float out_scale = 0) { \
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) { \
        if (input_shape0->GetDimCount() >= 2 && (input_shape0->GetDim(1) & 0x7)) \
            return ppl::common::RC_UNSUPPORTED; \
        if (input_shape1->GetDimCount() >= 2 && (input_shape1->GetDim(1) & 0x7)) \
            return ppl::common::RC_UNSUPPORTED; \
    } \
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, half>(stream, \
            input_shape0, (const half*)input0, input_shape1, \
            (const half*)input1, output_shape, (half*)output); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, float>(stream, \
            input_shape0, (const float*)input0, input_shape1, \
            (const float*)input1, output_shape, (float*)output); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT64) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, int64_t>(stream, \
            input_shape0, (const int64_t*)input0, input_shape1, \
            (const int64_t*)input1, output_shape, (int64_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT32) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, int32_t>(stream, \
            input_shape0, (const int32_t*)input0, input_shape1, \
            (const int32_t*)input1, output_shape, (int32_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT8) { \
        return PPLCUDAArithMeticForwardImpInt8<Arithmetic_##OPTYPE, int8_t>(stream, \
            input_shape0, (const int8_t*)input0, input_shape1, \
            (const int8_t*)input1, output_shape, (int8_t*)output, in_scale0, in_scale1, out_scale); \
    } else { \
        return ppl::common::RC_UNSUPPORTED; \
    } \
} 

INSTANT(Add);
INSTANT(Sub);
INSTANT(Mul);
INSTANT_LIMNHWC8(Div);
INSTANT_LIMNHWC8(Max);
INSTANT_LIMNHWC8(Min);
INSTANT_LIMNHWC8(Pow);
INSTANT_LIMNHWC8(PRelu);

#undef INSTANT
