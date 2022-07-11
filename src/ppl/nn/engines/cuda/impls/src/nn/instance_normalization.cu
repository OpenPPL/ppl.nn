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

#include "cudakernel/nn/instance_normalization.h"
#include "cudakernel/common/common.cuh"
#include "cudakernel/math/math.h"
#include "ppl/nn/common/tensor_shape.h"
#include <cuda_fp16.h>

template<typename T>
__device__ __forceinline__ void BlockDoubleReduceSum(T& val0, T& val1) {
    __shared__ T shared0[32];
    __shared__ T shared1[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val0 = WarpReduceSum(val0);
    val1 = WarpReduceSum(val1);
    if(lane == 0) {
        shared0[wid] = val0;
        shared1[wid] = val1;
    }
    __syncthreads();

    val0 = (lane < (blockDim.x >> 5)) ? shared0[lane] : (T)0.0f;
    val1 = (lane < (blockDim.x >> 5)) ? shared1[lane] : (T)0.0f;
    val0 = WarpReduceSum(val0);
    val1 = WarpReduceSum(val1);
    return;
}

template<typename T, typename TPar>
__global__ void ppl_cukernel_instancenorm(const T* in, const TPar* alpha,
                                const TPar* beta, T* out, const int channels,
                                const int HW, bool with_relu, const float eps = 1e-5) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int b_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    auto cur_in = in + b_idx * channels * HW + c_idx * HW;
    auto cur_out = out + b_idx * channels * HW + c_idx * HW;
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    for(auto tid = threadIdx.x; tid < HW; tid += blockDim.x) {
        float v = (float)__ldg(cur_in + tid);
        sum.x += v;
        sum.y += v * v;
    }
    //BlockReduceSum
    BlockDoubleReduceSum(sum.x, sum.y);
    float mean = sum.x / HW;
    float rstd = rsqrtf(sum.y / HW - mean * mean + float(eps));
    for(auto tid = threadIdx.x; tid < HW; tid += blockDim.x) {
        float out_val = ((float)__ldg(cur_in + tid) - mean) * rstd * 
                            (float)__ldg(alpha + c_idx) + (float)__ldg(beta + c_idx);
        if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
        cur_out[tid] = T(out_val);
    }
#endif
}

template<typename T, typename TPar>
__global__ void ppl_cukernel_instancenorm_nhwc(const T* in, const TPar* alpha,
                                const TPar* beta, T* out, const int pad_channels,
                                const int HW, bool with_relu, const float eps = 1e-5) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int b_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    auto cur_in = in + b_idx * pad_channels * HW + c_idx;
    auto cur_out = out + b_idx * pad_channels * HW + c_idx;
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    for(auto tid = threadIdx.x; tid < HW; tid += blockDim.x) {
        float v = (float)__ldg(cur_in + tid * pad_channels);
        sum.x += v;
        sum.y += v * v;
    }
    //BlockReduceSum
    BlockDoubleReduceSum(sum.x, sum.y);
    float mean = sum.x / HW;
    float rstd = rsqrtf(sum.y / HW - mean * mean + float(eps));
    for(auto tid = threadIdx.x; tid < HW; tid += blockDim.x) {
        float in_val = (float)__ldg(cur_in + tid * pad_channels);
        float out_val = (in_val - mean) * rstd * 
                        (float)__ldg(alpha + c_idx) + (float)__ldg(beta + c_idx);
        if (with_relu) out_val = out_val > 0.f ? out_val : 0.f;
        cur_out[tid * pad_channels] = T(out_val);
    }
#endif
}

__global__ void ppl_cukernel_instancenorm_nhwc_int8(const char* in, const float* alpha,
                                const float* beta, char* out, const int pad_channels,
                                const int HW, bool with_relu, float in_scale,
                                float out_scale, const float eps = 1e-5) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int b_idx = blockIdx.y;
    int c_idx = blockIdx.x;
    auto cur_in = in + b_idx * pad_channels * HW + c_idx;
    auto cur_out = out + b_idx * pad_channels * HW + c_idx;
    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    for(auto tid = threadIdx.x; tid < HW; tid += blockDim.x) {
        float v = (float)__ldg(cur_in + tid * pad_channels) * in_scale;
        sum.x += v;
        sum.y += v * v;
    }
    //BlockReduceSum
    BlockDoubleReduceSum(sum.x, sum.y);
    float mean = sum.x / HW;
    float rstd = rsqrtf(sum.y / HW - mean * mean + float(eps));
    for(auto tid = threadIdx.x; tid < HW; tid += blockDim.x) {
        float in_val = (float)__ldg(cur_in + tid * pad_channels) * in_scale;
        float out_val = float((in_val - mean) * rstd * 
                        (float)__ldg(alpha + c_idx) + (float)__ldg(beta + c_idx));
        int int_val = __float2int_rn(out_val * out_scale);
        char dst = int_val < -128 ? -128 : int_val > 127 ? 127 : (char)int_val; 
        if (with_relu) {
            dst = dst > 0 ? dst : 0;
        }
        cur_out[tid * pad_channels] = dst;
    }
#endif
}

ppl::common::RetCode PPLCUDAInstanceNormalizationForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* scale_shape,
    const void* scale,
    // share scale shape
    const void* B,
    ppl::nn::TensorShape* output_shape,
    void* output,
    float epsilon,
    float in_scale,
    float out_scale,
    bool with_relu)
{
    int dim_count = input_shape->GetDimCount();
    int batch     = input_shape->GetDim(0);
    int channels  = dim_count >= 2 ? input_shape->GetDim(1) : 1;
    int hw_count  = 1;
    for (int it = 2; it < dim_count; ++it)
        hw_count *= input_shape->GetDim(it);
    int64_t num_elems = output_shape->CalcElementsExcludingPadding();
    int block_size    = GetBlockSize(hw_count);
    dim3 grid_size(channels, batch, 1);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_instancenorm<float, float><<<grid_size, block_size, 0, stream>>>(
                (const float*)input, (const float*)scale, (const float*)B, (float*)output,
                channels, hw_count, with_relu, epsilon);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            ppl_cukernel_instancenorm<half, half><<<grid_size, block_size, 0, stream>>>(
                (const half*)input, (const half*)scale, (const half*)B, (half*)output,
                channels, hw_count, with_relu, epsilon);
        }
    } else {
        int pad_channels = dim_count >= 2 ? input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1) : 1;
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_instancenorm_nhwc<float, float><<<grid_size, block_size, 0, stream>>>(
                (const float*)input, (const float*)scale, (const float*)B, (float*)output,
                pad_channels, hw_count, with_relu, epsilon);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            ppl_cukernel_instancenorm_nhwc<half, half><<<grid_size, block_size, 0, stream>>>(
                (const half*)input, (const half*)scale, (const half*)B, (half*)output,
                pad_channels, hw_count, with_relu, epsilon);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
            ppl_cukernel_instancenorm_nhwc_int8<<<grid_size, block_size, 0, stream>>>(
                (const char*)input, (const float*)scale, (const float*)B, (char*)output,
                pad_channels, hw_count, with_relu, in_scale, out_scale, epsilon);
        }
    }
    return ppl::common::RC_SUCCESS;
}