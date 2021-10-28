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

#include "cudakernel/nn/batch_normalization.h"
#include "cudakernel/math/math.h"
#include "cudakernel/common/common.h"
#include "cudakernel/common/divmod_fast.h"
#include "ppl/nn/common/tensor_shape.h"
#include <cuda_fp16.h>

template <typename T>
__device__ T ppl_get_std(T var_val, T eps)
{
    return 1.f / sqrtf(var_val + eps);
}

template <>
__device__ half ppl_get_std<half>(half var_val, half eps)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hdiv(half(1), hsqrt(__hadd(var_val, eps)));
#else
    return half(0);
#endif
}

template <typename T>
__global__ void ppl_cukernel_batchnorm_withmeanvar(
    int64_t num_elems,
    DivModFast channel_fast,
    int channels,
    const T* input,
    const T* scale,
    const T* B,
    const T* mean,
    const T* var,
    float eps,
    T* output)
{
    typedef Math<T, T, T> OpMath;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    T t_eps          = (T)eps;
    int c_idx        = 0;
    int inner_offset = 0;
    channel_fast.divmod(index, c_idx, inner_offset);
    c_idx = c_idx % channels;

    T scale_val   = scale[c_idx];
    T B_val       = B[c_idx];
    T mean_val    = mean[c_idx];
    T var_val     = var[c_idx];
    T std         = ppl_get_std<T>(var_val, t_eps);
    output[index] = OpMath::add(OpMath::mul(OpMath::mul(OpMath::sub(input[index], mean_val), std), scale_val), B_val);
}

template <typename T>
__global__ void ppl_cukernel_batchnorm_withmeanvar_nhwc(
    int64_t num_elems,
    DivModFast channel_fast,
    int pad_channels,
    const T* input,
    const T* scale,
    const T* B,
    const T* mean,
    const T* var,
    float eps,
    T* output)
{
    typedef Math<T, T, T> OpMath;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    T t_eps          = (T)eps;
    int c_idx        = 0;
    int outer_offset = 0;
    channel_fast.divmod(index, outer_offset, c_idx);

    T scale_val        = scale[c_idx];
    T B_val            = B[c_idx];
    T mean_val         = mean[c_idx];
    T var_val          = var[c_idx];
    T std              = ppl_get_std<T>(var_val, t_eps);
    int nhwc_index     = outer_offset * pad_channels + c_idx;
    output[nhwc_index] = OpMath::add(OpMath::mul(OpMath::mul(OpMath::sub(input[nhwc_index], mean_val), std), scale_val), B_val);
}

ppl::common::RetCode PPLCUDABatchNormalizationForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* scale_shape,
    const void* scale,
    // share scale shape
    const void* B,
    const void* mean,
    const void* var,
    ppl::nn::TensorShape* output_shape,
    void* output,
    float epsilon)
{
    int dim_count = input_shape->GetDimCount();
    int batch     = input_shape->GetDim(0);
    int channels  = dim_count >= 2 ? input_shape->GetDim(1) : 1;
    int hw_count  = 1;
    for (int it = 2; it < dim_count; ++it)
        hw_count *= input_shape->GetDim(it);
    int64_t num_elems = output_shape->GetElementsExcludingPadding();
    int block_size    = 256;
    int grid_size     = DivUp(num_elems, block_size);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        DivModFast channel_fast(hw_count);
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_batchnorm_withmeanvar<float><<<grid_size, block_size, 0, stream>>>(num_elems, channel_fast, channels, (const float*)input, (const float*)scale, (const float*)B, (const float*)mean, (const float*)var, epsilon, (float*)output);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            ppl_cukernel_batchnorm_withmeanvar<half><<<grid_size, block_size, 0, stream>>>(num_elems, channel_fast, channels, (const half*)input, (const half*)scale, (const half*)B, (const half*)mean, (const half*)var, epsilon, (half*)output);
        }
    } else {
        DivModFast channel_fast(channels);
        int pad_channels = dim_count >= 2 ? input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1) : 1;
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_batchnorm_withmeanvar_nhwc<float><<<grid_size, block_size, 0, stream>>>(num_elems, channel_fast, pad_channels, (const float*)input, (const float*)scale, (const float*)B, (const float*)mean, (const float*)var, epsilon, (float*)output);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            ppl_cukernel_batchnorm_withmeanvar_nhwc<half><<<grid_size, block_size, 0, stream>>>(num_elems, channel_fast, pad_channels, (const half*)input, (const half*)scale, (const half*)B, (const half*)mean, (const half*)var, epsilon, (half*)output);
        }
    }
    return ppl::common::RC_SUCCESS;
}
