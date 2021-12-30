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

#include "cudakernel/unary/clip.h"
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <typename T>
__device__ __inline__ T ppl_scalar_clip(const T& in_val, float _min, float _max)
{
    float resf = (float)in_val;
    resf       = (resf > _min) ? resf : _min;
    resf       = (resf < _max) ? resf : _max;
    return (T)resf;
}

#endif

template <typename T>
__global__ void ppl_cukernel_clip_ndarray(
    const uint64_t num_elems,
    const T* input,
    T* output,
    float _min,
    float _max)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    output[index] = ppl_scalar_clip<T>(input[index], _min, _max);
#endif
}

template <typename T>
__global__ void ppl_cukernel_clip_nhwc(
    const uint64_t num_elems,
    int channels,
    int pad_channels,
    int chw,
    int hw,
    const T* input,
    T* output,
    float _min,
    float _max)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int chw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chw_idx >= chw)
        return;
    int c_idx     = chw_idx % channels;
    int hw_idx    = chw_idx / channels;
    int b_idx     = blockIdx.z;
    int64_t index = (b_idx * hw + hw_idx) * pad_channels + c_idx;

    output[index] = ppl_scalar_clip<T>(input[index], _min, _max);
#endif
}

template <>
__global__ void ppl_cukernel_clip_nhwc<float4>(
    const uint64_t num_elems,
    int channels,
    int pad_channels,
    int chw,
    int hw,
    const float4* input,
    float4* output,
    float _min,
    float _max)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    __shared__ float4 s_indata[256];
    __shared__ float4 s_outdata[256];
    int tid     = threadIdx.x;
    int chw_idx = blockIdx.x * blockDim.x + tid;
    if (chw_idx >= chw)
        return;
    int b_idx     = blockIdx.z;
    int64_t index = b_idx * hw * pad_channels + chw_idx;
    s_indata[tid] = input[index];

    half* s_indata_half  = reinterpret_cast<half*>(s_indata);
    half* s_outdata_half = reinterpret_cast<half*>(s_outdata);
    for (int it = 0; it < 8; it++) {
        int inner_idx             = tid * 8 + it;
        s_outdata_half[inner_idx] = ppl_scalar_clip<half>(s_indata_half[inner_idx], _min, _max);
    }
    output[index] = s_outdata[tid];
#endif
}

ppl::common::RetCode PPLCUDAClipForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    float _min,
    float _max)
{
    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int batch          = output_shape->GetDim(0);
    int channels       = output_shape->GetDim(1);
    int pad_channels   = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int height         = output_shape->GetDimCount() > 2 ? output_shape->GetDim(2) : 1;
    int width          = output_shape->GetDimCount() > 3 ? output_shape->GetDim(3) : 1;

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        int block_size = 256;
        int grid_size  = (num_elems + block_size - 1) / block_size;
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_clip_ndarray<float><<<grid_size, block_size, 0, stream>>>(num_elems,
                                                                                   (const float*)input,
                                                                                   (float*)output,
                                                                                   _min,
                                                                                   _max);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            ppl_cukernel_clip_ndarray<half><<<grid_size, block_size, 0, stream>>>(num_elems,
                                                                                  (const half*)input,
                                                                                  (half*)output,
                                                                                  _min,
                                                                                  _max);

        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
               output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        int block_size = 256;
        dim3 grid_size;
        int chw     = channels * height * width;
        grid_size.x = (chw + block_size - 1) / block_size;
        grid_size.z = batch;
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_clip_nhwc<float><<<grid_size, block_size, 0, stream>>>(
                num_elems, channels, pad_channels, channels * height * width, height * width, (const float*)input, (float*)output, _min, _max);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            if (channels & 0x7) {
                ppl_cukernel_clip_nhwc<half><<<grid_size, block_size, 0, stream>>>(
                    num_elems, channels, pad_channels, channels * height * width, height * width, (const half*)input, (half*)output, _min, _max);
            } else {
                int pad_chw = pad_channels * height * width;
                grid_size.x = (pad_chw + block_size - 1) / block_size;
                ppl_cukernel_clip_nhwc<float4><<<grid_size, block_size, 0, stream>>>(
                    num_elems >> 3, channels, pad_channels >> 3, pad_chw >> 3, height * width, (const float4*)input, (float4*)output, _min, _max);
            }
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}