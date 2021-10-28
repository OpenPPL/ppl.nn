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

#include "cudakernel/unary/log.h"
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <typename T>
__device__ __inline__ T ppl_scalar_log(const T &in_val)
{
    return log(in_val);
}

template <>
__device__ __inline__ half ppl_scalar_log<half>(const half &in_val)
{
    return __float2half(log(__half2float(in_val)));
}
#endif

template <typename T>
__global__ void ppl_cukernel_log_ndarray(
    const uint64_t num_elems,
    const T *input,
    T *output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    output[index] = ppl_scalar_log<T>(input[index]);
#endif
}

template <typename T>
__global__ void ppl_cukernel_log_nhwc(
    const uint64_t num_elems,
    int channels,
    int pad_channels,
    int chw,
    int hw,
    const T *input,
    T *output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int chw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chw_idx >= chw)
        return;
    int c_idx     = chw_idx % channels;
    int hw_idx    = chw_idx / channels;
    int b_idx     = blockIdx.z;
    int64_t index = (b_idx * hw + hw_idx) * pad_channels + c_idx;

    output[index] = ppl_scalar_log<T>(input[index]);
#endif
}

ppl::common::RetCode PPLCUDALogForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape *input_shape,
    const void *input,
    const ppl::nn::TensorShape *output_shape,
    void *output)
{
    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int batch          = output_shape->GetDim(0);
    int channels       = output_shape->GetDim(1);
    int pad_channels   = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int height         = output_shape->GetDim(2);
    int width          = output_shape->GetDim(3);

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        int block_size = 256;
        int grid_size  = (num_elems + block_size - 1) / block_size;
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_log_ndarray<float><<<grid_size, block_size, 0, stream>>>(num_elems,
                                                                                  (const float *)input,
                                                                                  (float *)output);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            ppl_cukernel_log_ndarray<half><<<grid_size, block_size, 0, stream>>>(num_elems,
                                                                                 (const half *)input,
                                                                                 (half *)output);

        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
        int block_size = 256;
        dim3 grid_size;
        int chw     = channels * height * width;
        grid_size.x = (chw + block_size - 1) / block_size;
        grid_size.z = batch;
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            ppl_cukernel_log_nhwc<float><<<grid_size, block_size, 0, stream>>>(
                num_elems, channels, pad_channels, channels * height * width, height * width, (const float *)input, (float *)output);
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            ppl_cukernel_log_nhwc<half><<<grid_size, block_size, 0, stream>>>(
                num_elems, channels, pad_channels, channels * height * width, height * width, (const half *)input, (half *)output);
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}