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

#include "cudakernel/unary/leakyrelu.h"
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <typename DataT>
__device__ __inline__ DataT ppl_scalar_leakyrelu(const DataT& in_val, float alpha);

template <>
__device__ __inline__ float ppl_scalar_leakyrelu<float>(const float& in_val, float alpha)
{
    float res;
    res = (in_val > 0) ? in_val : alpha * in_val;
    return res;
}

__device__ __inline__ int8_t ppl_scalar_leakyrelu_int8(const int8_t& in_val, float alpha, float in_scale, float out_scale)
{
    int8_t res;
    float res_f = (in_val > 0) ? in_val : alpha * in_val;
    res = round(res_f * in_scale / out_scale);
    return res;
}

template <>
__device__ __inline__ half ppl_scalar_leakyrelu<half>(const half& in_val, float alpha)
{
    half res;
    res = __hgt(in_val, 0) ? in_val : __hmul((half)alpha, in_val);
    return res;
}
#endif

template <typename DataT>
__global__ void ppl_cukernel_unary_leakyrelu(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output,
    float alpha)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    output[index] = ppl_scalar_leakyrelu<DataT>(in_val, alpha);
#endif
}

__global__ void ppl_cukernel_unary_leakyrelu(
    const uint64_t num_elems,
    const int8_t* input,
    int8_t* output,
    float alpha,
    float in_scale,
    float out_scale)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int8_t in_val  = input[index];
    output[index] = ppl_scalar_leakyrelu_int8(in_val, alpha, in_scale, out_scale);
#endif
}

ppl::common::RetCode PPLCUDAUnaryLeakyReluForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    float alpha,
    float in_scale,
    float out_scale)
{
    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int block_size     = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        ppl_cukernel_unary_leakyrelu<float><<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (float*)output, alpha);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        ppl_cukernel_unary_leakyrelu<half><<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, (half*)output, alpha);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        ppl_cukernel_unary_leakyrelu<<<grid_size, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output, alpha, in_scale, out_scale);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}
