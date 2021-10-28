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

#include "cudakernel/unary/not.h"
#include <cuda_fp16.h>

__device__ inline bool ppl_not_scalar(bool a)
{
    return (!a);
}

__global__ void ppl_cukernel_not_naive(
    const uint64_t num_elems,
    const bool *input,
    bool *output)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    output[index] = ppl_not_scalar(input[index]);
}

template <typename T>
__global__ void ppl_cukernel_not(
    int type_width,
    const uint64_t num_elems,
    const bool *input,
    bool *output)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int tid = threadIdx.x;
    __shared__ T transm[256];
    const T *input_ptr = reinterpret_cast<const T *>(input);
    transm[tid]        = input_ptr[index];

    bool *smem_ptr = reinterpret_cast<bool *>(transm + tid);
    for (int it = 0; it < type_width; ++it) {
        smem_ptr[it] = ppl_not_scalar(smem_ptr[it]);
    }
    T *output_ptr     = reinterpret_cast<T *>(output);
    output_ptr[index] = transm[tid];
}

ppl::common::RetCode PPLCUDANotForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape *input_shape,
    const bool *input,
    const ppl::nn::TensorShape *output_shape,
    bool *output)
{
    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int block_size     = 256;
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        if (!(num_elems & 0xf)) {
            int channel_shift  = 4;
            uint64_t grid_size = ((num_elems >> channel_shift) + block_size - 1) / block_size;
            int type_width     = 16;
            ppl_cukernel_not<float4><<<grid_size, block_size, 0, stream>>>(type_width, num_elems >> channel_shift, (const bool *)input, (bool *)output);
        } else if (!(num_elems & 0x7)) {
            int channel_shift  = 3;
            uint64_t grid_size = ((num_elems >> channel_shift) + block_size - 1) / block_size;
            int type_width     = 8;
            ppl_cukernel_not<float2><<<grid_size, block_size, 0, stream>>>(type_width, num_elems >> channel_shift, (const bool *)input, (bool *)output);
        } else if (!(num_elems & 0x3)) {
            int channel_shift  = 2;
            uint64_t grid_size = ((num_elems >> channel_shift) + block_size - 1) / block_size;
            int type_width     = 4;
            ppl_cukernel_not<float><<<grid_size, block_size, 0, stream>>>(type_width, num_elems >> channel_shift, (const bool *)input, (bool *)output);
        } else if (!(num_elems & 0x1)) {
            int channel_shift  = 1;
            uint64_t grid_size = ((num_elems >> channel_shift) + block_size - 1) / block_size;
            int type_width     = 2;
            ppl_cukernel_not<half><<<grid_size, block_size, 0, stream>>>(type_width, num_elems >> channel_shift, (const bool *)input, (bool *)output);
        } else {
            uint64_t grid_size = (num_elems + block_size - 1) / block_size;
            ppl_cukernel_not_naive<<<grid_size, block_size, 0, stream>>>(num_elems, (const bool *)input, (bool *)output);
        }
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}