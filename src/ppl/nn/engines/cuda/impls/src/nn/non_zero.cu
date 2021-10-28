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

#include "cudakernel/nn/non_zero.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include <memory>
#define NUM_THREADS_PER_BLOCK 256

template <typename srcT>
__device__ bool whether_true(srcT val)
{
    return val != (srcT)0;
}
template <>
__device__ bool whether_true<half>(half val)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __heq(val, 0);
#else
    return false;
#endif
}
template <typename srcT>
__global__ void count_nonzero_each_block(
    int64_t num_elems,
    const srcT* input,
    int32_t* output)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tid   = threadIdx.x;
    // if (index >= num_elems) return;
    __shared__ int reduce_smem[NUM_THREADS_PER_BLOCK];
    int count = 0;
    if (index < num_elems && whether_true(input[index]))
        ++count;
    reduce_smem[tid] = count;
    __syncthreads();
    for (int it = NUM_THREADS_PER_BLOCK / 2; it > 0; it = it >> 1) {
        if (tid < it) {
            reduce_smem[tid] += reduce_smem[tid + it];
            __syncthreads();
        }
    }
    if (tid == 0)
        output[blockIdx.x] = reduce_smem[0];
}

template <typename srcT>
__global__ void determine_nonzero_position(
    int64_t num_elems,
    int nonzero_elems,
    int num_dims,
    GArray<DivModFast> input_strides_fast,
    const srcT* input,
    int* pre_counts,
    int64_t* output)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tid   = threadIdx.x;
    __shared__ int reduce_smem[NUM_THREADS_PER_BLOCK];
    int count = 0;
    if (index < num_elems && whether_true(input[index]))
        ++count;
    reduce_smem[tid] = count;
    __syncthreads();
    int pre_count    = pre_counts[blockIdx.x];
    int pos_in_block = 0;
    for (int it = 0; it < tid; ++it) {
        pos_in_block += reduce_smem[it];
    }
    int res_pos = pre_count + pos_in_block;
    if (index < num_elems && whether_true(input[index])) {
        int remain = index, idx = 0;
        for (int it = 0; it < num_dims; ++it) {
            input_strides_fast[it].divmod(remain, idx, remain);
            output[res_pos] = (int64_t)idx;
            res_pos += nonzero_elems;
        }
    }
}

template <typename srcT>
void NonZeroImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const srcT* input,
    ppl::nn::TensorShape* output_shape,
    int64_t* output,
    int32_t* tempbuffer)
{
    // step 1: count each block
    int64_t max_elems = input_shape->GetElementsIncludingPadding();
    int num_blocks    = (max_elems + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    count_nonzero_each_block<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(max_elems,
                                                                               input,
                                                                               tempbuffer);
    std::unique_ptr<int32_t[]> host_count_each_block(new int32_t[num_blocks]);
    cudaMemcpy(host_count_each_block.get(), tempbuffer, sizeof(int32_t) * num_blocks, cudaMemcpyDeviceToHost);
    int pre_count = 0, nonzero_elems = 0;
    for (int it = 0; it < num_blocks; ++it) {
        nonzero_elems += host_count_each_block[it];
        host_count_each_block[it] = pre_count;
        pre_count                 = nonzero_elems;
    }
    cudaMemcpy(tempbuffer, host_count_each_block.get(), sizeof(int32_t) * num_blocks, cudaMemcpyHostToDevice);
    // step 2: calc result position
    int num_dims = input_shape->GetDimCount();
    GArray<DivModFast> input_strides_fast(num_dims);
    int64_t acc_input_stride = 1;
    for (int it = num_dims - 1; it >= 0; --it) {
        input_strides_fast[it] = DivModFast(acc_input_stride);
        acc_input_stride *= input_shape->GetDim(it);
    }
    determine_nonzero_position<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(max_elems,
                                                                                 nonzero_elems,
                                                                                 num_dims,
                                                                                 input_strides_fast,
                                                                                 input,
                                                                                 tempbuffer,
                                                                                 output);

    // step 3: change result count
    output_shape->SetDim(1, nonzero_elems);
}

ppl::common::RetCode PPLCUDANonZeroForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    int64_t* output,
    int32_t* tempbuffer)
{
    switch (input_shape->GetDataType()) {
        case ppl::common::DATATYPE_BOOL:
            NonZeroImp<bool>(stream, input_shape, (bool*)input, output_shape, output, tempbuffer);
            return ppl::common::RC_SUCCESS;
        case ppl::common::DATATYPE_FLOAT16:
            NonZeroImp<half>(stream, input_shape, (half*)input, output_shape, output, tempbuffer);
            return ppl::common::RC_SUCCESS;
        case ppl::common::DATATYPE_FLOAT32:
            NonZeroImp<float>(stream, input_shape, (float*)input, output_shape, output, tempbuffer);
            return ppl::common::RC_SUCCESS;
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
}

int64_t PPLNonZeroGetTempBufferSize(ppl::nn::TensorShape* input_shape)
{
    int64_t max_elems = input_shape->GetElementsIncludingPadding();
    int num_blocks    = (max_elems + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    return num_blocks * sizeof(int32_t);
}
