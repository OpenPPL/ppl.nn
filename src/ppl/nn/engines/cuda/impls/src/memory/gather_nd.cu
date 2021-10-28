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

#include "cudakernel/memory/gather_nd.h"
#include "cudakernel/common/divmod_fast.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/common/types.h"
#include <cuda_runtime.h>
#include <assert.h>
#include <vector>

template <typename T>
__global__ void ppl_cukernel_gather_nd(
    int64_t num_elems,
    DivModFast piece_size_fast,
    int64_t* piece_offsets,
    const T* input,
    T* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int piece_idx, offset;
    piece_size_fast.divmod(index, piece_idx, offset);
    int64_t base_offset = piece_offsets[piece_idx];
    output[index]       = input[base_offset + offset];
}

template <typename IndexT>
__global__ void ppl_cukernel_gather_nd_offset(
    int64_t num_pieces,
    DivModFast num_pieces_per_batch_fast,
    int batch_dim,
    int64_t* input_dims_gpu,
    int input_batch_stride,
    int64_t* input_strides_gpu,
    int indices_last_dim_size,
    const IndexT* indices_data,
    int64_t* piece_offsets)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_pieces)
        return;
    // batch offset
    int batch_idx             = num_pieces_per_batch_fast.div(index);
    int64_t batch_offset      = batch_idx * input_batch_stride;
    // inner offset
    const IndexT* indices_ptr = indices_data + index * indices_last_dim_size;
    int64_t rel_offset        = 0;
    for (int it = 0; it < indices_last_dim_size; ++it) {
        IndexT cor_val = indices_ptr[it];
        if (cor_val < 0)
            cor_val += input_dims_gpu[batch_dim + it];
        assert(cor_val >= 0 && cor_val < input_dims_gpu[batch_dim + it]);
        rel_offset += cor_val * input_strides_gpu[it];
    }
    piece_offsets[index] = batch_offset + rel_offset;
}

int64_t pplGatherNDGetTempBufferSize(
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* indices_shape,
    const void* indices)
{
    int num_input_dim   = input_shape->GetDimCount();
    int num_indices_dim = indices_shape->GetDimCount();
    int num_pieces      = indices_shape->GetElementsToDimensionIncludingPadding(num_indices_dim - 1);
    // pieces offsets and input strides and input_dims
    int64_t total_size  = (num_pieces + 2 * num_input_dim) * sizeof(int64_t);
    return total_size;
}

ppl::common::RetCode PPLCUDAGatherNDForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* indices_shape,
    const void* indices,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    int batch_dim)
{
    int num_batches           = input_shape->GetElementsToDimensionIncludingPadding(batch_dim);
    int input_batch_stride    = input_shape->GetElementsFromDimensionIncludingPadding(batch_dim);
    int num_indices_dim       = indices_shape->GetDimCount();
    int num_input_dim         = input_shape->GetDimCount();
    int indices_last_dim_size = indices_shape->GetDim(num_indices_dim - 1);
    int num_pieces            = indices_shape->GetElementsToDimensionIncludingPadding(num_indices_dim - 1);
    DivModFast num_pieces_per_batch_fast(num_pieces / num_batches);
    int piece_size = input_shape->GetElementsFromDimensionIncludingPadding(
        batch_dim + indices_last_dim_size);
    int block_size             = 256;
    // step 1: calcalute each piece's offset first
    int64_t* piece_offsets     = static_cast<int64_t*>(temp_buffer);
    int64_t* input_strides_gpu = piece_offsets + num_pieces;
    int64_t* input_dims_gpu    = input_strides_gpu + num_input_dim;
    std::vector<int64_t> input_strides(indices_last_dim_size);
    std::vector<int64_t> input_dims(num_input_dim);
    // dimension is partitioned as batch--indices_last_dim_size--piece_size
    int64_t acc_strides = piece_size;
    for (int it = 0; it < indices_last_dim_size; ++it) {
        input_strides[indices_last_dim_size - 1 - it] = acc_strides;
        acc_strides *= input_shape->GetDim(batch_dim + indices_last_dim_size - 1 - it);
    }
    for (int it = 0; it < num_input_dim; ++it)
        input_dims[it] = input_shape->GetDim(it);
    cudaMemcpyAsync(input_strides_gpu, input_strides.data(), sizeof(int64_t) * indices_last_dim_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(input_dims_gpu, input_dims.data(), sizeof(int64_t) * num_input_dim, cudaMemcpyHostToDevice, stream);
    int cal_offset_grid = (num_pieces + block_size - 1) / block_size;
    switch (ppl::common::GetSizeOfDataType(indices_shape->GetDataType())) {
        case sizeof(int32_t): {
            ppl_cukernel_gather_nd_offset<<<cal_offset_grid, block_size, 0, stream>>>(num_pieces, num_pieces_per_batch_fast, batch_dim, input_dims_gpu, input_batch_stride, input_strides_gpu, indices_last_dim_size, (const int32_t*)indices, piece_offsets);
            break;
        }
        case sizeof(int64_t): {
            ppl_cukernel_gather_nd_offset<<<cal_offset_grid, block_size, 0, stream>>>(num_pieces, num_pieces_per_batch_fast, batch_dim, input_dims_gpu, input_batch_stride, input_strides_gpu, indices_last_dim_size, (const int64_t*)indices, piece_offsets);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }

    // step2: begiin gather elements
    int64_t num_elems    = output_shape->GetElementsIncludingPadding();
    int gather_grid_size = (num_elems + block_size - 1) / block_size;
    DivModFast piece_size_fast(piece_size);

#define SWITCH_CASE(TYPE)                                                                                                                                  \
    case sizeof(TYPE): {                                                                                                                                   \
        ppl_cukernel_gather_nd<<<gather_grid_size, block_size, 0, stream>>>(num_elems, piece_size_fast, piece_offsets, (const TYPE*)input, (TYPE*)output); \
        return ppl::common::RC_SUCCESS;                                                                                                                    \
    }

    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        SWITCH_CASE(int8_t);
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }

#undef SWITCH_CASE
}
