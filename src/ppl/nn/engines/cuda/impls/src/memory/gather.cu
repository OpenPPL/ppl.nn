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

#include "cudakernel/memory/gather.h"
#include "cudakernel/common/divmod_fast.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include <memory>

__host__ __device__ __inline__ int get_indices_val(
    int indices_element_size,
    int offset,
    const void* indices)
{
    int res = 0;
    switch (indices_element_size) {
        case sizeof(int32_t):
            res = static_cast<const int32_t*>(indices)[offset];
            break;
        case sizeof(int64_t):
            res = static_cast<const int64_t*>(indices)[offset];
            break;
        default:
            break;
    }
    return res;
}

template <typename T>
__global__ void ppl_cukernel_gather(
    int64_t num_elems,
    DivModFast output_outer_block_fast,
    int input_axis_size,
    DivModFast output_inner_block_fast,
    const T* input,
    T* output,
    int indices_element_size,
    const void* indices)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int outer_idx, block_offset;
    output_outer_block_fast.divmod(index, outer_idx, block_offset);
    int indices_offset, inner_idx;
    output_inner_block_fast.divmod(block_offset, indices_offset, inner_idx);
    int64_t indices_idx = get_indices_val(indices_element_size, indices_offset, indices);
    // -d means distance from last dimension
    indices_idx         = indices_idx < 0 ? indices_idx + input_axis_size : indices_idx;
    if (indices_idx < 0 || indices_idx >= input_axis_size) {
        output[index] = 0;
        return;
    }
    int64_t input_idx = (outer_idx * input_axis_size + indices_idx) *
                            output_inner_block_fast.d_ +
                        inner_idx;
    output[index] = input[input_idx];
}

ppl::common::RetCode PPLCUDAGatherForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* indices_shape,
    const void* indices,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    int axis)
{
    int indices_element_size = ppl::common::GetSizeOfDataType(indices_shape->GetDataType());
    // special case, need further evaluement (performance is not usually better)
    if (axis == 0 && indices_shape->GetDimCount() == 1 && indices_shape->GetDim(0) == 1) {
        int indices_data_size = indices_shape->GetBytesIncludingPadding();
        std::unique_ptr<char[]> indices_data(new char[indices_data_size]);
        cudaMemcpy(indices_data.get(), indices, indices_data_size, cudaMemcpyDeviceToHost);
        int inner_size   = input_shape->GetBytesIncludingPadding() / input_shape->GetDim(0);
        int input_offset = get_indices_val(indices_element_size, 0, indices_data.get());
        cudaMemcpy(output, static_cast<const char*>(input) + input_offset * inner_size, output_shape->GetBytesIncludingPadding(), cudaMemcpyDeviceToDevice);
        return ppl::common::RC_SUCCESS;
    }
    int64_t num_elems      = output_shape->GetElementsIncludingPadding();
    int block_size         = 256;
    int grid_size          = (num_elems + block_size - 1) / block_size;
    // output dimension can be partitioned as outer--indices--inner. (before axis, axis, after axis)
    int output_inner_block = input_shape->GetElementsFromDimensionIncludingPadding(axis + 1);
    int input_axis_size    = input_shape->GetDim(axis);
    int indices_block_size = indices_shape->GetElementsIncludingPadding();
    int output_outer_block = indices_block_size * output_inner_block;

    DivModFast output_outer_block_fast(output_outer_block);
    DivModFast output_inner_block_fast(output_inner_block);

#define SWITCH_CASE(TYPE)                                                                                                                                                                                                       \
    case sizeof(TYPE): {                                                                                                                                                                                                        \
        ppl_cukernel_gather<<<grid_size, block_size, 0, stream>>>(num_elems, output_outer_block_fast, input_axis_size, output_inner_block_fast, (const TYPE*)input, (TYPE*)output, indices_element_size, (const void*)indices); \
        return ppl::common::RC_SUCCESS;                                                                                                                                                                                         \
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
