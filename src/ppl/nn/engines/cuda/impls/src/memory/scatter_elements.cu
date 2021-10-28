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

#include "cudakernel/memory/scatter_elements.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include <assert.h>

__device__ __inline__ int get_indices_val(
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
__global__ void ppl_cukernel_scatter_elements(
    int64_t num_updates,
    int num_updates_dim,
    GArray<DivModFast> updates_strides_fast,
    int axis,
    int input_axis_width,
    GArray<int64_t> input_strides,
    const T* updates,
    const T* input,
    T* output,
    int indices_element_size,
    const void* indices)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_updates)
        return;
    T update_val    = updates[index];
    int indices_val = get_indices_val(indices_element_size, index, indices);
    if (indices_val < 0)
        indices_val += input_axis_width;
    assert(indices_val >= 0 && indices_val < input_axis_width);

    int64_t output_offset = 0;
    int idx, remain = index;
    for (int it = 0; it < num_updates_dim; ++it) {
        updates_strides_fast[it].divmod(remain, idx, remain);
        if (it == axis) {
            idx = indices_val;
        }
        output_offset += idx * input_strides[it];
    }
    output[output_offset] = update_val;
}

ppl::common::RetCode PPLCUDAScatterElementsForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* indices_shape,
    const void* indices,
    const ppl::nn::TensorShape* updates_shape,
    const void* updates,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    int axis)
{
    int64_t num_elems = output_shape->GetElementsIncludingPadding();
    cudaMemcpyAsync(output, input, ppl::common::GetSizeOfDataType(input_shape->GetDataType()) * num_elems, cudaMemcpyDeviceToDevice, stream);

    int64_t num_updates = updates_shape->GetElementsIncludingPadding();
    int num_updates_dim = updates_shape->GetDimCount();
    GArray<DivModFast> updates_strides_fast(num_updates_dim);
    GArray<int64_t> input_strides(num_updates_dim);

    int64_t acc_updates_stride = 1;
    int64_t acc_input_stride   = 1;
    for (int it = num_updates_dim - 1; it >= 0; --it) {
        input_strides[it]        = acc_input_stride;
        updates_strides_fast[it] = DivModFast(acc_updates_stride);
        acc_input_stride *= input_shape->GetDim(it);
        acc_updates_stride *= updates_shape->GetDim(it);
    }

    int block_size           = 256;
    int grid_size            = (num_updates + block_size - 1) / block_size;
    int indices_element_size = ppl::common::GetSizeOfDataType(indices_shape->GetDataType());

    int input_axis_width = input_shape->GetDim(axis);

#define SWITCH_CASE(TYPE)                                                                                                                                                                                                                                                    \
    case sizeof(TYPE): {                                                                                                                                                                                                                                                     \
        ppl_cukernel_scatter_elements<<<grid_size, block_size, 0, stream>>>(num_updates, num_updates_dim, updates_strides_fast, axis, input_axis_width, input_strides, (const TYPE*)updates, (const TYPE*)input, (TYPE*)output, indices_element_size, (const void*)indices); \
        return ppl::common::RC_SUCCESS;                                                                                                                                                                                                                                      \
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
