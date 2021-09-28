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

#include "cudakernel/memory/expand.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include <algorithm>

#define MAX_BLOCK_DIM_YZ (65536)

template <typename T>
__global__ void ppl_cukernel_expand(
    int64_t num_elems,
    int num_output_dim,
    GArray<DivModFast> output_strides_fast,
    GArray<int64_t> input_strides,
    const T* input,
    T* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int64_t input_offset = 0;
    int idx, remain = index;
    for (int it = 0; it < num_output_dim; ++it) {
        output_strides_fast[it].divmod(remain, idx, remain);
        input_offset += idx * input_strides[it];
    }
    output[index] = input[input_offset];
}

template <typename T>
__global__ void ppl_cukernel_expand_one_broadcast(
    int64_t inner_elems,
    int axis_width,
    const T* input,
    T* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= inner_elems) return;

    int axis_idx = blockIdx.y;
    int outer_idx = blockIdx.z;
    int64_t input_index = outer_idx * inner_elems + index;
    int64_t output_index = outer_idx * axis_width * inner_elems + axis_idx * inner_elems + index;
    output[output_index] = input[input_index];
}

template <typename T>
__global__ void ppl_cukernel_expand_last_dim(
    int64_t inner_dim,
    const T* input,
    T* output)
{
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    if (index >= inner_dim)
        return;
    int blk_idx = blockIdx.x;
    int out_index = blk_idx * inner_dim + index;

    output[out_index] = input[blk_idx];
}

static void ppl_pad_tensor_shape(const ppl::nn::TensorShape *tensor_shape0,
                          const ppl::nn::TensorShape *tensor_shape1,
                          ppl::nn::TensorShape *pad_tensor_shape0,
                          ppl::nn::TensorShape *pad_tensor_shape1) {
    int max_dims = std::max(tensor_shape0->GetDimCount(), tensor_shape1->GetDimCount());
    if (pad_tensor_shape0->GetDimCount() < pad_tensor_shape1->GetDimCount()) {
        pad_tensor_shape0->SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape0->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape0->SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape0->SetDim(i, tensor_shape0->GetDim(i - offset));
        }
    } else {
        pad_tensor_shape1->SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape1->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape1->SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape1->SetDim(i, tensor_shape1->GetDim(i - offset));
        }
    }
}

static int ppl_get_num_broadcast_dims(const ppl::nn::TensorShape *tensor_shape0,
                            const ppl::nn::TensorShape *tensor_shape1,
                            int &aixs) {
    ppl::nn::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::nn::TensorShape pad_tensor_shape1 = *tensor_shape1;
    ppl_pad_tensor_shape(tensor_shape0, tensor_shape1,
            &pad_tensor_shape0, &pad_tensor_shape1);
    int dim_count = pad_tensor_shape0.GetDimCount();
    int num_broadcast_dims = 0;
    for(int it = 0; it < dim_count; ++it) {
        if (pad_tensor_shape0.GetDim(it) != pad_tensor_shape1.GetDim(it))
            ++num_broadcast_dims;
    }
    if (num_broadcast_dims == 1) {
        for(int it = 0; it < dim_count; ++it) {
            if (pad_tensor_shape0.GetDim(it) != pad_tensor_shape1.GetDim(it))
                aixs = it;
        }
    }
    return num_broadcast_dims;
}

ppl::common::RetCode PPLCUDAExpandForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output)
{
    int dim_count = output_shape->GetDimCount();
    uint64_t num_elems  = output_shape->GetElementsIncludingPadding();
    if (num_elems == input_shape->GetElementsIncludingPadding()) { // no expand, just copy, as reshape
        cudaMemcpyAsync(output, input, ppl::common::GetSizeOfDataType(input_shape->GetDataType()) * num_elems, cudaMemcpyDeviceToDevice, stream);
        return ppl::common::RC_SUCCESS;
    }

    int axis = 0;
    int num_broadcast_dims = ppl_get_num_broadcast_dims(input_shape, output_shape, axis);
    int inner_elems = 1; int outer_elems = 1;
    int32_t axis_width = output_shape->GetDim(axis);
    for(int it = dim_count - 1; it > axis; inner_elems *= output_shape->GetDim(it), --it);
    for(int it = 0; it < axis; outer_elems *= output_shape->GetDim(it), ++it);

    if (num_broadcast_dims == 1 && (axis == 0 || axis == dim_count - 1)) {  //only one broadcast dim, normal case
        if (axis == 0) { //just for-copy
            int iters = output_shape->GetDim(axis);
            int input_size = input_shape->GetElementsIncludingPadding() * ppl::common::GetSizeOfDataType(input_shape->GetDataType());
            char *output_ptr = static_cast<char*>(output);
            for(int it = 0; it < iters; ++it) {
                cudaMemcpyAsync(output_ptr + it * input_size, input, input_size, cudaMemcpyDeviceToDevice, stream);
            }
        } else if (axis == dim_count - 1) {
            int inner_dim = output_shape->GetDim(axis);
            int outer_dim = 1;
            for(int it = 0; it < axis; outer_dim *= output_shape->GetDim(it), ++it);
            int block_size = 256;
            dim3 grid_size(outer_dim, (inner_dim + block_size - 1) / block_size, 1);
        #define SWITCH_CASE(TYPE)                                                                                        \
            case sizeof(TYPE): {                                                                                         \
                ppl_cukernel_expand_last_dim<<<grid_size, block_size, 0, stream>>>(                                     \
                    inner_dim, (const TYPE *)input, (TYPE *)output);                                                    \
                return ppl::common::RC_SUCCESS;                                                                          \
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
        return ppl::common::RC_SUCCESS;
    } else if (num_broadcast_dims == 1 && (axis_width < MAX_BLOCK_DIM_YZ) && (outer_elems < MAX_BLOCK_DIM_YZ)) {
        if (ppl::common::GetSizeOfDataType(input_shape->GetDataType()) == 4 && !(inner_elems & 0xf)) {
            int block_size = 256;
            dim3 grid_size(((inner_elems >> 2) + block_size - 1) / block_size, axis_width, outer_elems); 
            ppl_cukernel_expand_one_broadcast<<<grid_size, block_size, 0, stream>>>(inner_elems >> 2,
                        axis_width, (const float4*)input, (float4*)output);
        } else {
            int block_size = 256;
            dim3 grid_size((inner_elems + block_size - 1) / block_size, axis_width, outer_elems); 

        #define SWITCH_CASE(TYPE)                                                                                        \
            case sizeof(TYPE): {                                                                                         \
            ppl_cukernel_expand_one_broadcast<<<grid_size, block_size, 0, stream>>>(inner_elems, \
                        axis_width, (const TYPE*)input, (TYPE*)output); \
                return ppl::common::RC_SUCCESS;                                                                          \
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
        return ppl::common::RC_SUCCESS;
    } else {
        uint32_t num_output_dim = output_shape->GetDimCount();
        GArray<DivModFast> output_strides_fast(num_output_dim);
        GArray<int64_t> input_strides(num_output_dim);

        ppl::nn::TensorShape pad_input_shape = *input_shape;
        if (pad_input_shape.GetDimCount() < num_output_dim) {
            pad_input_shape.SetDimCount(num_output_dim);
            // pad 1 to shape_min_pad's higher dim
            uint32_t offset = num_output_dim - input_shape->GetDimCount();
            for (uint32_t i = 0; i < offset; i++) {
                pad_input_shape.SetDim(i, 1);
            }
            for (uint32_t i = offset; i < num_output_dim; i++) {
                pad_input_shape.SetDim(i, input_shape->GetDim(i - offset));
            }
        }

        int64_t acc_output_stride = 1;
        int64_t acc_input_stride  = 1;
        for (int it = num_output_dim - 1; it >= 0; --it) {
            if (pad_input_shape.GetDim(it) == 1) {
                input_strides[it] = 0;
            } else {
                input_strides[it] = acc_input_stride;
            }
            output_strides_fast[it] = DivModFast(acc_output_stride);
            acc_input_stride *= pad_input_shape.GetDim(it);
            acc_output_stride *= output_shape->GetDim(it);
        }

        int block_size = 256;
        int grid_size  = (num_elems + block_size - 1) / block_size;

    #define SWITCH_CASE(TYPE)                                                                                        \
        case sizeof(TYPE): {                                                                                         \
            ppl_cukernel_expand<<<grid_size, block_size, 0, stream>>>(                                              \
                num_elems, num_output_dim, output_strides_fast, input_strides, (const TYPE *)input, (TYPE *)output); \
            return ppl::common::RC_SUCCESS;                                                                          \
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
}
