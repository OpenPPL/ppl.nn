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

#include "cudakernel/memory/concat.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "cudakernel/common/common.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"

#define NHWC8_ALIGNED_AXIS (8)

template <typename T>
__global__ void ppl_cukernel_concat(
    int64_t num_elems,
    const T* inputs,
    int64_t concat_size,
    int64_t top_axis_width,
    DivModFast num_elems_inner_fast,
    int axis_offset,
    T* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int outer_idx, inner_idx;
        num_elems_inner_fast.divmod(i, outer_idx, inner_idx);
        int64_t top_idx = inner_idx +
                          (outer_idx * top_axis_width + axis_offset) * concat_size;
        output[top_idx] = inputs[i];
    }
}

template <typename T1, typename T2>
__global__ void __launch_bounds__(256) ppl_cukernel_concat_two_inputs(
    int64_t num_elems,
    const T1* input0,
    const T1* input1,
    T2* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int tid = threadIdx.x;
        __shared__ T1 buffer[2 * 256];
        buffer[2 * tid]     = input0[i];
        buffer[2 * tid + 1] = input1[i];
        T2* buffer_ptr      = reinterpret_cast<T2*>(buffer);
        output[i]           = buffer_ptr[tid];
    }
}

template <typename T>
__global__ void ppl_cukernel_concat_nhwc(
    int64_t num_elems,
    int num_dims,
    int nhwc_axis,
    int axis_offset,
    GArray<DivModFast> input_strides_fast,
    GArray<int64_t> input_padded_strides,
    GArray<int64_t> output_padded_strides,
    const T* input,
    T* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int64_t output_offset = 0, input_offset = 0;
    int idx, remain                         = index;
    for (int it = 0; it < num_dims; ++it) {
        input_strides_fast[it].divmod(remain, idx, remain);
        input_offset += idx * input_padded_strides[it];
        idx = (it == nhwc_axis) ? idx + axis_offset : idx;
        output_offset += idx * output_padded_strides[it];
    }
    output[output_offset] = input[input_offset];
}

template <typename T1, typename T2>
__global__ void __launch_bounds__(256) ppl_cukernel_concat_nhwc_two_inputs(
    int64_t num_elems,
    int inner_dims,
    int axis_width0,
    int axis_width1,
    const T1* input0,
    const T1* input1,
    T2* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int inner_idx = i % inner_dims;
        int outer_idx = i / inner_dims;
        if (inner_idx >= axis_width0) {
            int input_offset = outer_idx * axis_width1 + (inner_idx - axis_width0);
            output[i]        = input1[input_offset];
        } else {
            int input_offset = outer_idx * axis_width0 + inner_idx;
            output[i]        = input0[input_offset];
        }
    }
}

template <typename T1, typename T2>
__global__ void __launch_bounds__(256) ppl_cukernel_concat_nhwc_two_inputs(
    int64_t num_elems,
    int inner_dims,
    int pad_inner_dims,
    int axis_width0,
    int pad_axis_width0,
    int axis_width1,
    int pad_axis_width1,
    const T1* input0,
    const T1* input1,
    T2* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int inner_idx = i % pad_inner_dims;
        int outer_idx = i / pad_inner_dims;
        // int output_offset = outer_idx * pad_inner_dims + inner_idx;
        if (inner_idx >= axis_width0) {
            int axis_offset  = inner_idx - axis_width0;
            int input_offset = outer_idx * pad_axis_width1 + axis_offset;
            output[i]        = axis_offset >= axis_width1 ? 0 : input1[input_offset];
        } else {
            int axis_offset  = inner_idx;
            int input_offset = outer_idx * pad_axis_width0 + axis_offset;
            output[i]        = axis_offset >= axis_width0 ? 0 : input0[input_offset];
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_concat_nhwc_nopadding(
    int64_t num_elems,
    const T* inputs,
    int64_t concat_size,
    int64_t top_axis_width,
    DivModFast num_elems_inner_fast,
    int axis_offset,
    T* output)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elems;
         i += (int64_t)blockDim.x * gridDim.x) {
        int outer_idx, inner_idx;
        num_elems_inner_fast.divmod(i, outer_idx, inner_idx);
        int64_t top_idx = inner_idx + (outer_idx * top_axis_width + axis_offset);
        output[top_idx] = inputs[i];
    }
}

bool IsConcatNoPadding(
    int axis,
    int num_inputs,
    int* input_dims[],
    int* input_padded_dims[],
    ppl::nn::TensorShape* output_shape,
    int mask)
{
    if (output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NHWC8 || axis != 1)
        return false;
    for (int i = 0; i < num_inputs; i++) {
        if (input_padded_dims[i][axis] - input_dims[i][axis] != 0)
            return false;
    }
    return true;
}

ppl::common::RetCode PPLCUDAConcatNoPaddingForwardImp(
    cudaStream_t stream,
    int axis,
    int num_inputs,
    int* input_dims[],
    int* input_padded_dims[],
    const void* inputs[],
    ppl::nn::TensorShape* output_shape,
    void* output,
    int mask)
{
    int64_t num_elems         = output_shape->GetElementsIncludingPadding() / output_shape->GetDim(axis);
    int64_t output_axis_width = output_shape->GetDim(axis);
    int64_t axis_offset       = 0;
#define SWITCH_CASE(TYPE)                                                                                            \
    case sizeof(TYPE): {                                                                                             \
        for (int j = 0; j < num_inputs; ++j) {                                                                       \
            int input_axis_width = input_dims[j][axis];                                                              \
            int num_in_elems     = num_elems * input_axis_width;                                                     \
            if (!(mask & (1 << j))) {                                                                                \
                if (num_in_elems > 0) {                                                                              \
                    DivModFast num_elems_inner_fast = DivModFast(input_axis_width);                                  \
                    int block_size                  = 256;                                                           \
                    int grid_size                   = (num_in_elems + block_size - 1) / block_size;                  \
                    ppl_cukernel_concat_nhwc_nopadding<<<grid_size, block_size, 0, stream>>>(num_in_elems,           \
                                                                                             (const TYPE*)inputs[j], \
                                                                                             num_in_elems,           \
                                                                                             output_axis_width,      \
                                                                                             num_elems_inner_fast,   \
                                                                                             axis_offset,            \
                                                                                             (TYPE*)output);         \
                }                                                                                                    \
            }                                                                                                        \
            axis_offset += input_axis_width;                                                                         \
        }                                                                                                            \
        return ppl::common::RC_SUCCESS;                                                                              \
    }

    switch (ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
        SWITCH_CASE(int8_t);
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#undef SWITCH_CASE
}
ppl::common::RetCode PPLCUDAConcatForwardImp(
    cudaStream_t stream,
    int axis,
    int num_inputs,
    int* input_dims[],
    int* input_padded_dims[],
    const void* inputs[],
    ppl::nn::TensorShape* output_shape,
    void* output,
    int mask)
{
    if (IsConcatNoPadding(axis, num_inputs, input_dims, input_padded_dims, output_shape, mask)) {
        return PPLCUDAConcatNoPaddingForwardImp(stream, axis, num_inputs, input_dims, input_padded_dims, inputs, output_shape, output, mask);
    }
    int num_dims     = output_shape->GetDimCount();
    int output_elems = output_shape->GetElementsIncludingPadding();
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        if (num_inputs == 2 && axis == (num_dims - 1) && input_dims[0][axis] == 1 && input_dims[1][axis] == 1) {
            int num_elems = 1;
            for (int it = 0; it < num_dims; num_elems *= input_dims[0][it], ++it)
                ;

#define SWITCH_CASE(TYPE1, TYPE2)                                                                     \
    case sizeof(TYPE1): {                                                                             \
        int block_size = 256;                                                                         \
        int grid_size  = (num_elems + block_size - 1) / block_size;                                   \
        ppl_cukernel_concat_two_inputs<<<grid_size, block_size, 0, stream>>>(num_elems,               \
                                                                             (const TYPE1*)inputs[0], \
                                                                             (const TYPE1*)inputs[1], \
                                                                             (TYPE2*)output);         \
        return ppl::common::RC_SUCCESS;                                                               \
    }
            switch (ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
                SWITCH_CASE(int8_t, int16_t);
                SWITCH_CASE(int16_t, int32_t);
                SWITCH_CASE(int32_t, int64_t);
                SWITCH_CASE(int64_t, float4);
                default:
                    return ppl::common::RC_UNSUPPORTED;
            }
#undef SWITCH_CASE
        } else {
            int64_t concat_size = 1;
            int64_t num_concats = 1;
            for (int i = num_dims - 1; i > axis; --i)
                concat_size *= input_dims[0][i];
            for (int i = 0; i < axis; ++i)
                num_concats *= input_dims[0][i];
            int axis_offset       = 0;
            int output_axis_width = output_shape->GetDim(axis);

#define SWITCH_CASE(TYPE)                                                                             \
    case sizeof(TYPE): {                                                                              \
        for (int j = 0; j < num_inputs; ++j) {                                                        \
            int input_axis_width = input_dims[j][axis];                                               \
            if (!(mask & (1 << j))) {                                                                 \
                int64_t input_concat_size = input_axis_width * concat_size;                           \
                int64_t num_elems         = input_concat_size * num_concats;                          \
                if (num_elems > 0) {                                                                  \
                    DivModFast num_elems_inner_fast = DivModFast(input_concat_size);                  \
                    int block_size                  = 256;                                            \
                    int grid_size                   = (num_elems + block_size - 1) / block_size;      \
                    ppl_cukernel_concat<<<grid_size, block_size, 0, stream>>>(num_elems,              \
                                                                              (const TYPE*)inputs[j], \
                                                                              concat_size,            \
                                                                              output_axis_width,      \
                                                                              num_elems_inner_fast,   \
                                                                              axis_offset,            \
                                                                              (TYPE*)output);         \
                }                                                                                     \
            }                                                                                         \
            axis_offset += input_axis_width;                                                          \
        }                                                                                             \
        return ppl::common::RC_SUCCESS;                                                               \
    }

            switch (ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
                SWITCH_CASE(int8_t);
                SWITCH_CASE(int16_t);
                SWITCH_CASE(int32_t);
                SWITCH_CASE(int64_t);
                default:
                    return ppl::common::RC_UNSUPPORTED;
            }
#undef SWITCH_CASE
        }
    } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
        // nhwc, axis == 1 means last dim
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16 && num_inputs == 2 &&
            axis == 1) {
            if (!(input_dims[0][axis] & 0x7) && !(input_dims[1][axis] & 0x7)) {
                int block_size    = 256;
                int channel_shift = 3;
                int grid_size     = ((output_elems >> channel_shift) + block_size - 1) / block_size;
                int axis_width0   = input_dims[0][axis] >> channel_shift;
                int axis_width1   = input_dims[1][axis] >> channel_shift;
                int inner_dims    = axis_width0 + axis_width1;
                ppl_cukernel_concat_nhwc_two_inputs<<<grid_size, block_size, 0, stream>>>(output_elems >> channel_shift,
                                                                                          inner_dims,
                                                                                          axis_width0,
                                                                                          axis_width1,
                                                                                          (const float4*)inputs[0],
                                                                                          (const float4*)inputs[1],
                                                                                          (float4*)output);
            } else {
                int block_size      = 256;
                int grid_size       = (output_elems + block_size - 1) / block_size;
                int axis_width0     = input_dims[0][axis];
                int pad_axis_width0 = Align(axis_width0, NHWC8_ALIGNED_AXIS);
                int axis_width1     = input_dims[1][axis];
                int pad_axis_width1 = Align(axis_width1, NHWC8_ALIGNED_AXIS);
                int inner_dims      = axis_width0 + axis_width1;
                int pad_inner_dims  = Align(inner_dims, NHWC8_ALIGNED_AXIS);
                ppl_cukernel_concat_nhwc_two_inputs<<<grid_size, block_size, 0, stream>>>(output_elems,
                                                                                          inner_dims,
                                                                                          pad_inner_dims,
                                                                                          axis_width0,
                                                                                          pad_axis_width0,
                                                                                          axis_width1,
                                                                                          pad_axis_width1,
                                                                                          (const int16_t*)inputs[0],
                                                                                          (const int16_t*)inputs[1],
                                                                                          (int16_t*)output);
            }
            return ppl::common::RC_SUCCESS;
        }
        int axis_offset = 0;
        std::vector<int32_t> nhwc_output_padded_dims(num_dims);
        nhwc_output_padded_dims[num_dims - 1] = output_shape->GetDim(1) +
                                                output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
        int jump_step = 0;
        for (int it = 0; it < num_dims - 1; ++it) {
            if (it == 1)
                jump_step = 1;
            nhwc_output_padded_dims[it] = output_shape->GetDim(it + jump_step);
        }
        GArray<int64_t> output_padded_strides(num_dims);
        int64_t acc_output_stride = 1;
        for (int it = num_dims - 1; it >= 0; --it) {
            output_padded_strides[it] = acc_output_stride;
            acc_output_stride *= nhwc_output_padded_dims[it];
        }

#define SWITCH_CASE(TYPE)                                                                                                                                                 \
    case sizeof(TYPE): {                                                                                                                                                  \
        for (int j = 0; j < num_inputs; ++j) {                                                                                                                            \
            if (!(mask & (1 << j))) {                                                                                                                                     \
                int nhwc_axis = (axis == 1) ? num_dims - 1 : axis - 1;                                                                                                    \
                nhwc_axis     = (axis == 0) ? 0 : nhwc_axis;                                                                                                              \
                std::vector<int32_t> nhwc_input_dims(num_dims);                                                                                                           \
                std::vector<int32_t> nhwc_input_padded_dims(num_dims);                                                                                                    \
                nhwc_input_dims[num_dims - 1]        = input_dims[j][1];                                                                                                  \
                nhwc_input_padded_dims[num_dims - 1] = input_padded_dims[j][1];                                                                                           \
                jump_step                            = 0;                                                                                                                 \
                for (int it = 0; it < num_dims - 1; ++it) {                                                                                                               \
                    if (it == 1)                                                                                                                                          \
                        jump_step = 1;                                                                                                                                    \
                    nhwc_input_dims[it]        = input_dims[j][it + jump_step];                                                                                           \
                    nhwc_input_padded_dims[it] = input_padded_dims[j][it + jump_step];                                                                                    \
                }                                                                                                                                                         \
                GArray<DivModFast> input_strides_fast(num_dims);                                                                                                          \
                GArray<int64_t> input_padded_strides(num_dims);                                                                                                           \
                int64_t acc_input_stride = 1, acc_input_padded_stride = 1;                                                                                                \
                for (int it = num_dims - 1; it >= 0; --it) {                                                                                                              \
                    input_strides_fast[it]   = DivModFast(acc_input_stride);                                                                                              \
                    input_padded_strides[it] = acc_input_padded_stride;                                                                                                   \
                    acc_input_stride *= nhwc_input_dims[it];                                                                                                              \
                    acc_input_padded_stride *= nhwc_input_padded_dims[it];                                                                                                \
                }                                                                                                                                                         \
                int input_axis_width = nhwc_input_dims[nhwc_axis];                                                                                                        \
                int64_t num_elems    = 1;                                                                                                                                 \
                for (int it = 0; it < num_dims; ++it)                                                                                                                     \
                    num_elems *= nhwc_input_dims[it];                                                                                                                     \
                int block_size = 256;                                                                                                                                     \
                int grid_size  = (num_elems + block_size - 1) / block_size;                                                                                               \
                ppl_cukernel_concat_nhwc<<<grid_size, block_size, 0, stream>>>(                                                                                           \
                    num_elems, num_dims, nhwc_axis, axis_offset, input_strides_fast, input_padded_strides, output_padded_strides, (const TYPE*)inputs[j], (TYPE*)output); \
                axis_offset += input_axis_width;                                                                                                                          \
            }                                                                                                                                                             \
        }                                                                                                                                                                 \
        return ppl::common::RC_SUCCESS;                                                                                                                                   \
    }

        switch (ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
            SWITCH_CASE(int8_t);
            SWITCH_CASE(int16_t);
            SWITCH_CASE(int32_t);
            SWITCH_CASE(int64_t);
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
#undef SWITCH_CASE
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}
