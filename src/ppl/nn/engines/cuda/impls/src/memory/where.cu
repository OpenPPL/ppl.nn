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

#include "cudakernel/memory/where.h"
#include "cudakernel/common/memory_utils.h"
#include "cudakernel/common/divmod_fast.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

enum BroadcastType {
    NoBroadcast      = 0,
    SimpleBroadcast  = 1,
    ComplexBroadcast = 2
};

template <typename T>
__global__ void ppl_cukernel_where_no_broadcast(
    int64_t num_elems,
    const bool* condition,
    const T* input_x,
    const T* input_y,
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    T out_val     = condition[index] ? input_x[index] : input_y[index];
    output[index] = out_val;
}

template <typename T>
__global__ void ppl_cukernel_where_vec_no_broadcast(
    int64_t num_elems,
    int num_vec_elems,
    const bool* condition,
    const T* input_x,
    const T* input_y,
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index *= num_vec_elems;
    if (index >= num_elems)
        return;

#pragma unroll
    for (int it = 0; it < num_vec_elems; ++it) {
        int acc_index     = index + it;
        T out_val         = condition[acc_index] ? input_x[acc_index] : input_y[acc_index];
        output[acc_index] = out_val;
    }
}

ppl::common::RetCode WhereForwardImpNoBroadcast(
    cudaStream_t stream,
    const ppl::nn::TensorShape* condition_shape,
    const bool* condition,
    const ppl::nn::TensorShape* input_x_shape,
    const void* input_x,
    const ppl::nn::TensorShape* input_y_shape,
    const void* input_y,
    const ppl::nn::TensorShape* output_shape,
    void* output)
{
    int64_t num_elems = output_shape->GetElementsIncludingPadding();

    constexpr int num_vec_elems = 4;
    bool vectorize              = (num_elems % num_vec_elems == 0);
    if (vectorize) {
        int block_size = 256;
        int grid_size  = ((num_elems / num_vec_elems) + block_size - 1) / block_size;

#define SWITCH_CASE(TYPE)                                                                                                 \
    case sizeof(TYPE): {                                                                                                  \
        ppl_cukernel_where_vec_no_broadcast<<<grid_size, block_size, 0, stream>>>(                                        \
            num_elems, num_vec_elems, (const bool*)condition, (const TYPE*)input_x, (const TYPE*)input_y, (TYPE*)output); \
        return ppl::common::RC_SUCCESS;                                                                                   \
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
        int block_size = 256;
        int grid_size  = (num_elems + block_size - 1) / block_size;

#define SWITCH_CASE(TYPE)                                                                                  \
    case sizeof(TYPE): {                                                                                   \
        ppl_cukernel_where_no_broadcast<<<grid_size, block_size, 0, stream>>>(                             \
            num_elems, (const bool*)condition, (const TYPE*)input_x, (const TYPE*)input_y, (TYPE*)output); \
        return ppl::common::RC_SUCCESS;                                                                    \
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
}

template <typename T, BroadcastType C_BType, BroadcastType X_BType, BroadcastType Y_BType>
__global__ void ppl_cukernel_where_simple_broadcast(
    int64_t num_elems,
    const bool* condition,
    const T* input_x,
    const T* input_y,
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int64_t c_index = (C_BType == BroadcastType::NoBroadcast) ? index : 0;
    int64_t x_index = (X_BType == BroadcastType::NoBroadcast) ? index : 0;
    int64_t y_index = (Y_BType == BroadcastType::NoBroadcast) ? index : 0;
    T out_val       = condition[c_index] ? input_x[x_index] : input_y[y_index];
    output[index]   = out_val;
}

template <BroadcastType C_BType, BroadcastType X_BType, BroadcastType Y_BType>
ppl::common::RetCode WhereForwardImpSimpleBroadcast(
    cudaStream_t stream,
    const ppl::nn::TensorShape* condition_shape,
    const bool* condition,
    const ppl::nn::TensorShape* input_x_shape,
    const void* input_x,
    const ppl::nn::TensorShape* input_y_shape,
    const void* input_y,
    const ppl::nn::TensorShape* output_shape,
    void* output)
{
    int64_t num_elems = output_shape->GetElementsIncludingPadding();

    int block_size = 256;
    int grid_size  = (num_elems + block_size - 1) / block_size;

#define SWITCH_CASE(TYPE)                                                                                      \
    case sizeof(TYPE): {                                                                                       \
        ppl_cukernel_where_simple_broadcast<TYPE, C_BType, X_BType, Y_BType>                                   \
            <<<grid_size, block_size, 0, stream>>>(                                                            \
                num_elems, (const bool*)condition, (const TYPE*)input_x, (const TYPE*)input_y, (TYPE*)output); \
        return ppl::common::RC_SUCCESS;                                                                        \
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

template <typename T, BroadcastType C_BType, BroadcastType X_BType, BroadcastType Y_BType>
__global__ void ppl_cukernel_where_complex_broadcast(
    int64_t num_elems,
    int num_dims,
    GArray<int64_t> condition_strides,
    const bool* condition,
    GArray<int64_t> input_x_strides,
    const T* input_x,
    GArray<int64_t> input_y_strides,
    const T* input_y,
    GArray<DivModFast> output_strides_fast,
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int64_t c_index = (C_BType == BroadcastType::NoBroadcast) ? index : 0;
    int64_t x_index = (X_BType == BroadcastType::NoBroadcast) ? index : 0;
    int64_t y_index = (Y_BType == BroadcastType::NoBroadcast) ? index : 0;

    int idx, remain = index;
    for (int it = 0; it < num_dims; ++it) {
        output_strides_fast[it].divmod(remain, idx, remain);
        if (C_BType == BroadcastType::ComplexBroadcast) {
            c_index += idx * condition_strides[it];
        }
        if (X_BType == BroadcastType::ComplexBroadcast) {
            x_index += idx * input_x_strides[it];
        }
        if (Y_BType == BroadcastType::ComplexBroadcast) {
            y_index += idx * input_y_strides[it];
        }
    }
    T out_val     = condition[c_index] ? input_x[x_index] : input_y[y_index];
    output[index] = out_val;
}

template <BroadcastType C_BType, BroadcastType X_BType, BroadcastType Y_BType>
ppl::common::RetCode WhereForwardImpComplexBroadcast(
    cudaStream_t stream,
    GArray<int64_t> condition_strides,
    const bool* condition,
    GArray<int64_t> input_x_strides,
    const void* input_x,
    GArray<int64_t> input_y_strides,
    const void* input_y,
    GArray<DivModFast> output_strides_fast,
    const ppl::nn::TensorShape* output_shape,
    void* output)
{
    int64_t num_elems = output_shape->GetElementsIncludingPadding();
    int num_dims      = output_shape->GetDimCount();

    int block_size = 256;
    int grid_size  = (num_elems + block_size - 1) / block_size;

#define SWITCH_CASE(TYPE)                                                                                                                                                                          \
    case sizeof(TYPE): {                                                                                                                                                                           \
        ppl_cukernel_where_complex_broadcast<TYPE, C_BType, X_BType, Y_BType>                                                                                                                      \
            <<<grid_size, block_size, 0, stream>>>(                                                                                                                                                \
                num_elems, num_dims, condition_strides, (const bool*)condition, input_x_strides, (const TYPE*)input_x, input_y_strides, (const TYPE*)input_y, output_strides_fast, (TYPE*)output); \
        return ppl::common::RC_SUCCESS;                                                                                                                                                            \
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

BroadcastType GetBroadcastType(int64_t test_num, int64_t ref_num)
{
    if (test_num == ref_num) { // no broadcast
        return BroadcastType::NoBroadcast;
    } else if (test_num == 1) { // scalar
        return BroadcastType::SimpleBroadcast;
    } else { // need recompute index
        return BroadcastType::ComplexBroadcast;
    }
}

ppl::common::RetCode PPLCUDAWhereForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* condition_shape,
    const bool* condition,
    const ppl::nn::TensorShape* input_x_shape,
    const void* input_x,
    const ppl::nn::TensorShape* input_y_shape,
    const void* input_y,
    const ppl::nn::TensorShape* output_shape,
    void* output)
{
    int64_t num_condition_elems = condition_shape->GetElementsIncludingPadding();
    int64_t num_input_x_elems   = input_x_shape->GetElementsIncludingPadding();
    int64_t num_input_y_elems   = input_y_shape->GetElementsIncludingPadding();
    int64_t num_elems           = output_shape->GetElementsIncludingPadding();
    int64_t num_dims            = output_shape->GetDimCount();

    BroadcastType condition_btype = GetBroadcastType(num_condition_elems, num_elems);
    BroadcastType input_x_btype   = GetBroadcastType(num_input_x_elems, num_elems);
    BroadcastType input_y_btype   = GetBroadcastType(num_input_y_elems, num_elems);
    if (condition_btype == BroadcastType::NoBroadcast &&
        input_x_btype == BroadcastType::NoBroadcast &&
        input_y_btype == BroadcastType::NoBroadcast) { // no broadcast
        return WhereForwardImpNoBroadcast(stream, condition_shape, condition, input_x_shape, input_x, input_y_shape, input_y, output_shape, output);
    } else if (condition_btype != BroadcastType::ComplexBroadcast &&
               input_x_btype != BroadcastType::ComplexBroadcast &&
               input_y_btype != BroadcastType::ComplexBroadcast) { // simple broadcast

#define HANDLE_SIMPLE_BROADCAST_Y(condition_btype, input_x_btype, input_y_btype)                                                                                                             \
    switch (input_y_btype) {                                                                                                                                                                 \
        case BroadcastType::NoBroadcast: {                                                                                                                                                   \
            return WhereForwardImpSimpleBroadcast<condition_btype,                                                                                                                           \
                                                  input_x_btype,                                                                                                                             \
                                                  BroadcastType::NoBroadcast>(stream, condition_shape, condition, input_x_shape, input_x, input_y_shape, input_y, output_shape, output);     \
            break;                                                                                                                                                                           \
        }                                                                                                                                                                                    \
        case BroadcastType::SimpleBroadcast: {                                                                                                                                               \
            return WhereForwardImpSimpleBroadcast<condition_btype,                                                                                                                           \
                                                  input_x_btype,                                                                                                                             \
                                                  BroadcastType::SimpleBroadcast>(stream, condition_shape, condition, input_x_shape, input_x, input_y_shape, input_y, output_shape, output); \
            break;                                                                                                                                                                           \
        }                                                                                                                                                                                    \
        default:                                                                                                                                                                             \
            return ppl::common::RC_UNSUPPORTED;                                                                                                                                              \
    }

#define HANDLE_SIMPLE_BROADCAST_XY(condition_btype, input_x_btype, input_y_btype)                     \
    switch (input_x_btype) {                                                                          \
        case BroadcastType::NoBroadcast: {                                                            \
            HANDLE_SIMPLE_BROADCAST_Y(condition_btype, BroadcastType::NoBroadcast, input_y_btype)     \
            break;                                                                                    \
        }                                                                                             \
        case BroadcastType::SimpleBroadcast: {                                                        \
            HANDLE_SIMPLE_BROADCAST_Y(condition_btype, BroadcastType::SimpleBroadcast, input_y_btype) \
            break;                                                                                    \
        }                                                                                             \
        default:                                                                                      \
            return ppl::common::RC_UNSUPPORTED;                                                       \
    }

        switch (condition_btype) {
            case BroadcastType::NoBroadcast: {
                HANDLE_SIMPLE_BROADCAST_XY(BroadcastType::NoBroadcast, input_x_btype, input_y_btype)
                break;
            }
            case BroadcastType::SimpleBroadcast: {
                HANDLE_SIMPLE_BROADCAST_XY(BroadcastType::SimpleBroadcast, input_x_btype, input_y_btype)
                break;
            }
            default:
                return ppl::common::RC_UNSUPPORTED;
        }

#undef HANDLE_SIMPLE_BROADCAST_XY
#undef HANDLE_SIMPLE_BROADCAST_Y
    } else {
        GArray<DivModFast> output_strides_fast(num_dims);
        GArray<int64_t> condition_strides(num_dims);
        GArray<int64_t> input_x_strides(num_dims);
        GArray<int64_t> input_y_strides(num_dims);

        ppl::nn::TensorShape temp_condition_shape = *condition_shape;
        if (temp_condition_shape.GetDimCount() < num_dims) {
            int old_dims = temp_condition_shape.GetDimCount();
            temp_condition_shape.SetDimCount(num_dims);
            for (int i = num_dims - 1; i >= 0; --i) {
                temp_condition_shape.SetDim(i, num_dims - i > old_dims ? 1 : temp_condition_shape.GetDim(old_dims - num_dims + i));
            }
        }

        ppl::nn::TensorShape temp_input_x_shape = *input_x_shape;
        if (temp_input_x_shape.GetDimCount() < num_dims) {
            int old_dims = temp_input_x_shape.GetDimCount();
            temp_input_x_shape.SetDimCount(num_dims);
            for (int i = num_dims - 1; i >= 0; --i) {
                temp_input_x_shape.SetDim(i, num_dims - i > old_dims ? 1 : temp_input_x_shape.GetDim(old_dims - num_dims + i));
            }
        }

        ppl::nn::TensorShape temp_input_y_shape = *input_y_shape;
        if (temp_input_x_shape.GetDimCount() < num_dims) {
            int old_dims = temp_input_y_shape.GetDimCount();
            temp_input_y_shape.SetDimCount(num_dims);
            for (int i = num_dims - 1; i >= 0; --i) {
                temp_input_y_shape.SetDim(i, num_dims - i > old_dims ? 1 : temp_input_y_shape.GetDim(old_dims - num_dims + i));
            }
        }

        int64_t acc_output_stride    = 1;
        int64_t acc_condition_stride = 1;
        int64_t acc_input_x_stride   = 1;
        int64_t acc_input_y_stride   = 1;
        for (int it = num_dims - 1; it >= 0; --it) {
            if (temp_condition_shape.GetDim(it) == 1) {
                condition_strides[it] = 0;
            } else {
                condition_strides[it] = acc_condition_stride;
            }
            if (temp_input_x_shape.GetDim(it) == 1) {
                input_x_strides[it] = 0;
            } else {
                input_x_strides[it] = acc_input_x_stride;
            }
            if (temp_input_y_shape.GetDim(it) == 1) {
                input_y_strides[it] = 0;
            } else {
                input_y_strides[it] = acc_input_y_stride;
            }
            output_strides_fast[it] = DivModFast(acc_output_stride);
            acc_condition_stride *= temp_condition_shape.GetDim(it);
            acc_input_x_stride *= temp_input_x_shape.GetDim(it);
            acc_input_y_stride *= temp_input_y_shape.GetDim(it);
            acc_output_stride *= output_shape->GetDim(it);
        }

#define HANDLE_COMPLEX_BROADCAST_Y(condition_btype, input_x_btype, input_y_btype)                                                                                                                                         \
    switch (input_y_btype) {                                                                                                                                                                                              \
        case BroadcastType::NoBroadcast: {                                                                                                                                                                                \
            return WhereForwardImpComplexBroadcast<condition_btype,                                                                                                                                                       \
                                                   input_x_btype,                                                                                                                                                         \
                                                   BroadcastType::NoBroadcast>(stream, condition_strides, condition, input_x_strides, input_x, input_y_strides, input_y, output_strides_fast, output_shape, output);      \
            break;                                                                                                                                                                                                        \
        }                                                                                                                                                                                                                 \
        case BroadcastType::SimpleBroadcast: {                                                                                                                                                                            \
            return WhereForwardImpComplexBroadcast<condition_btype,                                                                                                                                                       \
                                                   input_x_btype,                                                                                                                                                         \
                                                   BroadcastType::SimpleBroadcast>(stream, condition_strides, condition, input_x_strides, input_x, input_y_strides, input_y, output_strides_fast, output_shape, output);  \
            break;                                                                                                                                                                                                        \
        }                                                                                                                                                                                                                 \
        case BroadcastType::ComplexBroadcast: {                                                                                                                                                                           \
            return WhereForwardImpComplexBroadcast<condition_btype,                                                                                                                                                       \
                                                   input_x_btype,                                                                                                                                                         \
                                                   BroadcastType::ComplexBroadcast>(stream, condition_strides, condition, input_x_strides, input_x, input_y_strides, input_y, output_strides_fast, output_shape, output); \
            break;                                                                                                                                                                                                        \
        }                                                                                                                                                                                                                 \
    }

#define HANDLE_COMPLEX_BROADCAST_XY(condition_btype, input_x_btype, input_y_btype)                      \
    switch (input_x_btype) {                                                                            \
        case BroadcastType::NoBroadcast: {                                                              \
            HANDLE_COMPLEX_BROADCAST_Y(condition_btype, BroadcastType::NoBroadcast, input_y_btype)      \
            break;                                                                                      \
        }                                                                                               \
        case BroadcastType::SimpleBroadcast: {                                                          \
            HANDLE_COMPLEX_BROADCAST_Y(condition_btype, BroadcastType::SimpleBroadcast, input_y_btype)  \
            break;                                                                                      \
        }                                                                                               \
        case BroadcastType::ComplexBroadcast: {                                                         \
            HANDLE_COMPLEX_BROADCAST_Y(condition_btype, BroadcastType::ComplexBroadcast, input_y_btype) \
            break;                                                                                      \
        }                                                                                               \
    }

        switch (condition_btype) {
            case BroadcastType::NoBroadcast: {
                HANDLE_COMPLEX_BROADCAST_XY(BroadcastType::NoBroadcast, input_x_btype, input_y_btype)
                break;
            }
            case BroadcastType::SimpleBroadcast: {
                HANDLE_COMPLEX_BROADCAST_XY(BroadcastType::SimpleBroadcast, input_x_btype, input_y_btype)
                break;
            }
            case BroadcastType::ComplexBroadcast: {
                HANDLE_COMPLEX_BROADCAST_XY(BroadcastType::ComplexBroadcast, input_x_btype, input_y_btype)
                break;
            };
        }

#undef HANDLE_COMPLEX_BROADCAST_XY
#undef HANDLE_COMPLEX_BROADCAST_Y
    }
    return ppl::common::RC_UNSUPPORTED;
}
