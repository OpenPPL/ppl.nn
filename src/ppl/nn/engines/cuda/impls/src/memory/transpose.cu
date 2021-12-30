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

#include "cudakernel/memory/transpose.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "cudakernel/common/common.h"
#define DIM     32
#define MAX_DIM 65533

struct FastTransposeParam {
    int64_t n_outer  = 1;
    int64_t n_height = 1;
    int64_t n_width  = 1;
    int64_t n_inner  = 1;
    void reset()
    {
        n_outer  = 1;
        n_height = 1;
        n_width  = 1;
        n_inner  = 1;
    }
};

template <typename T>
__global__ void cuda_kernel_fast_trans(
    const T *input,
    FastTransposeParam param,
    T *output)
{
    __shared__ T share_val[DIM][DIM + 1];
    int64_t num = blockIdx.z;
    for (int n = num; n < param.n_outer; n += gridDim.z) {
        for (int t = blockIdx.y; t < DivUp(param.n_height, 32); t += gridDim.y) {
            int64_t idx_w = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t idx_h = t * blockDim.y + threadIdx.y;
            
            if (idx_w < param.n_width && idx_h < param.n_height) {
                int64_t offset                      = n * param.n_height * param.n_width + idx_h * param.n_width + idx_w;
                share_val[threadIdx.y][threadIdx.x] = input[offset];
            } else {
                share_val[threadIdx.y][threadIdx.x] = (T)0;
            }
            __syncthreads();
            idx_w = t * blockDim.y + threadIdx.x;
            idx_h = blockIdx.x * blockDim.x + threadIdx.y;
            if (idx_w < param.n_height && idx_h < param.n_width) {
                int64_t offset = n * param.n_height * param.n_width + idx_h * param.n_height + idx_w;
                output[offset] = share_val[threadIdx.x][threadIdx.y];
            }
        }
    }
}

bool FastTransposeSupport(
    FastTransposeParam *fast_param,
    const ppl::nn::TensorShape *input_shape,
    ppl::nn::common::TransposeParam param,
    const ppl::nn::TensorShape *output_shape)
{
    if (input_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY ||
        output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        return false;
    }
    int num_dims = input_shape->GetDimCount();
    for (int i = 0; i < num_dims; i++) {
        if (param.perm[i] == i) {
            fast_param->n_outer *= input_shape->GetDim(i);
            continue;
        } else {
            fast_param->n_height = input_shape->GetDim(i);
            for (int j = i + 1; j < num_dims; j++) {
                if (param.perm[j - 1] == j) {
                    fast_param->n_width *= input_shape->GetDim(j);
                } else {
                    return false;
                }
            }
            break;
        }
    }
    return true;
}

bool FastTransposeSupport2(
    FastTransposeParam *fast_param,
    const ppl::nn::TensorShape *input_shape,
    ppl::nn::common::TransposeParam param,
    const ppl::nn::TensorShape *output_shape)
{
    if (input_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY ||
        output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        return false;
    }
    int num_dims = input_shape->GetDimCount();
    for (int i = 0; i < num_dims; i++) {
        if (param.perm[i] == i) {
            fast_param->n_outer *= input_shape->GetDim(i);
            continue;
        } else {
            fast_param->n_width = input_shape->GetDim(num_dims - 1);
            fast_param->n_height = 1;
            for (int j = i; j < num_dims - 1; j++) {
                if (param.perm[j + 1] == j) {
                    fast_param->n_height *= input_shape->GetDim(j);
                } else {
                    return false;
                }
            }
            break;
        }
    }
    return true;
}
ppl::common::RetCode PPLCUDATransposeFastForwardImp(
    cudaStream_t stream,
    FastTransposeParam param,
    const ppl::nn::TensorShape *input_shape,
    const void *input,
    const ppl::nn::TensorShape *output_shape,
    void *output)
{
    dim3 dim_block(DIM, DIM, 1);
    int dimz = param.n_outer >= MAX_DIM ? MAX_DIM : param.n_outer;
    dim3 dim_grid(DivUp(param.n_width, DIM), DivUp(param.n_height, DIM), dimz);

#define SWITCH_CASE(TYPE)                                           \
    case sizeof(TYPE): {                                            \
        cuda_kernel_fast_trans<<<dim_grid, dim_block, 0, stream>>>( \
            (const TYPE *)input, param, (TYPE *)output);            \
        return ppl::common::RC_SUCCESS;                             \
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

template <typename T>
__global__ void cuda_kernel_middle_trans(
    const T *input,
    int64_t num_elems,
    FastTransposeParam param,
    T *output)
{
    int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_elems)
        return;
    int inner_idx  = tid % param.n_inner;
    int width_idx  = (tid / param.n_inner) % param.n_width;
    int height_idx = (tid / (param.n_inner * param.n_width)) % param.n_height;
    int outer_idx  = tid / (param.n_inner * param.n_width * param.n_height);

    int64_t offset = outer_idx * param.n_inner * param.n_width * param.n_height + height_idx * param.n_inner +
                     width_idx * param.n_height * param.n_inner + inner_idx;
    output[offset] = input[tid];
}

bool MiddleFastTransposeSupport(
    FastTransposeParam *fast_param,
    const ppl::nn::TensorShape *input_shape,
    ppl::nn::common::TransposeParam param,
    const ppl::nn::TensorShape *output_shape)
{
    if (input_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY ||
        output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        return false;
    }
    fast_param->reset();
    int num_dims    = input_shape->GetDimCount();
    int height_axis = 0;
    int width_axis  = num_dims - 1;
    for (int i = 0; i < num_dims && param.perm[i] == i; fast_param->n_outer *= input_shape->GetDim(i), height_axis = i + 1, i++)
        ;
    for (int i = num_dims - 1; i >= 0 && param.perm[i] == i; fast_param->n_inner *= input_shape->GetDim(i), width_axis = i - 1, i--)
        ;
    if (width_axis <= height_axis)
        return false;
    fast_param->n_height *= input_shape->GetDim(height_axis);
    fast_param->n_width *= input_shape->GetDim(width_axis);
    if (width_axis - height_axis != 1)
        return false;
    return true;
}

ppl::common::RetCode PPLCUDATransposeMiddleFastForwardImp(
    cudaStream_t stream,
    FastTransposeParam param,
    const ppl::nn::TensorShape *input_shape,
    const void *input,
    const ppl::nn::TensorShape *output_shape,
    void *output)
{
    const int block_size = 256;
    dim3 dim_block(block_size, 1, 1);
    int64_t num_elems = output_shape->GetElementsIncludingPadding();
    dim3 dim_grid(DivUp(num_elems, block_size), 1, 1);

#define SWITCH_CASE(TYPE)                                             \
    case sizeof(TYPE): {                                              \
        cuda_kernel_middle_trans<<<dim_grid, dim_block, 0, stream>>>( \
            (const TYPE *)input, num_elems, param, (TYPE *)output);   \
        return ppl::common::RC_SUCCESS;                               \
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

template <typename T>
__global__ void ppl_cukernel_transpose(
    int64_t num_elems,
    int num_dims,
    GArray<DivModFast> input_strides_fast,
    GArray<int64_t> output_flip_strides,
    const T *input,
    T *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int64_t output_offset = 0;
    int idx, remain = index;
    for (int it = 0; it < num_dims; ++it) {
        input_strides_fast[it].divmod(remain, idx, remain);
        output_offset += idx * output_flip_strides[it];
    }
    output[output_offset] = input[index];
}

template <typename T>
__global__ void ppl_cukernel_transpose_nhwc(
    int64_t num_elems,
    int num_dims,
    GArray<DivModFast> input_strides_fast,
    GArray<int64_t> input_strides,
    GArray<int64_t> output_flip_strides,
    const T *input,
    T *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int64_t input_offset  = 0;
    int64_t output_offset = 0;
    int idx, remain = index;
    for (int it = 0; it < num_dims; ++it) {
        input_strides_fast[it].divmod(remain, idx, remain);
        input_offset += idx * input_strides[it];
        output_offset += idx * output_flip_strides[it];
    }
    output[output_offset] = input[input_offset];
}

ppl::common::RetCode PPLCUDATransposeForwardImp(
    cudaStream_t stream,
    ppl::nn::common::TransposeParam param,
    const ppl::nn::TensorShape *input_shape,
    const void *input,
    const ppl::nn::TensorShape *output_shape,
    void *output)
{
    FastTransposeParam fast_param;
    if (FastTransposeSupport(&fast_param, input_shape, param, output_shape)) {
        return PPLCUDATransposeFastForwardImp(stream, fast_param, input_shape, input, output_shape, output);
    } else if (FastTransposeSupport2(&fast_param, input_shape, param, output_shape)) {
        return PPLCUDATransposeFastForwardImp(stream, fast_param, input_shape, input, output_shape, output);
    } else if (MiddleFastTransposeSupport(&fast_param, input_shape, param, output_shape)) {
        return PPLCUDATransposeMiddleFastForwardImp(stream, fast_param, input_shape, input, output_shape, output);
    }
    int num_dims      = output_shape->GetDimCount();
    int64_t num_elems = output_shape->GetElementsExcludingPadding();

    GArray<DivModFast> input_strides_fast(num_dims);
    GArray<int64_t> input_strides(num_dims);
    GArray<int64_t> output_strides(num_dims);
    int64_t acc_output_stride = 1;
    int64_t acc_input_stride  = 1;
    for (int it = num_dims - 1; it >= 0; --it) {
        input_strides_fast[it] = DivModFast(acc_input_stride);
        output_strides[it]     = acc_output_stride;
        acc_input_stride *= input_shape->GetDim(it);
        acc_output_stride *= output_shape->GetDim(it);
    }
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
        acc_input_stride  = 1;
        acc_output_stride = 1;
        for (int it = num_dims - 1; it >= 0; --it) {
            if (it == num_dims - 1) {
                input_strides[1]  = acc_input_stride;
                output_strides[1] = acc_output_stride;
                acc_input_stride *= input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
                acc_output_stride *= output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
            } else if (it == 0) {
                input_strides[it]  = acc_input_stride;
                output_strides[it] = acc_output_stride;
                acc_input_stride *= input_shape->GetDim(it);
                acc_output_stride *= output_shape->GetDim(it);
            } else {
                input_strides[it + 1]  = acc_input_stride;
                output_strides[it + 1] = acc_output_stride;
                acc_input_stride *= input_shape->GetDim(it + 1);
                acc_output_stride *= output_shape->GetDim(it + 1);
            }
        }
    }
    GArray<int64_t> output_flip_strides(num_dims);
    for (int i = 0; i < num_dims; ++i) {
        for (int j = 0; j < num_dims; ++j) {
            if (param.perm[j] == i) {
                output_flip_strides[i] = output_strides[j];
            }
        }
    }

    int block_size = 256;
    int grid_size  = (num_elems + block_size - 1) / block_size;

#define SWITCH_CASE(TYPE)                                                                                                          \
    case sizeof(TYPE): {                                                                                                           \
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {                                                      \
            ppl_cukernel_transpose_nhwc<<<grid_size, block_size, 0, stream>>>(                                                     \
                num_elems, num_dims, input_strides_fast, input_strides, output_flip_strides, (const TYPE *)input, (TYPE *)output); \
        } else {                                                                                                                   \
            ppl_cukernel_transpose<<<grid_size, block_size, 0, stream>>>(                                                          \
                num_elems, num_dims, input_strides_fast, output_flip_strides, (const TYPE *)input, (TYPE *)output);                \
        }                                                                                                                          \
        return ppl::common::RC_SUCCESS;                                                                                            \
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