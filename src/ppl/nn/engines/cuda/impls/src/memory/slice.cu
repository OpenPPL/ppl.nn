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

#include "cudakernel/memory/slice.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

#define MAX_DIM_SIZE SLICE_PARAM_MAX_DIM_SIZE

template <typename T>
__global__ void ppl_cukernel_slice(
    int64_t num_elems,
    int num_dims,
    SliceKernelParam param,
    GArray<int64_t> input_strides,
    const T* input,
    GArray<int64_t> output_strides,
    GArray<DivModFast> output_strides_fast,
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int output_idx[MAX_DIM_SIZE];
    int input_idx[MAX_DIM_SIZE];
    int idx, remain = index;
    for (int it = 0; it < num_dims; ++it) {
        output_strides_fast[it].divmod(remain, idx, remain);
        output_idx[it] = idx;
    }

    // copy output_idx to input_idx
    for (int it = 0; it < num_dims; ++it)
        input_idx[it] = output_idx[it];

    // calc input_idx according to axes[]
    for (int it = 0; it < param.axes_num; ++it) {
        int axis        = param.axes[it];
        input_idx[axis] = output_idx[axis] * param.steps[it] + param.starts[it];
    }

    int64_t input_offset  = 0;
    int64_t output_offset = 0;
    for (int it = 0; it < num_dims; ++it) {
        input_offset += input_idx[it] * input_strides[it];
        output_offset += output_idx[it] * output_strides[it];
    }
    output[output_offset] = input[input_offset];
}

ppl::common::RetCode PPLCUDASliceForwardImp(
    cudaStream_t stream,
    SliceKernelParam param,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output)
{
    if (output_shape->GetElementsIncludingPadding() == 0)
        return ppl::common::RC_SUCCESS;
    int block_size     = 256;
    uint64_t num_elems = output_shape->GetElementsExcludingPadding();
    int grid_size      = (num_elems + block_size - 1) / block_size;
    int num_dims       = output_shape->GetDimCount();
    GArray<int64_t> input_strides(num_dims);
    GArray<int64_t> output_strides(num_dims);
    GArray<DivModFast> output_strides_fast(num_dims);
    int64_t acc_output_stride = 1;
    int64_t acc_input_stride  = 1;
    for (int it = num_dims - 1; it >= 0; --it) {
        input_strides[it]       = acc_input_stride;
        output_strides[it]      = acc_output_stride;
        output_strides_fast[it] = DivModFast(acc_output_stride);
        acc_input_stride *= input_shape->GetDim(it);
        acc_output_stride *= output_shape->GetDim(it);
    }
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
        acc_output_stride = 1;
        acc_input_stride  = 1;
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

#define SWITCH_CASE(TYPE)                                                                                                       \
    case sizeof(TYPE): {                                                                                                        \
        ppl_cukernel_slice<<<grid_size, block_size, 0, stream>>>(                                                               \
            num_elems, num_dims, param, input_strides, (const TYPE*)input, output_strides, output_strides_fast, (TYPE*)output); \
        return ppl::common::RC_SUCCESS;                                                                                         \
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
