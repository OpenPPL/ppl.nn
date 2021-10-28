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

#include "cudakernel/memory/tile.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void ppl_cukernel_tile(
    int64_t num_elems,
    int num_dims,
    GArray<DivModFast> input_dims_fast,
    GArray<int64_t> input_strides,
    const T* input,
    GArray<DivModFast> output_strides_fast,
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int64_t input_offset = 0;
    int idx, remain = index;
    for (int it = 0; it < num_dims; ++it) {
        output_strides_fast[it].divmod(remain, idx, remain);
        int quo, in_idx;
        input_dims_fast[it].divmod(idx, quo, in_idx);
        input_offset += input_strides[it] * in_idx;
    }
    output[index] = input[input_offset];
}

ppl::common::RetCode PPLCUDATileForwardImp(
    cudaStream_t stream,
    TileParam param,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output)
{
    int block_size     = 256;
    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int grid_size      = (num_elems + block_size - 1) / block_size;
    int num_dims       = output_shape->GetDimCount();
    GArray<int64_t> input_strides(num_dims);
    GArray<DivModFast> input_dims_fast(num_dims);
    GArray<DivModFast> output_strides_fast(num_dims);
    int64_t acc_output_stride = 1;
    int64_t acc_input_stride  = 1;
    for (int it = num_dims - 1; it >= 0; --it) {
        input_strides[it]       = acc_input_stride;
        input_dims_fast[it]     = input_shape->GetDim(it);
        output_strides_fast[it] = DivModFast(acc_output_stride);
        acc_input_stride *= input_shape->GetDim(it);
        acc_output_stride *= output_shape->GetDim(it);
    }

#define SWITCH_CASE(TYPE)                                                                                                 \
    case sizeof(TYPE): {                                                                                                  \
        ppl_cukernel_tile<<<grid_size, block_size, 0, stream>>>(                                                          \
            num_elems, num_dims, input_dims_fast, input_strides, (const TYPE*)input, output_strides_fast, (TYPE*)output); \
        return ppl::common::RC_SUCCESS;                                                                                   \
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
