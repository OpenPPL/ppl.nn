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

#include "cudakernel/memory/channel_shuffle.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "cudakernel/common/common.h"

template <typename T>
__global__ void ppl_cukernel_channel_shuffle(
    int64_t num_elems,
    int32_t group,
    int32_t channels_per_group,
    GArray<DivModFast> input_strides_fast,
    const T* input,
    T* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int64_t output_offset = 0;
    int n_idx, c_idx, hw_idx, remain = index;

    input_strides_fast[0].divmod(remain, n_idx, remain);
    output_offset += (index - remain);
    input_strides_fast[1].divmod(remain, c_idx, remain);
    hw_idx = remain;
    int out_c_idx = c_idx % channels_per_group * group + c_idx / channels_per_group; 
    output_offset += out_c_idx * input_strides_fast[1].d_ + hw_idx;

    output[output_offset] = input[index];
}

template <typename T>
__global__ void ppl_cukernel_channel_shuffle_nhwc(
    int64_t num_elems,
    int32_t group,
    int channels_per_group,
    int pad_channels,
    DivModFast channels_fast,
    const T *input,
    T *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int64_t input_offset = 0;
    int64_t output_offset = 0;
    int nhw_idx, c_idx, remain = index;
    channels_fast.divmod(remain, nhw_idx, c_idx);
    int out_c_idx = c_idx % channels_per_group * group + c_idx / channels_per_group; 
    input_offset += nhw_idx * pad_channels + c_idx;
    output_offset += nhw_idx * pad_channels + out_c_idx;

    output[output_offset] = input[input_offset];
}

ppl::common::RetCode PPLCUDAChannelShuffleForwardImp(
    cudaStream_t stream,
    int group,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output)
{
    // num_dims must be equal to 4
    int num_dims      = output_shape->GetDimCount();
    int64_t num_elems = output_shape->GetElementsExcludingPadding();

    // for ndarray layout
    int num_input_strides_dims = num_dims - 2;
    GArray<DivModFast> input_strides_fast(num_input_strides_dims);
    int elems_hw = input_shape->GetDim(2) * input_shape->GetDim(3);
    input_strides_fast[1] = DivModFast(elems_hw);
    int elems_chw = input_shape->GetDim(1) * elems_hw;
    input_strides_fast[0] = DivModFast(elems_chw);
    // for nhwc layout
    int pad_channels = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
    DivModFast channels_fast(input_shape->GetDim(1));

    int block_size = 256;
    int grid_size  = (num_elems + block_size - 1) / block_size;
    int channels_per_group = input_shape->GetDim(1) / group;

    #define SWITCH_CASE(TYPE)                                                                                       \
    case sizeof(TYPE): {                                                                                            \
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC){                                         \
            ppl_cukernel_channel_shuffle_nhwc<<<grid_size, block_size, 0, stream>>>(                                \
                num_elems, group, channels_per_group, pad_channels, channels_fast,                                  \
                (const TYPE *)input, (TYPE *)output);                                                               \
        } else {                                                                                                    \
            ppl_cukernel_channel_shuffle<<<grid_size, block_size, 0, stream>>>(                                     \
                num_elems, group, channels_per_group, input_strides_fast, (const TYPE *)input, (TYPE *)output);     \
        }                                                                                                           \
        return ppl::common::RC_SUCCESS;                                                                             \
    }                                                                                                               \


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