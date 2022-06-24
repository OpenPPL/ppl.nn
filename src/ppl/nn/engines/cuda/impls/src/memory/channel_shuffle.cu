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

#define ROUND_BY_SCALE(output, out_scale, input, in_scale)      \
    do {                                                        \
        int res = round(input * in_scale / out_scale);          \
        if(res > 127) res = 127;                                \
        else if(res < -128) res = -128;                         \
        output =res;                                            \
    } while(false);


template <typename T>
__global__ void ppl_cukernel_channel_shuffle(
    int64_t num_elems_pad,
    int channel,
    int pad_channel,
    GArray<DivModFast> input_strides_fast,
    const T* input,
    T* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 2 * num_elems_pad)
        return;
    int64_t output_offset = 0;
    int n_idx, c_idx, hw_idx, remain = index;
    int chw = input_strides_fast[0].d_;
    int hw = input_strides_fast[1].d_;

    input_strides_fast[0].divmod(remain, n_idx, remain); // index / chw
    output_offset += (index - remain);
    input_strides_fast[1].divmod(remain, c_idx, remain); // index / hw
    hw_idx = remain;

    if (c_idx >= channel)
        output[index] = 0.0f;
    else {
        if ((c_idx & 1) == 0) {
            int32_t input_index = n_idx * chw + c_idx / 2 * hw + hw_idx;
            output[index] = input[input_index];
        } else {
            int32_t input_index = n_idx * chw + (channel + c_idx) / 2 * hw + hw_idx;
            output[index] = input[input_index];
        }
    }
}

__global__ void ppl_cukernel_channel_shuffle_int8(
    int64_t num_elems_pad,
    int channel,
    int pad_channel,
    GArray<DivModFast> input_strides_fast,
    const int8_t* input,
    int8_t* output,
    float in_scale,
    float out_scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 2 * num_elems_pad)
        return;
    int64_t output_offset = 0;
    int n_idx, c_idx, hw_idx, remain = index;
    int chw = input_strides_fast[0].d_;
    int hw = input_strides_fast[1].d_;

    input_strides_fast[0].divmod(remain, n_idx, remain); // index / chw
    output_offset += (index - remain);
    input_strides_fast[1].divmod(remain, c_idx, remain); // index / hw
    hw_idx = remain;

    if (c_idx >= channel)
        output[index] = 0;
    else {
        if ((c_idx & 1) == 0) {
            int32_t input_index = n_idx * chw + c_idx / 2 * hw + hw_idx;
            ROUND_BY_SCALE(output[index], out_scale, input[input_index], in_scale);
        } else {
            int32_t input_index = n_idx * chw + (channel + c_idx) / 2 * hw + hw_idx;
            ROUND_BY_SCALE(output[index], out_scale, input[input_index], in_scale);
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_channel_shuffle_nhwc(
    int64_t num_elems_pad,
    int channel,
    int pad_channel,
    DivModFast channels_fast,
    const T *input,
    T *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems_pad)
        return;

    int nhw_idx, c_idx, remain = index;
    channels_fast.divmod(remain, nhw_idx, c_idx);

    if (c_idx >= channel)
        output[index] = 0.0f;
    else {
        if ((c_idx & 1) == 0) {
            int32_t input_index = nhw_idx * pad_channel + c_idx / 2;
            output[index] = input[input_index];
        } else {
            int32_t input_index = nhw_idx * pad_channel + (channel + c_idx) / 2;
            output[index] = input[input_index];
        }
    }
}

__global__ void ppl_cukernel_channel_shuffle_nhwc_int8(
    int64_t num_elems_pad,
    int channel,
    int pad_channel,
    DivModFast channels_fast,
    const int8_t *input,
    int8_t *output,
    float in_scale,
    float out_scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems_pad)
        return;

    int nhw_idx, c_idx, remain = index;
    channels_fast.divmod(remain, nhw_idx, c_idx);

    if (c_idx >= channel)
        output[index] = 0;
    else {
        if ((c_idx & 1) == 0) {
            int32_t input_index = nhw_idx * pad_channel + c_idx / 2;
            ROUND_BY_SCALE(output[index], out_scale, input[input_index], in_scale);
        } else {
            int32_t input_index = nhw_idx * pad_channel + (channel + c_idx) / 2;
            ROUND_BY_SCALE(output[index], out_scale, input[input_index], in_scale);
        }
    }
}

ppl::common::RetCode PPLCUDAChannelShuffleForwardImp(
    cudaStream_t stream,
    int group,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    float in_scale,
    float out_scale)
{
    // num_dims must be equal to 4
    int num_dims      = output_shape->GetDimCount();
    int64_t num_elems_pad = output_shape->CalcElementsIncludingPadding();

    // for ndarray layout without padding
    int num_input_strides_dims = num_dims - 2;
    GArray<DivModFast> input_strides_fast(num_input_strides_dims);
    int elems_hw = input_shape->GetDim(2) * input_shape->GetDim(3);
    input_strides_fast[1] = DivModFast(elems_hw);
    int elems_chw = input_shape->GetDim(1) * elems_hw;
    input_strides_fast[0] = DivModFast(elems_chw);
    // for nhwc layout
    int channel = input_shape->GetDim(1);
    int pad_channel = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
    DivModFast channels_fast(pad_channel);

    int block_size = 256;
    int grid_size  = (num_elems_pad + block_size - 1) / block_size;

    #define SWITCH_CASE(TYPE)                                                                                       \
    case sizeof(TYPE): {                                                                                            \
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||                                       \
            output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16){                                       \
            ppl_cukernel_channel_shuffle_nhwc<<<grid_size, block_size, 0, stream>>>(                                \
                num_elems_pad, channel, pad_channel, channels_fast, (const TYPE *)input, (TYPE *)output);           \
        } else {                                                                                                    \
            ppl_cukernel_channel_shuffle<<<grid_size, block_size, 0, stream>>>(                                     \
                num_elems_pad, channel, pad_channel, input_strides_fast, (const TYPE *)input, (TYPE *)output);      \
        }                                                                                                           \
        return ppl::common::RC_SUCCESS;                                                                             \
    }                                                                                                               \


    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        case sizeof(int8_t): {                                                                                            
            if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
                output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16){                                         
                ppl_cukernel_channel_shuffle_nhwc_int8<<<grid_size, block_size, 0, stream>>>(                                
                    num_elems_pad, channel, pad_channel, channels_fast, (const int8_t *)input, (int8_t *)output, in_scale, out_scale);                                           
            } else {                                                                                                    
                ppl_cukernel_channel_shuffle_int8<<<grid_size, block_size, 0, stream>>>(                                     
                    num_elems_pad, channel, pad_channel, input_strides_fast, (int8_t *)input, (int8_t *)output, in_scale, out_scale);
            }                                                                                                           
            return ppl::common::RC_SUCCESS;                                                                             
        }
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#undef SWITCH_CASE
}

template <typename T>
__global__ void ppl_cukernel_fuse_channel_shuffle(
    int64_t num_elems_pad,
    int channel,
    int pad_channel,
    GArray<DivModFast> input_strides_fast,
    const T* input0,
    const T* input1,
    T* output0,
    T* output1)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 2 * num_elems_pad)
        return;
    int64_t output_offset = 0;
    int n_idx, c_idx, hw_idx, remain = index;
    int chw = input_strides_fast[0].d_;
    int hw = input_strides_fast[1].d_;

    input_strides_fast[0].divmod(remain, n_idx, remain); // index / chw
    output_offset += (index - remain);
    input_strides_fast[1].divmod(remain, c_idx, remain); // index / hw
    hw_idx = remain;

    if (index < num_elems_pad) {
        if (c_idx >= channel)
            output0[index] = 0.0f;
        else {
            int32_t input_index = n_idx * chw + c_idx / 2 * hw + hw_idx;
            if ((c_idx & 1) == 0) {
                output0[index] = input0[input_index];
            } else {
                output0[index] = input1[input_index];
            }
        }
    } else {
        if (c_idx - pad_channel >= channel)
            output1[index - num_elems_pad] = 0.0f;
        else {
            // if channel is odd, the first output1 element is from input1
            // if channel is even, the first output1 element is from input0
            int32_t input_index = n_idx * chw + (channel + c_idx) / 2 * hw + hw_idx;
            if (((channel + c_idx) & 1) == 0) {
                output1[index - num_elems_pad] = input0[input_index - num_elems_pad];
            } else {
                output1[index - num_elems_pad] = input1[input_index - num_elems_pad];
            }
        }
    }
}

__global__ void ppl_cukernel_fuse_channel_shuffle_int8(
    int64_t num_elems_pad,
    int channel,
    int pad_channel,
    GArray<DivModFast> input_strides_fast,
    const int8_t* input0,
    const int8_t* input1,
    int8_t* output0,
    int8_t* output1,
    float in_scale0,
    float in_scale1,
    float out_scale0,
    float out_scale1)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 2 * num_elems_pad)
        return;
    int64_t output_offset = 0;
    int n_idx, c_idx, hw_idx, remain = index;
    int chw = input_strides_fast[0].d_;
    int hw = input_strides_fast[1].d_;

    input_strides_fast[0].divmod(remain, n_idx, remain); // index / chw
    output_offset += (index - remain);
    input_strides_fast[1].divmod(remain, c_idx, remain); // index / hw
    hw_idx = remain;

    if (index < num_elems_pad) {
        if (c_idx >= channel)
            output0[index] = 0;
        else {
            int32_t input_index = n_idx * chw + c_idx / 2 * hw + hw_idx;
            if ((c_idx & 1) == 0) {
                ROUND_BY_SCALE(output0[index], out_scale0, input0[input_index], in_scale0)
            } else {
                ROUND_BY_SCALE(output0[index], out_scale0, input1[input_index], in_scale1)
            }
        }
    } else {
        if (c_idx - pad_channel >= channel)
            output1[index - num_elems_pad] = 0;
        else {
            // if channel is odd, the first output1 element is from input1
            // if channel is even, the first output1 element is from input0
            int32_t input_index = n_idx * chw + (channel + c_idx) / 2 * hw + hw_idx;
            if (((channel + c_idx) & 1) == 0) {
                ROUND_BY_SCALE(output1[index - num_elems_pad], out_scale1, input0[input_index - num_elems_pad], in_scale0)
            } else {
                ROUND_BY_SCALE(output1[index - num_elems_pad], out_scale1, input1[input_index - num_elems_pad], in_scale1)
            }
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_fuse_channel_shuffle_nhwc(
    int64_t num_elems_pad,
    int channel,
    int pad_channel,
    DivModFast channels_fast,
    const T *input0,
    const T *input1,
    T *output0,
    T *output1)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 2 * num_elems_pad)
        return;

    int nhw_idx, c_idx, remain = index;
    channels_fast.divmod(remain, nhw_idx, c_idx);

    if (index < num_elems_pad) {
        if (c_idx >= channel)
            output0[index] = 0.0f;
        else {
            int32_t input_index = nhw_idx * pad_channel + c_idx / 2;
            if ((c_idx & 1) == 0) {
                output0[index] = input0[input_index];
            } else {
                output0[index] = input1[input_index];
            }
        }
    } else {
        if (c_idx - pad_channel >= channel)
            output1[index - num_elems_pad] = 0.0f;
        else {
            // if channel is odd, the first output1 element is from input1
            // if channel is even, the first output1 element is from input0
            int32_t input_index = nhw_idx * pad_channel + (channel + c_idx) / 2;
            if (((channel + c_idx) & 1) == 0) {
                output1[index - num_elems_pad] = input0[input_index - num_elems_pad];
            } else {
                output1[index - num_elems_pad] = input1[input_index - num_elems_pad];
            }
        }
    }
}

__global__ void ppl_cukernel_fuse_channel_shuffle_nhwc_int8(
    int64_t num_elems_pad,
    int channel,
    int pad_channel,
    DivModFast channels_fast,
    const int8_t *input0,
    const int8_t *input1,
    int8_t *output0,
    int8_t *output1,
    float in_scale0,
    float in_scale1,
    float out_scale0,
    float out_scale1)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 2 * num_elems_pad)
        return;

    int nhw_idx, c_idx, remain = index;
    channels_fast.divmod(remain, nhw_idx, c_idx);

    if (index < num_elems_pad) {
        if (c_idx >= channel)
            output0[index] = 0.0f;
        else {
            int32_t input_index = nhw_idx * pad_channel + c_idx / 2;
            if ((c_idx & 1) == 0) {
                ROUND_BY_SCALE(output0[index], out_scale0, input0[input_index], in_scale0)
            } else {
                ROUND_BY_SCALE(output0[index], out_scale0, input1[input_index], in_scale1)
            }
        }
    } else {
        if (c_idx - pad_channel >= channel)
            output1[index - num_elems_pad] = 0.0f;
        else {
            // if channel is odd, the first output1 element is from input1
            // if channel is even, the first output1 element is from input0
            int32_t input_index = nhw_idx * pad_channel + (channel + c_idx) / 2;
            if (((channel + c_idx) & 1) == 0) {
                ROUND_BY_SCALE(output1[index - num_elems_pad], out_scale1, input0[input_index - num_elems_pad], in_scale0)
            } else {
                ROUND_BY_SCALE(output1[index - num_elems_pad], out_scale1, input1[input_index - num_elems_pad], in_scale1)
            }
        }
    }
}

ppl::common::RetCode PPLCUDAFuseChannelShuffleForwardImp(
    cudaStream_t stream,
    int group,
    const ppl::nn::TensorShape* input_shape,
    const void* input0,
    const void* input1,
    const ppl::nn::TensorShape* output_shape,
    void* output0,
    void* output1,
    float in_scale0,
    float in_scale1,
    float out_scale0,
    float out_scale1)
{
    // num_dims must be equal to 4
    int num_dims      = output_shape->GetDimCount();
    int64_t num_elems_pad = output_shape->CalcElementsIncludingPadding();

    // for ndarray layout without padding
    int num_input_strides_dims = num_dims - 2;
    GArray<DivModFast> input_strides_fast(num_input_strides_dims);
    int elems_hw = input_shape->GetDim(2) * input_shape->GetDim(3);
    input_strides_fast[1] = DivModFast(elems_hw);
    int elems_chw = input_shape->GetDim(1) * elems_hw;
    input_strides_fast[0] = DivModFast(elems_chw);
    // for nhwc layout
    int channel = input_shape->GetDim(1);
    int pad_channel = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
    DivModFast channels_fast(pad_channel);

    int block_size = 256;
    int grid_size  = (2 * num_elems_pad + block_size - 1) / block_size;

    #define SWITCH_CASE(TYPE)                                                                                       \
    case sizeof(TYPE): {                                                                                            \
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||                                       \
            output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {                                         \
            ppl_cukernel_fuse_channel_shuffle_nhwc<<<grid_size, block_size, 0, stream>>>(                                \
                num_elems_pad, channel, pad_channel, channels_fast,                                  \
                (const TYPE *)input0, (const TYPE *) input1, (TYPE *)output0, (TYPE *)output1);                     \
        } else {                                                                                                    \
            ppl_cukernel_fuse_channel_shuffle<<<grid_size, block_size, 0, stream>>>(                                     \
                num_elems_pad, channel, pad_channel, input_strides_fast, (const TYPE *)input0, (const TYPE *)input1,\
                (TYPE *)output0, (TYPE *)output1);                                                                  \
        }                                                                                                           \
        return ppl::common::RC_SUCCESS;                                                                             \
    }                                                                                                               \


    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        case sizeof(int8_t): {                                                                                            
            if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
                output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16){                                         
                ppl_cukernel_fuse_channel_shuffle_nhwc_int8<<<grid_size, block_size, 0, stream>>>(                                
                    num_elems_pad, channel, pad_channel, channels_fast,                                  
                    (const int8_t *)input0, (const int8_t *) input1, (int8_t *)output0, (int8_t *)output1,
                    in_scale0, in_scale1, out_scale0, out_scale1);            
            } else {                                                                                                    
                ppl_cukernel_fuse_channel_shuffle_int8<<<grid_size, block_size, 0, stream>>>(                                     
                    num_elems_pad, channel, pad_channel, input_strides_fast, (const int8_t *)input0, (const int8_t *)input1,
                    (int8_t *)output0, (int8_t *)output1, in_scale0, in_scale1, out_scale0, out_scale1);                                                                  
            }                                                                                                           
            return ppl::common::RC_SUCCESS;                                                                             
        }        
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#undef SWITCH_CASE
}