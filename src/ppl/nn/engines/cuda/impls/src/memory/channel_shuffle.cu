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
    int64_t num_elems_pad,
    int32_t group,
    int32_t channels_per_group,
    GArray<DivModFast> input_strides_fast,
    const T* input,
    T* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems_pad)
        return;
    int64_t output_offset = 0;
    int n_idx, c_idx, hw_idx, remain = index;

    input_strides_fast[0].divmod(remain, n_idx, remain);
    output_offset += (index - remain);
    input_strides_fast[1].divmod(remain, c_idx, remain);
    hw_idx = remain;
    int out_c_idx = c_idx % channels_per_group * group + c_idx / channels_per_group; 
    output_offset += out_c_idx * input_strides_fast[1].d_ + hw_idx;

    output[output_offset] = index >= num_elems ? 0 : input[index];
}

__global__ void ppl_cukernel_channel_shuffle_int8(
    int64_t num_elems,
    int64_t num_elems_pad,
    int32_t group,
    int32_t channels_per_group,
    GArray<DivModFast> input_strides_fast,
    const int8_t* input,
    int8_t* output,
    float in_scale,
    float out_scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems_pad)
        return;
    int64_t output_offset = 0;
    int n_idx, c_idx, hw_idx, remain = index;

    input_strides_fast[0].divmod(remain, n_idx, remain);
    output_offset += (index - remain);
    input_strides_fast[1].divmod(remain, c_idx, remain);
    hw_idx = remain;
    int out_c_idx = c_idx % channels_per_group * group + c_idx / channels_per_group; 
    output_offset += out_c_idx * input_strides_fast[1].d_ + hw_idx;

    int res = round(input[index] * in_scale / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;

    output[output_offset] = index >= num_elems ? 0 : res;
}

template <typename T>
__global__ void ppl_cukernel_channel_shuffle_nhwc(
    int64_t num_elems,
    int64_t num_elems_pad,
    int32_t group,
    int channels_per_group,
    int pad_channels,
    DivModFast channels_fast,
    const T *input,
    T *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems_pad)
        return;
    int64_t input_offset = 0;
    int64_t output_offset = 0;
    int nhw_idx, c_idx, remain = index;
    channels_fast.divmod(remain, nhw_idx, c_idx);
    int out_c_idx = c_idx % channels_per_group * group + c_idx / channels_per_group; 
    input_offset += nhw_idx * pad_channels + c_idx;
    output_offset += nhw_idx * pad_channels + out_c_idx;

    output[output_offset] = index >= num_elems ? 0 : input[input_offset];
}

__global__ void ppl_cukernel_channel_shuffle_nhwc_int8(
    int64_t num_elems,
    int64_t num_elems_pad,
    int32_t group,
    int channels_per_group,
    int pad_channels,
    DivModFast channels_fast,
    const int8_t *input,
    int8_t *output,
    float in_scale,
    float out_scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems_pad)
        return;
    int64_t input_offset = 0;
    int64_t output_offset = 0;
    int nhw_idx, c_idx, remain = index;
    channels_fast.divmod(remain, nhw_idx, c_idx);
    int out_c_idx = c_idx % channels_per_group * group + c_idx / channels_per_group; 
    input_offset += nhw_idx * pad_channels + c_idx;
    output_offset += nhw_idx * pad_channels + out_c_idx;

    int res = round(input[input_offset] * in_scale / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;

    output[output_offset] = index >= num_elems ? 0 : res;
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
    int64_t num_elems = output_shape->GetElementsExcludingPadding();
    int64_t num_elems_pad = output_shape->GetElementsExcludingPadding();

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
    int grid_size  = (num_elems_pad + block_size - 1) / block_size;
    int channels_per_group = input_shape->GetDim(1) / group;

    #define SWITCH_CASE(TYPE)                                                                                       \
    case sizeof(TYPE): {                                                                                            \
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||                                       \
            output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16){                                       \
            ppl_cukernel_channel_shuffle_nhwc<<<grid_size, block_size, 0, stream>>>(                                \
                num_elems, num_elems_pad, group, channels_per_group, pad_channels, channels_fast,                                  \
                (const TYPE *)input, (TYPE *)output);                                           \
        } else {                                                                                                    \
            ppl_cukernel_channel_shuffle<<<grid_size, block_size, 0, stream>>>(                                     \
                num_elems, num_elems_pad, group, channels_per_group, input_strides_fast, (const TYPE *)input, (TYPE *)output);     \
        }                                                                                                           \
        return ppl::common::RC_SUCCESS;                                                                             \
    }                                                                                                               \


    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        case sizeof(int8_t): {                                                                                            
            if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
                output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16){                                         
                ppl_cukernel_channel_shuffle_nhwc_int8<<<grid_size, block_size, 0, stream>>>(                                
                    num_elems, num_elems_pad, group, channels_per_group, pad_channels, channels_fast,                                  
                    (const int8_t *)input, (int8_t *)output, in_scale, out_scale);                                           
            } else {                                                                                                    
                ppl_cukernel_channel_shuffle_int8<<<grid_size, block_size, 0, stream>>>(                                     
                    num_elems, num_elems_pad, group, channels_per_group, input_strides_fast, (int8_t *)input, (int8_t *)output, in_scale, out_scale);
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
    int64_t num_elems,
    int64_t num_elems_pad,
    int32_t group,
    int32_t channels_per_group,
    GArray<DivModFast> input_strides_fast,
    const T* input1,
    const T* input2,
    T* output1,
    T* output2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 2 * num_elems_pad)
        return;
    int64_t output_offset = 0;
    int n_idx, c_idx, hw_idx, remain = index;
    int hw = input_strides_fast[1].d_;

    input_strides_fast[0].divmod(remain, n_idx, remain); // index / chw
    output_offset += (index - remain);
    input_strides_fast[1].divmod(remain, c_idx, remain); // index / hw
    hw_idx = remain;
    int out_c_idx = c_idx % channels_per_group * group + c_idx / channels_per_group; 
    output_offset += out_c_idx * input_strides_fast[1].d_ + hw_idx;
    int out_div_hw = output_offset / hw;
    int in_div_hw = index / hw;

    if(out_div_hw % 2) {
        if(in_div_hw % 2) {
            output2[(out_div_hw - 1) / 2 * hw + hw_idx] = index >= 2 * num_elems ? 0 : input2[(in_div_hw - 1) / 2 * hw + hw_idx];
        } else {
            output2[(out_div_hw - 1) / 2 * hw + hw_idx] = index >= 2 * num_elems ? 0 : input1[in_div_hw / 2 * hw + hw_idx];
        }
    } else {
        if(in_div_hw % 2) {
            output1[out_div_hw / 2 * hw + hw_idx] = index >= 2 * num_elems ? 0 : input2[(in_div_hw - 1) / 2 * hw + hw_idx];
        } else {
            output1[out_div_hw / 2 * hw + hw_idx] = index >= 2 * num_elems ? 0 : input1[in_div_hw / 2 * hw + hw_idx];
        }
    }
}

__global__ void ppl_cukernel_fuse_channel_shuffle_int8(
    int64_t num_elems,
    int64_t num_elems_pad,
    int32_t group,
    int32_t channels_per_group,
    GArray<DivModFast> input_strides_fast,
    const int8_t* input1,
    const int8_t* input2,
    int8_t* output1,
    int8_t* output2,
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
    int hw = input_strides_fast[1].d_;

    input_strides_fast[0].divmod(remain, n_idx, remain); // index / chw
    output_offset += (index - remain);
    input_strides_fast[1].divmod(remain, c_idx, remain); // index / hw
    hw_idx = remain;
    int out_c_idx = c_idx % channels_per_group * group + c_idx / channels_per_group; 
    output_offset += out_c_idx * input_strides_fast[1].d_ + hw_idx;
    int out_div_hw = output_offset / hw;
    int in_div_hw = index / hw;

    if(out_div_hw % 2) {
        if(in_div_hw % 2) {
            int res = round(input2[(in_div_hw - 1) / 2 * hw + hw_idx] * in_scale1 / out_scale1);
            if(res > 127) res = 127;
            else if(res < -128) res = -128;
            output2[(out_div_hw - 1) / 2 * hw + hw_idx] = index >= 2 * num_elems ? 0 : res;
        } else {
            int res = round(input1[in_div_hw / 2 * hw + hw_idx] * in_scale0 / out_scale1);
            if(res > 127) res = 127;
            else if(res < -128) res = -128;
            output2[(out_div_hw - 1) / 2 * hw + hw_idx] = index >= 2 * num_elems ? 0 : res;
        }
    } else {
        if(in_div_hw % 2) {
            int res = round(input2[(in_div_hw - 1) / 2 * hw + hw_idx] * in_scale1 / out_scale0);
            if(res > 127) res = 127;
            else if(res < -128) res = -128;
            output1[out_div_hw / 2 * hw + hw_idx] = index >= 2 * num_elems ? 0 : res;
        } else {
            int res = round(input1[in_div_hw / 2 * hw + hw_idx] * in_scale0 / out_scale0);
            if(res > 127) res = 127;
            else if(res < -128) res = -128;
            output1[out_div_hw / 2 * hw + hw_idx] = index >= 2 * num_elems ? 0 : res;
        }
    }
}

template <typename T>
__global__ void ppl_cukernel_fuse_channel_shuffle_nhwc(
    int64_t num_elems,
    int64_t num_elems_pad,
    int32_t group,
    int channels_per_group,
    int pad_channels,
    DivModFast channels_fast,
    const T *input1,
    const T *input2,
    T *output1,
    T *output2,
    int elems_nhw,
    int elems_c)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 2 * num_elems_pad)
        return;
    int64_t input_offset = 0;
    int64_t output_offset = 0;
    int nhw_idx, c_idx, remain = index;
    channels_fast.divmod(remain, nhw_idx, c_idx);
    int out_c_idx = c_idx % channels_per_group * group + c_idx / channels_per_group; 
    input_offset += nhw_idx * 2 * elems_c + c_idx;
    output_offset += nhw_idx * 2 * elems_c + out_c_idx;
    if(output_offset % (2 * elems_c) >= elems_c) {
        if(input_offset % (2 * elems_c) >= elems_c) {
            output2[nhw_idx * pad_channels + out_c_idx - elems_c] = index >= 2 * num_elems ? 0 : input2[nhw_idx * pad_channels + c_idx - elems_c];
        } else {
            output2[nhw_idx * pad_channels + out_c_idx - elems_c] = index >= 2 * num_elems ? 0 : input1[nhw_idx * pad_channels + c_idx];
        }
    } else {
        if(input_offset % (2 * elems_c) >= elems_c) {
            output1[nhw_idx * pad_channels + out_c_idx] = index >= 2 * num_elems ? 0 : input2[nhw_idx * pad_channels + c_idx - elems_c];
        } else {
            output1[nhw_idx * pad_channels + out_c_idx] = index >= 2 * num_elems ? 0 : input1[nhw_idx * pad_channels + c_idx];
        }
    }
}

__global__ void ppl_cukernel_fuse_channel_shuffle_nhwc_int8(
    int64_t num_elems,
    int64_t num_elems_pad,
    int32_t group,
    int channels_per_group,
    int pad_channels,
    DivModFast channels_fast,
    const int8_t *input1,
    const int8_t *input2,
    int8_t *output1,
    int8_t *output2,
    int elems_nhw,
    int elems_c,
    float in_scale0,
    float in_scale1,
    float out_scale0,
    float out_scale1)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 2 * num_elems_pad)
        return;
    int64_t input_offset = 0;
    int64_t output_offset = 0;
    int nhw_idx, c_idx, remain = index;
    channels_fast.divmod(remain, nhw_idx, c_idx);
    int out_c_idx = c_idx % channels_per_group * group + c_idx / channels_per_group; 
    input_offset += nhw_idx * 2 * elems_c + c_idx;
    output_offset += nhw_idx * 2 * elems_c + out_c_idx;
    if(output_offset % (2 * elems_c) >= elems_c) {
        if(input_offset % (2 * elems_c) >= elems_c) {
            int res = round(input2[nhw_idx * pad_channels + c_idx - elems_c] * in_scale1 / out_scale1);
            if(res > 127) res = 127;
            else if(res < -128) res = -128;
            output2[nhw_idx * pad_channels + out_c_idx - elems_c] = index >= 2 * num_elems ? 0 : res;
        } else {
            int res = round(input1[nhw_idx * pad_channels + c_idx] * in_scale0 / out_scale1);
            if(res > 127) res = 127;
            else if(res < -128) res = -128;
            output2[nhw_idx * pad_channels + out_c_idx - elems_c] = index >= 2 * num_elems ? 0 : res;
        }
    } else {
        if(input_offset % (2 * elems_c) >= elems_c) {
            int res = round(input2[nhw_idx * pad_channels + c_idx - elems_c] * in_scale1 / out_scale0);
            if(res > 127) res = 127;
            else if(res < -128) res = -128;
            output1[nhw_idx * pad_channels + out_c_idx] = index >= 2 * num_elems ? 0 : res;
        } else {
            int res = round(input1[nhw_idx * pad_channels + c_idx] * in_scale0 / out_scale0);
            if(res > 127) res = 127;
            else if(res < -128) res = -128;
            output1[nhw_idx * pad_channels + out_c_idx] = index >= 2 * num_elems ? 0 : res;
        }
    }
}

ppl::common::RetCode PPLCUDAFuseChannelShuffleForwardImp(
    cudaStream_t stream,
    int group,
    const ppl::nn::TensorShape* input_shape,
    const void* input1,
    const void* input2,
    const ppl::nn::TensorShape* output_shape,
    void* output1,
    void* output2,
    float in_scale0,
    float in_scale1,
    float out_scale0,
    float out_scale1)
{
    // num_dims must be equal to 4
    int num_dims      = output_shape->GetDimCount();
    int64_t num_elems = output_shape->GetElementsExcludingPadding();
    int64_t num_elems_pad = output_shape->GetElementsExcludingPadding();

    // for ndarray layout
    int num_input_strides_dims = num_dims - 2;
    GArray<DivModFast> input_strides_fast(num_input_strides_dims);
    int elems_hw = input_shape->GetDim(2) * input_shape->GetDim(3);
    input_strides_fast[1] = DivModFast(elems_hw);
    int elems_chw = 2 * input_shape->GetDim(1) * elems_hw;
    input_strides_fast[0] = DivModFast(elems_chw);
    // for nhwc layout
    int pad_channels = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
    DivModFast channels_fast(2 * input_shape->GetDim(1));
    int elems_nhw = elems_hw * input_shape->GetDim(0);
    int elems_c = input_shape->GetDim(1);

    int block_size = 256;
    int grid_size  = (2 * num_elems_pad + block_size - 1) / block_size;
    int channels_per_group = (2 * input_shape->GetDim(1)) / group;

    #define SWITCH_CASE(TYPE)                                                                                       \
    case sizeof(TYPE): {                                                                                            \
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||                                       \
            output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {                                         \
            ppl_cukernel_fuse_channel_shuffle_nhwc<<<grid_size, block_size, 0, stream>>>(                                \
                num_elems, num_elems_pad, group, channels_per_group, pad_channels, channels_fast,                                  \
                (const TYPE *)input1, (const TYPE *) input2, (TYPE *)output1, (TYPE *)output2, elems_nhw, elems_c);                     \
        } else {                                                                                                    \
            ppl_cukernel_fuse_channel_shuffle<<<grid_size, block_size, 0, stream>>>(                                     \
                num_elems, num_elems_pad, group, channels_per_group, input_strides_fast, (const TYPE *)input1, (const TYPE *)input2,\
                (TYPE *)output1, (TYPE *)output2);                                                                  \
        }                                                                                                           \
        return ppl::common::RC_SUCCESS;                                                                             \
    }                                                                                                               \


    switch (ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
        case sizeof(int8_t): {                                                                                            
            if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
                output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16){                                         
                ppl_cukernel_fuse_channel_shuffle_nhwc_int8<<<grid_size, block_size, 0, stream>>>(                                
                    num_elems, num_elems_pad, group, channels_per_group, pad_channels, channels_fast,                                  
                    (const int8_t *)input1, (const int8_t *) input2, (int8_t *)output1, (int8_t *)output2, elems_nhw, elems_c,
                    in_scale0, in_scale1, out_scale0, out_scale1);            
            } else {                                                                                                    
                ppl_cukernel_fuse_channel_shuffle_int8<<<grid_size, block_size, 0, stream>>>(                                     
                    num_elems, num_elems_pad, group, channels_per_group, input_strides_fast, (const int8_t *)input1, (const int8_t *)input2,
                    (int8_t *)output1, (int8_t *)output2, in_scale0, in_scale1, out_scale0, out_scale1);                                                                  
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