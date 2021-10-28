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

#include "cudakernel/nn/unpooling.h"
#include "ppl/common/types.h"

template <typename T>
__global__ void ppl_cukernel_max_unpool(
    const int num_elems,
    const T* input,
    const int batch,
    const int channels,
    const int pad_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    T* output,
    bool use_bottom_mask,
    const int64_t* bottom_mask,
    int layout_coef)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int in_widx  = index % in_width;
    int in_hidx  = (index / in_width) % in_height;
    int c_idx    = (index / in_width / in_height) % channels;
    int main_c   = c_idx / layout_coef;
    int remain_c = c_idx % layout_coef;
    int n_idx    = index / in_width / in_height / channels;
    int in_index = n_idx * pad_channels * in_height * in_width +
                   main_c * in_height * in_width * layout_coef +
                   in_hidx * in_width * layout_coef + in_widx * layout_coef + remain_c;
    if (use_bottom_mask) {
        const int mask_index = int(bottom_mask[in_index]);
        T in_val             = input[in_index];
        // int out_widx         = mask_index % out_width;
        // int out_hidx         = (mask_index / out_width) % out_height;
        // int out_index        = n_idx * pad_channels * out_height * out_width +
        // main_c * out_height * out_width * layout_coef +
        // out_hidx * out_width * layout_coef + out_widx * layout_coef + remain_c;
        output[mask_index]   = in_val;
    } else {
        T in_val      = input[in_index];
        int out_hidx  = max(0, min(in_hidx * stride_h - pad_h, out_height - 1));
        int out_widx  = max(0, min(in_widx * stride_w - pad_w, out_width - 1));
        int out_index = n_idx * pad_channels * out_height * out_width +
                        main_c * out_height * out_width * layout_coef +
                        out_hidx * out_width * layout_coef + out_widx * layout_coef + remain_c;
        output[out_index] = in_val;
    }
}

template <typename T>
__global__ void ppl_cukernel_max_unpool_nhwc(
    const int num_elems,
    const T* input,
    const int batch,
    const int channels,
    const int pad_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    T* output,
    bool use_bottom_mask,
    const int64_t* bottom_mask)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int cidx    = index % channels;
    int remain  = index / channels;
    int in_widx = remain % in_width;
    remain /= in_width;
    int in_hidx = remain % in_height;
    remain /= in_height;
    int n_idx    = remain / channels;
    int in_index = n_idx * in_height * in_width * pad_channels +
                   in_hidx * in_width * pad_channels +
                   in_widx * pad_channels + cidx;
    if (use_bottom_mask) {
        const int mask_index = int(bottom_mask[in_index]);
        T in_val             = input[in_index];
        int out_widx         = mask_index % out_width;
        int out_hidx         = (mask_index / out_width) % out_height;
        int out_index        = n_idx * out_height * out_width * pad_channels +
                        out_hidx * out_width * pad_channels +
                        out_widx * pad_channels + cidx;
        output[out_index] = in_val;
    } else {
        T in_val      = input[in_index];
        int out_hidx  = max(0, min(in_hidx * stride_h - pad_h, out_height - 1));
        int out_widx  = max(0, min(in_widx * stride_w - pad_w, out_width - 1));
        int out_index = n_idx * out_height * out_width * pad_channels +
                        out_hidx * out_width * pad_channels +
                        out_widx * pad_channels + cidx;
        output[out_index] = in_val;
    }
}

ppl::common::RetCode PPLCUDAMaxUnpoolForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output,
    bool use_bottom_mask,
    const int64_t* bottom_mask,
    int unpool,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int kernel_h,
    int kernel_w)
{
    int top_elems = output_shape->GetElementsIncludingPadding();
    cudaMemset((void*)output, 0, ppl::common::GetSizeOfDataType(output_shape->GetDataType()) * top_elems);
    int batch        = output_shape->GetDim(0);
    int channels     = output_shape->GetDim(1);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding1(1);
    int out_height   = output_shape->GetDim(2);
    int out_width    = output_shape->GetDim(3);
    int in_height    = input_shape->GetDim(2);
    int in_width     = input_shape->GetDim(3);
    int block_size   = 256;
    int num_elems    = in_width * in_height * channels * batch;
    int grid_size    = (num_elems + block_size - 1) / block_size;

    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
#define SWITCH_CASE(TYPE)                                                                                                                                                                                                 \
    case sizeof(TYPE): {                                                                                                                                                                                                  \
        ppl_cukernel_max_unpool_nhwc<<<grid_size, block_size, 0, stream>>>(                                                                                                                                               \
            num_elems, (const TYPE*)input, batch, channels, pad_channels, in_height, in_width, out_height, out_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, (TYPE*)output, use_bottom_mask, bottom_mask); \
        return ppl::common::RC_SUCCESS;                                                                                                                                                                                   \
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
    } else {
        int layout_coef = 1;
#define SWITCH_CASE(TYPE)                                                                                                                                                                                                              \
    case sizeof(TYPE): {                                                                                                                                                                                                               \
        ppl_cukernel_max_unpool<<<grid_size, block_size, 0, stream>>>(                                                                                                                                                                 \
            num_elems, (const TYPE*)input, batch, channels, pad_channels, in_height, in_width, out_height, out_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, (TYPE*)output, use_bottom_mask, bottom_mask, layout_coef); \
        return ppl::common::RC_SUCCESS;                                                                                                                                                                                                \
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
