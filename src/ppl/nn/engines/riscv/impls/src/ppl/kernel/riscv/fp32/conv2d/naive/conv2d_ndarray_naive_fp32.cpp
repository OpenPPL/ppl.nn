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

#include "ppl/common/log.h"
#include "ppl/kernel/riscv/common/math.h"
#include "ppl/kernel/riscv/fp32/conv2d/naive/conv2d_ndarray_naive_fp32.h"
#include <cstring>

namespace ppl { namespace kernel { namespace riscv {

static size_t conv_naive_cto8c_get_cvt_filter_size_fp32(
    int64_t flt_h,
    int64_t flt_w,
    int64_t channels,
    int64_t group,
    int64_t num_outs)
{
    int64_t channels_per_group = channels / group;
    int64_t num_outs_per_group = num_outs / group;
    int64_t num_elem           = flt_h * flt_w * channels_per_group * num_outs_per_group * group;
    return num_elem * sizeof(float);
}

static void conv_naive_cto8c_cvt_filter_fp32(
    const float* filter,
    int64_t flt_h,
    int64_t flt_w,
    int64_t num_outs,
    int64_t channels,
    int64_t group,
    float* filter_cvt)
{
    int64_t channels_per_group = channels / group;
    int64_t num_outs_per_group = num_outs / group;
    int64_t num_elem           = flt_h * flt_w * channels_per_group * num_outs_per_group * group;
    for (int64_t i = 0; i < num_elem; i++) {
        filter_cvt[i] = filter[i];
    }
}

static size_t conv_naive_cto8c_get_temp_buffer_size_fp32(
    int64_t src_h,
    int64_t src_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t flt_h,
    int64_t flt_w,
    int64_t channels,
    int64_t group,
    int64_t num_outs)
{
    int64_t src_pad_h = src_h + 2 * padding_h;
    int64_t src_pad_w = src_w + 2 * padding_w;

    int64_t dst_h = (src_pad_h - flt_h + stride_h) / stride_h;
    int64_t dst_w = (src_pad_w - flt_w + stride_w) / stride_w;

    size_t src_pad_size = src_pad_h * src_pad_w * channels * sizeof(float);

    return src_pad_size;
}

static void conv_naive_padding_src_cvt_fp32(
    int64_t src_h,
    int64_t src_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t channels,
    const float* src,
    float* src_cvt)
{
    int64_t src_pad_h = src_h + 2 * pad_h;
    int64_t src_pad_w = src_w + 2 * pad_w;

    for (int64_t c = 0; c < channels; c += 1) {
        for (int64_t hi = 0; hi < src_pad_h; hi += 1) {
            for (int64_t wi = 0; wi < src_pad_w; wi += 1) {
                if (hi < pad_h || hi >= pad_h + src_h || wi < pad_w || wi >= pad_w + src_w) {
                    src_cvt[c * src_pad_h * src_pad_w + hi * src_pad_w + wi] = 0;
                } else {
                    src_cvt[c * src_pad_h * src_pad_w + hi * src_pad_w + wi] =
                        src[c * src_h * src_w + (hi - pad_h) * src_w + (wi - pad_w)];
                }
            }
        }
    }
}

static void conv_naive_kernel_riscv_fp32(
    const float* src,
    int64_t src_h,
    int64_t src_w,
    int64_t dst_h,
    int64_t dst_w,
    int64_t flt_h,
    int64_t flt_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t channels,
    int64_t num_outs,
    int64_t num_threads,
    const float* filter,
    const float* bias,
    float* dst)
{
    int64_t src_channel_stride    = src_h * src_w;
    int64_t filter_channel_stride = flt_h * flt_w;
    int64_t dst_out_stride        = dst_h * dst_w;

    for (int64_t i = 0; i < num_outs; ++i) {
        auto dst_per_out = dst + i * dst_out_stride;
        for (int64_t dst_h_loc = 0; dst_h_loc < dst_h; ++dst_h_loc) {
            for (int64_t dst_w_loc = 0; dst_w_loc < dst_w; ++dst_w_loc) {
                dst_per_out[dst_h_loc * dst_w + dst_w_loc] = bias[i];
            }
        }
    }

    for (int64_t i = 0; i < num_outs; ++i) {
        auto dst_per_out = dst + i * dst_out_stride;
        for (int64_t c = 0; c < channels; ++c) {
            auto src_per_channel = src + c * src_channel_stride;

            for (int64_t dst_h_loc = 0; dst_h_loc < dst_h; ++dst_h_loc) {
                for (int64_t dst_w_loc = 0; dst_w_loc < dst_w; ++dst_w_loc) {
                    for (int64_t flt_h_loc = 0; flt_h_loc < flt_h; ++flt_h_loc) {
                        for (int64_t flt_w_loc = 0; flt_w_loc < flt_w; ++flt_w_loc) {
                            auto src_h_loc = dst_h_loc * stride_h + flt_h_loc;
                            auto src_w_loc = dst_w_loc * stride_w + flt_w_loc;
                            dst_per_out[dst_h_loc * dst_w + dst_w_loc] +=
                                src_per_channel[src_h_loc * src_w + src_w_loc] * filter[flt_h_loc * flt_w + flt_w_loc];
                            // LOG(DEBUG) << dst_per_out[dst_h_loc * dst_w + dst_w_loc] << " " <<
                            // src_per_channel[src_h_loc * src_w + src_w_loc] << " " << filter[flt_h_loc * flt_w +
                            // flt_w_loc];
                        }
                    }
                }
            }
            filter += filter_channel_stride;
        }
    }

    for (int64_t i = 0; i < num_outs; i += 1) {
        for (int64_t h = 0; h < dst_h; h += 1) {
            for (int64_t w = 0; w < dst_w; w += 1) {
                LOG(DEBUG) << i * dst_h * dst_w + h * dst_w + w << " " << dst[i * dst_h * dst_w + h * dst_w + w];
            }
        }
    }
}

static void conv_naive_cto8c_general_riscv_fp32(
    const float* src,
    int64_t src_h,
    int64_t src_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t flt_h,
    int64_t flt_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t group,
    int64_t channels,
    int64_t num_outs,
    int64_t batch,
    int64_t channel_blk_size,
    int64_t h_blk_size,
    int64_t w_blk_size,
    int64_t num_threads,
    const float* filter,
    const float* bias,
    float* dst,
    float* temp_buffer,

    bool with_relu,
    bool reflect_pad)
{
    int64_t src_pad_h = src_h + 2 * padding_h;
    int64_t src_pad_w = src_w + 2 * padding_w;

    int64_t dst_h = (src_pad_h - flt_h + stride_h) / stride_h;
    int64_t dst_w = (src_pad_w - flt_w + stride_w) / stride_w;

    int64_t channels_per_group = channels / group;
    int64_t num_outs_per_group = num_outs / group;

    int64_t src_batch_stride      = channels * src_h * src_w;
    int64_t dst_batch_stride      = num_outs * dst_h * dst_w;
    int64_t temp_src_batch_stride = channels * src_pad_h * src_pad_w;

    int64_t flt_group_stride  = num_outs_per_group * channels_per_group * flt_h * flt_w;
    int64_t bias_group_stride = num_outs_per_group;
    auto temp_src             = temp_buffer;

    for (int64_t b = 0; b < batch; b++) {
        auto src_per_batch = src + b * src_batch_stride;
        auto src_per_group = src_per_batch;
        auto dst_per_batch = dst + b * dst_batch_stride;

        for (int64_t g = 0; g < group; g++) {
            auto temp_dst_per_group = dst_per_batch + g * num_outs_per_group * dst_h * dst_w;
            auto filter_per_group   = filter + g * flt_group_stride;
            auto bias_per_group     = bias + g * bias_group_stride;

            conv_naive_padding_src_cvt_fp32(src_h, src_w, padding_h, padding_w, channels_per_group, src_per_group, temp_src);
            conv_naive_kernel_riscv_fp32(
                temp_src,
                src_pad_h,
                src_pad_w,
                dst_h,
                dst_w,
                flt_h,
                flt_w,
                stride_h,
                stride_w,
                channels_per_group,
                num_outs_per_group,
                num_threads,
                filter_per_group,
                bias_per_group,
                temp_dst_per_group);

            src_per_group += channels_per_group * src_h * src_w;
        }
    }
}

uint64_t conv2d_ndarray_naive_fp32_runtime_executor::cal_temp_buffer_size()
{
    LOG(DEBUG) << "ndarray naive conv: cal temp buffer size";

    size_t temp_buffer_size = conv_naive_cto8c_get_temp_buffer_size_fp32(
        src_shape_->GetDim(2), // src_h
        src_shape_->GetDim(3), // src_w
        conv_param_->pad_h,
        conv_param_->pad_w,
        conv_param_->stride_h,
        conv_param_->stride_w,
        conv_param_->kernel_h,
        conv_param_->kernel_w,
        conv_param_->channels,
        conv_param_->group,
        conv_param_->num_output);

    return temp_buffer_size;
}

void conv2d_ndarray_naive_fp32_runtime_executor::adjust_tunning_param() {}

ppl::common::RetCode conv2d_ndarray_naive_fp32_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    adjust_tunning_param();
    LOG(DEBUG) << "ndarray naive conv: prepare";
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_ndarray_naive_fp32_runtime_executor::execute()
{
    const conv2d_common_param& cp = *conv_param_;

    LOG(DEBUG) << "ndarray naive conv: execute";
    if (src_ == nullptr || cvt_bias_ == nullptr || cvt_filter_ == nullptr || temp_buffer_ == nullptr ||
        dst_ == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }

    conv_naive_cto8c_general_riscv_fp32(
        src_,
        src_shape_->GetDim(2), // src_h
        src_shape_->GetDim(3), // src_w
        conv_param_->pad_h,
        conv_param_->pad_w,
        conv_param_->kernel_h,
        conv_param_->kernel_w,
        conv_param_->stride_h,
        conv_param_->stride_w,
        conv_param_->group,
        conv_param_->channels,
        conv_param_->num_output,
        src_shape_->GetDim(0), // batch
        0,
        0,
        0,
        0,
        cvt_filter_,
        cvt_bias_,
        dst_,
        (float*)temp_buffer_,

        false,
        false);

    return ppl::common::RC_SUCCESS;
}

bool conv2d_ndarray_naive_fp32_offline_manager::is_supported()
{
    return true;
}

ppl::common::RetCode conv2d_ndarray_naive_fp32_offline_manager::fast_init_tunning_param()
{
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_ndarray_naive_fp32_offline_manager::pick_best_tunning_param(
    const float* src,
    const float* filter,
    float* dst,
    ppl::common::TensorShape& src_shape,
    ppl::common::TensorShape& dst_shape)
{
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_ndarray_naive_fp32_offline_manager::gen_cvt_weights(
    const float* filter,
    const float* bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }
    LOG(DEBUG) << "ndarray naive conv: gen cvt weights";

    const int64_t num_output = param_.num_output;
    const int64_t channels   = param_.channels;
    const int64_t kernel_h   = param_.kernel_h;
    const int64_t kernel_w   = param_.kernel_w;
    const int64_t num_group  = param_.group;

    {
        cvt_bias_size_ = round_up(num_output, 4);
        cvt_bias_      = (float*)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
        memcpy(cvt_bias_, bias, num_output * sizeof(float));
        memset(cvt_bias_ + num_output, 0.f, (cvt_bias_size_ - num_output) * sizeof(float));
    }

    {
        cvt_filter_size_ = conv_naive_cto8c_get_cvt_filter_size_fp32(kernel_h, kernel_w, channels, num_group, num_output);

        cvt_filter_ = (float*)allocator_->Alloc(cvt_filter_size_);
        conv_naive_cto8c_cvt_filter_fp32(filter, kernel_h, kernel_w, num_output, channels, num_group, cvt_filter_);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv
