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
#include "ppl/kernel/riscv/fp32/conv2d/direct_gemm/vec128/conv2d_n4cx_direct_gemm_fp32_vec128.h"
#include "ppl/kernel/riscv/fp32/conv2d/tile_gemm/conv2d_generic_tile_gemm_fp32_vec128.h"
#include "ppl/kernel/riscv/fp32/conv2d/common/conv2d_shell_fp32.h"

namespace ppl { namespace kernel { namespace riscv {

uint64_t conv2d_n4cx_direct_gemm_fp32_runtime_executor::cal_temp_buffer_size()
{
    return 4;
}

void conv2d_n4cx_direct_gemm_fp32_runtime_executor::adjust_tunning_param() {}

ppl::common::RetCode conv2d_n4cx_direct_gemm_fp32_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    adjust_tunning_param();
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n4cx_direct_gemm_fp32_runtime_executor::execute()
{
    const conv2d_common_param& cp = *conv_param_;

    if (src_ == nullptr || cvt_bias_ == nullptr || cvt_filter_ == nullptr || temp_buffer_ == nullptr ||
        dst_ == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }

    int64_t pad_channels   = round_up(conv_param_->channels, 4);
    int64_t pad_num_output = round_up(conv_param_->num_output, 4);
    int64_t dst_h          = dst_shape_->GetDim(2);
    int64_t dst_w          = dst_shape_->GetDim(3);
    auto gemm_func         = conv2d_gemm_select_xcto4c_kernel_fp32_vec128<4, true>(pad_num_output, dst_h * dst_w);
    gemm_func(cvt_filter_, src_, dst_, pad_num_output, dst_h * dst_w, pad_channels);

    conv2d_n4cx_mem_dst_blk_trans_fp32_vec128<false>(
        dst_,
        dst_h,
        dst_w,

        dst_,
        dst_h,
        dst_w,

        pad_num_output,
        dst_h,
        dst_w,

        cvt_bias_);

    return ppl::common::RC_SUCCESS;
}

bool conv2d_n4cx_direct_gemm_fp32_offline_manager::is_supported()
{
    return true;
}

ppl::common::RetCode conv2d_n4cx_direct_gemm_fp32_offline_manager::fast_init_tunning_param()
{
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n4cx_direct_gemm_fp32_offline_manager::pick_best_tunning_param(
    const float* src,
    const float* filter,
    float* dst,
    ppl::common::TensorShape& src_shape,
    ppl::common::TensorShape& dst_shape)
{
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n4cx_direct_gemm_fp32_offline_manager::gen_cvt_weights(const float* filter,
                                                                                   const float* bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    if (param_.kernel_h != 1 || param_.kernel_w != 1 || param_.group != 1 || param_.stride_h != 1 ||
        param_.stride_w != 1 || param_.pad_h != 0 || param_.pad_w != 0) {
        return ppl::common::RC_UNSUPPORTED;
    }

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
        cvt_filter_size_ = conv2d_nxcx_conv_tile_gemm_get_cvt_filter_size_fp32_vec128<4>(kernel_h, kernel_w, channels, num_output, num_group);

        cvt_filter_ = (float*)allocator_->Alloc(cvt_filter_size_);
        conv2d_nxcx_conv_tile_gemm_cvt_filter_fp32_vec128<4>(filter, kernel_h, kernel_w, num_output, channels, num_group, round_up(num_output, 4), round_up(channels, 4), cvt_filter_);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv
