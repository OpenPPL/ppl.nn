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

#include <new>
#include <immintrin.h>
#include <string.h>

#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/common/sse_tools.h"
#include "ppl/kernel/x86/fp32/conv2d/depthwise/sse/conv2d_n8cx_depthwise_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/depthwise/sse/conv2d_n8cx_depthwise_kernel_fp32_sse.h"
#include "ppl/common/sys.h"

#define ASSUME_L2_BYTES() (256 * 1024)
#define ASSUME_L2_WAYS()  4
#define ASSUME_L3_BYTES() (2048 * 1024)
#define L2_RATIO()        0.251
#define L3_RATIO()        0.501

#define PADDING_POLICY_NOPAD() 0
#define PADDING_POLICY_PREPAD() 1

namespace ppl { namespace kernel { namespace x86 {

void conv2d_n8cx_depthwise_fp32_sse_executor::init_preproc_param()
{
    schedule_param_.padded_ch = round_up(conv_param_->group, CH_DT_BLK());
    schedule_param_.ow_kr_blk = MAX_OW_RF();
}

void conv2d_n8cx_depthwise_fp32_sse_executor::cal_kernel_tunning_param()
{
    const conv2d_fp32_param &cp = *conv_param_;
    kernel_schedule_param &sp   = schedule_param_;

    const int64_t num_thread = PPL_OMP_MAX_THREADS();
    const int64_t batch      = src_shape_->GetDim(0);
    const int64_t src_h      = src_shape_->GetDim(2);
    const int64_t src_w      = src_shape_->GetDim(3);
    const int64_t dst_h      = dst_shape_->GetDim(2);
    const int64_t dst_w      = dst_shape_->GetDim(3);
    const int64_t ext_kernel_w = (cp.kernel_w - 1) * cp.dilation_w + 1;

    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (ASSUME_L3_BYTES() * num_thread) : ppl::common::GetCpuCacheL3()) * L3_RATIO() / sizeof(float);

    const int64_t src_len     = batch * sp.padded_ch * src_h * src_w;
    const int64_t dst_len     = batch * sp.padded_ch * dst_h * dst_w;
    const int64_t sum_src_len = (conv_param_->fuse_flag & conv_fuse_flag::sum) ? int64_t(batch) * sp.padded_ch * dst_h * dst_w : 0;
    const int64_t tot_data_len = src_len + dst_len + sum_src_len;

    if (tot_data_len < l3_cap_all_core
        && dst_w <= 14
        && cp.stride_w < dst_w && cp.pad_w != 0
        && cp.dilation_w < dst_w
        && cp.sparse_level() < 0.4f
        && num_thread < 8) {
        sp.padding_policy = PADDING_POLICY_PREPAD();
    } else {
        sp.padding_policy = PADDING_POLICY_NOPAD();
    }

    sp.unroll_ow_start = -1;
    sp.unroll_ow_end = -1;
    if (sp.padding_policy == PADDING_POLICY_NOPAD()) {
        for (int64_t ow = 0; ow < dst_w; ++ow) {
            if (ow * cp.stride_w - cp.pad_w >= 0) {
                sp.unroll_ow_start = ow;
                break;
            }
        }
        for (int64_t ow = dst_w - 1; ow >= 0; --ow) {
            if (ow * cp.stride_w - cp.pad_w + ext_kernel_w <= src_w) {
                sp.unroll_ow_end = ow + 1;
                break;
            }
        }
        if (sp.unroll_ow_start >= sp.unroll_ow_end || sp.unroll_ow_start < 0 || sp.unroll_ow_end < 0) {
            sp.unroll_ow_start = sp.unroll_ow_end = dst_w;
        }
    } else {
        sp.unroll_ow_start = 0;
        sp.unroll_ow_end = dst_w;
    }

    sp.use_nt_store = 0;
    if (tot_data_len > l3_cap_all_core * 3) {
        sp.use_nt_store = 1;
    }
}

uint64_t conv2d_n8cx_depthwise_fp32_sse_executor::cal_temp_buffer_size()
{
    if (schedule_param_.padding_policy == PADDING_POLICY_NOPAD()) {
        return 64u;
    } else {
        const int64_t src_h         = src_shape_->GetDim(2);
        const int64_t src_w         = src_shape_->GetDim(3);
        const uint64_t padded_src_hw = uint64_t(src_h) * (src_w + 2 * conv_param_->pad_w);
        return padded_src_hw * CH_DT_BLK() * PPL_OMP_MAX_THREADS() * sizeof(float);
    }
}

ppl::common::RetCode conv2d_n8cx_depthwise_fp32_sse_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_depthwise_fp32_sse_executor::execute()
{
    if (!conv_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_) || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const conv2d_fp32_param &cp     = *conv_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int64_t batch = src_shape_->GetDim(0);
    const int64_t src_h = src_shape_->GetDim(2);
    const int64_t src_w = src_shape_->GetDim(3);
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);
    const int64_t padded_src_w = (src_w + 2 * cp.pad_w);

    const int64_t padded_src_c = round_up(src_shape_->GetDim(1), CH_DT_BLK());
    const int64_t padded_dst_c = round_up(dst_shape_->GetDim(1), CH_DT_BLK());

    const int64_t ext_kernel_h = (cp.kernel_h - 1) * cp.dilation_h + 1;
    const int64_t ext_kernel_w = (cp.kernel_w - 1) * cp.dilation_w + 1;

    const int64_t src_b_stride = padded_src_c * src_h * src_w;
    const int64_t src_h_stride = src_w * CH_DT_BLK();
    const int64_t src_sw_stride = cp.stride_w * CH_DT_BLK();
    const int64_t dst_b_stride = padded_dst_c * dst_h * dst_w;
    const int64_t padded_src_h_stride = (src_w + 2 * cp.pad_w) * CH_DT_BLK();

    const bool with_sum = cp.fuse_flag & conv_fuse_flag::sum;
    const bool with_relu  = cp.fuse_flag & conv_fuse_flag::relu;
    const bool with_relu6 = cp.fuse_flag & conv_fuse_flag::relu6;

    int64_t sum_src_b_stride = 0;
    if (with_sum) {
        sum_src_b_stride = int64_t(round_up(sum_src_shape_->GetDim(1), CH_DT_BLK())) * dst_h * dst_w;
    }

    int64_t share_param[SHAR_PARAM_LEN()];
    share_param[SRC_SW_STRIDE_IDX()] = src_sw_stride;
    share_param[SRC_DH_STRIDE_IDX()] = cp.dilation_h * (sp.padding_policy == PADDING_POLICY_PREPAD() ? padded_src_h_stride : src_h_stride);
    share_param[SRC_DW_STRIDE_IDX()] = cp.dilation_w * CH_DT_BLK();
    share_param[KW_IDX()] = cp.kernel_w;
    {
        uint64_t kernel_flags = 0;
        if (with_sum) kernel_flags |= KERNEL_FLAG_SUM();
        if (with_relu) kernel_flags |= KERNEL_FLAG_RELU();
        if (with_relu6) kernel_flags |= KERNEL_FLAG_RELU6();
        share_param[FLAGS_IDX()] = kernel_flags;
    }
    const int32_t nt_store_sel = sp.use_nt_store;
    const int32_t stride_w_sel = cp.stride_w > 2 ? 0: cp.stride_w;

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t bc = 0; bc < batch * sp.padded_ch; bc += CH_DT_BLK()) {
        int64_t private_param[PRIV_PARAM_LEN()];
        const int64_t b           = bc / sp.padded_ch;
        const int64_t c           = bc % sp.padded_ch;
        const float *base_src     = src_ + b * src_b_stride + c * src_h * src_w;
        const float *base_sum_src = sum_src_ + b * sum_src_b_stride + c * dst_h * dst_w;
        float *base_dst           = dst_ + b * dst_b_stride + c * dst_h * dst_w;

        PICK_PARAM(const float*, private_param, FLT_IDX()) = cvt_filter_ + c * cp.kernel_h * cp.kernel_w;
        PICK_PARAM(const float*, private_param, BIAS_IDX()) = cvt_bias_ + c;

        int64_t base_src_h_stride = src_h_stride;
        if (sp.padding_policy == PADDING_POLICY_PREPAD()) {
            const int64_t padded_src_hw = int64_t(src_h) * padded_src_w;
            float *padded_src = reinterpret_cast<float*>(temp_buffer_) + PPL_OMP_THREAD_ID() * padded_src_hw * CH_DT_BLK();
            float *l_padded_src = padded_src;
            for (int64_t ih = 0; ih < src_h; ++ih) {
                memset32_sse(l_padded_src, 0, cp.pad_w * CH_DT_BLK());
                l_padded_src += cp.pad_w * CH_DT_BLK();
                memcpy32_sse(l_padded_src, base_src, base_src_h_stride);
                l_padded_src += base_src_h_stride;
                base_src += base_src_h_stride;
                memset32_sse(l_padded_src, 0, cp.pad_w * CH_DT_BLK());
                l_padded_src += cp.pad_w * CH_DT_BLK();
            }
            base_src = padded_src + cp.pad_w * CH_DT_BLK();
            base_src_h_stride = padded_src_h_stride;
        }

        const int64_t ow_unroll_len  = sp.unroll_ow_end - sp.unroll_ow_start;
        const int64_t ow_unroll_body = round(ow_unroll_len, sp.ow_kr_blk);
        const int64_t ow_unroll_tail = ow_unroll_len - ow_unroll_body;

        for (int64_t oh = 0; oh < dst_h; ++oh) {
            const int64_t ih = oh * cp.stride_h - cp.pad_h;
            if (cp.dilation_h == 1) {
                private_param[KH_START_IDX()] = min<int64_t>(max<int64_t>(0 - ih, 0), cp.kernel_h - 1);
                private_param[KH_END_IDX()]   = max<int64_t>(min<int64_t>(src_h - ih, cp.kernel_h), 0);
            } else {
                private_param[KH_START_IDX()] = div_up(min<int64_t>(max<int64_t>(0 - ih, 0), ext_kernel_h - 1), cp.dilation_h);
                private_param[KH_END_IDX()]   = div_up(max<int64_t>(min<int64_t>(src_h - ih, ext_kernel_h), 0), cp.dilation_h);
            }

            PICK_PARAM(const float*, private_param, SRC_IDX())     = base_src + ih * base_src_h_stride - cp.pad_w * CH_DT_BLK();
            PICK_PARAM(const float*, private_param, SUM_SRC_IDX()) = base_sum_src + oh * dst_w * CH_DT_BLK();
            PICK_PARAM(float*, private_param, DST_IDX())           = base_dst + oh * dst_w * CH_DT_BLK();

            for (int64_t ow = 0; ow < sp.unroll_ow_start; ++ow) {
                const int64_t iw = ow * cp.stride_w - cp.pad_w;
                if (cp.dilation_w == 1) { // avoid too much index compute
                    private_param[KW_START_IDX()] = min<int64_t>(max<int64_t>(0 - iw, 0), cp.kernel_w - 1);
                    private_param[KW_END_IDX()]   = max<int64_t>(min<int64_t>(src_w - iw, cp.kernel_w), 0);
                } else {
                    private_param[KW_START_IDX()] = div_up(min<int64_t>(max<int64_t>(0 - iw, 0), ext_kernel_w - 1), cp.dilation_w);
                    private_param[KW_END_IDX()]   = div_up(max<int64_t>(min<int64_t>(src_w - iw, ext_kernel_w), 0), cp.dilation_w);
                }
                conv2d_n8cx_depthwise_kernel_fp32_sse_pad_table[nt_store_sel](share_param, private_param);
            }

            if (ow_unroll_body) {
                private_param[OW_IDX()] = ow_unroll_body;
                conv2d_n8cx_depthwise_kernel_fp32_sse_blk_table[nt_store_sel][stride_w_sel][sp.ow_kr_blk - 1](share_param, private_param);
            }
            if (ow_unroll_tail) {
                private_param[OW_IDX()] = ow_unroll_tail;
                conv2d_n8cx_depthwise_kernel_fp32_sse_blk_table[nt_store_sel][stride_w_sel][ow_unroll_tail - 1](share_param, private_param);
            }

            for (int64_t ow = sp.unroll_ow_end; ow < dst_w; ++ow) {
                const int64_t iw = ow * cp.stride_w - cp.pad_w;
                if (cp.dilation_w == 1) {
                    private_param[KW_START_IDX()] = min<int64_t>(max<int64_t>(0 - iw, 0), cp.kernel_w - 1);
                    private_param[KW_END_IDX()]   = max<int64_t>(min<int64_t>(src_w - iw, cp.kernel_w), 0);
                } else {
                    private_param[KW_START_IDX()] = div_up(min<int64_t>(max<int64_t>(0 - iw, 0), ext_kernel_w - 1), cp.dilation_w);
                    private_param[KW_END_IDX()]   = div_up(max<int64_t>(min<int64_t>(src_w - iw, ext_kernel_w), 0), cp.dilation_w);
                }
                conv2d_n8cx_depthwise_kernel_fp32_sse_pad_table[nt_store_sel](share_param, private_param);
            }
        }
    }
    if (sp.use_nt_store) {
        PRAGMA_OMP_PARALLEL()
        {
            _mm_sfence();
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_depthwise_fp32_sse_manager::gen_cvt_weights(const float *filter, const float *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t channels  = param_.group;
    const int64_t padded_ch = round_up(channels, CH_DT_BLK());

    cvt_bias_size_ = padded_ch;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }
    memcpy(cvt_bias_, bias, channels * sizeof(float));
    memset(cvt_bias_ + channels, 0, (padded_ch - channels) * sizeof(float));

    cvt_filter_size_ = padded_ch * param_.kernel_h * param_.kernel_w;
    cvt_filter_      = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    ppl::nn::TensorShape filter_shape;
    filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
    filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    filter_shape.Reshape({1, channels, param_.kernel_h, param_.kernel_w});

    return reorder_ndarray_n8cx_fp32(&filter_shape, filter, cvt_filter_);
}

bool conv2d_n8cx_depthwise_fp32_sse_manager::is_supported()
{
    return param_.is_depthwise();
}

conv2d_fp32_executor *conv2d_n8cx_depthwise_fp32_sse_manager::gen_executor()
{
    return new conv2d_n8cx_depthwise_fp32_sse_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86
