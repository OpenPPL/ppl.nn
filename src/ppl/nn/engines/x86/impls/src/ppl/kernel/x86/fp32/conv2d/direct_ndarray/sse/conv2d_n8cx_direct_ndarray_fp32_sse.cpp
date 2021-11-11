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
#include <nmmintrin.h>
#include <string.h>

#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/common/sse_tools.h"
#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/sse/conv2d_n8cx_direct_ndarray_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/sse/conv2d_n8cx_direct_ndarray_kernel_fp32_sse.h"
#include "ppl/common/sys.h"

#define ASSUME_L2_BYTES() (256 * 1024)
#define ASSUME_L2_WAYS()  4
#define ASSUME_L3_BYTES() (2048 * 1024)
#define L2_RATIO()        0.251
#define L3_RATIO()        0.501

#define OW_KR_BLK() MAX_OW_RF()
#define OC_KR_BLK() (2 * OC_DT_BLK())
#define OC_L2_BLK_MAX() (4 * OC_DT_BLK())

namespace ppl { namespace kernel { namespace x86 {

void conv2d_n8cx_direct_ndarray_fp32_sse_executor::init_preproc_param()
{
    schedule_param_.ic_per_gp = conv_param_->channels / conv_param_->group;
    schedule_param_.oc_per_gp = conv_param_->num_output / conv_param_->group;
    schedule_param_.padded_oc = round_up(schedule_param_.oc_per_gp, OC_DT_BLK());
}

void conv2d_n8cx_direct_ndarray_fp32_sse_executor::cal_kernel_tunning_param()
{
    const conv2d_fp32_param &cp = *conv_param_;
    kernel_schedule_param &sp   = schedule_param_;

    const int64_t num_thread = PPL_OMP_MAX_THREADS();
    const int64_t batch      = src_shape_->GetDim(0);
    const int64_t src_h      = src_shape_->GetDim(2);
    const int64_t src_w      = src_shape_->GetDim(3);
    const int64_t dst_h      = dst_shape_->GetDim(2);
    const int64_t dst_w      = dst_shape_->GetDim(3);

    sp.unroll_ow_start = -1;
    sp.unroll_ow_end = -1;
    for (int64_t ow = 0; ow < dst_w; ++ow) {
        if (ow * cp.stride_w - cp.pad_w >= 0) {
            sp.unroll_ow_start = ow;
            break;
        }
    }
    for (int64_t ow = dst_w - 1; ow >= 0; --ow) {
        if (ow * cp.stride_w - cp.pad_w + cp.kernel_w <= src_w) {
            sp.unroll_ow_end = ow + 1;
            break;
        }
    }
    if (sp.unroll_ow_start >= sp.unroll_ow_end || sp.unroll_ow_start < 0 || sp.unroll_ow_end < 0) {
        sp.unroll_ow_start = sp.unroll_ow_end = dst_w;
    }

    sp.oc_l2_blk = min<int64_t>(sp.padded_oc, OC_L2_BLK_MAX());

    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (ASSUME_L3_BYTES() * num_thread) : ppl::common::GetCpuCacheL3()) * L3_RATIO() / sizeof(float);

    sp.use_nt_store           = 0;
    const int64_t src_len     = int64_t(batch) * cp.channels * src_h * src_w;
    const int64_t dst_len     = int64_t(batch) * cp.group * sp.padded_oc * dst_h * dst_w;
    const int64_t sum_src_len = (conv_param_->fuse_flag & conv_fuse_flag::SUM) ? int64_t(batch) * cp.group * sp.padded_oc * dst_h * dst_w : 0;
    if (src_len + dst_len + sum_src_len > l3_cap_all_core * 3) {
        sp.use_nt_store = 1;
    }
}

uint64_t conv2d_n8cx_direct_ndarray_fp32_sse_executor::cal_temp_buffer_size()
{
    return 64u;
}

ppl::common::RetCode conv2d_n8cx_direct_ndarray_fp32_sse_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::SUM) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_direct_ndarray_fp32_sse_executor::execute()
{
    if (!conv_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || ((conv_param_->fuse_flag & conv_fuse_flag::SUM) && !sum_src_) || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const conv2d_fp32_param &cp     = *conv_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int64_t batch = src_shape_->GetDim(0);
    const int64_t src_h = src_shape_->GetDim(2);
    const int64_t src_w = src_shape_->GetDim(3);
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);

    const int64_t src_b_stride = int64_t(src_shape_->GetDim(1)) * src_h * src_w;
    const int64_t src_g_stride = int64_t(sp.ic_per_gp) * src_h * src_w;
    const int64_t src_c_stride = int64_t(src_h) * src_w;
    const int64_t dst_b_stride = int64_t(round_up(dst_shape_->GetDim(1), OC_DT_BLK())) * dst_h * dst_w;
    const int64_t dst_g_stride = int64_t(sp.padded_oc) * dst_h * dst_w;
    const int64_t dst_ocb_stride = int64_t(dst_h) * dst_w * OC_DT_BLK();
    const int64_t flt_g_stride = int64_t(sp.padded_oc) * sp.ic_per_gp * cp.kernel_h * cp.kernel_w;
    const int64_t flt_ocb_stride = int64_t(sp.ic_per_gp) * cp.kernel_h * cp.kernel_w * OC_DT_BLK();

    const bool with_sum = cp.fuse_flag & conv_fuse_flag::SUM;
    const bool with_relu  = cp.fuse_flag & conv_fuse_flag::RELU;
    const bool with_relu6 = cp.fuse_flag & conv_fuse_flag::RELU6;

    int64_t sum_src_b_stride = 0;
    if (with_sum) {
        sum_src_b_stride = int64_t(round_up(sum_src_shape_->GetDim(1), OC_DT_BLK())) * dst_h * dst_w;
    }

    int64_t share_param[SHAR_PARAM_LEN()];
    share_param[CHANNELS_IDX()] = sp.ic_per_gp;
    share_param[SRC_C_STRIDE_IDX()] = src_c_stride;
    share_param[SRC_H_STRIDE_IDX()] = src_w;
    share_param[HIS_OCB_STRIDE_IDX()] = dst_ocb_stride;
    share_param[DST_OCB_STRIDE_IDX()] = dst_ocb_stride;
    share_param[FLT_OCB_STRIDE_IDX()] = flt_ocb_stride;
    share_param[SW_IDX()] = cp.stride_w;
    share_param[KH_IDX()] = cp.kernel_h;
    share_param[KW_IDX()] = cp.kernel_w;
    share_param[KW_IDX()] = cp.kernel_w;
    {
        uint64_t kernel_flags = 0;
        if (with_sum) kernel_flags |= KERNEL_FLAG_AD_BIAS();
        if (!with_sum) kernel_flags |= KERNEL_FLAG_LD_BIAS();
        if (with_relu) kernel_flags |= KERNEL_FLAG_RELU();
        if (with_relu6) kernel_flags |= KERNEL_FLAG_RELU6();
        share_param[FLAGS_IDX()] = kernel_flags;
    }
    const int32_t nt_store_sel = sp.use_nt_store;

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(4)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t g = 0; g < cp.group; ++g) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR()
#endif
            for (int64_t ocl2 = 0; ocl2 < sp.padded_oc; ocl2 += sp.oc_l2_blk) {
                for (int64_t oh = 0; oh < dst_h; ++oh) {
                    int64_t private_param[PRIV_PARAM_LEN()];
                    const int64_t ocl2_eff = min<int64_t>(sp.padded_oc - ocl2, sp.oc_l2_blk);
                    const int64_t ih       = oh * cp.stride_h - cp.pad_h;
                    private_param[KH_START_IDX()] = min<int64_t>(max<int64_t>(0 - ih, 0), cp.kernel_h - 1);
                    private_param[KH_END_IDX()]   = max<int64_t>(min<int64_t>(src_h - ih, cp.kernel_h), 0);
                    for (int64_t oc = ocl2; oc < ocl2 + ocl2_eff; oc += OC_KR_BLK()) {
                        const int64_t oc_eff = min<int64_t>(ocl2 + ocl2_eff - oc, OC_KR_BLK());
                        const int64_t oc_sel = div_up(oc_eff, OC_DT_BLK()) - 1;
                        PICK_PARAM(const float*, private_param, SRC_IDX()) = src_ + b * src_b_stride + g * src_g_stride + ih * src_w - cp.pad_w;
                        PICK_PARAM(const float*, private_param, HIS_IDX()) = sum_src_ + b * sum_src_b_stride + g * dst_g_stride + oc * dst_h * dst_w + oh * dst_w * OC_DT_BLK();
                        PICK_PARAM(float*, private_param, DST_IDX())       = dst_ + b * dst_b_stride + g * dst_g_stride + oc * dst_h * dst_w + oh * dst_w * OC_DT_BLK();
                        PICK_PARAM(const float*, private_param, FLT_IDX())  = cvt_filter_ + g * flt_g_stride + oc * sp.ic_per_gp * cp.kernel_h * cp.kernel_w;
                        PICK_PARAM(const float*, private_param, BIAS_IDX()) = cvt_bias_ + g * sp.padded_oc + oc;

                        for (int64_t ow = 0; ow < sp.unroll_ow_start; ++ow) {
                            const int64_t iw              = ow * cp.stride_w - cp.pad_w;
                            private_param[KW_START_IDX()] = min<int64_t>(max<int64_t>(0 - iw, 0), cp.kernel_w - 1);
                            private_param[KW_END_IDX()]   = max<int64_t>(min<int64_t>(src_w - iw, cp.kernel_w), 0);
                            conv2d_n8cx_direct_ndarray_kernel_fp32_sse_pad_table[nt_store_sel][oc_sel](share_param, private_param);
                        }

                        const int64_t ow_unroll_len  = sp.unroll_ow_end - sp.unroll_ow_start;
                        const int64_t ow_unroll_body = round(ow_unroll_len, OW_KR_BLK());
                        const int64_t ow_unroll_tail = ow_unroll_len - ow_unroll_body;
                        if (ow_unroll_body) {
                            private_param[OW_IDX()] = ow_unroll_body;
                            switch (oc_sel) {
                                case 0: conv2d_n8cx_direct_ndarray_kernel_fp32_sse_o8_table[nt_store_sel][OW_KR_BLK() - 1](share_param, private_param); break;
                                case 1: conv2d_n8cx_direct_ndarray_kernel_fp32_sse_o16_table[nt_store_sel][OW_KR_BLK() - 1](share_param, private_param); break;
                            }
                        }
                        if (ow_unroll_tail) {
                            private_param[OW_IDX()] = ow_unroll_tail;
                            switch (oc_sel) {
                                case 0: conv2d_n8cx_direct_ndarray_kernel_fp32_sse_o8_table[nt_store_sel][ow_unroll_tail - 1](share_param, private_param); break;
                                case 1: conv2d_n8cx_direct_ndarray_kernel_fp32_sse_o16_table[nt_store_sel][ow_unroll_tail - 1](share_param, private_param); break;
                            }
                        }

                        for (int64_t ow = sp.unroll_ow_end; ow < dst_w; ++ow) {
                            const int64_t iw       = ow * cp.stride_w - cp.pad_w;
                            private_param[KW_START_IDX()] = min<int64_t>(max<int64_t>(0 - iw, 0), cp.kernel_w - 1);
                            private_param[KW_END_IDX()]   = max<int64_t>(min<int64_t>(src_w - iw, cp.kernel_w), 0);
                            conv2d_n8cx_direct_ndarray_kernel_fp32_sse_pad_table[nt_store_sel][oc_sel](share_param, private_param);
                        }
                    }
                }
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

ppl::common::RetCode conv2d_n8cx_direct_ndarray_fp32_sse_manager::gen_cvt_weights(const float *filter, const float *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t oc_per_gp = param_.num_output / param_.group;
    const int64_t ic_per_gp = param_.channels / param_.group;
    const int64_t padded_oc = round_up(oc_per_gp, OC_DT_BLK());

    cvt_bias_size_ = param_.group * padded_oc;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    for (int64_t g = 0; g < param_.group; ++g) {
        memcpy(cvt_bias_ + g * padded_oc, bias + g * oc_per_gp, oc_per_gp * sizeof(float));
        memset(cvt_bias_ + g * padded_oc + oc_per_gp, 0, (padded_oc - oc_per_gp) * sizeof(float));
    }

    ppl::nn::TensorShape filter_shape;
    filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
    filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    filter_shape.Reshape({1, oc_per_gp, ic_per_gp * param_.kernel_h * param_.kernel_w , 1});

    cvt_filter_size_ = param_.group * padded_oc * ic_per_gp * param_.kernel_h * param_.kernel_w;
    cvt_filter_      = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }
    ppl::common::RetCode status = ppl::common::RC_OTHER_ERROR;
    for (int64_t g = 0; g < param_.group; ++g) {
        status = reorder_ndarray_n8cx_fp32(
                    &filter_shape,
                    filter + g * oc_per_gp * ic_per_gp * param_.kernel_h * param_.kernel_w,
                    cvt_filter_ + g * padded_oc * ic_per_gp * param_.kernel_h * param_.kernel_w);
        if (status != ppl::common::RC_SUCCESS)
            break;
    }

    return status;
}

bool conv2d_n8cx_direct_ndarray_fp32_sse_manager::is_supported()
{
    bool small_channels = param_.channels / param_.group < 2 * OC_DT_BLK();
    bool aligned_num_output = param_.group == 1 || param_.num_output / param_.group % OC_DT_BLK() == 0;
    return small_channels && aligned_num_output && param_.dilation_h == 1 && param_.dilation_w == 1;
}

conv2d_fp32_executor *conv2d_n8cx_direct_ndarray_fp32_sse_manager::gen_executor()
{
    return new conv2d_n8cx_direct_ndarray_fp32_sse_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86
