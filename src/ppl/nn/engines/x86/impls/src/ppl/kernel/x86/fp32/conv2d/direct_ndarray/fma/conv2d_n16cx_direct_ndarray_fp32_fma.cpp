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
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/fma/conv2d_n16cx_direct_ndarray_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/fma/conv2d_n16cx_direct_ndarray_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/array_param_helper.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

static const int64_t ASSUME_L2_BYTES = 256 * 1024;
static const int64_t ASSUME_L2_WAYS = 4;
static const int64_t ASSUME_L3_BYTES = 2048 * 1024;
static const float L2_RATIO = 0.251f;
static const float L3_RATIO = 0.501f;

static const int64_t OC_DATA_BLK = conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_DATA_BLK;
static const int64_t OC_REG_ELTS = conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::OC_REG_ELTS;
static const int64_t OW_KER_BLK = conv2d_n16cx_direct_ndarray_kernel_fp32_fma::config::MAX_W_BLK;

static const int64_t OC_L2_BLK_MAX = 4 * OC_DATA_BLK;
static const int64_t OW_L2_BLK_MAX = 96;

void conv2d_n16cx_direct_ndarray_fp32_fma_executor::init_preproc_param()
{
    schedule_param_.ic_per_grp = conv_param_->channels / conv_param_->group;
    schedule_param_.oc_per_grp = conv_param_->num_output / conv_param_->group;
    schedule_param_.padded_oc = round_up(schedule_param_.oc_per_grp, OC_DATA_BLK);
}

void conv2d_n16cx_direct_ndarray_fp32_fma_executor::cal_kernel_tunning_param()
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

    sp.oc_l2_blk = min<int64_t>(sp.padded_oc, OC_L2_BLK_MAX);
    sp.ow_l2_blk = dst_w;
    if (sp.ow_l2_blk >= 2 * OW_L2_BLK_MAX) sp.ow_l2_blk = round_up(OW_L2_BLK_MAX, OW_KER_BLK);
    else if (sp.ow_l2_blk > 1.5 * OW_L2_BLK_MAX) sp.ow_l2_blk = round_up(div_up(sp.ow_l2_blk, 2), OW_KER_BLK);

    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (ASSUME_L3_BYTES * num_thread) : ppl::common::GetCpuCacheL3()) * L3_RATIO / sizeof(float);

    sp.use_nt_store           = 0;
    const int64_t src_len     = int64_t(batch) * cp.channels * src_h * src_w;
    const int64_t dst_len     = int64_t(batch) * cp.group * sp.padded_oc * dst_h * dst_w;
    const int64_t sum_src_len = (conv_param_->fuse_flag & conv_fuse_flag::SUM) ? int64_t(batch) * cp.group * sp.padded_oc * dst_h * dst_w : 0;
    if (src_len + dst_len + sum_src_len > l3_cap_all_core * 3) {
        sp.use_nt_store = 1;
    }
}

uint64_t conv2d_n16cx_direct_ndarray_fp32_fma_executor::cal_temp_buffer_size()
{
    return 0;
}

ppl::common::RetCode conv2d_n16cx_direct_ndarray_fp32_fma_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::SUM) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n16cx_direct_ndarray_fp32_fma_executor::execute()
{
    if (!conv_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || ((conv_param_->fuse_flag & conv_fuse_flag::SUM) && !sum_src_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (schedule_param_.use_nt_store) {
        return execute_inner<true>();
    } else {
        return execute_inner<false>();
    }
}

template <bool use_nt_store>
ppl::common::RetCode conv2d_n16cx_direct_ndarray_fp32_fma_executor::execute_inner()
{
    const conv2d_fp32_param &cp     = *conv_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int64_t batch = src_shape_->GetDim(0);
    const int64_t src_h = src_shape_->GetDim(2);
    const int64_t src_w = src_shape_->GetDim(3);
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);

    const int64_t src_b_stride = src_shape_->GetDim(1) * src_h * src_w;
    const int64_t src_g_stride = sp.ic_per_grp * src_h * src_w;
    const int64_t src_c_stride = src_h * src_w;
    const int64_t dst_b_stride = round_up(dst_shape_->GetDim(1), OC_DATA_BLK) * dst_h * dst_w;
    const int64_t dst_g_stride = sp.padded_oc * dst_h * dst_w;
    const int64_t flt_c_stride = cp.kernel_h * cp.kernel_w * OC_DATA_BLK;
    const int64_t padded_reg_oc = round_up(sp.oc_per_grp, OC_REG_ELTS);

    const bool with_relu  = cp.fuse_flag & conv_fuse_flag::RELU;
    const bool with_relu6 = cp.fuse_flag & conv_fuse_flag::RELU6;
    const bool with_sum   = cp.fuse_flag & conv_fuse_flag::SUM;

    int64_t sum_src_b_stride = 0;
    if (with_sum) {
        sum_src_b_stride = int64_t(round_up(sum_src_shape_->GetDim(1), OC_DATA_BLK)) * dst_h * dst_w;
    }

    int64_t kernel_flags = 0;
    if (with_relu)  kernel_flags |= conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::RELU;
    if (with_relu6) kernel_flags |= conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::RELU6;
    if (with_sum)   kernel_flags |= conv2d_n16cx_direct_ndarray_kernel_fp32_fma::flag::SUM;

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(5)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t g = 0; g < cp.group; ++g) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR()
#endif
            for (int64_t ocl2 = 0; ocl2 < padded_reg_oc; ocl2 += sp.oc_l2_blk) {
                for (int64_t oh = 0; oh < dst_h; ++oh) {
                    for (int64_t owl2 = 0; owl2 < dst_w; owl2 += sp.ow_l2_blk) {
                        int64_t kernel_param[conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::LENGTH];
                        conv2d_n16cx_direct_ndarray_kernel_fp32_fma ker(kernel_param);
                        array_param_helper ker_p(kernel_param);

                        const int64_t ocl2_eff = min<int64_t>(padded_reg_oc - ocl2, sp.oc_l2_blk);
                        const int64_t owl2_eff = min<int64_t>(dst_w - owl2, sp.ow_l2_blk);
                        const int64_t ih       = oh * cp.stride_h - cp.pad_h;
                        const int64_t iwl2     = owl2 * cp.stride_w - cp.pad_w;
                        const int64_t kh_start = min<int64_t>(max<int64_t>(0 - ih, 0), cp.kernel_h);
                        const int64_t kh_end   = max<int64_t>(min<int64_t>(src_h - ih, cp.kernel_h), 0);

                        const int64_t nt_store_sel   = sp.use_nt_store;
                        int64_t unroll_owl2_start = max(sp.unroll_ow_start, owl2);
                        int64_t unroll_owl2_end   = min(sp.unroll_ow_end, owl2 + owl2_eff);
                        if (unroll_owl2_start >= unroll_owl2_end || unroll_owl2_start < 0 || unroll_owl2_end < 0) {
                            unroll_owl2_start = unroll_owl2_end = owl2 + owl2_eff;
                        }
                        const int64_t owl2_unroll_len  = unroll_owl2_end - unroll_owl2_start;
                        const int64_t owl2_unroll_body = round(owl2_unroll_len, OW_KER_BLK);
                        const int64_t owl2_unroll_tail = owl2_unroll_len - owl2_unroll_body;

                        const float *base_src      = src_ + b * src_b_stride + g * src_g_stride + ih * src_w + iwl2;
                        const float *base_sum_src  = sum_src_ + b * sum_src_b_stride + g * dst_g_stride + ocl2 * dst_h * dst_w + oh * dst_w * OC_DATA_BLK + owl2 * OC_DATA_BLK;
                        float *base_dst            = dst_ + b * dst_b_stride + g * dst_g_stride + ocl2 * dst_h * dst_w + oh * dst_w * OC_DATA_BLK + owl2 * OC_DATA_BLK;
                        const float *base_flt      = cvt_filter_ + (g * sp.padded_oc + ocl2) * sp.ic_per_grp * cp.kernel_h * cp.kernel_w;
                        const float *base_bias     = cvt_bias_ + g * sp.padded_oc + ocl2;

                        ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::CHANNELS_IDX)     = sp.ic_per_grp;
                        ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KH_IDX)           = cp.kernel_h;
                        ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KW_IDX)           = cp.kernel_w;
                        ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SW_IDX)           = cp.stride_w;
                        ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KH_START_IDX)     = kh_start;
                        ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KH_END_IDX)       = kh_end;
                        ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SRC_H_STRIDE_IDX) = src_w;
                        ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SRC_C_STRIDE_IDX) = src_c_stride;
                        ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::FLT_C_STRIDE_IDX) = flt_c_stride;
                        ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::FLAGS_IDX)        = kernel_flags;

                        ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::FLT_PTR_IDX)  = base_flt;
                        ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::BIAS_PTR_IDX) = base_bias;
                        for (int64_t oc = ocl2; oc < ocl2 + ocl2_eff; oc += OC_DATA_BLK) {
                            const int64_t oc_eff = min<int64_t>(ocl2 + ocl2_eff - oc, OC_DATA_BLK);
                            const int64_t oc_reg = div_up(oc_eff, OC_REG_ELTS);
                            ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SRC_PTR_IDX)     = base_src;
                            ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::SUM_SRC_PTR_IDX) = base_sum_src;
                            ker_p.pick<float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::DST_PTR_IDX)           = base_dst;

                            for (int64_t ow = owl2; ow < unroll_owl2_start; ++ow) {
                                const int64_t iw       = ow * cp.stride_w - cp.pad_w;
                                const int64_t kw_start = min<int64_t>(max<int64_t>(0 - iw, 0), cp.kernel_w);
                                const int64_t kw_end   = max<int64_t>(min<int64_t>(src_w - iw, cp.kernel_w), 0);
                                ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KW_START_IDX) = kw_start;
                                ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KW_END_IDX)   = kw_end;
                                ker.execute_border(nt_store_sel, oc_reg);
                            }

                            if (owl2_unroll_body) {
                                ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::DST_WIDTH_IDX) = owl2_unroll_body;
                                ker.execute(nt_store_sel, oc_reg, OW_KER_BLK);
                            }
                            if (owl2_unroll_tail) {
                                ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::DST_WIDTH_IDX) = owl2_unroll_tail;
                                ker.execute(nt_store_sel, oc_reg, owl2_unroll_tail);
                            }

                            for (int64_t ow = unroll_owl2_end; ow < owl2 + owl2_eff; ++ow) {
                                const int64_t iw       = ow * cp.stride_w - cp.pad_w;
                                const int64_t kw_start = min<int64_t>(max<int64_t>(0 - iw, 0), cp.kernel_w);
                                const int64_t kw_end   = max<int64_t>(min<int64_t>(src_w - iw, cp.kernel_w), 0);
                                ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KW_START_IDX) = kw_start;
                                ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::KW_END_IDX)   = kw_end;
                                ker.execute_border(nt_store_sel, oc_reg);
                            }
                            ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::FLT_PTR_IDX)  += OC_DATA_BLK * sp.ic_per_grp * cp.kernel_h * cp.kernel_w;
                            ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_fma::param_def::BIAS_PTR_IDX) += OC_DATA_BLK;
                            base_sum_src += OC_DATA_BLK * dst_h * dst_w;
                            base_dst     += OC_DATA_BLK * dst_h * dst_w;
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

ppl::common::RetCode conv2d_n16cx_direct_ndarray_fp32_fma_manager::gen_cvt_weights(const float *filter, const float *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t oc_per_grp = param_.num_output / param_.group;
    const int64_t ic_per_grp = param_.channels / param_.group;
    const int64_t padded_oc = round_up(oc_per_grp, OC_DATA_BLK);

    cvt_bias_size_ = param_.group * padded_oc;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    for (int64_t g = 0; g < param_.group; ++g) {
        memcpy(cvt_bias_ + g * padded_oc, bias + g * oc_per_grp, oc_per_grp * sizeof(float));
        memset(cvt_bias_ + g * padded_oc + oc_per_grp, 0, (padded_oc - oc_per_grp) * sizeof(float));
    }

    ppl::nn::TensorShape filter_shape;
    filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
    filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    filter_shape.Reshape({1, oc_per_grp, ic_per_grp * param_.kernel_h * param_.kernel_w , 1});

    cvt_filter_size_ = param_.group * padded_oc * ic_per_grp * param_.kernel_h * param_.kernel_w;
    cvt_filter_      = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }
    ppl::common::RetCode status = ppl::common::RC_OTHER_ERROR;
    for (int64_t g = 0; g < param_.group; ++g) {
        status = reorder_ndarray_n16cx_fp32_avx(
                    &filter_shape,
                    filter + g * oc_per_grp * ic_per_grp * param_.kernel_h * param_.kernel_w,
                    cvt_filter_ + g * padded_oc * ic_per_grp * param_.kernel_h * param_.kernel_w);
        if (status != ppl::common::RC_SUCCESS)
            break;
    }

    return status;
}

bool conv2d_n16cx_direct_ndarray_fp32_fma_manager::is_supported()
{
    bool small_channels = param_.channels / param_.group < OC_DATA_BLK;
    bool aligned_num_output = param_.group == 1 || param_.num_output / param_.group % OC_DATA_BLK == 0;
    return small_channels && aligned_num_output && param_.dilation_h == 1 && param_.dilation_w == 1;
}

conv2d_fp32_executor *conv2d_n16cx_direct_ndarray_fp32_fma_manager::gen_executor()
{
    return new conv2d_n16cx_direct_ndarray_fp32_fma_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86
