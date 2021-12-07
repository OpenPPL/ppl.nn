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

#include <string.h>
#include <vector>

#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/fma/conv2d_n16cx_gemm_direct_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/fma/conv2d_n16cx_gemm_direct_kernel_fp32_fma.h"
#include "ppl/kernel/x86/fp32/pd_conv2d/fma/pd_conv2d_n16cx_gemm_direct_fp32_fma.h"
#include "ppl/kernel/x86/fp32/pd_conv2d/fma/pd_conv2d_n16cx_depthwise_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/array_param_helper.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

static const int64_t ASSUME_L2_BYTES = 256 * 1024;
static const int64_t ASSUME_L2_WAYS = 4;
static const int64_t ASSUME_L3_BYTES = 2048 * 1024;
static const float L2_RATIO = 0.251f;
static const float L3_RATIO = 0.501f;

static const int64_t IC_DATA_BLK = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::IC_DATA_BLK;
static const int64_t OC_DATA_BLK = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_DATA_BLK;
static const int64_t OC_REG_ELTS = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS;

static const int64_t IC_L2_BLK_MAX_LARGE = 16 * IC_DATA_BLK; // preserve for tuning
static const int64_t IC_L2_BLK_MAX_SMALL = 16 * IC_DATA_BLK;
static const float IC_L2_BLK_TAIL_RATIO = 0.251f;
static const int64_t OC_L2_BLK_MAX_LARGE = 16 * OC_DATA_BLK; // preserve for tuning
static const int64_t OC_L2_BLK_MAX_SMALL = 16 * OC_DATA_BLK;
static const int64_t OC_L2_BLK_MIN = 1 * OC_DATA_BLK;
static const int64_t OH_L2_BLK_MIN = 32;

static const int64_t GD_KERNEL_BLK_MAX = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::MAX_S_REGS;
static const int64_t GD_KERNEL_BLK_MIN = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::MAX_S_REGS - 2;

static const int64_t EXEC_MODE_FUSE = 0;
static const int64_t EXEC_MODE_SEPARATE = 1;

int64_t pd_conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_ic_l2_blk(const conv2d_fp32_param &param)
{
    return conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_ic_l2_blk(param);
}

void pd_conv2d_n16cx_gemm_direct_fp32_fma_executor::init_preproc_param()
{
    auto gd_param = conv2d_executor_->conv_param();
    schedule_param_.ic_per_grp = gd_param->channels / gd_param->group;
    schedule_param_.oc_per_grp = gd_param->num_output / gd_param->group;
    schedule_param_.padded_ic = round_up(schedule_param_.ic_per_grp, IC_DATA_BLK);
    schedule_param_.padded_oc = round_up(schedule_param_.oc_per_grp, OC_DATA_BLK);
    schedule_param_.gd_ker_blk = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::MAX_S_REGS;
    schedule_param_.dw_ker_blk = pd_conv2d_n16cx_depthwise_kernel_fp32_fma::config::MAX_W_REGS;

    inter_shape_.Reshape(src_shape_->GetDims(), src_shape_->GetDimCount());
    inter_shape_.SetDim(1, dst_shape_->GetDim(1));
    inter_shape_.SetDataType(src_shape_->GetDataType());
    inter_shape_.SetDataFormat(src_shape_->GetDataFormat());

    conv2d_executor_->set_src_shape(src_shape_);
    conv2d_executor_->set_dst_shape(&inter_shape_);
    depthwise_conv2d_executor_->set_src_shape(&inter_shape_);
    depthwise_conv2d_executor_->set_dst_shape(dst_shape_);
}

void pd_conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_kernel_tunning_param()
{
    const conv2d_fp32_param &gd_p = *conv2d_executor_->conv_param();
    const conv2d_fp32_param &dw_p = *depthwise_conv2d_executor_->conv_param();
    kernel_schedule_param &sp   = schedule_param_;

    const int64_t num_thread = PPL_OMP_MAX_THREADS();
    const int64_t batch      = src_shape_->GetDim(0);
    const int64_t src_h      = src_shape_->GetDim(2);
    const int64_t src_w      = src_shape_->GetDim(3);
    const int64_t dst_h      = dst_shape_->GetDim(2);
    const int64_t dst_w      = dst_shape_->GetDim(3);

    const float l2_cap_per_core = (ppl::common::GetCpuCacheL2() == 0 ? ASSUME_L2_BYTES : ppl::common::GetCpuCacheL2()) * L2_RATIO / sizeof(float);
    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (ASSUME_L3_BYTES * num_thread) : ppl::common::GetCpuCacheL3()) * L3_RATIO / sizeof(float);

    sp.ic_l2_blk = cal_ic_l2_blk(gd_p);
    sp.ic_l2_cnt = div_up(sp.padded_ic, sp.ic_l2_blk);

    sp.mb_l3_blk = min(batch, num_thread);
    sp.grp_l3_blk = 1;

    sp.gd_ker_blk = GD_KERNEL_BLK_MAX;
#define REDUN_S(S, S_BLK) (float(round_up(S, S_BLK)) / (S)-1.0f)
    if (REDUN_S(src_w, sp.gd_ker_blk) > 0.201f) {
        for (int64_t gd_ker_blk = GD_KERNEL_BLK_MAX - 1; gd_ker_blk >= GD_KERNEL_BLK_MIN; --gd_ker_blk) {
            if (REDUN_S(src_w, gd_ker_blk) < REDUN_S(src_w, sp.gd_ker_blk)) {
                sp.gd_ker_blk = gd_ker_blk;
            }
        }
    }
#undef REDUN_S
    if (sp.padded_oc > sp.padded_ic) {
        sp.oc_l2_blk = min(OC_L2_BLK_MAX_LARGE, sp.padded_oc);
    } else {
        sp.oc_l2_blk = min(OC_L2_BLK_MAX_SMALL, sp.padded_oc);
    }

    const int64_t oc_thread = div_up(num_thread, sp.grp_l3_blk * sp.mb_l3_blk);
    if (sp.padded_oc / oc_thread < sp.oc_l2_blk) {
        sp.oc_l2_blk = round_up(max(sp.padded_oc / oc_thread, OC_L2_BLK_MIN), OC_DATA_BLK);
    }

    sp.oh_l2_blk = dst_h;
    const int64_t oh_thread = div_up(num_thread, sp.grp_l3_blk * sp.mb_l3_blk * div_up(sp.padded_oc, sp.oc_l2_blk));
    if (oh_thread > 1) {
        sp.oh_l2_blk = max(dst_h / oh_thread, OH_L2_BLK_MIN);
    }

    const int64_t inter_buffer_len = dw_p.kernel_h * (dw_p.pad_w * 2 + src_shape_->GetDim(3)) * sp.oc_l2_blk;
    const int64_t feature_map_len = batch * (sp.padded_ic * gd_p.group * src_h * src_w + sp.padded_oc * gd_p.group * dst_h * dst_w);
    const bool large_inter_cost = inter_buffer_len > (l2_cap_per_core / L2_RATIO); // inter buffer oversized
    const bool small_feature_map = feature_map_len < (l2_cap_per_core * num_thread * 2); // data already in L2
    const bool small_src_w = (src_w < 2 * sp.gd_ker_blk && feature_map_len < (l2_cap_per_core * num_thread * 3 + l3_cap_all_core))
                          || (src_w < 4 * sp.gd_ker_blk && feature_map_len < (l2_cap_per_core * num_thread * 3)); // weak kernel performance
    const bool dense_conv = src_w <= 4 * sp.gd_ker_blk && dw_p.sparse_level() < 0.04f; // (sh1*sw1)/(kh5*kw5), weak kernel performance
    if (small_src_w || large_inter_cost || small_feature_map || dense_conv) {
        sp.mode = EXEC_MODE_SEPARATE;
    } else {
        sp.mode = EXEC_MODE_FUSE;
    }

    sp.use_nt_store = 0;
    if (batch * gd_p.group * sp.padded_oc * dst_h * dst_w > l3_cap_all_core * 3) {
        sp.use_nt_store = 1;
    }
}

uint64_t pd_conv2d_n16cx_gemm_direct_fp32_fma_executor::cal_temp_buffer_size()
{
    if (schedule_param_.mode == EXEC_MODE_SEPARATE) {
        schedule_param_.gd_temp_buffer_size = round_up(conv2d_executor_->cal_temp_buffer_size(), PPL_X86_CACHELINE_BYTES());
        schedule_param_.dw_temp_buffer_size = round_up(depthwise_conv2d_executor_->cal_temp_buffer_size(), PPL_X86_CACHELINE_BYTES());
        return schedule_param_.gd_temp_buffer_size + schedule_param_.dw_temp_buffer_size + inter_shape_.GetBytesIncludingPadding();
    } else {
        const conv2d_fp32_param &dw_p = *depthwise_conv2d_executor_->conv_param();
        const uint64_t inter_buffer_size = (uint64_t)dw_p.kernel_h * (dw_p.pad_w * 2 + src_shape_->GetDim(3)) * schedule_param_.oc_l2_blk * sizeof(float);
        return inter_buffer_size * PPL_OMP_MAX_THREADS();
    }
}

ppl::common::RetCode pd_conv2d_n16cx_gemm_direct_fp32_fma_executor::prepare()
{
    bool gd_prepare_ready = conv2d_executor_ && conv2d_executor_->conv_param();
    bool dw_prepare_ready = depthwise_conv2d_executor_ && depthwise_conv2d_executor_->conv_param();
    if (!gd_prepare_ready || !dw_prepare_ready || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();

    if (schedule_param_.mode == EXEC_MODE_SEPARATE) {
        conv2d_executor_->prepare();
        depthwise_conv2d_executor_->prepare();
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode pd_conv2d_n16cx_gemm_direct_fp32_fma_executor::execute() {
    if (schedule_param_.mode == EXEC_MODE_SEPARATE) {
        return separate_execute();
    }
    if (schedule_param_.mode == EXEC_MODE_FUSE) {
        return fuse_execute();
    }
    return ppl::common::RC_INVALID_VALUE;
}

ppl::common::RetCode pd_conv2d_n16cx_gemm_direct_fp32_fma_executor::separate_execute()
{
    if (!conv2d_executor_ || !depthwise_conv2d_executor_ || !src_ || !dst_ || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *gd_temp_buffer = (uint8_t *)temp_buffer_;
    uint8_t *dw_temp_buffer = temp_buffer_ != nullptr ? gd_temp_buffer + schedule_param_.gd_temp_buffer_size : nullptr;
    float *inter_buffer = temp_buffer_ != nullptr ? (float*)(dw_temp_buffer + schedule_param_.dw_temp_buffer_size) : nullptr;
    conv2d_executor_->set_src(src_);
    conv2d_executor_->set_dst(inter_buffer);
    conv2d_executor_->set_temp_buffer(gd_temp_buffer);
    depthwise_conv2d_executor_->set_src(inter_buffer);
    depthwise_conv2d_executor_->set_dst(dst_);
    depthwise_conv2d_executor_->set_temp_buffer(dw_temp_buffer);

    auto ret = conv2d_executor_->execute();
    if (ppl::common::RC_SUCCESS != ret) {
        return ret;
    }
    ret = depthwise_conv2d_executor_->execute();
    return ret;
}

ppl::common::RetCode pd_conv2d_n16cx_gemm_direct_fp32_fma_executor::fuse_execute()
{
    bool gd_execute_ready = conv2d_executor_ && conv2d_executor_->conv_param() && conv2d_executor_->cvt_filter() && conv2d_executor_->cvt_bias();
    bool dw_execute_ready = depthwise_conv2d_executor_ && depthwise_conv2d_executor_->conv_param() && depthwise_conv2d_executor_->cvt_filter() && depthwise_conv2d_executor_->cvt_bias();
    if (!gd_execute_ready || !dw_execute_ready || !src_ || !dst_ || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    auto gd_e = conv2d_executor_;
    auto dw_e = depthwise_conv2d_executor_;
    const conv2d_fp32_param &gd_p   = *gd_e->conv_param();
    const conv2d_fp32_param &dw_p   = *dw_e->conv_param();
    const kernel_schedule_param &sp = schedule_param_;

    const int64_t batch         = src_shape_->GetDim(0);
    const int64_t src_h         = src_shape_->GetDim(2);
    const int64_t src_w         = src_shape_->GetDim(3);
    const int64_t dst_h         = dst_shape_->GetDim(2);
    const int64_t dst_w         = dst_shape_->GetDim(3);
    const int64_t padded_reg_oc = round_up(sp.oc_per_grp, OC_REG_ELTS);

    const int64_t src_b_stride    = round_up(src_shape_->GetDim(1), IC_DATA_BLK) * src_h * src_w;
    const int64_t src_g_stride    = sp.padded_ic * src_h * src_w;
    const int64_t src_icb_stride  = src_h * src_w * IC_DATA_BLK;
    const int64_t src_h_stride    = src_w * IC_DATA_BLK;
    const int64_t inter_h_stride  = (src_w + 2 * dw_p.pad_w) * OC_DATA_BLK;
    const int64_t inter_oc_stride = dw_p.kernel_h * (src_w + dw_p.pad_w * 2);
    const int64_t gd_flt_g_stride = sp.ic_l2_cnt * sp.padded_oc * sp.ic_l2_blk;

    const int64_t dw_flt_ocb_stride = dw_p.kernel_h * dw_p.kernel_w * OC_DATA_BLK;
    const int64_t dst_b_stride      = round_up(dst_shape_->GetDim(1), OC_DATA_BLK) * dst_h * dst_w;
    const int64_t dst_ocb_stride    = dst_h * dst_w * OC_DATA_BLK;
    const int64_t dst_h_stride      = dst_w * OC_DATA_BLK;

    const bool gd_with_relu  = gd_p.fuse_flag & conv_fuse_flag::RELU;
    const bool gd_with_relu6 = gd_p.fuse_flag & conv_fuse_flag::RELU6;
    const bool dw_with_relu  = dw_p.fuse_flag & conv_fuse_flag::RELU;
    const bool dw_with_relu6 = dw_p.fuse_flag & conv_fuse_flag::RELU6;

    const int64_t spec_stride_w_sel = dw_p.stride_w < 3 ? dw_p.stride_w : 0;
    const uint64_t inter_buffer_len = (uint64_t)inter_oc_stride * sp.oc_l2_blk;

    PRAGMA_OMP_PARALLEL_FOR() // Init padding zeros
    for (int64_t t = 0; t < PPL_OMP_MAX_THREADS(); ++t) {
        float *inter_buffer = (float*)temp_buffer_ + inter_buffer_len * PPL_OMP_THREAD_ID();
        for (int64_t oc = 0; oc < sp.oc_l2_blk; oc += OC_DATA_BLK) {
            for (int64_t kh = 0; kh < dw_p.kernel_h; ++kh) {
                memset32_avx(inter_buffer, 0, dw_p.pad_w * OC_DATA_BLK);
                inter_buffer += dw_p.pad_w * OC_DATA_BLK;
                inter_buffer += src_w * OC_DATA_BLK;
                memset32_avx(inter_buffer, 0, dw_p.pad_w * OC_DATA_BLK);
                inter_buffer += dw_p.pad_w * OC_DATA_BLK;
            }
        }
    }

    for (int64_t mbl3 = 0; mbl3 < batch; mbl3 += sp.mb_l3_blk) {
        const int64_t mbl3_eff = min(batch - mbl3, sp.mb_l3_blk);
        for (int64_t grpl3 = 0; grpl3 < gd_p.group; grpl3 += sp.grp_l3_blk) {
            const int64_t grpl3_eff = min(gd_p.group - grpl3, sp.grp_l3_blk);
#ifdef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(4)
#endif
            for (int64_t g = grpl3; g < grpl3 + grpl3_eff; ++g) {
                for (int64_t b = mbl3; b < mbl3 + mbl3_eff; ++b) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                    PRAGMA_OMP_PARALLEL_FOR()
#endif
                    for (int64_t ocl2 = 0; ocl2 < padded_reg_oc; ocl2 += sp.oc_l2_blk) {
                        for (int64_t ohl2 = 0; ohl2 < dst_h; ohl2 += sp.oh_l2_blk) {
                            const int64_t ocl2_eff = min(padded_reg_oc - ocl2, sp.oc_l2_blk);
                            const int64_t ohl2_eff = min(dst_h - ohl2, sp.oh_l2_blk);
                            const int64_t iw_body  = round(src_w, sp.gd_ker_blk);
                            const int64_t iw_tail  = src_w - iw_body;
                            const int64_t ow_body  = round(dst_w, sp.dw_ker_blk);
                            const int64_t ow_tail  = dst_w - ow_body;

                            float *inter_buffer = (float*)temp_buffer_ + inter_buffer_len * PPL_OMP_THREAD_ID();
                            int64_t ih_scroll   = 0;

                            int64_t gd_ker_param[conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::LENGTH];
                            array_param_helper gd_ker_p(gd_ker_param);
                            conv2d_n16cx_gemm_direct_kernel_fp32_fma gd_ker(gd_ker_param);

                            std::vector<float*> base_dw_src_ptr_kh_list(dw_p.kernel_h, nullptr);
                            std::vector<float*> dw_src_ptr_kh_list(dw_p.kernel_h, nullptr);
                            int64_t dw_ker_param[pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::LENGTH];
                            array_param_helper dw_ker_p(dw_ker_param);
                            pd_conv2d_n16cx_depthwise_kernel_fp32_fma dw_ker(dw_ker_param);

                            dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::KW_IDX)            = dw_p.kernel_w;
                            dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::SRC_SW_STRIDE_IDX) = dw_p.stride_w * OC_DATA_BLK;
                            gd_ker_p.pick<int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SRC_ICB_STRIDE_IDX) = src_icb_stride;
                            for (int64_t oh = ohl2; oh < ohl2 + ohl2_eff; ++oh) {
                                const int64_t ih_offset   = oh * dw_p.stride_h - dw_p.pad_h;
                                const int64_t ih_start    = max<int64_t>(ih_offset, 0);
                                const int64_t ih_end      = min<int64_t>(ih_offset + dw_p.kernel_h, src_h);
                                const int64_t dw_kh_start = min<int64_t>(max<int64_t>(0 - ih_offset, 0), dw_p.kernel_h - 1);
                                const int64_t dw_kh_end   = max<int64_t>(min<int64_t>(src_h - ih_offset, dw_p.kernel_h), 0);
                                ih_scroll                 = max(ih_start, ih_scroll);

                                for (int64_t icl2 = 0; icl2 < sp.padded_ic; icl2 += sp.ic_l2_blk) {
                                    const int64_t icl2_eff = min(sp.ic_per_grp - icl2, sp.ic_l2_blk);
                                    const bool is_first_ic = icl2 == 0;
                                    const bool is_last_ic  = (icl2 + sp.ic_l2_blk >= sp.ic_per_grp);
                                    int64_t ker_flags      = 0;

                                    if (is_first_ic) {
                                        ker_flags |= conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::LOAD_BIAS;
                                    }
                                    if (is_last_ic) {
                                        if (gd_with_relu) ker_flags |= conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::RELU;
                                        if (gd_with_relu6) ker_flags |= conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::RELU6;
                                    }

                                    const float *base_src  = src_ + b * src_b_stride + g * src_g_stride + icl2 * src_h * src_w + ih_scroll * src_h_stride;
                                    const float *base_flt  = gd_e->cvt_filter() + g * gd_flt_g_stride + icl2 * sp.padded_oc + ocl2 * sp.ic_l2_blk;
                                    const float *base_bias = gd_e->cvt_bias() + g * sp.padded_oc + ocl2;

                                    gd_ker_p.pick<int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::CHANNELS_IDX) = icl2_eff;
                                    gd_ker_p.pick<int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::FLAGS_IDX)    = ker_flags;
                                    for (int64_t ih = ih_scroll; ih < ih_end; ++ih) {
                                        float *base_dst = inter_buffer + dw_p.pad_w * OC_DATA_BLK + (ih % dw_p.kernel_h) * inter_h_stride;
                                        gd_ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::FLT_PTR_IDX)  = base_flt;
                                        gd_ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::BIAS_PTR_IDX) = base_bias;
                                        for (int64_t oc = ocl2; oc < ocl2 + ocl2_eff; oc += OC_DATA_BLK) {
                                            const int64_t oc_eff = min(ocl2 + ocl2_eff - oc, OC_DATA_BLK);
                                            const int64_t oc_reg = div_up(oc_eff, OC_REG_ELTS);
                                            if (iw_body) {
                                                gd_ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SRC_PTR_IDX) = base_src;
                                                gd_ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::HIS_PTR_IDX) = base_dst;
                                                gd_ker_p.pick<float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::DST_PTR_IDX)       = base_dst;
                                                gd_ker_p.pick<int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SPACE_IDX)        = iw_body;
                                                gd_ker.execute(0, oc_reg, sp.gd_ker_blk);
                                            }
                                            if (iw_tail) {
                                                gd_ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SRC_PTR_IDX) = base_src + iw_body * IC_DATA_BLK;
                                                gd_ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::HIS_PTR_IDX) = base_dst + iw_body * OC_DATA_BLK;
                                                gd_ker_p.pick<float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::DST_PTR_IDX)       = base_dst + iw_body * OC_DATA_BLK;
                                                gd_ker_p.pick<int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SPACE_IDX)        = iw_tail;
                                                gd_ker.execute(0, oc_reg, iw_tail);
                                            }
                                            gd_ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::FLT_PTR_IDX)  += OC_DATA_BLK * sp.ic_l2_blk;
                                            gd_ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::BIAS_PTR_IDX) += OC_DATA_BLK;
                                            base_dst += inter_oc_stride * OC_DATA_BLK;
                                        }
                                        base_src += src_h_stride;
                                    }
                                }
                                ih_scroll = ih_end;
                                { // dw session
                                    const int64_t dw_oc    = g * sp.padded_oc + ocl2;
                                    const float *base_flt  = dw_e->cvt_filter() + dw_oc * dw_p.kernel_h * dw_p.kernel_w;
                                    const float *base_bias = dw_e->cvt_bias() + dw_oc;
                                    float *base_dst        = dst_ + b * dst_b_stride + dw_oc * dst_h * dst_w + oh * dst_h_stride;
                                    int64_t ker_flags      = 0;
                                    if (dw_with_relu)  ker_flags |= pd_conv2d_n16cx_depthwise_kernel_fp32_fma::flag::RELU;
                                    if (dw_with_relu6) ker_flags |= pd_conv2d_n16cx_depthwise_kernel_fp32_fma::flag::RELU6;
                                    for (int64_t kh = dw_kh_start; kh < dw_kh_end; ++kh) {
                                        const int64_t ih = ih_offset + kh;
                                        base_dw_src_ptr_kh_list[kh] = inter_buffer + (ih % dw_p.kernel_h) * inter_h_stride;
                                    }
                                    dw_ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::FLT_PTR_IDX)  = base_flt;
                                    dw_ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::BIAS_PTR_IDX) = base_bias;
                                    dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::KH_START_IDX)      = dw_kh_start;
                                    dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::KH_END_IDX)        = dw_kh_end;
                                    dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::FLAGS_IDX)         = ker_flags;

                                    for (int64_t oc = 0; oc < ocl2_eff; oc += OC_DATA_BLK) {
                                        for (int64_t kh = dw_kh_start; kh < dw_kh_end; ++kh) {
                                            dw_src_ptr_kh_list[kh] = base_dw_src_ptr_kh_list[kh] + oc * inter_oc_stride;
                                        }
                                        dw_ker_p.pick<float**>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::SRC_PTR_KH_LIST_IDX) = dw_src_ptr_kh_list.data();
                                        dw_ker_p.pick<float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::DST_PTR_IDX)          = base_dst;
                                        if (ow_body) {
                                            dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::DST_WIDTH_IDX) = ow_body;
                                            dw_ker.execute(sp.use_nt_store, spec_stride_w_sel, sp.dw_ker_blk);
                                        }
                                        if (ow_tail) {
                                            dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::DST_WIDTH_IDX) = ow_tail;
                                            dw_ker.execute(sp.use_nt_store, spec_stride_w_sel, ow_tail);
                                        }
                                        dw_ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::FLT_PTR_IDX)  += dw_flt_ocb_stride;
                                        dw_ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_fma::param_def::BIAS_PTR_IDX) += OC_DATA_BLK;
                                        base_dst += dst_ocb_stride;
                                    }
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
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86