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

#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/avx512/conv2d_n16cx_direct_ndarray_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/avx512/conv2d_n16cx_direct_ndarray_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/pd_conv2d/avx512/pd_conv2d_n16cx_direct_ndarray_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/pd_conv2d/avx512/pd_conv2d_n16cx_depthwise_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/common/array_param_helper.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

static const int64_t ASSUME_L2_BYTES = 256 * 1024;
static const int64_t ASSUME_L2_WAYS = 4;
static const int64_t ASSUME_L3_BYTES = 2048 * 1024;
static const float L2_RATIO = 0.251f;
static const float L3_RATIO = 0.501f;

static const int64_t OC_DATA_BLK = conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::config::OC_DATA_BLK;

static const int64_t OC_L2_BLK_MAX = 4 * OC_DATA_BLK;
static const int64_t OH_L2_BLK_MIN = 32;

void pd_conv2d_n16cx_direct_ndarray_fp32_avx512_executor::init_preproc_param()
{
    auto dr_param = conv2d_executor_->conv_param();
    schedule_param_.ic_per_grp = dr_param->channels / dr_param->group;
    schedule_param_.oc_per_grp = dr_param->num_output / dr_param->group;
    schedule_param_.padded_oc = round_up(schedule_param_.oc_per_grp, OC_DATA_BLK);
    schedule_param_.dr_ker_blk = conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::config::MAX_W_BLK;
    schedule_param_.dw_ker_blk = pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::config::MAX_W_BLK;
    schedule_param_.oc_ker_blk = conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::config::MAX_OC_BLK;

    inter_shape_.SetDimCount(src_shape_->GetDimCount());
    inter_shape_.SetDim(0, src_shape_->GetDim(0));
    inter_shape_.SetDim(1, dr_param->num_output);
    const int64_t dr_ekh = (dr_param->kernel_h - 1) * dr_param->dilation_h + 1;
    const int64_t dr_ekw = (dr_param->kernel_w - 1) * dr_param->dilation_w + 1;
    const int64_t inter_h = ((src_shape_->GetDim(2) + 2 * dr_param->pad_h - dr_ekh) / dr_param->stride_h + 1);
    const int64_t inter_w = ((src_shape_->GetDim(3) + 2 * dr_param->pad_w - dr_ekw) / dr_param->stride_w + 1);
    inter_shape_.SetDim(2, inter_h);
    inter_shape_.SetDim(3, inter_w);
    inter_shape_.SetDataType(ppl::common::DATATYPE_FLOAT32);
    inter_shape_.SetDataFormat(ppl::common::DATAFORMAT_N16CX);

    conv2d_executor_->set_src_shape(src_shape_);
    conv2d_executor_->set_dst_shape(&inter_shape_);
    depthwise_conv2d_executor_->set_src_shape(&inter_shape_);
    depthwise_conv2d_executor_->set_dst_shape(dst_shape_);
}

void pd_conv2d_n16cx_direct_ndarray_fp32_avx512_executor::cal_kernel_tunning_param()
{
    const conv2d_fp32_param &dr_p = *conv2d_executor_->conv_param();
    const conv2d_fp32_param &dw_p = *depthwise_conv2d_executor_->conv_param();
    kernel_schedule_param &sp   = schedule_param_;

    const int64_t num_thread = PPL_OMP_MAX_THREADS();
    const int64_t batch      = src_shape_->GetDim(0);
    const int64_t src_h      = src_shape_->GetDim(2);
    const int64_t src_w      = src_shape_->GetDim(3);
    const int64_t dst_h      = dst_shape_->GetDim(2);
    const int64_t dst_w      = dst_shape_->GetDim(3);
    const int64_t inter_w    = inter_shape_.GetDim(3);

    const float l2_cap_per_core = (ppl::common::GetCpuCacheL2() == 0 ? ASSUME_L2_BYTES : ppl::common::GetCpuCacheL2()) * L2_RATIO / sizeof(float);
    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (ASSUME_L3_BYTES * num_thread) : ppl::common::GetCpuCacheL3()) * L3_RATIO / sizeof(float);

    sp.mb_l3_blk = min(batch, num_thread);
    sp.grp_l3_blk = 1;

    sp.dr_unroll_w_start = -1;
    sp.dr_unroll_w_end = -1;
    for (int64_t iw = 0; iw < inter_w; ++iw) {
        if (iw * dr_p.stride_w - dr_p.pad_w >= 0) {
            sp.dr_unroll_w_start = iw;
            break;
        }
    }
    for (int64_t iw = inter_w - 1; iw >= 0; --iw) {
        if (iw * dr_p.stride_w - dr_p.pad_w + dr_p.kernel_w <= src_w) {
            sp.dr_unroll_w_end = iw + 1;
            break;
        }
    }
    if (sp.dr_unroll_w_start >= sp.dr_unroll_w_end || sp.dr_unroll_w_start < 0 || sp.dr_unroll_w_end < 0) {
        sp.dr_unroll_w_start = sp.dr_unroll_w_end = inter_w;
    }

    sp.oc_l2_blk = min(OC_L2_BLK_MAX, sp.padded_oc);

    sp.oh_l2_blk = dst_h;
    const int64_t oh_thread = div_up(num_thread, sp.grp_l3_blk * sp.mb_l3_blk * div_up(sp.padded_oc, sp.oc_l2_blk));
    if (oh_thread > 1) {
        sp.oh_l2_blk = max(dst_h / oh_thread, OH_L2_BLK_MIN);
    }

    sp.use_nt_store = 0;
    if (batch * dr_p.group * sp.padded_oc * dst_h * dst_w > l3_cap_all_core * 3) {
        sp.use_nt_store = 1;
    }

    const int64_t inter_buffer_len = dw_p.kernel_h * (dw_p.pad_w * 2 + inter_w) * sp.oc_l2_blk;
    const int64_t feature_map_len = batch * (sp.ic_per_grp * dr_p.group * src_h * src_w + sp.padded_oc * dr_p.group * dst_h * dst_w);
    const bool large_inter_cost = inter_buffer_len > (l2_cap_per_core / L2_RATIO); // inter buffer oversized
    const bool small_feature_map = feature_map_len < (l2_cap_per_core * num_thread * 2); // data already in L2
    const bool small_inter_w = (inter_w < sp.dr_ker_blk && feature_map_len < (l2_cap_per_core * num_thread * 3 + l3_cap_all_core))
                          || (inter_w < 2 * sp.dr_ker_blk && feature_map_len < (l2_cap_per_core * num_thread * 3)); // weak kernel performance
    const bool dense_conv = inter_w <= 2 * sp.dr_ker_blk && dw_p.sparse_level() < 0.04f; // (sh1*sw1)/(kh5*kw5), weak kernel performance
    if (small_inter_w || large_inter_cost || small_feature_map || dense_conv) {
        mode_ = pd_conv2d_fp32_mode::SEPARATE;
    } else {
        mode_ = pd_conv2d_fp32_mode::FUSE;
    }
}

uint64_t pd_conv2d_n16cx_direct_ndarray_fp32_avx512_executor::cal_temp_buffer_size()
{
    if (mode_ == pd_conv2d_fp32_mode::SEPARATE) {
        schedule_param_.dr_temp_buffer_size = round_up(conv2d_executor_->cal_temp_buffer_size(), PPL_X86_CACHELINE_BYTES());
        schedule_param_.dw_temp_buffer_size = round_up(depthwise_conv2d_executor_->cal_temp_buffer_size(), PPL_X86_CACHELINE_BYTES());
        return schedule_param_.dr_temp_buffer_size + schedule_param_.dw_temp_buffer_size + inter_shape_.GetBytesIncludingPadding();
    } else {
        const conv2d_fp32_param &dw_p = *depthwise_conv2d_executor_->conv_param();
        const uint64_t inter_buffer_size = (uint64_t)dw_p.kernel_h * (dw_p.pad_w * 2 + inter_shape_.GetDim(3)) * schedule_param_.oc_l2_blk * sizeof(float);
        return inter_buffer_size * PPL_OMP_MAX_THREADS();
    }
}

ppl::common::RetCode pd_conv2d_n16cx_direct_ndarray_fp32_avx512_executor::prepare()
{
    bool dr_prepare_ready = conv2d_executor_ && conv2d_executor_->conv_param();
    bool dw_prepare_ready = depthwise_conv2d_executor_ && depthwise_conv2d_executor_->conv_param();
    if (!dr_prepare_ready || !dw_prepare_ready || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();

    if (mode_ == pd_conv2d_fp32_mode::SEPARATE) {
        auto ret = conv2d_executor_->prepare();
        if (ppl::common::RC_SUCCESS != ret) {
            return ret;
        }
        ret = depthwise_conv2d_executor_->prepare();
        if (ppl::common::RC_SUCCESS != ret) {
            return ret;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode pd_conv2d_n16cx_direct_ndarray_fp32_avx512_executor::execute() {
    if (mode_ == pd_conv2d_fp32_mode::SEPARATE) {
        return separate_execute();
    }
    if (mode_ == pd_conv2d_fp32_mode::FUSE) {
        return fuse_execute();
    }
    return ppl::common::RC_INVALID_VALUE;
}

ppl::common::RetCode pd_conv2d_n16cx_direct_ndarray_fp32_avx512_executor::separate_execute()
{
    if (!conv2d_executor_ || !depthwise_conv2d_executor_ || !src_ || !dst_ || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }
    uint8_t *dr_temp_buffer = (uint8_t *)temp_buffer_;
    uint8_t *dw_temp_buffer = temp_buffer_ != nullptr ? dr_temp_buffer + schedule_param_.dr_temp_buffer_size : nullptr;
    float *inter_buffer = temp_buffer_ != nullptr ? (float*)(dw_temp_buffer + schedule_param_.dw_temp_buffer_size) : nullptr;
    conv2d_executor_->set_src(src_);
    conv2d_executor_->set_dst(inter_buffer);
    conv2d_executor_->set_temp_buffer(dr_temp_buffer);
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

ppl::common::RetCode pd_conv2d_n16cx_direct_ndarray_fp32_avx512_executor::fuse_execute()
{
    bool dr_execute_ready = conv2d_executor_ && conv2d_executor_->conv_param() && conv2d_executor_->cvt_filter() && conv2d_executor_->cvt_bias();
    bool dw_execute_ready = depthwise_conv2d_executor_ && depthwise_conv2d_executor_->conv_param() && depthwise_conv2d_executor_->cvt_filter() && depthwise_conv2d_executor_->cvt_bias();
    if (!dr_execute_ready || !dw_execute_ready || !src_ || !dst_ || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    auto dr_e = conv2d_executor_;
    auto dw_e = depthwise_conv2d_executor_;
    const conv2d_fp32_param &dr_p   = *dr_e->conv_param();
    const conv2d_fp32_param &dw_p   = *dw_e->conv_param();
    const kernel_schedule_param &sp = schedule_param_;

    const int64_t batch         = src_shape_->GetDim(0);
    const int64_t src_h         = src_shape_->GetDim(2);
    const int64_t src_w         = src_shape_->GetDim(3);
    const int64_t dst_h         = dst_shape_->GetDim(2);
    const int64_t dst_w         = dst_shape_->GetDim(3);
    const int64_t inter_h       = inter_shape_.GetDim(2);
    const int64_t inter_w       = inter_shape_.GetDim(3);

    const int64_t src_b_stride     = src_shape_->GetDim(1) * src_h * src_w;
    const int64_t src_g_stride     = sp.ic_per_grp * src_h * src_w;
    const int64_t src_c_stride     = src_h * src_w;
    const int64_t dr_flt_c_stride  = dr_p.kernel_h * dr_p.kernel_w * OC_DATA_BLK;
    const int64_t dr_flt_oc_stride = sp.ic_per_grp * dr_p.kernel_h * dr_p.kernel_w;

    const int64_t inter_h_stride  = (inter_w + 2 * dw_p.pad_w) * OC_DATA_BLK;
    const int64_t inter_oc_stride = dw_p.kernel_h * (inter_w + dw_p.pad_w * 2);

    const int64_t dw_flt_ocb_stride = dw_p.kernel_h * dw_p.kernel_w * OC_DATA_BLK;
    const int64_t dst_b_stride      = round_up(dst_shape_->GetDim(1), OC_DATA_BLK) * dst_h * dst_w;
    const int64_t dst_ocb_stride    = dst_h * dst_w * OC_DATA_BLK;
    const int64_t dst_h_stride      = dst_w * OC_DATA_BLK;

    const bool dr_with_relu  = dr_p.fuse_flag & conv_fuse_flag::RELU;
    const bool dr_with_relu6 = dr_p.fuse_flag & conv_fuse_flag::RELU6;
    const bool dw_with_relu  = dw_p.fuse_flag & conv_fuse_flag::RELU;
    const bool dw_with_relu6 = dw_p.fuse_flag & conv_fuse_flag::RELU6;

    int64_t dr_ker_flags = 0;
    if (dr_with_relu)  dr_ker_flags |= conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::RELU;
    if (dr_with_relu6) dr_ker_flags |= conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::flag::RELU6;

    int64_t dw_ker_flags = 0;
    if (dw_with_relu)  dw_ker_flags |= pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::flag::RELU;
    if (dw_with_relu6) dw_ker_flags |= pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::flag::RELU6;

    const int64_t spec_stride_w_sel = dw_p.stride_w < 3 ? dw_p.stride_w : 0;
    const uint64_t inter_buffer_len = (uint64_t)inter_oc_stride * sp.oc_l2_blk;

    PRAGMA_OMP_PARALLEL_FOR() // Init padding zeros
    for (int64_t t = 0; t < PPL_OMP_MAX_THREADS(); ++t) {
        float *inter_buffer = (float*)temp_buffer_ + inter_buffer_len * PPL_OMP_THREAD_ID();
        for (int64_t oc = 0; oc < sp.oc_l2_blk; oc += OC_DATA_BLK) {
            for (int64_t kh = 0; kh < dw_p.kernel_h; ++kh) {
                memset32_avx(inter_buffer, 0, dw_p.pad_w * OC_DATA_BLK);
                inter_buffer += dw_p.pad_w * OC_DATA_BLK;
                inter_buffer += inter_w * OC_DATA_BLK;
                memset32_avx(inter_buffer, 0, dw_p.pad_w * OC_DATA_BLK);
                inter_buffer += dw_p.pad_w * OC_DATA_BLK;
            }
        }
    }

    for (int64_t mbl3 = 0; mbl3 < batch; mbl3 += sp.mb_l3_blk) {
        const int64_t mbl3_eff = min(batch - mbl3, sp.mb_l3_blk);
        for (int64_t grpl3 = 0; grpl3 < dr_p.group; grpl3 += sp.grp_l3_blk) {
            const int64_t grpl3_eff = min(dr_p.group - grpl3, sp.grp_l3_blk);
#ifdef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(4)
#endif
            for (int64_t g = grpl3; g < grpl3 + grpl3_eff; ++g) {
                for (int64_t b = mbl3; b < mbl3 + mbl3_eff; ++b) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                    PRAGMA_OMP_PARALLEL_FOR()
#endif
                    for (int64_t ocl2 = 0; ocl2 < sp.padded_oc; ocl2 += sp.oc_l2_blk) {
                        for (int64_t ohl2 = 0; ohl2 < dst_h; ohl2 += sp.oh_l2_blk) {
                            const int64_t ocl2_eff = min(sp.padded_oc - ocl2, sp.oc_l2_blk);
                            const int64_t ohl2_eff = min(dst_h - ohl2, sp.oh_l2_blk);
                            const int64_t ow_body  = round(dst_w, sp.dw_ker_blk);
                            const int64_t ow_tail  = dst_w - ow_body;

                            float *inter_buffer = (float*)temp_buffer_ + inter_buffer_len * PPL_OMP_THREAD_ID();
                            int64_t ih_scroll   = 0;

                            int64_t dr_ker_param[conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::LENGTH];
                            array_param_helper dr_ker_p(dr_ker_param);
                            conv2d_n16cx_direct_ndarray_kernel_fp32_avx512 dr_ker(dr_ker_param);

                            std::vector<float*> base_dw_src_ptr_kh_list(dw_p.kernel_h, nullptr);
                            std::vector<float*> dw_src_ptr_kh_list(dw_p.kernel_h, nullptr);
                            int64_t dw_ker_param[pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::LENGTH];
                            array_param_helper dw_ker_p(dw_ker_param);
                            pd_conv2d_n16cx_depthwise_kernel_fp32_avx512 dw_ker(dw_ker_param);

                            dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::KW_IDX)            = dw_p.kernel_w;
                            dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::SRC_SW_STRIDE_IDX) = dw_p.stride_w * OC_DATA_BLK;
                            dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::FLAGS_IDX)         = dw_ker_flags;

                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::CHANNELS_IDX)           = sp.ic_per_grp;
                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KH_IDX)                 = dr_p.kernel_h;
                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KW_IDX)                 = dr_p.kernel_w;
                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SW_IDX)                 = dr_p.stride_w;
                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SRC_H_STRIDE_IDX)       = src_w;
                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SRC_C_STRIDE_IDX)       = src_c_stride;
                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLT_C_STRIDE_IDX)       = dr_flt_c_stride;
                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SUM_SRC_OCB_STRIDE_IDX) = inter_oc_stride * OC_DATA_BLK;
                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::DST_OCB_STRIDE_IDX)     = inter_oc_stride * OC_DATA_BLK;
                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLT_OCB_STRIDE_IDX)     = dr_flt_oc_stride * OC_DATA_BLK;
                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLAGS_IDX)              = dr_ker_flags;
                            for (int64_t oh = ohl2; oh < ohl2 + ohl2_eff; ++oh) {
                                const int64_t ih_offset   = oh * dw_p.stride_h - dw_p.pad_h;
                                const int64_t ih_start    = max<int64_t>(ih_offset, 0);
                                const int64_t ih_end      = min<int64_t>(ih_offset + dw_p.kernel_h, inter_h);
                                const int64_t dw_kh_start = min<int64_t>(max<int64_t>(0 - ih_offset, 0), dw_p.kernel_h - 1);
                                const int64_t dw_kh_end   = max<int64_t>(min<int64_t>(inter_h - ih_offset, dw_p.kernel_h), 0);
                                ih_scroll                 = max(ih_start, ih_scroll);

                                for (int64_t ih = ih_scroll; ih < ih_end; ++ih) {
                                    const int64_t eh          = ih * dr_p.stride_h - dr_p.pad_h;
                                    const int64_t dr_kh_start = min<int64_t>(max<int64_t>(0 - eh, 0), dr_p.kernel_h - 1);
                                    const int64_t dr_kh_end   = max<int64_t>(min<int64_t>(src_h - eh, dr_p.kernel_h), 0);

                                    const int64_t iw_unroll_len  = sp.dr_unroll_w_end - sp.dr_unroll_w_start;
                                    const int64_t iw_unroll_body = round(iw_unroll_len, sp.dr_ker_blk);
                                    const int64_t iw_unroll_tail = iw_unroll_len - iw_unroll_body;

                                    const float *base_src      = src_ + b * src_b_stride + g * src_g_stride + eh * src_w - dr_p.pad_w;
                                    float *base_dst            = inter_buffer + dw_p.pad_w * OC_DATA_BLK + (ih % dw_p.kernel_h) * inter_h_stride;
                                    const float *base_flt      = dr_e->cvt_filter() + g * sp.padded_oc * sp.ic_per_grp * dr_p.kernel_h * dr_p.kernel_w;
                                    const float *base_bias     = dr_e->cvt_bias() + g * sp.padded_oc;

                                    dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KH_START_IDX)      = dr_kh_start;
                                    dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KH_END_IDX)        = dr_kh_end;
                                    dr_ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLT_PTR_IDX)  = base_flt;
                                    dr_ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::BIAS_PTR_IDX) = base_bias;
                                    for (int64_t oc = ocl2; oc < ocl2 + ocl2_eff; oc += sp.oc_ker_blk) {
                                        const int64_t oc_eff = min<int64_t>(ocl2 + ocl2_eff - oc, sp.oc_ker_blk);
                                        const int64_t oc_reg = div_up(oc_eff, OC_DATA_BLK);
                                        dr_ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::SRC_PTR_IDX)     = base_src;
                                        dr_ker_p.pick<float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::DST_PTR_IDX)           = base_dst;

                                        for (int64_t iw = 0; iw < sp.dr_unroll_w_start; ++iw) {
                                            const int64_t ew          = iw * dr_p.stride_w - dr_p.pad_w;
                                            const int64_t dr_kw_start = min<int64_t>(max<int64_t>(0 - ew, 0), dr_p.kernel_w - 1);
                                            const int64_t dr_kw_end   = max<int64_t>(min<int64_t>(src_w - ew, dr_p.kernel_w), 0);
                                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KW_START_IDX) = dr_kw_start;
                                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KW_END_IDX)   = dr_kw_end;
                                            dr_ker.execute_border(0, oc_reg);
                                        }

                                        if (iw_unroll_body) {
                                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::DST_WIDTH_IDX) = iw_unroll_body;
                                            dr_ker.execute(0, oc_reg, sp.dr_ker_blk);
                                        }
                                        if (iw_unroll_tail) {
                                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::DST_WIDTH_IDX) = iw_unroll_tail;
                                            dr_ker.execute(0, oc_reg, iw_unroll_tail);
                                        }

                                        for (int64_t iw = sp.dr_unroll_w_end; iw < inter_w; ++iw) {
                                            const int64_t ew          = iw * dr_p.stride_w - dr_p.pad_w;
                                            const int64_t dr_kw_start = min<int64_t>(max<int64_t>(0 - ew, 0), dr_p.kernel_w - 1);
                                            const int64_t dr_kw_end   = max<int64_t>(min<int64_t>(src_w - ew, dr_p.kernel_w), 0);
                                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KW_START_IDX) = dr_kw_start;
                                            dr_ker_p.pick<int64_t>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::KW_END_IDX)   = dr_kw_end;
                                            dr_ker.execute_border(0, oc_reg);
                                        }
                                        dr_ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::FLT_PTR_IDX)  += sp.oc_ker_blk * dr_flt_oc_stride;
                                        dr_ker_p.pick<const float*>(conv2d_n16cx_direct_ndarray_kernel_fp32_avx512::param_def::BIAS_PTR_IDX) += sp.oc_ker_blk;
                                        base_dst += sp.oc_ker_blk * inter_oc_stride;
                                    }
                                }
                                ih_scroll = ih_end;
                                { // dw session
                                    const int64_t dw_oc    = g * sp.padded_oc + ocl2;
                                    const float *base_flt  = dw_e->cvt_filter() + dw_oc * dw_p.kernel_h * dw_p.kernel_w;
                                    const float *base_bias = dw_e->cvt_bias() + dw_oc;
                                    float *base_dst        = dst_ + b * dst_b_stride + dw_oc * dst_h * dst_w + oh * dst_h_stride;
                                    for (int64_t kh = dw_kh_start; kh < dw_kh_end; ++kh) {
                                        const int64_t ih = ih_offset + kh;
                                        base_dw_src_ptr_kh_list[kh] = inter_buffer + (ih % dw_p.kernel_h) * inter_h_stride;
                                    }
                                    dw_ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::FLT_PTR_IDX)  = base_flt;
                                    dw_ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::BIAS_PTR_IDX) = base_bias;
                                    dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::KH_START_IDX)      = dw_kh_start;
                                    dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::KH_END_IDX)        = dw_kh_end;

                                    for (int64_t oc = 0; oc < ocl2_eff; oc += OC_DATA_BLK) {
                                        for (int64_t kh = dw_kh_start; kh < dw_kh_end; ++kh) {
                                            dw_src_ptr_kh_list[kh] = base_dw_src_ptr_kh_list[kh] + oc * inter_oc_stride;
                                        }
                                        dw_ker_p.pick<float**>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::SRC_PTR_KH_LIST_IDX) = dw_src_ptr_kh_list.data();
                                        dw_ker_p.pick<float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::DST_PTR_IDX)          = base_dst;
                                        if (ow_body) {
                                            dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::DST_WIDTH_IDX) = ow_body;
                                            dw_ker.execute(sp.use_nt_store, spec_stride_w_sel, sp.dw_ker_blk);
                                        }
                                        if (ow_tail) {
                                            dw_ker_p.pick<int64_t>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::DST_WIDTH_IDX) = ow_tail;
                                            dw_ker.execute(sp.use_nt_store, spec_stride_w_sel, ow_tail);
                                        }
                                        dw_ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::FLT_PTR_IDX)  += dw_flt_ocb_stride;
                                        dw_ker_p.pick<const float*>(pd_conv2d_n16cx_depthwise_kernel_fp32_avx512::param_def::BIAS_PTR_IDX) += OC_DATA_BLK;
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