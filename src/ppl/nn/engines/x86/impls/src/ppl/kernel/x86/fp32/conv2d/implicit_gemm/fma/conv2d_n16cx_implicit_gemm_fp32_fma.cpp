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
#include "ppl/kernel/x86/fp32/conv2d/implicit_gemm/fma/conv2d_n16cx_implicit_gemm_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/implicit_gemm/fma/conv2d_n16cx_implicit_gemm_kernel_fp32_fma.h"
#include "ppl/common/sys.h"

#define ASSUME_L2_BYTES() (256 * 1024)
#define ASSUME_L2_WAYS()  4
#define ASSUME_L3_BYTES() (2048 * 1024)
#define L2_RATIO()        0.251
#define L3_RATIO()        0.501

#define IH_TR_BLK_MIN()        2
#define IW_TR_BLK_MIN()        8
#define IW_TR_BLK_TAIL_RATIO() 0.251
#define OH_L3_BLK_MID()        12
#define OH_L3_BLK_MIN()        1
#define OH_L3_BLK_TAIL_RATIO() 0.251
#define OW_L3_BLK_MIN(SP)      (8 * (SP).ow_kr_blk)
#define OW_L3_BLK_TAIL_RATIO() 0.251
#define HW_L3_BLK_MAX()        2048
#define OH_L2_BLK_TAIL_RATIO() 0.667
#define OW_L2_BLK_TAIL_RATIO() 0.667
#define IC_L2_BLK_MAX()        (16 * CH_DT_BLK())
#define IC_L2_BLK_TAIL_RATIO() 0.334
#define OC_L2_BLK_MAX()        (8 * CH_DT_BLK())
#define OC_L2_BLK_MIN()        (4 * CH_DT_BLK())
#define OC_L2_BLK_TAIL_RATIO() 0 // 0.667
#define THREAD_TAIL_RATIO()    0.8
#define THREAD_BODY_ROUND()    4
#define CACHE_ALIGNED_HW()     64
#define CACHE_CONFLICT_HW()    (ASSUME_L2_BYTES() / ASSUME_L2_WAYS() / CH_DT_BLK() / sizeof(float))

#define COMPUTE_POLICY_NO_TRANS()    0
#define COMPUTE_POLICY_BLOCK_TRANS() 1

#define TIMER_COUNT()  3
#define SRCTR_TIMER()  0
#define KERNEL_TIMER() 1
#define FENCE_TIMER()  2

namespace ppl { namespace kernel { namespace x86 {

bool conv2d_n16cx_implicit_gemm_fp32_fma_executor::init_profiler()
{
#ifdef PPL_X86_KERNEL_TIMING
    profiler_.init(TIMER_COUNT());
    return true;
#else
    return false;
#endif
}

void conv2d_n16cx_implicit_gemm_fp32_fma_executor::clear_profiler()
{
#ifdef PPL_X86_KERNEL_TIMING
    profiler_.clear();
#endif
}

std::string conv2d_n16cx_implicit_gemm_fp32_fma_executor::export_profiler()
{
#ifdef PPL_X86_KERNEL_TIMING
    static const char *timer_name[TIMER_COUNT()] = {
        "srctr",
        "kernel",
        "fence"};
    return profiler_.export_csv(timer_name, false);
#else
    return "";
#endif
}

int32_t conv2d_n16cx_implicit_gemm_fp32_fma_executor::cal_ic_l2_blk(const conv2d_fp32_param &param)
{
    const int32_t ic_per_gp = param.channels / param.group;
    const int32_t padded_ic = round_up(ic_per_gp, CH_DT_BLK());

    int32_t ic_l2_blk = min(div_up(IC_L2_BLK_MAX(), param.kernel_h * param.kernel_w * CH_DT_BLK()) * CH_DT_BLK(), padded_ic);
    if (mod_up(padded_ic, ic_l2_blk) < IC_L2_BLK_TAIL_RATIO() * ic_l2_blk) {
        ic_l2_blk = round_up(padded_ic / (padded_ic / ic_l2_blk), CH_DT_BLK());
    }

    return ic_l2_blk;
}

void conv2d_n16cx_implicit_gemm_fp32_fma_executor::init_preproc_param()
{
    schedule_param_.ic_per_gp = conv_param_->channels / conv_param_->group;
    schedule_param_.oc_per_gp = conv_param_->num_output / conv_param_->group;
    schedule_param_.padded_ic = round_up(schedule_param_.ic_per_gp, CH_DT_BLK());
    schedule_param_.padded_oc = round_up(schedule_param_.oc_per_gp, CH_DT_BLK());

    const int64_t dst_w       = dst_shape_->GetDim(3);
    schedule_param_.ow_kr_blk = BLK1X6_OW_RF();
#define REDUN_W(W, W_BLK) (float(round_up(W, W_BLK)) / (W)-1.0f)
    if (REDUN_W(dst_w, schedule_param_.ow_kr_blk) > 0.201f) {
        for (int32_t ow_blk = BLK1X6_OW_RF() - 1; ow_blk >= BLK1X6_OW_RF() - 2; --ow_blk) {
            if (REDUN_W(dst_w, ow_blk) < REDUN_W(dst_w, schedule_param_.ow_kr_blk)) {
                schedule_param_.ow_kr_blk = ow_blk;
            }
        }
    }
#undef REDUN_W
    schedule_param_.oc_kr_blk = BLK1X6_OC_RF() * OC_RF_BLK();
    schedule_param_.cur_batch = 0;
    schedule_param_.cur_group = 0;
    schedule_param_.cur_ic    = 0;
}

void conv2d_n16cx_implicit_gemm_fp32_fma_executor::cal_kernel_tunning_param()
{
    const conv2d_fp32_param &cp = *conv_param_;
    kernel_schedule_param &sp   = schedule_param_;

    const int32_t num_thread = PPL_OMP_MAX_THREADS();
    const int32_t batch      = src_shape_->GetDim(0);
    const int32_t src_h      = src_shape_->GetDim(2);
    const int32_t src_w      = src_shape_->GetDim(3);
    const int32_t dst_h      = dst_shape_->GetDim(2);
    const int32_t dst_w      = dst_shape_->GetDim(3);

    const int32_t ext_kernel_h = (cp.kernel_h - 1) * cp.dilation_h + 1;
    const int32_t ext_kernel_w = (cp.kernel_w - 1) * cp.dilation_w + 1;
    const int32_t padded_src_h = src_h + 2 * cp.pad_h;
    const int32_t padded_src_w = src_w + 2 * cp.pad_w;

    const float l2_cap_per_core = (ppl::common::GetCpuCacheL2() == 0 ? ASSUME_L2_BYTES() : ppl::common::GetCpuCacheL2()) * L2_RATIO() / sizeof(float);
    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (ASSUME_L3_BYTES() * num_thread) : ppl::common::GetCpuCacheL3()) * L3_RATIO() / sizeof(float);

    sp.ic_l2_blk = cal_ic_l2_blk(cp);
    sp.ic_l2_cnt = div_up(sp.padded_ic, sp.ic_l2_blk);

    const int32_t flt_space_per_ker  = sp.ic_l2_blk * cp.kernel_h * cp.kernel_w * sp.oc_kr_blk;
    const int32_t oc_l2_blk_by_space = round_up(int32_t(l2_cap_per_core / flt_space_per_ker * sp.oc_kr_blk), CH_DT_BLK());
    sp.oc_l2_blk                     = min(min(max(oc_l2_blk_by_space, OC_L2_BLK_MIN()), OC_L2_BLK_MAX()), sp.padded_oc);
    if (mod_up(sp.padded_oc, sp.oc_l2_blk) < OC_L2_BLK_TAIL_RATIO() * sp.oc_l2_blk) {
        sp.oc_l2_blk = round_up(sp.padded_oc / max(sp.padded_oc / sp.oc_l2_blk, 1), CH_DT_BLK());
    }

    sp.mb_l3_blk = min(batch, num_thread);
    sp.gp_l3_blk = 1;

    sp.oh_l3_blk = dst_h;
    sp.ow_l3_blk = round_up(dst_w, sp.ow_kr_blk);
    sp.ih_l3_blk = (sp.oh_l3_blk - 1) * cp.stride_h + ext_kernel_h;
    sp.iw_l3_blk = (sp.ow_l3_blk - 1) * cp.stride_w + ext_kernel_w;

    const int32_t small_channels = static_cast<int32_t>(sp.padded_ic <= CH_DT_BLK() && sp.padded_oc <= CH_DT_BLK());
#define CACHE_CONFLICT(H, W) (((H) * (W)) % CACHE_CONFLICT_HW() == 0)
#define CACHE_ALIGNED(H, W)  (((H) * (W)) % CACHE_ALIGNED_HW() == 0)
    while (sp.ih_l3_blk * sp.iw_l3_blk > HW_L3_BLK_MAX() * (small_channels + 1)) {
        if (sp.oh_l3_blk > (cp.sparse_level() > 1.001f ? OH_L3_BLK_MIN() : OH_L3_BLK_MID())) {
            sp.oh_l3_blk -= 1;
        } else if (sp.ow_l3_blk > (OW_L3_BLK_MIN(sp))) {
            sp.ow_l3_blk -= sp.ow_kr_blk;
        } else {
            break;
        }
        sp.ih_l3_blk = (sp.oh_l3_blk - 1) * cp.stride_h + ext_kernel_h;
        sp.iw_l3_blk = (sp.ow_l3_blk - 1) * cp.stride_w + ext_kernel_w;
    }
    sp.oh_l3_blk = min(sp.oh_l3_blk, dst_h);
    sp.ow_l3_blk = min(sp.ow_l3_blk, dst_w);

    if (mod_up(dst_h, sp.oh_l3_blk) < OH_L3_BLK_TAIL_RATIO() * sp.oh_l3_blk) {
        sp.oh_l3_blk = dst_h / max(dst_h / sp.oh_l3_blk, 1);
    }
    if (mod_up(dst_w, sp.ow_l3_blk) < OW_L3_BLK_TAIL_RATIO() * sp.ow_l3_blk) {
        sp.ow_l3_blk = round_up(dst_w / max(dst_w / sp.ow_l3_blk, 1), sp.ow_kr_blk);
    }
    sp.ih_l3_blk = (sp.oh_l3_blk - 1) * cp.stride_h + ext_kernel_h;
    sp.iw_l3_blk = (sp.ow_l3_blk - 1) * cp.stride_w + ext_kernel_w;

    if (sp.oh_l3_blk == dst_h && sp.ow_l3_blk == dst_w && !CACHE_CONFLICT(padded_src_h, padded_src_w)) {
        sp.ih_l3_blk = padded_src_h;
        sp.iw_l3_blk = padded_src_w;
    }

    sp.ih_l3_buf = sp.ih_l3_blk;
    sp.iw_l3_buf = sp.iw_l3_blk;

    sp.prefetch_src = 1;
    if (cp.pad_h == 0 && cp.pad_w == 0) {
        if (CACHE_CONFLICT(src_h, src_w)) {
            sp.compute_policy = COMPUTE_POLICY_BLOCK_TRANS();
            sp.prefetch_src   = 0;
        } else {
            sp.compute_policy = COMPUTE_POLICY_NO_TRANS();
            if (CACHE_ALIGNED(src_h, src_w)) {
                sp.prefetch_src = 0;
            }
        }
    } else {
        sp.compute_policy = COMPUTE_POLICY_BLOCK_TRANS();
        sp.prefetch_src   = 0;
    }
    if (cp.sparse_level() > 1.001f) {
        sp.prefetch_src = 0;
    }

    if (CACHE_CONFLICT(sp.ih_l3_buf, sp.iw_l3_buf)) {
        ++sp.iw_l3_buf; // deep dark fantasy
    }

    sp.use_nt_store = 0;
    if (batch * cp.group * sp.padded_oc * dst_h * dst_w > l3_cap_all_core * 2) {
        sp.use_nt_store = 1;
    }

#undef CACHE_CONFLICT
#undef CACHE_ALIGNED

#ifdef PPL_USE_X86_OMP
#ifdef PPL_USE_X86_OMP_COLLAPSE
    if (num_thread != 1 && num_thread != sp.mb_l3_blk * sp.gp_l3_blk &&
        (mod_up(sp.mb_l3_blk * sp.gp_l3_blk * div_up(sp.padded_oc, sp.oc_l2_blk), num_thread) < THREAD_TAIL_RATIO() * num_thread ||
         mod_up(sp.padded_oc, sp.oc_l2_blk) < OC_L2_BLK_TAIL_RATIO() * sp.oc_l2_blk)) {
        const int32_t src_space_per_group = sp.ic_l2_blk * sp.ih_l3_buf * sp.iw_l3_buf;
        while (num_thread != sp.mb_l3_blk * sp.gp_l3_blk &&
               (THREAD_BODY_ROUND() * num_thread > sp.mb_l3_blk * sp.gp_l3_blk * div_up(sp.padded_oc, sp.oc_l2_blk) ||
                mod_up(sp.mb_l3_blk * sp.gp_l3_blk * div_up(sp.padded_oc, sp.oc_l2_blk), num_thread) < THREAD_TAIL_RATIO() * num_thread ||
                mod_up(sp.padded_oc, sp.oc_l2_blk) < OC_L2_BLK_TAIL_RATIO() * sp.oc_l2_blk)) {
            if (sp.gp_l3_blk < cp.group && sp.mb_l3_blk * (sp.gp_l3_blk + 1) * src_space_per_group < l3_cap_all_core) {
                ++sp.gp_l3_blk;
            } else if (sp.oc_l2_blk > OC_L2_BLK_MIN()) {
                sp.oc_l2_blk -= CH_DT_BLK();
            } else {
                break;
            }
        }
    }
#else
    if (num_thread != 1 && num_thread != div_up(sp.padded_oc, sp.oc_l2_blk)) {
        while (THREAD_BODY_ROUND() * num_thread > div_up(sp.padded_oc, sp.oc_l2_blk) ||
               mod_up(div_up(sp.padded_oc, sp.oc_l2_blk), num_thread) < THREAD_TAIL_RATIO() * num_thread ||
               mod_up(sp.padded_oc, sp.oc_l2_blk) < OC_L2_BLK_TAIL_RATIO() * sp.oc_l2_blk) {
            if (sp.oc_l2_blk > OC_L2_BLK_MIN()) {
                sp.oc_l2_blk -= CH_DT_BLK();
            } else {
                break;
            }
        }
    }
#endif
#endif
}

void conv2d_n16cx_implicit_gemm_fp32_fma_executor::cal_kernel_threading_param(const int32_t batch, const int32_t group, const int32_t ic)
{
    const conv2d_fp32_param &cp = *conv_param_;
    kernel_schedule_param &sp   = schedule_param_;

    const bool bg_changed = sp.cur_batch != batch || sp.cur_group != group;
    const bool ic_changed = sp.cur_ic != ic;

    sp.cur_batch = batch;
    sp.cur_group = group;
    sp.cur_ic    = ic;

#ifdef PPL_USE_X86_OMP
    const int32_t num_thread = PPL_OMP_MAX_THREADS();
#endif

    if (bg_changed) {
        sp.oh_l2_blk = cp.sparse_level() > 1.001f ? 1 : sp.oh_l3_blk;
        sp.ow_l2_blk = round_up(sp.ow_l3_blk, sp.ow_kr_blk);
#ifdef PPL_USE_X86_OMP_COLLAPSE
        // Split the block for load-balancing
        if (num_thread != 1 && num_thread != batch * group &&
            (mod_up(batch * group * div_up(sp.padded_oc, sp.oc_l2_blk), num_thread) < THREAD_TAIL_RATIO() * num_thread ||
             mod_up(sp.padded_oc, sp.oc_l2_blk) < OC_L2_BLK_TAIL_RATIO() * sp.oc_l2_blk)) {
            const int32_t num_tasks_bgo = batch * group * div_up(sp.padded_oc, sp.oc_l2_blk);
            while (
                THREAD_BODY_ROUND() * num_thread > num_tasks_bgo * div_up(sp.oh_l3_blk, sp.oh_l2_blk) * div_up(sp.ow_l3_blk, sp.ow_l2_blk) ||
                mod_up(num_tasks_bgo * div_up(sp.oh_l3_blk, sp.oh_l2_blk) * div_up(sp.ow_l3_blk, sp.ow_l2_blk), num_thread) < THREAD_TAIL_RATIO() * num_thread ||
                mod_up(sp.oh_l3_blk, sp.oh_l2_blk) < OH_L2_BLK_TAIL_RATIO() * sp.oh_l2_blk ||
                mod_up(sp.ow_l3_blk, sp.ow_l2_blk) < OW_L2_BLK_TAIL_RATIO() * sp.ow_l2_blk) {
                if (sp.oh_l2_blk > 1) {
                    sp.oh_l2_blk -= 1;
                } else if (sp.ow_l2_blk > sp.ow_kr_blk) {
                    sp.ow_l2_blk -= sp.ow_kr_blk;
                } else {
                    break;
                }
            }
        }
#endif
        sp.oh_l2_blk = min(sp.oh_l3_blk, sp.oh_l2_blk);
        sp.ow_l2_blk = min(sp.ow_l3_blk, sp.ow_l2_blk);
    }

    if (sp.compute_policy == COMPUTE_POLICY_BLOCK_TRANS() && (bg_changed || ic_changed)) {
        sp.ih_tr_blk = sp.ih_l3_blk;
        sp.iw_tr_blk = sp.iw_l3_blk;
#ifdef PPL_USE_X86_OMP_COLLAPSE
        const int32_t num_tasks_bgi = batch * group * div_up(ic, CH_DT_BLK());
        if (num_tasks_bgi < num_thread) {
            sp.ih_tr_blk = max(sp.ih_tr_blk / div_up(num_thread, num_tasks_bgi), IH_TR_BLK_MIN());
            if (num_tasks_bgi * div_up(sp.ih_l3_blk, sp.ih_tr_blk) < num_thread) {
                sp.iw_tr_blk = max(sp.iw_tr_blk / div_up(num_thread, num_tasks_bgi * div_up(sp.ih_l3_blk, sp.ih_tr_blk)), IW_TR_BLK_MIN());
            }
            if (mod_up(sp.iw_l3_blk, sp.iw_tr_blk) < IW_TR_BLK_TAIL_RATIO() * sp.iw_tr_blk) {
                sp.iw_tr_blk = sp.iw_l3_blk / (sp.iw_l3_blk / sp.iw_tr_blk);
            }
        }
#endif
    }
}

uint64_t conv2d_n16cx_implicit_gemm_fp32_fma_executor::cal_temp_buffer_size()
{
    const kernel_schedule_param &sp = schedule_param_;

    uint64_t src_trans_size = 64u; // avoid empty
    if (sp.compute_policy == COMPUTE_POLICY_BLOCK_TRANS()) {
        src_trans_size = uint64_t(sp.mb_l3_blk * sp.gp_l3_blk) * sp.ic_l2_blk * sp.ih_l3_buf * sp.iw_l3_buf * sizeof(float);
    }

    return src_trans_size;
}

ppl::common::RetCode conv2d_n16cx_implicit_gemm_fp32_fma_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();
    cal_kernel_tunning_param();
    cal_kernel_threading_param(schedule_param_.mb_l3_blk, schedule_param_.gp_l3_blk, schedule_param_.ic_l2_blk);

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n16cx_implicit_gemm_fp32_fma_executor::execute()
{
    if (!conv_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_) || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const conv2d_fp32_param &cp     = *conv_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int32_t batch = src_shape_->GetDim(0);
    const int32_t src_h = src_shape_->GetDim(2);
    const int32_t src_w = src_shape_->GetDim(3);
    const int32_t dst_h = dst_shape_->GetDim(2);
    const int32_t dst_w = dst_shape_->GetDim(3);

    const int32_t padded_src_c = round_up(src_shape_->GetDim(1), CH_DT_BLK());
    const int32_t padded_dst_c = round_up(dst_shape_->GetDim(1), CH_DT_BLK());

    const int32_t ext_kernel_h = (cp.kernel_h - 1) * cp.dilation_h + 1;
    const int32_t ext_kernel_w = (cp.kernel_w - 1) * cp.dilation_w + 1;
    const int64_t padded_rf_oc = round_up(sp.oc_per_gp, OC_RF_BLK());

    const int64_t src_h_stride   = int64_t(src_w) * CH_DT_BLK();
    const int64_t src_g_stride   = int64_t(sp.padded_ic) * src_h * src_w;
    const int64_t src_b_stride   = int64_t(padded_src_c) * src_h * src_w;
    const int64_t src_icb_stride = int64_t(src_h) * src_w * CH_DT_BLK();
    const int64_t dst_h_stride   = int64_t(dst_w) * CH_DT_BLK();
    const int64_t dst_g_stride   = int64_t(sp.padded_oc) * dst_h * dst_w;
    const int64_t dst_b_stride   = int64_t(padded_dst_c) * dst_h * dst_w;
    const int64_t flt_g_stride   = int64_t(sp.ic_l2_cnt) * sp.padded_oc * cp.kernel_h * cp.kernel_w * sp.ic_l2_blk;
    const int64_t flt_k_stride   = int64_t(sp.ic_l2_blk) * CH_DT_BLK();
    const int64_t bias_g_stride  = sp.padded_oc;

    const bool with_sum   = cp.fuse_flag & conv_fuse_flag::sum;
    const bool with_relu  = cp.fuse_flag & conv_fuse_flag::relu;
    const bool with_relu6 = cp.fuse_flag & conv_fuse_flag::relu6;

    int64_t sum_src_b_stride = 0;
    if (with_sum) {
        sum_src_b_stride = int64_t(round_up(sum_src_shape_->GetDim(1), CH_DT_BLK())) * dst_h * dst_w;
    }

    float *src_trans                   = (float *)temp_buffer_;
    const int64_t src_trans_h_stride   = int64_t(sp.iw_l3_buf) * CH_DT_BLK();
    const int64_t src_trans_g_stride   = int64_t(sp.ih_l3_buf) * sp.iw_l3_buf * sp.ic_l2_blk;
    const int64_t src_trans_b_stride   = int64_t(sp.gp_l3_blk) * sp.ih_l3_buf * sp.iw_l3_buf * sp.ic_l2_blk;
    const int64_t src_trans_icb_stride = int64_t(sp.ih_l3_buf) * sp.iw_l3_buf * CH_DT_BLK();

    int64_t share_param[SHAR_PARAM_LEN()];
    share_param[KH_IDX()] = cp.kernel_h;
    share_param[KW_IDX()] = cp.kernel_w;
    for (int64_t gpl3 = 0; gpl3 < cp.group; gpl3 += sp.gp_l3_blk) {
        const int64_t gpl3_eff = min<int64_t>(cp.group - gpl3, sp.gp_l3_blk);
        for (int64_t mbl3 = 0; mbl3 < batch; mbl3 += sp.mb_l3_blk) {
            const int64_t mbl3_eff = min<int64_t>(batch - mbl3, sp.mb_l3_blk);
            for (int64_t icl2 = 0; icl2 < sp.ic_per_gp; icl2 += sp.ic_l2_blk) {
                const int64_t icl2_eff = min<int64_t>(sp.ic_per_gp - icl2, sp.ic_l2_blk);
                const bool is_first_ic = icl2 == 0;
                const bool is_last_ic  = (icl2 + sp.ic_l2_blk >= sp.ic_per_gp);

                cal_kernel_threading_param(mbl3_eff, gpl3_eff, icl2_eff);

                const float *base_src = src_;
                const float *base_his = dst_;
                const float *base_flt = cvt_filter_;
                float *base_dst       = dst_;

                int64_t base_src_b_stride   = src_b_stride;
                int64_t base_src_g_stride   = src_g_stride;
                int64_t base_src_icb_stride = src_icb_stride;
                int64_t base_src_h_stride   = src_h_stride;
                int64_t base_his_b_stride   = dst_b_stride;
                int64_t base_dst_b_stride   = dst_b_stride;
                uint64_t kernel_flags       = 0;
                if (is_first_ic) {
                    if (with_sum) {
                        base_his          = sum_src_;
                        base_his_b_stride = sum_src_b_stride;
                        kernel_flags |= KERNEL_FLAG_AD_BIAS();
                    } else {
                        kernel_flags |= KERNEL_FLAG_LD_BIAS();
                    }
                }
                if (is_last_ic) {
                    if (with_relu) {
                        kernel_flags |= KERNEL_FLAG_RELU();
                    } else if (with_relu6) {
                        kernel_flags |= KERNEL_FLAG_RELU6();
                    }
                }

                base_src += mbl3 * base_src_b_stride + gpl3 * base_src_g_stride + icl2 * src_h * src_w;
                base_his += mbl3 * base_his_b_stride + gpl3 * dst_g_stride;
                base_dst += mbl3 * base_dst_b_stride + gpl3 * dst_g_stride;
                base_flt += gpl3 * flt_g_stride + icl2 * sp.padded_oc * cp.kernel_h * cp.kernel_w;

                for (int64_t ohl3 = 0; ohl3 < dst_h; ohl3 += sp.oh_l3_blk) {
                    const int64_t ohl3_eff = min<int64_t>(dst_h - ohl3, sp.oh_l3_blk);
                    const int64_t ihl3     = ohl3 * cp.stride_h;
                    const int64_t ihl3_eff = (ohl3_eff - 1) * cp.stride_h + ext_kernel_h;
                    for (int64_t owl3 = 0; owl3 < dst_w; owl3 += sp.ow_l3_blk) {
                        const int64_t owl3_eff = min<int64_t>(dst_w - owl3, sp.ow_l3_blk);
                        const int64_t iwl3     = owl3 * cp.stride_w;
                        const int64_t iwl3_eff = (owl3_eff - 1) * cp.stride_w + ext_kernel_w;
                        const float *tile_src  = base_src + (ihl3 - cp.pad_h) * base_src_h_stride + (iwl3 - cp.pad_w) * CH_DT_BLK();
                        const float *tile_his  = base_his + ohl3 * dst_h_stride + owl3 * CH_DT_BLK();
                        float *tile_dst        = base_dst + ohl3 * dst_h_stride + owl3 * CH_DT_BLK();

                        int64_t tile_src_b_stride   = base_src_b_stride;
                        int64_t tile_src_g_stride   = base_src_g_stride;
                        int64_t tile_src_icb_stride = base_src_icb_stride;
                        int64_t tile_src_h_stride   = base_src_h_stride;
                        if (sp.compute_policy == COMPUTE_POLICY_BLOCK_TRANS()) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
                            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(5)
#endif
                            for (int64_t b = 0; b < mbl3_eff; ++b) {
                                for (int64_t g = 0; g < gpl3_eff; ++g) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                                    PRAGMA_OMP_PARALLEL_FOR()
#endif
                                    for (int64_t icb = 0; icb < div_up(icl2_eff, CH_DT_BLK()); ++icb) {
                                        for (int64_t iht = 0; iht < ihl3_eff; iht += sp.ih_tr_blk) {
                                            for (int64_t iwt = 0; iwt < iwl3_eff; iwt += sp.iw_tr_blk) {
#ifdef PPL_X86_KERNEL_TIMING
                                                profiler_.tic(SRCTR_TIMER());
#endif
                                                const int64_t iht_eff           = min<int64_t>(ihl3_eff - iht, sp.ih_tr_blk);
                                                const int64_t iwt_eff           = min<int64_t>(iwl3_eff - iwt, sp.iw_tr_blk);
                                                const float *l_tile_src         = tile_src + b * tile_src_b_stride + g * tile_src_g_stride + icb * tile_src_icb_stride + iht * tile_src_h_stride + iwt * CH_DT_BLK();
                                                float *l_src_trans              = src_trans + b * src_trans_b_stride + g * src_trans_g_stride + icb * src_trans_icb_stride + iht * src_trans_h_stride + iwt * CH_DT_BLK();
                                                const int64_t mapped_iht        = ihl3 + iht - cp.pad_h;
                                                const int64_t top_pad_ih_len    = min<int64_t>(max<int64_t>(0 - mapped_iht, 0), iht_eff);
                                                const int64_t bottom_pad_ih_len = max<int64_t>(mapped_iht + iht_eff - max<int64_t>(src_h, mapped_iht), 0);
                                                const int64_t body_ih_len       = iht_eff - top_pad_ih_len - bottom_pad_ih_len;
                                                const int64_t mapped_iwt        = iwl3 + iwt - cp.pad_w;
                                                const int64_t right_pad_iw_len  = min<int64_t>(max<int64_t>(0 - mapped_iwt, 0), iwt_eff);
                                                const int64_t left_pad_iw_len   = max<int64_t>(mapped_iwt + iwt_eff - max<int64_t>(src_w, mapped_iwt), 0);
                                                const int64_t body_iw_len       = iwt_eff - right_pad_iw_len - left_pad_iw_len;
                                                for (int64_t ih = 0; ih < top_pad_ih_len; ++ih) {
                                                    memset32_avx(l_src_trans, 0, iwt_eff * CH_DT_BLK());
                                                    l_src_trans += src_trans_h_stride;
                                                    l_tile_src += tile_src_h_stride;
                                                }
                                                for (int64_t ih = 0; ih < body_ih_len - 1; ++ih) {
                                                    const float *w_tile_src = l_tile_src;
                                                    float *w_src_trans      = l_src_trans;
                                                    memset32_avx(w_src_trans, 0, right_pad_iw_len * CH_DT_BLK());
                                                    w_src_trans += right_pad_iw_len * CH_DT_BLK();
                                                    w_tile_src += right_pad_iw_len * CH_DT_BLK();
                                                    memcpy32_avx(w_src_trans, w_tile_src, body_iw_len * CH_DT_BLK());
                                                    w_src_trans += body_iw_len * CH_DT_BLK();
                                                    memset32_avx(w_src_trans, 0, left_pad_iw_len * CH_DT_BLK());
                                                    l_src_trans += src_trans_h_stride;
                                                    l_tile_src += tile_src_h_stride;
                                                }
                                                if (body_ih_len > 0) {
                                                    const float *w_tile_src = l_tile_src;
                                                    float *w_src_trans      = l_src_trans;
                                                    memset32_avx(w_src_trans, 0, right_pad_iw_len * CH_DT_BLK());
                                                    w_src_trans += right_pad_iw_len * CH_DT_BLK();
                                                    w_tile_src += right_pad_iw_len * CH_DT_BLK();
                                                    memcpy32_avx(w_src_trans, w_tile_src, body_iw_len * CH_DT_BLK());
                                                    w_src_trans += body_iw_len * CH_DT_BLK();
                                                    memset32_avx(w_src_trans, 0, left_pad_iw_len * CH_DT_BLK());
                                                    l_src_trans += src_trans_h_stride;
                                                    l_tile_src += tile_src_h_stride;
                                                }
                                                for (int64_t ih = 0; ih < bottom_pad_ih_len; ++ih) {
                                                    memset32_avx(l_src_trans, 0, iwt_eff * CH_DT_BLK());
                                                    l_src_trans += src_trans_h_stride;
                                                    l_tile_src += tile_src_h_stride;
                                                }
#ifdef PPL_X86_KERNEL_TIMING
                                                profiler_.toc(SRCTR_TIMER());
#endif
                                            }
                                        }
                                    }
                                }
                            }
                            tile_src            = src_trans;
                            tile_src_b_stride   = src_trans_b_stride;
                            tile_src_g_stride   = src_trans_g_stride;
                            tile_src_icb_stride = src_trans_icb_stride;
                            tile_src_h_stride   = src_trans_h_stride;
                        }
                        share_param[CHANNELS_IDX()]       = icl2_eff;
                        share_param[SRC_ICB_STRIDE_IDX()] = tile_src_icb_stride;
                        share_param[SRC_SH_STRIDE_IDX()]  = cp.stride_h * tile_src_h_stride;
                        share_param[SRC_SW_STRIDE_IDX()]  = cp.stride_w * CH_DT_BLK();
                        share_param[SRC_DH_STRIDE_IDX()]  = cp.dilation_h * tile_src_h_stride - cp.kernel_w * cp.dilation_w * CH_DT_BLK();
                        share_param[SRC_DW_STRIDE_IDX()]  = cp.dilation_w * CH_DT_BLK();
                        share_param[HIS_H_STRIDE_IDX()]   = dst_h_stride;
                        share_param[DST_H_STRIDE_IDX()]   = dst_h_stride;
                        share_param[FLT_K_STRIDE_IDX()]   = flt_k_stride;
                        share_param[FLAGS_IDX()]          = kernel_flags;
#ifdef PPL_USE_X86_OMP_COLLAPSE
                        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(5)
#endif
                        for (int64_t b = 0; b < mbl3_eff; ++b) {
                            for (int64_t g = 0; g < gpl3_eff; ++g) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                                PRAGMA_OMP_PARALLEL_FOR()
#endif
                                for (int64_t ocl2 = 0; ocl2 < padded_rf_oc; ocl2 += sp.oc_l2_blk) {
                                    for (int64_t ohl2 = 0; ohl2 < ohl3_eff; ohl2 += sp.oh_l2_blk) {
                                        for (int64_t owl2 = 0; owl2 < owl3_eff; owl2 += sp.ow_l2_blk) {
#ifdef PPL_X86_KERNEL_TIMING
                                            profiler_.tic(KERNEL_TIMER());
#endif
                                            int64_t kernel_param[PRIV_PARAM_LEN()];
                                            const int64_t ocl2_eff = min<int64_t>(padded_rf_oc - ocl2, sp.oc_l2_blk);
                                            const int64_t ohl2_eff = min<int64_t>(ohl3_eff - ohl2, sp.oh_l2_blk);
                                            const int64_t owl2_eff = min<int64_t>(owl3_eff - owl2, sp.ow_l2_blk);
                                            const int64_t ow_body  = round(owl2_eff, sp.ow_kr_blk);
                                            const int64_t ow_tail  = owl2_eff - ow_body;
                                            const int64_t ihl2     = ohl2 * cp.stride_h;
                                            const int64_t iwl2     = owl2 * cp.stride_w;

                                            const bool small_stride_optimal = cp.stride_w == 1 || cp.stride_w == 2;
                                            const int64_t stride_sel        = small_stride_optimal ? cp.stride_w : 0;
                                            const int64_t prefetch_sel      = sp.prefetch_src;
                                            const int64_t nt_store_sel      = sp.use_nt_store;

                                            *reinterpret_cast<const float **>(kernel_param + FLT_IDX())  = base_flt + g * flt_g_stride + ocl2 * sp.ic_l2_blk * cp.kernel_h * cp.kernel_w;
                                            *reinterpret_cast<const float **>(kernel_param + BIAS_IDX()) = cvt_bias_ + (gpl3 + g) * bias_g_stride + ocl2;
                                            for (int64_t oc = ocl2; oc < ocl2 + ocl2_eff; oc += CH_DT_BLK()) {
                                                const int64_t oc_eff = min<int64_t>(ocl2 + ocl2_eff - oc, CH_DT_BLK());
                                                const int64_t oc_sel = div_up(oc_eff, OC_RF_BLK()) - 1;
                                                const float *l_src   = tile_src + b * tile_src_b_stride + g * tile_src_g_stride + ihl2 * tile_src_h_stride + iwl2 * CH_DT_BLK();
                                                const float *l_his   = tile_his + b * base_his_b_stride + g * dst_g_stride + oc * dst_h * dst_w + ohl2 * dst_w * CH_DT_BLK() + owl2 * CH_DT_BLK();
                                                float *l_dst         = tile_dst + b * base_dst_b_stride + g * dst_g_stride + oc * dst_h * dst_w + ohl2 * dst_w * CH_DT_BLK() + owl2 * CH_DT_BLK();

                                                kernel_param[OH_IDX()] = ohl2_eff;
                                                if (ow_body) {
                                                    *reinterpret_cast<const float **>(kernel_param + SRC_IDX()) = l_src;
                                                    *reinterpret_cast<const float **>(kernel_param + HIS_IDX()) = l_his;
                                                    *reinterpret_cast<float **>(kernel_param + DST_IDX())       = l_dst;

                                                    kernel_param[OW_IDX()] = ow_body;
                                                    conv2d_n16cx_implicit_gemm_kernel_fp32_fma_blk1x6_table[stride_sel][nt_store_sel][prefetch_sel][oc_sel][sp.ow_kr_blk - 1](kernel_param, share_param);
                                                }
                                                if (ow_tail) {
                                                    *reinterpret_cast<const float **>(kernel_param + SRC_IDX()) = l_src + ow_body * cp.stride_w * CH_DT_BLK();
                                                    *reinterpret_cast<const float **>(kernel_param + HIS_IDX()) = l_his + ow_body * CH_DT_BLK();
                                                    *reinterpret_cast<float **>(kernel_param + DST_IDX())       = l_dst + ow_body * CH_DT_BLK();

                                                    kernel_param[OW_IDX()] = ow_tail;
                                                    conv2d_n16cx_implicit_gemm_kernel_fp32_fma_blk1x6_table[stride_sel][nt_store_sel][prefetch_sel][oc_sel][ow_tail - 1](kernel_param, share_param);
                                                }
                                                *reinterpret_cast<const float **>(kernel_param + FLT_IDX()) += CH_DT_BLK() * sp.ic_l2_blk * cp.kernel_h * cp.kernel_w;
                                                *reinterpret_cast<const float **>(kernel_param + BIAS_IDX()) += CH_DT_BLK();
                                            }
#ifdef PPL_X86_KERNEL_TIMING
                                            profiler_.toc(KERNEL_TIMER());
#endif
                                        }
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
#ifdef PPL_X86_KERNEL_TIMING
                    profiler_.tic(FENCE_TIMER());
#endif
                    _mm_sfence();
#ifdef PPL_X86_KERNEL_TIMING
                    profiler_.toc(FENCE_TIMER());
#endif
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n16cx_implicit_gemm_fp32_fma_manager::gen_cvt_weights(const float *filter, const float *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int32_t oc_per_gp = param_.num_output / param_.group;
    const int32_t padded_oc = round_up(oc_per_gp, CH_DT_BLK());
    const int32_t ic_l2_blk = conv2d_n16cx_implicit_gemm_fp32_fma_executor::cal_ic_l2_blk(param_);

    cvt_bias_size_ = param_.group * padded_oc;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    for (int32_t g = 0; g < param_.group; ++g) {
        memcpy(cvt_bias_ + g * padded_oc, bias + g * oc_per_gp, oc_per_gp * sizeof(float));
        memset(cvt_bias_ + g * padded_oc + oc_per_gp, 0, (padded_oc - oc_per_gp) * sizeof(float));
    }

    cvt_filter_size_ = reorder_goidhw_gIOdhwB16i16o_fp32_get_dst_size(
        param_.group, param_.num_output, param_.channels,
        1, param_.kernel_h, param_.kernel_w, ic_l2_blk);
    cvt_filter_size_ /= sizeof(float);
    cvt_filter_      = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    return reorder_goidhw_gIOdhwB16i16o_fp32(
        filter, param_.group, param_.num_output, param_.channels,
        1, param_.kernel_h, param_.kernel_w, ic_l2_blk, cvt_filter_);
}

bool conv2d_n16cx_implicit_gemm_fp32_fma_manager::is_supported()
{
    if (param_.is_pointwise() && param_.stride_h == 1 && param_.stride_w == 1) {
        return false;
    }
    bool aligned_channels   = param_.channels / param_.group % CH_DT_BLK() == 0;
    bool aligned_num_output = param_.num_output / param_.group % CH_DT_BLK() == 0;
    return (param_.group == 1) || (aligned_channels && aligned_num_output);
}

conv2d_fp32_executor *conv2d_n16cx_implicit_gemm_fp32_fma_manager::gen_executor()
{
    return new conv2d_n16cx_implicit_gemm_fp32_fma_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86
