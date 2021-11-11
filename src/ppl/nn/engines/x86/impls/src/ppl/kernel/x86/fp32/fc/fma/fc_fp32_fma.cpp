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
#include <string.h>

#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/fp32/fc/fma/fc_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/im2col_gemm/fma/conv_gemm_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/common/sys.h"

#define CH_DT_BLK() M6_N_DT_BLK()
#define OC_RF_BLK() N_RF_BLK()
#define OC_RF_CNT() M6_N_RF()
#define B_KR_BLK()  6

#define IC_L2_BLK_MAX()        (64 * CH_DT_BLK())
#define IC_TR_THR_MAX()        4
#define IC_L2_BLK_TAIL_RATIO() 0.501
#define OC_L2_BLK_MAX()        (2 * CH_DT_BLK())

namespace ppl { namespace kernel { namespace x86 {

int32_t fc_fp32_fma_executor::cal_ic_l2_blk(const fc_fp32_param &param)
{
    const int32_t padded_ic = round_up(param.channels, CH_DT_BLK());

    int32_t ic_l2_blk = min(IC_L2_BLK_MAX(), padded_ic);
    if (mod_up(padded_ic, ic_l2_blk) < IC_L2_BLK_TAIL_RATIO() * ic_l2_blk) {
        ic_l2_blk = round_up(padded_ic / (padded_ic / ic_l2_blk), CH_DT_BLK());
    }

    return ic_l2_blk;
}

void fc_fp32_fma_executor::cal_kernel_tunning_param()
{
    schedule_param_.ic_l2_blk = cal_ic_l2_blk(*fc_param_);
    schedule_param_.ic_l2_cnt = div_up(fc_param_->channels, schedule_param_.ic_l2_blk);
    schedule_param_.oc_l2_blk = round_up(min<int32_t>(fc_param_->num_output, OC_L2_BLK_MAX()), CH_DT_BLK());

    schedule_param_.multi_batch  = src_shape_->GetDim(0) != 1;
    schedule_param_.unaligned_oc = fc_param_->num_output % OC_RF_BLK() != 0;
}

uint64_t fc_fp32_fma_executor::cal_temp_buffer_size()
{
    const uint64_t dst_buf_size = schedule_param_.unaligned_oc ? (src_shape_->GetDim(0) * CH_DT_BLK() * sizeof(float)) : 0;

    uint64_t src_trans_size = 0;
    if (schedule_param_.multi_batch) {
        src_trans_size = src_shape_->GetDim(0) * schedule_param_.ic_l2_blk * sizeof(float);
    }

    return max<uint64_t>(src_trans_size + dst_buf_size, 64u);
}

ppl::common::RetCode fc_fp32_fma_executor::prepare()
{
    if (!fc_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    cal_kernel_tunning_param();

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode fc_fp32_fma_executor::execute()
{
    if (!fc_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const fc_fp32_param &fp         = *fc_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int32_t batch     = src_shape_->GetDim(0);
    const int32_t padded_oc = round_up(fp.num_output, CH_DT_BLK());

    const int64_t src_b_stride     = fp.channels;
    const int64_t dst_b_stride     = fp.num_output;
    const int64_t dst_buf_b_stride = CH_DT_BLK();

    const bool with_relu = fp.fuse_flag & fc_fuse_flag::RELU;

    int64_t src_trans_size = 0;
    if (sp.multi_batch) {
        src_trans_size = src_shape_->GetDim(0) * schedule_param_.ic_l2_blk;
    }

    float *src_trans    = (float *)temp_buffer_;
    float *base_dst_buf = src_trans + src_trans_size;

    for (int64_t icl2 = 0; icl2 < fp.channels; icl2 += sp.ic_l2_blk) {
        const int64_t icl2_eff        = min<int64_t>(fp.channels - icl2, sp.ic_l2_blk);
        const int64_t padded_icl2_eff = round_up(icl2_eff, CH_DT_BLK());
        const bool is_first_ic        = icl2 == 0;
        const bool is_last_ic         = (icl2 + sp.ic_l2_blk >= fp.channels);
        uint64_t kernel_flags         = 0;
        if (!is_first_ic) {
            kernel_flags |= KERNEL_FLAG_LOAD_C();
        }
        if (is_last_ic) {
            kernel_flags |= KERNEL_FLAG_ADD_V();
            if (with_relu) {
                kernel_flags |= KERNEL_FLAG_RELU();
            }
        }
        const float *base_src = src_ + icl2;
        const float *base_flt = cvt_filter_ + icl2 * padded_oc;
        if (sp.multi_batch) {
            int64_t ic_tr_blk = CH_DT_BLK();
            if (div_up(icl2_eff, ic_tr_blk) > IC_TR_THR_MAX()) {
                ic_tr_blk = round_up(icl2_eff / IC_TR_THR_MAX(), CH_DT_BLK());
            }
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t ict = 0; ict < icl2_eff; ict += ic_tr_blk) {
                const int64_t ict_eff = min<int64_t>(icl2_eff - ict, ic_tr_blk);
                for (int64_t icb = 0; icb < ict_eff; icb += CH_DT_BLK()) {
                    for (int64_t bb = 0; bb < batch; bb += B_KR_BLK()) {
                        const int64_t bb_eff  = min<int64_t>(batch - bb, B_KR_BLK());
                        const int64_t icb_eff = min<int64_t>(ict_eff - icb, CH_DT_BLK());
                        const float *l_src    = base_src + bb * src_b_stride + (ict + icb);
                        float *l_src_trans    = src_trans + bb * padded_icl2_eff + (ict + icb) * bb_eff;
                        for (int64_t b = 0; b < bb_eff; ++b) {
                            memcpy32_avx(l_src_trans, l_src, icb_eff);
                            memset32_avx(l_src_trans + icb_eff, 0, (CH_DT_BLK() - icb_eff));
                            l_src += src_b_stride;
                            l_src_trans += CH_DT_BLK();
                        }
                    }
                }
            }
        }

        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t ocl2 = 0; ocl2 < fp.num_output; ocl2 += sp.oc_l2_blk) {
            const int64_t ocl2_eff = min<int64_t>(fp.num_output - ocl2, sp.oc_l2_blk);
            int64_t priv_param[PRIV_PARAM_LEN()];
            int64_t shar_param[SHAR_PARAM_LEN()];
            PICK_PARAM(int64_t, shar_param, K_IDX())       = icl2_eff;
            PICK_PARAM(int64_t, shar_param, FLAGS_IDX())   = kernel_flags;
            PICK_PARAM(const float *, priv_param, B_IDX()) = base_flt + ocl2 * sp.ic_l2_blk;
            PICK_PARAM(const float *, priv_param, V_IDX()) = cvt_bias_ + ocl2;
            float *base_dst                                = dst_ + ocl2;
            for (int64_t oc = 0; oc < ocl2_eff; oc += CH_DT_BLK()) {
                const int64_t oc_eff    = min<int64_t>(ocl2_eff - oc, CH_DT_BLK());
                const int64_t oc_sel    = div_up(oc_eff, OC_RF_BLK()) - 1;
                const bool oc_unaligned = oc_eff % OC_RF_BLK() != 0;
                float *oc_dst           = base_dst;
                int64_t oc_dst_b_stride = dst_b_stride;
                if (oc_unaligned) {
                    oc_dst          = base_dst_buf;
                    oc_dst_b_stride = dst_buf_b_stride;
                }
                PICK_PARAM(float *, priv_param, C_IDX())          = oc_dst;
                PICK_PARAM(int64_t, shar_param, C_M_STRIDE_IDX()) = oc_dst_b_stride;
                PICK_PARAM(const float *, priv_param, A_IDX())    = sp.multi_batch ? src_trans : base_src;

                const int64_t b_body = round(batch, B_KR_BLK());
                const int64_t b_tail = batch - b_body;
                if (sp.multi_batch && b_body) {
                    PICK_PARAM(int64_t, shar_param, A_MBLK_STRIDE_IDX()) = B_KR_BLK() * padded_icl2_eff;
                    PICK_PARAM(int64_t, shar_param, A_KBLK_STRIDE_IDX()) = B_KR_BLK() * CH_DT_BLK();
                    PICK_PARAM(int64_t, priv_param, M_IDX())             = b_body;
                    conv_gemm_kernel_fp32_fma_table[oc_sel][B_KR_BLK() - 1](priv_param, shar_param);
                    PICK_PARAM(const float *, priv_param, A_IDX()) += b_body * padded_icl2_eff;
                    PICK_PARAM(float *, priv_param, C_IDX()) += b_body * oc_dst_b_stride;
                }
                if (b_tail) {
                    PICK_PARAM(int64_t, shar_param, A_KBLK_STRIDE_IDX()) = b_tail * CH_DT_BLK();
                    PICK_PARAM(int64_t, priv_param, M_IDX())             = b_tail;
                    conv_gemm_kernel_fp32_fma_table[oc_sel][b_tail - 1](priv_param, shar_param);
                }

                if (oc_unaligned && is_last_ic) {
                    float *l_dst     = base_dst;
                    float *l_dst_buf = base_dst_buf;
                    for (int64_t b = 0; b < batch; ++b) {
                        memcpy32_avx(l_dst, l_dst_buf, oc_eff);
                        l_dst += dst_b_stride;
                        l_dst_buf += dst_buf_b_stride;
                    }
                }
                PICK_PARAM(const float *, priv_param, B_IDX()) += CH_DT_BLK() * sp.ic_l2_blk;
                PICK_PARAM(const float *, priv_param, V_IDX()) += CH_DT_BLK();
                base_dst += CH_DT_BLK();
            }
        }
    }

    return common::RC_SUCCESS;
}

ppl::common::RetCode fc_fp32_fma_manager::gen_cvt_weights(const float *filter, const float *bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int32_t padded_oc = round_up(param_.num_output, CH_DT_BLK());
    const int32_t ic_l2_blk = fc_fp32_fma_executor::cal_ic_l2_blk(param_);

    cvt_bias_size_ = padded_oc;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    memcpy(cvt_bias_, bias, param_.num_output * sizeof(float));
    memset(cvt_bias_ + param_.num_output, 0, (padded_oc - param_.num_output) * sizeof(float));

    cvt_filter_size_ = reorder_goidhw_gIOdhwB16i16o_fp32_get_dst_size(
        1, param_.num_output, param_.channels, 1, 1, 1, ic_l2_blk);
    cvt_filter_size_ /= sizeof(float);
    cvt_filter_ = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    return reorder_goidhw_gIOdhwB16i16o_fp32(
        filter, 1, param_.num_output, param_.channels, 1, 1, 1, ic_l2_blk, cvt_filter_);
}

fc_fp32_executor *fc_fp32_fma_manager::gen_executor()
{
    return new fc_fp32_fma_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86
