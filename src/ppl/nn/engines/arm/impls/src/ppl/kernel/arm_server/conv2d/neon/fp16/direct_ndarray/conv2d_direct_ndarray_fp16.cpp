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

#ifdef PPLNN_USE_ARMV8_2_FP16

#include "ppl/kernel/arm_server/conv2d/neon/fp16/direct_ndarray/conv2d_direct_ndarray_fp16.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp16/direct_ndarray/conv2d_direct_ndarray_h1wx_kernel.h"
#include "ppl/kernel/arm_server/conv2d/neon/fp16/direct_ndarray/conv2d_direct_ndarray_h1w1_kernel.h"

#include <arm_neon.h>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#if defined PPL_USE_ARM_SERVER_OMP
#include <omp.h>
#endif

#include "ppl/common/arm/sysinfo.h"
#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

#define CBLK()  8
#define ICBLK() CBLK()
#define OCBLK() CBLK()

uint64_t conv2d_direct_ndarray_fp16_runtime_executor::cal_temp_buffer_size()
{
    return 0;
}

void conv2d_direct_ndarray_fp16_runtime_executor::adjust_schedule_param()
{
    return;
}

ppl::common::RetCode conv2d_direct_ndarray_fp16_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    adjust_schedule_param();
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_direct_ndarray_fp16_runtime_executor::execute()
{
    PRAGMA_OMP_PARALLEL()
    {
        const conv2d_param &cp                              = *conv_param_;
        const conv2d_direct_ndarray_fp16_kernel_param &kp   = ker_param_;
        const conv2d_direct_ndarray_fp16_schedule_param &sp = sched_param_;

        const __fp16 *input                         = (const __fp16 *)src_;
        const __fp16 *cvt_filter                    = (const __fp16 *)cvt_filter_;
        const __fp16 *bias                          = (const __fp16 *)cvt_bias_;
        __fp16 *output                              = (__fp16 *)dst_;
        __fp16 *sum                                 = (__fp16 *)sum_;
        const int64_t src_h                         = src_shape_->GetDim(2);
        const int64_t src_w                         = src_shape_->GetDim(3);
        const int64_t channels                      = src_shape_->GetDim(1);
        const int64_t num_output                    = cp.num_output;
        const int64_t dst_h                         = dst_shape_->GetDim(2);
        const int64_t dst_w                         = dst_shape_->GetDim(3);
        const int64_t flt_h                         = cp.kernel_h;
        const int64_t flt_w                         = cp.kernel_w;
        const int64_t pad_h                         = cp.pad_h;
        const int64_t pad_w                         = cp.pad_w;
        const int64_t strd_h                        = cp.stride_h;
        const int64_t strd_w                        = cp.stride_w;
        const int64_t dltn_h                        = cp.dilation_h;
        const int64_t dltn_w                        = cp.dilation_w;
        const int64_t num_batch                     = src_shape_->GetDim(0);

        int64_t ow_inner_start = std::max((int64_t)0, DIV_CEIL((pad_w - 0 * dltn_w), strd_w)); // inclusive
        int64_t ow_inner_end   = std::min((int64_t)dst_w, DIV_CEIL((src_w + pad_w - (flt_w - 1) * dltn_w), strd_w)); // exclusive
        ow_inner_start         = std::min(ow_inner_start, dst_w);
        ow_inner_end           = std::max(ow_inner_end, ow_inner_start);

        const int64_t num_output_pck = CEIL8(num_output);

        const int64_t dst_tile_h = kp.dst_tile_h;
        const int64_t dst_tile_w = kp.dst_tile_w;

        const int64_t ocblk2 = kp.ocblk2;
        const int64_t ic_tile = sp.ic_tile;

        const int64_t input_hw_num        = src_h * src_w;
        const int64_t input_chw_num       = channels * input_hw_num;
        const int64_t output_hw_num       = dst_h * dst_w;
        const int64_t output_batch_stride = num_output_pck * output_hw_num;
        const int64_t output_hwcb_num     = output_hw_num * CBLK();
        const int64_t output_wcb_num      = dst_w * CBLK();
        const int64_t flt_ichw_num        = channels * flt_h * flt_w;
        const int64_t flt_ic_stride       = flt_h * flt_w * ocblk2;

#if not defined PPL_USE_ARM_SERVER_OMP
        for (int64_t batch_id = 0; batch_id < num_batch; batch_id++) {
            const __fp16 *input_batch_base = input + batch_id * input_chw_num;
            for (int64_t ic_l1 = 0; ic_l1 < channels; ic_l1 += ic_tile) {
                const int64_t ic_remain     = std::min(ic_tile, channels - ic_l1);
                const uint32_t fuse_flag    = (ic_l1 + ic_tile >= channels) ? cp.fuse_flag : static_cast<uint32_t>(conv_fuse_flag::NONE);
                const __fp16 *input_ic_base = input_batch_base + ic_l1 * input_hw_num;
                for (int64_t oc_l1 = 0; oc_l1 < num_output_pck; oc_l1 += ocblk2) {
                    const __fp16 *filter_cc_base     = cvt_filter + oc_l1 * flt_ichw_num + ic_l1 * flt_ic_stride;
                    const __fp16 *const bias_oc_base = (ic_l1 == 0) ? (bias + oc_l1) : nullptr;
                    const int64_t oc_remains         = std::min(ocblk2, num_output_pck - oc_l1);
                    const ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_kernel_func_t *const conv_direct_kernel_func_table =
                        (oc_remains > OCBLK()) ? ppl_arm_server_kernel_fp16_conv_direct_ndarray_oc16_kernel_func_table : ppl_arm_server_kernel_fp16_conv_direct_ndarray_oc8_kernel_func_table;
                    const ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_h1w1_kernel_func_t conv_direct_kernel_h1w1_func =
                        (oc_remains > OCBLK()) ? ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_h1w1_kernel<16> : ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_h1w1_kernel<8>;
                    for (int64_t oh = 0; oh < dst_h; oh += dst_tile_h) {
                        __fp16 *output_h_base = output + batch_id * output_batch_stride + oc_l1 * output_hw_num + oh * output_wcb_num;
                        __fp16 *sum_h_base    = sum + batch_id * output_batch_stride + oc_l1 * output_hw_num + oh * output_wcb_num;
#else
        for (int64_t ic_l1 = 0; ic_l1 < channels; ic_l1 += ic_tile) {
            const uint32_t fuse_flag = (ic_l1 + ic_tile >= channels) ? cp.fuse_flag : 0;

            PRAGMA_OMP_FOR_COLLAPSE(3)
            for (int64_t batch_id = 0; batch_id < num_batch; batch_id++) {
                for (int64_t oc_l1 = 0; oc_l1 < num_output_pck; oc_l1 += ocblk2) {
                    for (int64_t oh = 0; oh < dst_h; oh += dst_tile_h) {
                        const __fp16 *input_ic_base      = input + batch_id * input_chw_num + ic_l1 * input_hw_num;
                        __fp16 *output_h_base            = output + batch_id * output_batch_stride + oc_l1 * output_hw_num + oh * output_wcb_num;
                        __fp16 *sum_h_base               = sum + batch_id * output_batch_stride + oc_l1 * output_hw_num + oh * output_wcb_num;
                        const __fp16 *filter_cc_base     = cvt_filter + oc_l1 * flt_ichw_num + ic_l1 * flt_ic_stride;
                        const __fp16 *const bias_oc_base = (ic_l1 == 0) ? (bias + oc_l1) : nullptr;
                        const int64_t ic_remain          = std::min(ic_tile, channels - ic_l1);
                        const int64_t oc_remains         = std::min(ocblk2, num_output_pck - oc_l1);
                        const ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_kernel_func_t *const conv_direct_kernel_func_table =
                            (oc_remains > OCBLK()) ? ppl_arm_server_kernel_fp16_conv_direct_ndarray_oc16_kernel_func_table : ppl_arm_server_kernel_fp16_conv_direct_ndarray_oc8_kernel_func_table;
                        const ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_h1w1_kernel_func_t conv_direct_kernel_h1w1_func =
                            (oc_remains > OCBLK()) ? ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_h1w1_kernel<16> : ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_h1w1_kernel<8>;
#endif
                        const int64_t ih           = -pad_h + oh * strd_h;
                        int64_t flt_h_start         = DIV_CEIL(std::max((int64_t)0, -ih), dltn_h); // std::max((int64_t)0, DIV_CEIL((pad_h-oh*strd_h), dltn_h));
                        int64_t flt_h_end           = std::min(flt_h, DIV_CEIL((src_h - ih), dltn_h));
                        int64_t flt_h_valid         = flt_h_end - flt_h_start;
                        const __fp16 *input_h_base = input_ic_base + ih * src_w;

                        for (int64_t ow = 0; ow < ow_inner_start; ow++) {
                            const int64_t iw   = -pad_w + ow * strd_w;
                            int64_t flt_w_start = DIV_CEIL(std::max((int64_t)0, -iw), dltn_w);
                            int64_t flt_w_end   = std::min(flt_w, DIV_CEIL((src_w - iw), dltn_w));
                            conv_direct_kernel_h1w1_func(
                                input_h_base + iw,
                                filter_cc_base,
                                bias_oc_base,
                                output_h_base + ow * OCBLK(),
                                sum_h_base + ow * OCBLK(),
                                input_hw_num,
                                ic_remain,
                                flt_h_start,
                                flt_h_end,
                                flt_w_start,
                                flt_w_end,
                                flt_w,
                                flt_ic_stride,
                                dltn_h * src_w,
                                dltn_w,
                                output_hwcb_num,
                                fuse_flag);
                        } // close loop over ow(1/3):head

                        const __fp16 *input_kh_base = input_h_base + flt_h_start * dltn_h * src_w;
                        for (int64_t ow = ow_inner_start; ow < ow_inner_end; ow += dst_tile_w) {
                            const int64_t ow_len = std::min(dst_tile_w, ow_inner_end - ow);
                            const int64_t iw     = -pad_w + ow * strd_w;
                            conv_direct_kernel_func_table[ow_len](
                                input_kh_base + iw,
                                filter_cc_base + (flt_h_start * flt_w * OCBLK() * 2),
                                bias_oc_base,
                                output_h_base + ow * OCBLK(),
                                sum_h_base + ow * OCBLK(),
                                src_h,
                                src_w,
                                ic_remain,
                                flt_h_valid,
                                flt_w,
                                strd_w,
                                dltn_h,
                                dltn_w,
                                flt_ic_stride,
                                output_hwcb_num,
                                fuse_flag);

                        } // close loop over ow(2/3):body

                        for (int64_t ow = ow_inner_end; ow < dst_w; ow++) {
                            const int64_t iw   = -pad_w + ow * strd_w;
                            int64_t flt_w_start = DIV_CEIL(std::max((int64_t)0, -iw), dltn_w);
                            int64_t flt_w_end   = std::min(flt_w, DIV_CEIL((src_w - iw), dltn_w));

                            conv_direct_kernel_h1w1_func(
                                input_h_base + iw,
                                filter_cc_base,
                                bias_oc_base,
                                output_h_base + ow * CBLK(),
                                sum_h_base + ow * OCBLK(),
                                input_hw_num,
                                ic_remain,
                                flt_h_start,
                                flt_h_end,
                                flt_w_start,
                                flt_w_end,
                                flt_w,
                                flt_ic_stride,
                                dltn_h * src_w,
                                dltn_w,
                                output_hwcb_num,
                                fuse_flag);
                        } // close loop over ow(3/3):tail

                    } // close loop over oh
                } // close loop over ic l1 section
            } // close loop over oc l1 section
        } // close loop over batch
    }
    return ppl::common::RC_SUCCESS;
}

bool conv2d_direct_ndarray_fp16_offline_manager::is_supported()
{
    return true;
}

std::vector<int64_t>  conv2d_direct_ndarray_fp16_offline_manager::get_schedule_param() const
{
    std::vector<int64_t> sp = { sched_param_.ic_tile };
    return sp;
}

ppl::common::RetCode conv2d_direct_ndarray_fp16_offline_manager::set_schedule_param(const std::vector<int64_t>& sp)
{
    if (sp.size() != 1) {
        return fast_init_schedule_param();
    }
    sched_param_.ic_tile = sp[0];
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_direct_ndarray_fp16_offline_manager::fast_init_schedule_param()
{
    sched_param_.ic_tile = 128;
    if (sched_param_.ic_tile != 128) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_direct_ndarray_fp16_offline_manager::pick_best_schedule_param(
    const ppl::common::TensorShape &src_shape,
    void *src,
    void *cvt_bias,
    const ppl::common::TensorShape &dst_shape,
    void *dst,
    bool tune_sp,
    double &run_time)
{
    return fast_init_schedule_param();
}

// NOTE: (oc, ic, kh, kw) -> (oc/8, ic, kh, kw, 8oc)
static inline int64_t ppl_arm_server_kernel_fp16_conv_direct_n8cx_get_converted_filter_size(
    const int64_t channels,
    const int64_t num_output,
    const int64_t flt_h,
    const int64_t flt_w)
{
    return CEIL128(((num_output + 15) & (~15)) * channels * flt_h * flt_w * sizeof(__fp16)) + 128;
}

// NOTE: (oc, ic, kh, kw) -> (oc/16, ic, kh, kw, 16oc)
static void ppl_arm_server_kernel_fp16_conv_direct_n8cx_convert_filter(
    const __fp16 *filter,
    __fp16 *converted_filter,
    const int64_t channels,
    const int64_t num_output,
    const int64_t flt_h,
    const int64_t flt_w)
{
    const int64_t ocs = OCBLK() * 2;
    for (int64_t oc = 0; oc < num_output; oc++) {
        for (int64_t ic = 0; ic < channels; ic++) {
            for (int64_t kh = 0; kh < flt_h; kh++) {
                for (int64_t kw = 0; kw < flt_w; kw++) {
                    const int64_t cvt_index = (oc / ocs) * channels * flt_h * flt_w * ocs +
                                              ic * flt_h * flt_w * ocs +
                                              kh * flt_w * ocs +
                                              kw * ocs +
                                              oc % ocs;
                    converted_filter[cvt_index] = filter[oc * channels * flt_h * flt_w + ic * flt_h * flt_w + kh * flt_w + kw];
                }
            }
        }
    }

    for (int64_t oc = num_output; oc < CEIL8(num_output); oc++) {
        for (int64_t ic = 0; ic < channels; ic++) {
            for (int64_t kh = 0; kh < flt_h; kh++) {
                for (int64_t kw = 0; kw < flt_w; kw++) {
                    const int64_t cvt_index = (oc / ocs) * channels * flt_h * flt_w * ocs +
                                              ic * flt_h * flt_w * ocs +
                                              kh * flt_w * ocs +
                                              kw * ocs +
                                              oc % ocs;
                    converted_filter[cvt_index] = 0.0f;
                }
            }
        }
    }
}

ppl::common::RetCode conv2d_direct_ndarray_fp16_offline_manager::try_fuse(conv_fuse_flag_t fuse_type)
{
    return ((fuse_type | conv_fuse_flag::HSWISH) || (fuse_type | conv_fuse_flag::PRELU )) ?
        ppl::common::RC_UNSUPPORTED : ppl::common::RC_SUCCESS;
}

// should be called after init_schedule_param
ppl::common::RetCode conv2d_direct_ndarray_fp16_offline_manager::generate_cvt_weights_shapes(
    ppl::common::TensorShape &cvt_filter_shape,
    ppl::common::TensorShape &cvt_bias_shape)
{
    const int64_t num_output = param_.num_output;
    const int64_t channels   = param_.channels;
    const int64_t kernel_h   = param_.kernel_h;
    const int64_t kernel_w   = param_.kernel_w;

    cvt_bias_size_ = CEIL8(num_output) * sizeof(__fp16);
    cvt_bias_shape.SetDimCount(1);
    cvt_bias_shape.SetDim(0, cvt_bias_size_/sizeof(__fp16));
    cvt_bias_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    cvt_bias_shape.SetDataType(ppl::common::DATATYPE_FLOAT16);

    cvt_filter_size_ = ppl_arm_server_kernel_fp16_conv_direct_n8cx_get_converted_filter_size(
        channels, num_output, kernel_h, kernel_w);
    cvt_filter_shape.SetDimCount(1);
    cvt_filter_shape.SetDim(0, cvt_filter_size_/sizeof(__fp16));
    cvt_filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    cvt_filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT16);

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_direct_ndarray_fp16_offline_manager::generate_cvt_weights(
    const void *filter,
    const void *bias,
    void* new_filter,
    void* new_bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t num_output = param_.num_output;
    const int64_t channels   = param_.channels;
    const int64_t kernel_h   = param_.kernel_h;
    const int64_t kernel_w   = param_.kernel_w;

    if (!bias && new_bias) {
        cvt_bias_ = new_bias;
    } else if (bias && new_bias) {
        cvt_bias_ = new_bias;
        int64_t padding_offset_bytes = num_output * sizeof(__fp16);
        int64_t padding_bytes        = (CEIL8(num_output) - num_output) * sizeof(__fp16);
        memcpy(cvt_bias_, bias, num_output * sizeof(__fp16));
        memset((uint8_t *)cvt_bias_ + padding_offset_bytes, 0, padding_bytes);
    } else {
        cvt_bias_ = allocator_->Alloc(cvt_bias_size_);
        memset(cvt_bias_, 0, cvt_bias_size_);
        is_bias_owner_ = true;
    }

    cvt_filter_ = new_filter;
    ppl_arm_server_kernel_fp16_conv_direct_n8cx_convert_filter(
        (const __fp16 *)filter,
        (__fp16 *)cvt_filter_,
        channels,
        num_output,
        kernel_h,
        kernel_w);
    
    return ppl::common::RC_SUCCESS;
}

conv2d_runtime_executor *conv2d_direct_ndarray_fp16_offline_manager::gen_executor()
{
    return new conv2d_direct_ndarray_fp16_runtime_executor(&param_, cvt_filter_, cvt_bias_, sched_param_);
}

#undef CBLK
#undef ICBLK
#undef OCBLK

}}}}; // namespace ppl::kernel::arm_server::neon

#endif
