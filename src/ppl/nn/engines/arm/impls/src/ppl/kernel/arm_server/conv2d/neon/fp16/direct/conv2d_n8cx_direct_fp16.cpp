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

#include "ppl/kernel/arm_server/conv2d/neon/fp16/direct/conv2d_n8cx_direct_fp16.h"

#include <arm_neon.h>
#include <chrono>
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

#include "ppl/kernel/arm_server/conv2d/neon/fp16/utils/conv2d_utils_fp16.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

#define CBLK()  8
#define ICBLK() CBLK()
#define OCBLK() CBLK()

template<const int64_t ocblk, const int64_t dst_tile_w>
void ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func(
    const __fp16 *input_base,
    const __fp16 *filter_base,
    const __fp16 *bias_base,
    __fp16       *output_base,
    __fp16       *sum_base,
    int64_t       ic_tile_pck,
    const int64_t flt_h,
    const int64_t flt_w,
    const int64_t flt_next_w_bytes,
    const int64_t flt_next_hw_bytes,
    const int64_t dst_ocblk_offset_byte,
    const int64_t src_icblk_offset_byte,
    const int64_t src_filter_row_offset_byte,
    const int64_t src_filter_elem_offset_byte,
    const int64_t src_out_elem_offset_byte,
    const uint32_t fuse_flag);

#define DST_TILE_W() 10
#include "conv2d_direct_kernel_fp16.inc"
#undef DST_TILE_W
#define DST_TILE_W() 9
#include "conv2d_direct_kernel_fp16.inc"
#undef DST_TILE_W
#define DST_TILE_W() 8
#include "conv2d_direct_kernel_fp16.inc"
#undef DST_TILE_W
#define DST_TILE_W() 7
#include "conv2d_direct_kernel_fp16.inc"
#undef DST_TILE_W
#define DST_TILE_W() 6
#include "conv2d_direct_kernel_fp16.inc"
#undef DST_TILE_W
#define DST_TILE_W() 5
#include "conv2d_direct_kernel_fp16.inc"
#undef DST_TILE_W
#define DST_TILE_W() 4
#include "conv2d_direct_kernel_fp16.inc"
#undef DST_TILE_W
#define DST_TILE_W() 3
#include "conv2d_direct_kernel_fp16.inc"
#undef DST_TILE_W
#define DST_TILE_W() 2
#include "conv2d_direct_kernel_fp16.inc"
#undef DST_TILE_W
#define DST_TILE_W() 1
#include "conv2d_direct_kernel_fp16.inc"
#undef DST_TILE_W

typedef void (*ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_t)(
    const __fp16 *input_base,
    const __fp16 *filter_base,
    const __fp16 *bias_base,
    __fp16 *output_base,
    __fp16 *sum_base,
    int64_t ic_block_ceil8,
    const int64_t flt_h,
    const int64_t flt_w,
    const int64_t flt_diff_w_x_icv_x_ocs_bytes,
    const int64_t flt_diff_h_x_flt_w_x_icv_x_ocs_bytes,
    const int64_t dst_h_x_dst_w_x_ocv_bytes,
    const int64_t src_h_x_src_w_x_icv_bytes,
    const int64_t dltn_h_x_src_w_x_icv_bytes,
    const int64_t dltn_w_x_icv_bytes,
    const int64_t strd_w_x_icv_bytes,
    const uint32_t fuse_flag);

#define OW_CASE() 10
static ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_t ppl_arm_server_kernel_fp16_conv_direct_kernels_oc16[OW_CASE() + 1] =
    {
        nullptr,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<16, 1>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<16, 2>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<16, 3>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<16, 4>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<16, 5>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<16, 6>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<16, 7>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<16, 8>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<16, 9>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<16, 10>,
};

static ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_t ppl_arm_server_kernel_fp16_conv_direct_kernels_oc8[OW_CASE() + 1] =
    {
        nullptr,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<8, 1>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<8, 2>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<8, 3>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<8, 4>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<8, 5>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<8, 6>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<8, 7>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<8, 8>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<8, 9>,
        ppl_arm_server_kernel_fp16_conv_direct_n8cx_h1wx_func<8, 10>,
};
#undef OW_CASE

uint64_t conv2d_n8cx_direct_fp16_runtime_executor::cal_temp_buffer_size()
{
    const conv2d_param &cp = *conv_param_;
    const int64_t ic_g_pck = CEIL8(cp.channels / cp.group);
    const int64_t oc_g_pck = CEIL8(cp.num_output / cp.group);

    const int64_t src_h     = src_shape_->GetDim(2);
    const int64_t src_w     = src_shape_->GetDim(3);
    const int64_t dst_h     = dst_shape_->GetDim(2);
    const int64_t dst_w     = dst_shape_->GetDim(3);
    const int64_t num_batch = src_shape_->GetDim(0);

    uint64_t input_gbuf_size  = num_batch * ic_g_pck * src_h * src_w * sizeof(__fp16);
    uint64_t output_gbuf_size = num_batch * oc_g_pck * dst_h * dst_w * sizeof(__fp16);

    return input_gbuf_size + output_gbuf_size;
}

void conv2d_n8cx_direct_fp16_runtime_executor::adjust_schedule_param()
{
    return;
}

ppl::common::RetCode conv2d_n8cx_direct_fp16_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    adjust_schedule_param();
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_direct_fp16_runtime_executor::execute()
{
    const conv2d_param &cp                           = *conv_param_;
    const conv2d_n8cx_direct_fp16_kernel_param   &kp = ker_param_;
    const conv2d_n8cx_direct_fp16_schedule_param &sp = sched_param_;

    const __fp16 *input                              = (const __fp16 *)src_;
    const __fp16 *cvt_filter                         = (const __fp16 *)cvt_filter_;
    const __fp16 *bias                               = (const __fp16 *)cvt_bias_;
    __fp16 *output                                   = (__fp16 *)dst_;
    __fp16 *sum                                      = (__fp16 *)sum_;
    __fp16 *tmp_buffer                               = (__fp16 *)temp_buffer_;
    const int64_t src_h                              = src_shape_->GetDim(2);
    const int64_t src_w                              = src_shape_->GetDim(3);
    const int64_t channels                           = src_shape_->GetDim(1);
    const int64_t num_output                         = cp.num_output;
    const int64_t dst_h                              = dst_shape_->GetDim(2);
    const int64_t dst_w                              = dst_shape_->GetDim(3);
    const int64_t flt_h                              = cp.kernel_h;
    const int64_t flt_w                              = cp.kernel_w;
    const int64_t pad_h                              = cp.pad_h;
    const int64_t pad_w                              = cp.pad_w;
    const int64_t strd_h                             = cp.stride_h;
    const int64_t strd_w                             = cp.stride_w;
    const int64_t dltn_h                             = cp.dilation_h;
    const int64_t dltn_w                             = cp.dilation_w;
    const int64_t num_batch                          = src_shape_->GetDim(0);

    PRAGMA_OMP_PARALLEL()
    {
        const int64_t src_c_pck = CEIL8(channels);
        const int64_t dst_c_pck = CEIL8(num_output);

        const int64_t ic_group = channels / cp.group;
        const int64_t oc_group = num_output / cp.group;
        const int64_t ic_g_pck = CEIL8(ic_group);
        const int64_t oc_g_pck = CEIL8(oc_group);

        const int64_t dst_tile_h = kp.dst_tile_h;
        const int64_t dst_tile_w = kp.dst_tile_w;

        const int64_t ocblk2  = kp.ocblk2;
        const int64_t ic_tile = sp.ic_tile;

        const int64_t icblk_bytes = 16;
        const int64_t ocblk_bytes = 16;

        const int64_t src_icblk_offset_byte    = src_h * src_w * icblk_bytes;
        const int64_t dst_ocblk_offset_byte    = dst_h * dst_w * ocblk_bytes;
        const int64_t src_filter_w_offset_byte = dltn_w * icblk_bytes;
        const int64_t src_output_w_offset_byte = strd_w * icblk_bytes;
        const int64_t src_filter_h_offset_byte = src_w * dltn_h * icblk_bytes;
        const int64_t icv_x_ocs_bytes          = ocblk2 * icblk_bytes;
        const int64_t flt_w_x_icv_x_ocs_bytes  = flt_w * ocblk2 * icblk_bytes;

        const int64_t single_batch_input_size  = src_c_pck * src_h * src_w;
        const int64_t single_batch_output_size = dst_c_pck * dst_h * dst_w;

        const bool use_in_gbuf                   = (cp.group > 1 && ic_g_pck != ic_group);
        const bool use_out_gbuf                  = (cp.group > 1 && oc_g_pck != oc_group);
        const int64_t input_group_buffer_offset  = num_batch * ic_g_pck * src_h * src_w;
        __fp16 *input_gbuf                       = tmp_buffer;
        __fp16 *output_gbuf                      = input_gbuf + input_group_buffer_offset;

        int64_t ow_inner_start = std::max((int64_t)0, DIV_CEIL((pad_w - 0 * dltn_w), strd_w)); // inclusive
        int64_t ow_inner_end   = std::min((int64_t)dst_w, DIV_CEIL((src_w + pad_w - (flt_w - 1) * dltn_w), strd_w)); // exclusive
        ow_inner_start         = std::min(ow_inner_start, dst_w);
        ow_inner_end           = std::max(ow_inner_end, ow_inner_start);

        uint32_t kernel_fuse_type = cp.fuse_flag;
        if (use_out_gbuf && (cp.fuse_flag & conv_fuse_flag::SUM)) {
            kernel_fuse_type = conv_fuse_flag::NONE;
        }

        for (int64_t g = 0; g < cp.group; g++) {
            int64_t in_b_stride  = single_batch_input_size;
            int64_t out_b_stride = single_batch_output_size;

            const __fp16 *cvt_filter_g_base = cvt_filter + g * CEIL(oc_group, ocblk2) * ic_g_pck * flt_h * flt_w;
            const __fp16 *bias_g_base       = bias + g * oc_group;

            const __fp16 *kernel_input = input + g * ic_group * src_h * src_w;
            __fp16 *kernel_output      = output + g * oc_group * dst_h * dst_w;
            if (use_in_gbuf) {
                in_b_stride  = ic_g_pck * src_h * src_w;
                kernel_input = input_gbuf;
                for (int64_t b = 0; b < num_batch; b++) {
                    conv2d_n8cx_load_group_fp16(
                        input + b * single_batch_input_size,
                        input_gbuf + b * in_b_stride,
                        src_h * src_w,
                        ic_group,
                        g,
                        0);
                }
                PRAGMA_OMP_BARRIER()
            }
            if (use_out_gbuf) {
                out_b_stride  = oc_g_pck * dst_h * dst_w;
                kernel_output = output_gbuf;
            }
#if not defined PPL_USE_ARM_SERVER_OMP
            for (int64_t batch_id = 0; batch_id < num_batch; batch_id++) {
                const __fp16 *input_batch_base_ptr = kernel_input + batch_id * in_b_stride;
                __fp16 *output_batch_base_ptr      = kernel_output + batch_id * out_b_stride;
                __fp16 *sum_bg_base_ptr            = sum + batch_id * single_batch_output_size + g * oc_group * dst_h * dst_w;
                for (int64_t ic_l1 = 0; ic_l1 < ic_g_pck; ic_l1 += ic_tile) {
                    const int64_t ic_remain  = ppl::kernel::arm_server::min(ic_tile, ic_g_pck - ic_l1);
                    const uint32_t fuse_flag = (ic_l1 + ic_tile >= ic_g_pck) ? kernel_fuse_type : static_cast<uint32_t>(conv_fuse_flag::NONE);
                    for (int64_t oc_l1 = 0; oc_l1 < oc_g_pck; oc_l1 += ocblk2) {
                        const __fp16 *const bias_ptr = (ic_l1 == 0) ? (bias_g_base + oc_l1) : nullptr;
                        const int64_t oc_remains     = ppl::kernel::arm_server::min(ocblk2, oc_g_pck - oc_l1);
                        const ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_t *const conv_direct_kernel_func_table =
                            (oc_remains > OCBLK()) ? ppl_arm_server_kernel_fp16_conv_direct_kernels_oc16 : ppl_arm_server_kernel_fp16_conv_direct_kernels_oc8;
                        for (int64_t oh = 0; oh < dst_h; oh += dst_tile_h) {
#else
            for (int64_t ic_l1 = 0; ic_l1 < ic_g_pck; ic_l1 += ic_tile) {
                const uint32_t fuse_flag = (ic_l1 + ic_tile >= ic_g_pck) ? kernel_fuse_type : static_cast<uint32_t>(conv_fuse_flag::NONE);
                PRAGMA_OMP_FOR_COLLAPSE(3)
                for (int64_t batch_id = 0; batch_id < num_batch; batch_id++) {
                    for (int64_t oc_l1 = 0; oc_l1 < oc_g_pck; oc_l1 += ocblk2) {
                        for (int64_t oh = 0; oh < dst_h; oh += dst_tile_h) {
                            const __fp16 *input_batch_base_ptr = kernel_input + batch_id * in_b_stride;
                            __fp16 *output_batch_base_ptr      = kernel_output + batch_id * out_b_stride;
                            __fp16 *sum_bg_base_ptr            = sum + batch_id * single_batch_output_size + g * oc_group * dst_h * dst_w;
                            const __fp16 *const bias_ptr       = (ic_l1 == 0) ? (bias_g_base + oc_l1) : nullptr;
                            const int64_t ic_remain            = ppl::kernel::arm_server::min(ic_tile, ic_g_pck - ic_l1);
                            const int64_t oc_remains           = ppl::kernel::arm_server::min(ocblk2, oc_g_pck - oc_l1);
                            const ppl_arm_server_kernel_fp16_conv_gen_direct_kernel_t *const conv_direct_kernel_func_table =
                                (oc_remains > 8) ? ppl_arm_server_kernel_fp16_conv_direct_kernels_oc16 : ppl_arm_server_kernel_fp16_conv_direct_kernels_oc8;
#endif
                            const int64_t ih           = -pad_h + oh * strd_h;
                            int64_t flt_h_start         = DIV_CEIL(std::max((int64_t)0, -ih), dltn_h);
                            int64_t flt_h_end           = std::min(flt_h, DIV_CEIL((src_h - ih), dltn_h));
                            flt_h_end                   = std::max(flt_h_end, flt_h_start);
                            const int64_t flt_h_skipped = flt_h - (flt_h_end - flt_h_start);
                            if (flt_h_skipped == flt_h) continue;

                            if (0 < ow_inner_start) {
                                int64_t prv_ow         = 0;
                                int64_t ow             = 0;
                                int64_t prv_flt_w_start = -1;
                                int64_t prv_flt_w_end   = -1;
                                for (; ow < ow_inner_start + 1; ow++) {
                                    const int64_t iw               = -pad_w + ow * strd_w;
                                    int64_t flt_w_start             = DIV_CEIL(std::max((int64_t)0, -iw), dltn_w);
                                    int64_t flt_w_end               = std::min(flt_w, DIV_CEIL((src_w - iw), dltn_w));
                                    flt_w_end                       = std::max(flt_w_end, flt_w_start);
                                    if (prv_flt_w_start != flt_w_start || prv_flt_w_end != flt_w_end || ow - prv_ow == dst_tile_w || ow == ow_inner_start) {
                                        const int64_t prv_flt_w_skipped = flt_w - (prv_flt_w_end - prv_flt_w_start);
                                        if (prv_flt_w_skipped < flt_w && ow > prv_ow) {
                                            const int64_t iw_iter = -pad_w + prv_ow * strd_w + prv_flt_w_start * dltn_w;
                                            conv_direct_kernel_func_table[ow - prv_ow](
                                                input_batch_base_ptr + ic_l1 * src_h * src_w + (ih + flt_h_start * dltn_h) * src_w * CBLK() + iw_iter * CBLK(),
                                                cvt_filter_g_base + oc_l1 * ic_g_pck * flt_h * flt_w + ic_l1 * flt_h * flt_w * ocblk2 + flt_h_start * flt_w * CBLK() * ocblk2 + prv_flt_w_start * CBLK() * ocblk2,
                                                bias_ptr,
                                                output_batch_base_ptr + oc_l1 * dst_h * dst_w + oh * dst_w * CBLK() + prv_ow * CBLK(),
                                                sum_bg_base_ptr + oc_l1 * dst_h * dst_w + oh * dst_w * CBLK() + prv_ow * CBLK(),
                                                ic_remain,
                                                flt_h_end - flt_h_start,
                                                prv_flt_w_end - prv_flt_w_start,
                                                prv_flt_w_skipped * icv_x_ocs_bytes,
                                                flt_h_skipped * flt_w_x_icv_x_ocs_bytes,
                                                dst_ocblk_offset_byte,
                                                src_icblk_offset_byte,
                                                src_filter_h_offset_byte,
                                                src_filter_w_offset_byte,
                                                src_output_w_offset_byte,
                                                fuse_flag);
                                        }
                                        prv_ow          = ow;
                                        prv_flt_w_start = flt_w_start;
                                        prv_flt_w_end   = flt_w_end;
                                    }
                                }
                            }
                            for (int64_t ow = ow_inner_start; ow < ow_inner_end; ow += dst_tile_w) {
                                const int64_t ow_len = std::min(dst_tile_w, ow_inner_end - ow);
                                const int64_t iw     = -pad_w + ow * strd_w;

                                conv_direct_kernel_func_table[ow_len](
                                    input_batch_base_ptr + ic_l1 * src_h * src_w + (ih + flt_h_start * dltn_h) * src_w * CBLK() + iw * CBLK(),
                                    cvt_filter_g_base + oc_l1 * ic_g_pck * flt_h * flt_w + ic_l1 * flt_h * flt_w * ocblk2 + flt_h_start * flt_w * CBLK() * ocblk2,
                                    bias_ptr,
                                    output_batch_base_ptr + oc_l1 * dst_w * dst_h + oh * dst_w * CBLK() + ow * CBLK(),
                                    sum_bg_base_ptr + oc_l1 * dst_w * dst_h + oh * dst_w * CBLK() + ow * CBLK(),
                                    ic_remain,
                                    flt_h_end - flt_h_start,
                                    flt_w,
                                    0,
                                    flt_h_skipped * flt_w_x_icv_x_ocs_bytes,
                                    dst_ocblk_offset_byte,
                                    src_icblk_offset_byte,
                                    src_filter_h_offset_byte,
                                    src_filter_w_offset_byte,
                                    src_output_w_offset_byte,
                                    fuse_flag);
                            }
                            if (ow_inner_end < dst_w) {
                                int64_t prv_ow         = ow_inner_end;
                                int64_t ow             = ow_inner_end;
                                int64_t prv_flt_w_start = -1;
                                int64_t prv_flt_w_end   = -1;
                                for (; ow < dst_w + 1; ow++) {
                                    const int64_t iw   = -pad_w + ow * strd_w;
                                    int64_t flt_w_start = DIV_CEIL(std::max((int64_t)0, -iw), dltn_w);
                                    int64_t flt_w_end   = std::min(flt_w, DIV_CEIL((src_w - iw), dltn_w));
                                    flt_w_end           = std::max(flt_w_end, flt_w_start);
                                    if (prv_flt_w_start != flt_w_start || prv_flt_w_end != flt_w_end || ow - prv_ow == 10 || ow == dst_w) {
                                        const int64_t prv_flt_w_skipped = flt_w - (prv_flt_w_end - prv_flt_w_start);
                                        if (prv_flt_w_skipped < flt_w && ow > prv_ow) {
                                            const int64_t iw_iter = -pad_w + prv_ow * strd_w + prv_flt_w_start * dltn_w;
                                            conv_direct_kernel_func_table[ow - prv_ow](
                                                input_batch_base_ptr + ic_l1 * src_h * src_w + (ih + flt_h_start * dltn_h) * src_w * CBLK() + iw_iter * CBLK(),
                                                cvt_filter_g_base + oc_l1 * ic_g_pck * flt_h * flt_w + ic_l1 * flt_h * flt_w * ocblk2 + flt_h_start * flt_w * CBLK() * ocblk2 + prv_flt_w_start * CBLK() * ocblk2,
                                                bias_ptr,
                                                output_batch_base_ptr + oc_l1 * dst_h * dst_w + oh * dst_w * CBLK() + prv_ow * CBLK(),
                                                sum_bg_base_ptr + oc_l1 * dst_h * dst_w + oh * dst_w * CBLK() + prv_ow * CBLK(),
                                                ic_remain,
                                                flt_h_end - flt_h_start,
                                                prv_flt_w_end - prv_flt_w_start,
                                                prv_flt_w_skipped * icv_x_ocs_bytes,
                                                flt_h_skipped * flt_w_x_icv_x_ocs_bytes,
                                                dst_ocblk_offset_byte,
                                                src_icblk_offset_byte,
                                                src_filter_h_offset_byte,
                                                src_filter_w_offset_byte,
                                                src_output_w_offset_byte,
                                                fuse_flag);
                                        }
                                        prv_ow         = ow;
                                        prv_flt_w_start = flt_w_start;
                                        prv_flt_w_end   = flt_w_end;
                                    }
                                }
                            }
                        } // close loop over oh
                    } // close loop over ic tile
                } // close loop over oc block x2
            } // close loop over batch

            if (use_out_gbuf) {
                for (int64_t b = 0; b < num_batch; b++) {
                    conv2d_n8cx_store_group_fp16(
                        output_gbuf + b * out_b_stride,
                        output + b * single_batch_output_size,
                        sum + b * single_batch_output_size,
                        dst_h * dst_w,
                        oc_group,
                        g,
                        0,
                        cp.fuse_flag);
                }
                PRAGMA_OMP_BARRIER()
            }
        } // close loop over group
    }
    return ppl::common::RC_SUCCESS;
}

bool conv2d_n8cx_direct_fp16_offline_manager::is_supported()
{
    return true;
}

std::vector<int64_t>  conv2d_n8cx_direct_fp16_offline_manager::get_schedule_param() const
{
    std::vector<int64_t> sp = { sched_param_.ic_tile };
    return sp;
}

ppl::common::RetCode conv2d_n8cx_direct_fp16_offline_manager::set_schedule_param(const std::vector<int64_t>& sp)
{
    if (sp.size() != 1) {
        return fast_init_schedule_param();
    }
    sched_param_.ic_tile = sp[0];
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_direct_fp16_offline_manager::fast_init_schedule_param()
{
    sched_param_.ic_tile = 128;
    if (sched_param_.ic_tile != 128) {
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

static inline int64_t ppl_arm_server_kernel_fp16_conv_direct_n8cx_get_converted_filter_size(
    const int64_t group,
    const int64_t in_c,
    const int64_t out_c,
    const int64_t ker_h,
    const int64_t ker_w)
{
    const int64_t ic_group  = in_c / group;
    const int64_t oc_group  = out_c / group;
    const int64_t ic_g_pck  = CEIL8(ic_group);
    const int64_t oc_g_pck2 = CEIL16(oc_group);
    return CEIL128(group * oc_g_pck2 * ic_g_pck * ker_h * ker_w * sizeof(__fp16)) + ker_w * 2 * OCBLK() * ICBLK() * sizeof(__fp16) + 128;
}

ppl::common::RetCode conv2d_n8cx_direct_fp16_offline_manager::pick_best_schedule_param(
    const ppl::common::TensorShape &src_shape,
    void *src,
    void *cvt_bias,
    const ppl::common::TensorShape &dst_shape,
    void *dst,
    bool tune_sp,
    double &run_time)
{
    const int64_t num_output = param_.num_output;
    const int64_t channels   = param_.channels;
    const int64_t kernel_h   = param_.kernel_h;
    const int64_t kernel_w   = param_.kernel_w;

    if (src_shape.GetDimCount() < 4) {
        return ppl::common::RC_INVALID_VALUE;
    }

    // auto conv_exe = dynamic_cast<conv2d_n8cx_direct_fp16_runtime_executor*>(gen_executor());
    auto conv_exe = conv2d_n8cx_direct_fp16_runtime_executor(&param_, cvt_filter_, cvt_bias_, sched_param_);
    conv_exe.set_src(src);
    conv_exe.set_src_shape(&src_shape);
    conv_exe.set_dst(dst);
    conv_exe.set_dst_shape(&dst_shape);
    conv_exe.set_cvt_bias(cvt_bias);

    size_t cvt_filter_size = ppl_arm_server_kernel_fp16_conv_direct_n8cx_get_converted_filter_size(
        param_.group, channels, num_output, kernel_h, kernel_w);
    __fp16 *cvt_filter     = (__fp16 *)allocator_->Alloc(cvt_filter_size);
    conv_exe.set_cvt_filter(cvt_filter);

    std::vector<int64_t> candidate_ic_tile_list = {128};
    if (tune_sp) {
        candidate_ic_tile_list = {32, 64, 128, 192, 256};
    }

    conv_exe.set_dst(dst);
    size_t tmp_buf_size = 0;
    for (auto ic_tile : candidate_ic_tile_list) {
        conv_exe.sched_param_.ic_tile = ic_tile;
        conv_exe.adjust_schedule_param();

        const size_t new_size = conv_exe.cal_temp_buffer_size();
        if (new_size > tmp_buf_size) {
            tmp_buf_size = new_size;
        }
    }
    __fp16 *tmp_buffer    = (__fp16 *)allocator_->Alloc(tmp_buf_size);
    conv_exe.set_temp_buffer(tmp_buffer);

    int64_t best_ic_tile   = 128;
    int64_t best_run_time = std::numeric_limits<int64_t>::max();

    const int num_warmup_iter    = 1;
    const int num_benchmark_iter = 3;
    for (auto ic_tile : candidate_ic_tile_list) {
        conv_exe.sched_param_.ic_tile = ic_tile;
        conv_exe.prepare();

        for (int i = 0; i < num_warmup_iter; i++) {
            conv_exe.execute();
        }

        auto begin_ts = std::chrono::system_clock::now();
        for (int i = 0; i < num_benchmark_iter; i++) {
            conv_exe.execute();
        }
        auto end_ts = std::chrono::system_clock::now();

        int64_t elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - begin_ts).count();
        if (elapsed_time < best_run_time) {
            best_ic_tile  = ic_tile;
            best_run_time = elapsed_time;
        }

        if (ic_tile >= channels / param_.group) break;
    }

    allocator_->Free(cvt_filter);
    allocator_->Free(tmp_buffer);

    sched_param_.ic_tile = best_ic_tile;
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    LOG(INFO) << "choose sp param ic: " << sched_param_.ic_tile;
    LOG(INFO) << "best run time: " << best_run_time / num_benchmark_iter / 1000 << " ms";
#endif/*  */
    run_time = (double)best_run_time / (double)num_benchmark_iter;
    return ppl::common::RC_SUCCESS;
}

// NOTE: (oc, ic, kh, kw) -> (oc/16, ic/8, kh, kw, 8ic, 16oc)
static void ppl_arm_server_kernel_fp16_conv_direct_n8cx_convert_filter(
    const __fp16 *filter,
    __fp16 *converted_filter,
    const int64_t group,
    const int64_t in_c,
    const int64_t out_c,
    const int64_t flt_h,
    const int64_t flt_w)
{
    const int64_t ocblk2    = OCBLK() * 2;
    const int64_t ic_group  = in_c / group;
    const int64_t oc_group  = out_c / group;
    const int64_t ic_g_pck  = CEIL8(ic_group);
    const int64_t oc_g_pck2 = CEIL16(oc_group);

    for (int64_t g = 0; g < group; g++) {
        const __fp16 *filter_g_base = filter + g * oc_group * ic_group * flt_h * flt_w;
        __fp16 *cvt_filter_g_base   = converted_filter + g * oc_g_pck2 * ic_g_pck * flt_h * flt_w;

        for (int64_t oc = 0; oc < oc_group; oc++) {
            for (int64_t ic = 0; ic < ic_group; ic++) {
                for (int64_t kh = 0; kh < flt_h; kh++) {
                    for (int64_t kw = 0; kw < flt_w; kw++) {
                        const int64_t cvt_index = (oc / ocblk2) * DIV_CEIL(ic_group, ICBLK()) * flt_h * flt_w * ICBLK() * ocblk2 +
                                                  (ic / ICBLK()) * flt_h * flt_w * ICBLK() * ocblk2 +
                                                  kh * flt_w * ICBLK() * ocblk2 +
                                                  kw * ICBLK() * ocblk2 +
                                                  (ic % ICBLK()) * ocblk2 +
                                                  oc % ocblk2;
                        cvt_filter_g_base[cvt_index] = filter_g_base[oc * ic_group * flt_h * flt_w + ic * flt_h * flt_w + kh * flt_w + kw];
                    }
                }
            }
            for (int64_t ic = ic_group; ic < ic_g_pck; ic++) {
                for (int64_t kh = 0; kh < flt_h; kh++) {
                    for (int64_t kw = 0; kw < flt_w; kw++) {
                        const int64_t cvt_index = (oc / ocblk2) * DIV_CEIL(ic_group, ICBLK()) * flt_h * flt_w * ICBLK() * ocblk2 +
                                                  (ic / ICBLK()) * flt_h * flt_w * ICBLK() * ocblk2 +
                                                  kh * flt_w * ICBLK() * ocblk2 +
                                                  kw * ICBLK() * ocblk2 +
                                                  (ic % ICBLK()) * ocblk2 +
                                                  oc % ocblk2;
                        cvt_filter_g_base[cvt_index] = 0.0f;
                    }
                }
            }
        }

        for (int64_t oc = oc_group; oc < oc_g_pck2; oc++) {
            for (int64_t ic = 0; ic < ic_g_pck; ic++) {
                for (int64_t kh = 0; kh < flt_h; kh++) {
                    for (int64_t kw = 0; kw < flt_w; kw++) {
                        const int64_t cvt_index = (oc / ocblk2) * DIV_CEIL(ic_group, ICBLK()) * flt_h * flt_w * ICBLK() * ocblk2 +
                                                  (ic / ICBLK()) * flt_h * flt_w * ICBLK() * ocblk2 +
                                                  kh * flt_w * ICBLK() * ocblk2 +
                                                  kw * ICBLK() * ocblk2 +
                                                  (ic % ICBLK()) * ocblk2 +
                                                  oc % ocblk2;
                        cvt_filter_g_base[cvt_index] = 0.0f;
                    }
                }
            }
        }
    }
}

ppl::common::RetCode conv2d_n8cx_direct_fp16_offline_manager::try_fuse(conv_fuse_flag_t fuse_type)
{
    return ((fuse_type | conv_fuse_flag::HSWISH) || (fuse_type | conv_fuse_flag::PRELU )) ?
        ppl::common::RC_UNSUPPORTED : ppl::common::RC_SUCCESS;
}

// should be called after init_schedule_param
ppl::common::RetCode conv2d_n8cx_direct_fp16_offline_manager::generate_cvt_weights_shapes(
    ppl::common::TensorShape &cvt_filter_shape,
    ppl::common::TensorShape &cvt_bias_shape)
{
    const int64_t group  = param_.group;
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
        group, channels, num_output, kernel_h, kernel_w);
    cvt_filter_shape.SetDimCount(1);
    cvt_filter_shape.SetDim(0, cvt_filter_size_/sizeof(__fp16));
    cvt_filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    cvt_filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT16);

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_direct_fp16_offline_manager::generate_cvt_weights(
    const void *filter,
    const void *bias,
    void* new_filter,
    void* new_bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t group  = param_.group;
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
        std::memcpy(cvt_bias_, bias, num_output * sizeof(__fp16));
        std::memset((uint8_t *)cvt_bias_ + padding_offset_bytes, 0, padding_bytes);
    } else {
        cvt_bias_ = allocator_->Alloc(cvt_bias_size_);
        std::memset(cvt_bias_, 0, cvt_bias_size_);
        is_bias_owner_ = true;
    }

    cvt_filter_ = new_filter;
    ppl_arm_server_kernel_fp16_conv_direct_n8cx_convert_filter(
        (const __fp16 *)filter,
        (__fp16 *)cvt_filter_,
        group,
        channels,
        num_output,
        kernel_h,
        kernel_w);
    
    return ppl::common::RC_SUCCESS;
}

conv2d_runtime_executor *conv2d_n8cx_direct_fp16_offline_manager::gen_executor()
{
    return new conv2d_n8cx_direct_fp16_runtime_executor(&param_, cvt_filter_, cvt_bias_, sched_param_);
}

#undef ICBLK
#undef OCBLK

}}}}; // namespace ppl::kernel::arm_server::neon

#endif
