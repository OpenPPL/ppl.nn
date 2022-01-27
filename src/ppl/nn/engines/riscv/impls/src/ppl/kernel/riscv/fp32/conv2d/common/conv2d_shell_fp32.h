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

#ifndef __ST_PPL_KERNEL_RISCV_FP16_CONV2D_COMMON_CONV_SHELL_H_
#define __ST_PPL_KERNEL_RISCV_FP16_CONV2D_COMMON_CONV_SHELL_H_

#include "ppl/kernel/riscv/common/math.h"
#include "ppl/common/log.h"
namespace ppl { namespace kernel { namespace riscv {

template <typename T>
using conv2d_per_group_fp32_func_type_t = void (*)(
    const float* src,
    const float* filter,
    const float* bias,
    float* temp_buffer,
    float* dst,

    int64_t src_h,
    int64_t src_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t flt_h,
    int64_t flt_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t hole_h,
    int64_t hole_w,
    int64_t dst_h,
    int64_t dst_w,
    int64_t ic,
    int64_t oc,

    T tunning_info);

using conv2d_get_real_filter_size_func_type_t = int64_t (*)(const int64_t flt_size);

static void conv2d_shell_divide_src_for_group_fp32(
    const float* src,
    int64_t src_hw,
    int64_t ic_per_gp,
    int64_t group,
    float* pad_src)
{
    const int64_t atom_c = 4;

    int64_t pad_ic_per_gp = round_up(ic_per_gp, atom_c);
    auto pad_src_per_gp   = pad_src;

    for (int64_t g = 0; g < group; g++) {
        int64_t ci = 0;
        for (; ci <= ic_per_gp - atom_c; ci += atom_c) {
            for (int64_t hwi = 0; hwi < src_hw; hwi++) {
                for (int64_t cj = 0; cj < atom_c; cj++) {
                    int64_t ci_gp_idx = ic_per_gp * g + ci + cj;
                    pad_src_per_gp[ci * src_hw + hwi * atom_c + cj] =
                        src[(ci_gp_idx / atom_c) * (src_hw * atom_c) + hwi * atom_c + (ci_gp_idx % atom_c)];
                }
            }
        }
        if (ci != ic_per_gp) {
            int64_t num_ic_left = ic_per_gp - ci;
            for (int64_t hwi = 0; hwi < src_hw; hwi++) {
                int64_t cj = 0;
                for (; cj < num_ic_left; cj++) {
                    int64_t ci_gp_idx = ic_per_gp * g + ci + cj;
                    pad_src_per_gp[ci * src_hw + hwi * atom_c + cj] =
                        src[(ci_gp_idx / atom_c) * (src_hw * atom_c) + hwi * atom_c + (ci_gp_idx % atom_c)];
                }
                for (; cj < atom_c; cj++) {
                    pad_src_per_gp[ci * src_hw + hwi * atom_c + cj] = 0.0f;
                }
            }
        }
        pad_src_per_gp += pad_ic_per_gp * src_hw;
    }
}

static void conv2d_shell_merge_dst_for_group_fp32(
    const float* pad_dst,
    int64_t dst_hw,
    int64_t oc_per_gp,
    int64_t group,
    float* dst)
{
    const int64_t atom_c = 4;

    int64_t pad_oc_per_gp = round_up(oc_per_gp, atom_c);
    int64_t oc            = oc_per_gp * group;

    int64_t ci = 0;
    for (; ci <= oc - atom_c; ci += atom_c) {
        for (int64_t hwi = 0; hwi < dst_hw; hwi++) {
            for (int64_t cj = 0; cj < atom_c; cj += 1) {
                int64_t gp_idx          = (ci + cj) / oc_per_gp;
                int64_t gp_inner_oc_idx = (ci + cj) % oc_per_gp;
                dst[ci * dst_hw + hwi * atom_c + cj] =
                    pad_dst[gp_idx * pad_oc_per_gp * dst_hw + (gp_inner_oc_idx / atom_c) * dst_hw * atom_c +
                            hwi * atom_c + gp_inner_oc_idx % atom_c];
            }
        }
    }

    if (ci != oc) {
        int64_t num_oc_left = oc - ci;
        for (int64_t hwi = 0; hwi < dst_hw; hwi += 1) {
            int64_t cj = 0;
            for (; cj < num_oc_left; cj += 1) {
                int64_t gp_idx          = (ci + cj) / oc_per_gp;
                int64_t gp_inner_oc_idx = (ci + cj) % oc_per_gp;
                dst[ci * dst_hw + hwi * atom_c + cj] =
                    pad_dst[gp_idx * pad_oc_per_gp * dst_hw + (gp_inner_oc_idx / atom_c) * dst_hw * atom_c +
                            hwi * atom_c + gp_inner_oc_idx % atom_c];
            }
            for (; cj < atom_c; cj += 1) {
                dst[ci * dst_hw + hwi * atom_c + cj] = 0.0f;
            }
        }
    }
}

template <typename T,
          int64_t atom_ic,
          conv2d_get_real_filter_size_func_type_t get_real_filter_size,
          conv2d_per_group_fp32_func_type_t<T> conv_per_group>
static void conv2d_shell_fp32(
    const float* src,
    const float* filter,
    const float* bias,
    float* temp_buffer,
    float* dst,

    int64_t src_h,
    int64_t src_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t flt_h,
    int64_t flt_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t hole_h,
    int64_t hole_w,
    int64_t ic,
    int64_t oc,
    int64_t group,
    int64_t batch,

    T tunning_info)
{
    const int64_t atom_oc = 4;

    int64_t flt_h_with_hole = hole_h * (flt_h - 1) + 1;
    int64_t flt_w_with_hole = hole_w * (flt_w - 1) + 1;
    int64_t src_pad_h       = src_h + 2 * pad_h;
    int64_t src_pad_w       = src_w + 2 * pad_w;
    int64_t dst_h           = (src_pad_h - flt_h_with_hole + stride_h) / stride_h;
    int64_t dst_w           = (src_pad_w - flt_w_with_hole + stride_w) / stride_w;

    int64_t ic_per_gp     = ic / group;
    int64_t oc_per_gp     = oc / group;
    int64_t pad_ic_per_gp = round_up(ic_per_gp, atom_ic);
    int64_t pad_oc_per_gp = round_up(oc_per_gp, atom_oc);

    int64_t pad_ic = round_up(ic, atom_ic);
    int64_t pad_oc = round_up(oc, atom_oc);

    int64_t src_batch_stride = pad_ic * src_h * src_w;
    int64_t dst_batch_stride = pad_oc * dst_h * dst_w;

    if (group == 1) {
        for (int64_t i = 0; i < batch; i++) {
            auto src_per_batch_ptr = src + i * src_batch_stride;
            auto dst_per_batch_ptr = dst + i * dst_batch_stride;

            conv_per_group(
                src_per_batch_ptr,
                filter,
                bias,
                temp_buffer,
                dst_per_batch_ptr,

                src_h,
                src_w,
                pad_h,
                pad_w,
                flt_h,
                flt_w,
                stride_h,
                stride_w,
                hole_h,
                hole_w,
                dst_h,
                dst_w,
                ic_per_gp,
                oc_per_gp,

                tunning_info);
        }
    } else {
        int64_t real_flt_h = get_real_filter_size(flt_h);
        int64_t real_flt_w = get_real_filter_size(flt_w);

        int64_t src_pad_size_per_gp = pad_ic_per_gp * src_h * src_w;
        int64_t dst_pad_size_per_gp = pad_oc_per_gp * dst_h * dst_w;
        int64_t filter_gp_stride    = pad_ic_per_gp * pad_oc_per_gp * real_flt_h * real_flt_w;

        int64_t src_div_per_batch_size = 0;
        if (ic_per_gp % atom_ic != 0) {
            src_div_per_batch_size = group * src_pad_size_per_gp;
        }

        int64_t dst_div_per_batch_size = 0;
        if (oc_per_gp % atom_oc != 0) {
            dst_div_per_batch_size = group * dst_pad_size_per_gp;
        }

        auto src_div_loc      = temp_buffer;
        auto dst_div_loc      = src_div_loc + src_div_per_batch_size;
        auto conv_temp_buffer = dst_div_loc + dst_div_per_batch_size;

        for (int64_t i = 0; i < batch; i += 1) {
            auto src_per_batch_ptr = src + i * src_batch_stride;
            if (ic_per_gp % atom_ic != 0) {
                conv2d_shell_divide_src_for_group_fp32(src_per_batch_ptr, src_h * src_w, ic_per_gp, group, src_div_loc);
                src_per_batch_ptr = src_div_loc;
            }

            auto dst_per_batch_ptr = dst + i * dst_batch_stride;
            if (oc_per_gp % atom_oc != 0) {
                dst_per_batch_ptr = dst_div_loc;
            }

            for (int64_t g = 0; g < group; g += 1) {
                auto src_per_gp_ptr    = src_per_batch_ptr + g * src_pad_size_per_gp;
                auto dst_per_gp_ptr    = dst_per_batch_ptr + g * dst_pad_size_per_gp;
                auto filter_per_gp_ptr = filter + g * filter_gp_stride;
                auto bias_per_gp_ptr   = bias + g * oc_per_gp;

                conv_per_group(
                    src_per_gp_ptr,
                    filter_per_gp_ptr,
                    bias_per_gp_ptr,
                    conv_temp_buffer,
                    dst_per_gp_ptr,

                    src_h,
                    src_w,
                    pad_h,
                    pad_w,
                    flt_h,
                    flt_w,
                    stride_h,
                    stride_w,
                    hole_h,
                    hole_w,
                    dst_h,
                    dst_w,
                    ic_per_gp,
                    oc_per_gp,

                    tunning_info);
            }
            if (oc_per_gp % atom_oc != 0) {
                conv2d_shell_merge_dst_for_group_fp32(
                    dst_div_loc,
                    dst_h * dst_w,
                    oc_per_gp,
                    group,
                    dst + i * dst_batch_stride);
            }
        }
    }
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP16_CONV2D_COMMON_CONV_SHELL_H_
