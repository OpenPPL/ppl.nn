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

#ifndef __ST_PPL_KERNEL_RISCV_FP32_CONV2D_TILE_GEMM_CONV2D_GENERIC_TILE_GEMM_FP32_VEC128_H_
#define __ST_PPL_KERNEL_RISCV_FP32_CONV2D_TILE_GEMM_CONV2D_GENERIC_TILE_GEMM_FP32_VEC128_H_

#include "ppl/kernel/riscv/common/math.h"
#include "ppl/kernel/riscv/fp32/conv2d/common/conv2d_gemm_kernel_fp32.h"
#include "ppl/kernel/riscv/fp32/conv2d/common/conv2d_mem_fp32.h"
#include <cstring>

namespace ppl { namespace kernel { namespace riscv {

struct conv2d_nxcx_conv_tile_gemm_tunning_info {
    int64_t tile_gemm_m_blk;
    int64_t tile_gemm_k_blk;
    int64_t tile_gemm_dst_h_blk;
    int64_t tile_gemm_dst_w_blk;
    int64_t num_threads;
};

template <int64_t atom_c>
void conv2d_nxcx_tile_gemm_src_blk_im2col_fp32_vec128(
    const float* src,
    int64_t src_h,
    int64_t src_w,
    int64_t dst_h,
    int64_t dst_w,
    int64_t flt_h,
    int64_t flt_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t hole_h,
    int64_t hole_w,
    int64_t channels,
    int64_t tile_gemm_dst_h_beg,
    int64_t tile_gemm_dst_h_blk,
    int64_t tile_gemm_dst_w_beg,
    int64_t tile_gemm_dst_w_blk,
    float* src_trans)
{
    int64_t pad_channels = round_up(channels, atom_c);
    int64_t src_h_beg    = tile_gemm_dst_h_beg * stride_h - pad_h;
    int64_t src_w_beg    = tile_gemm_dst_w_beg * stride_w - pad_w;

    int64_t src_h_stride       = atom_c * src_w;
    int64_t src_channel_stride = src_h * src_h_stride;
    int64_t img_h_stride       = stride_h * src_h_stride;

    int64_t num_dst_w_blk_elem = tile_gemm_dst_w_blk * atom_c;

    for (int64_t ic = 0; ic < pad_channels; ic += atom_c) {
        for (int64_t kh = 0, kh_with_hole = 0; kh < flt_h; kh++, kh_with_hole += hole_h) {
            for (int64_t kw = 0, kw_with_hole = 0; kw < flt_w; kw++, kw_with_hole += hole_w) {
                int64_t src_h_loc = src_h_beg + kh_with_hole;
                int64_t dst_h_loc = 0;

                // top padding
                for (; src_h_loc < 0 && dst_h_loc < tile_gemm_dst_h_blk; src_h_loc += stride_h, dst_h_loc += 1) {
                    memset(src_trans, 0.f, num_dst_w_blk_elem * sizeof(float));
                    src_trans += num_dst_w_blk_elem;
                }

                auto src_img = src + src_h_stride * src_h_loc;
                for (; src_h_loc < src_h && dst_h_loc < tile_gemm_dst_h_blk; src_h_loc += stride_h, dst_h_loc += 1) {
                    int64_t src_w_loc = src_w_beg + kw_with_hole;
                    int64_t dst_w_loc = 0;
                    // left padding
                    for (; src_w_loc < 0 && dst_w_loc < tile_gemm_dst_w_blk; src_w_loc += stride_w, dst_w_loc += 1) {
                        if (atom_c == 1) {
                            src_trans[0] = 0.0f;
                        } else {
                            memset(src_trans, 0.f, atom_c * sizeof(float));
                        }

                        src_trans += atom_c;
                    }

                    for (; src_w_loc < src_w && dst_w_loc < tile_gemm_dst_w_blk; src_w_loc += stride_w, dst_w_loc += 1) {
                        if (atom_c == 1) {
                            src_trans[0] = src_img[src_w_loc];
                        } else {
                            memcpy(src_trans, src_img + src_w_loc * atom_c, atom_c * sizeof(float));
                        }

                        src_trans += atom_c;
                    }

                    // right padding
                    for (; dst_w_loc < tile_gemm_dst_w_blk; dst_w_loc += 1) {
                        if (atom_c == 1) {
                            src_trans[0] = 0.0f;
                        } else {
                            memset(src_trans, 0.f, atom_c * sizeof(float));
                        }

                        src_trans += atom_c;
                    }
                    src_img += img_h_stride;
                }

                // bottom padding
                for (; dst_h_loc < tile_gemm_dst_h_blk; dst_h_loc += 1) {
                    memset(src_trans, 0.f, num_dst_w_blk_elem * sizeof(float));
                    src_trans += num_dst_w_blk_elem;
                }
            }
        }
        src += src_channel_stride;
    }
}

template <int64_t atom_ic>
void conv2d_nxcx_conv_tile_gemm_riscv_per_group_fp32_vec128(
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
    conv2d_nxcx_conv_tile_gemm_tunning_info tunning_info)
{
    const int64_t atom_oc = 4;

    int64_t tile_gemm_m_blk     = round_up(tunning_info.tile_gemm_m_blk, atom_oc);
    int64_t tile_gemm_k_blk     = round_up(tunning_info.tile_gemm_k_blk, atom_ic);
    int64_t tile_gemm_dst_h_blk = min(dst_h, tunning_info.tile_gemm_dst_h_blk);
    int64_t tile_gemm_dst_w_blk = min(dst_w, tunning_info.tile_gemm_dst_w_blk);

    int64_t pad_ic  = round_up(ic, atom_ic);
    int64_t pad_oc  = round_up(oc, atom_oc);
    int64_t total_m = pad_oc;
    int64_t total_k = flt_h * flt_w * pad_ic;

    int64_t src_trans_size = total_k * tile_gemm_dst_h_blk * tile_gemm_dst_w_blk;
    auto src_trans         = temp_buffer;
    auto dst_blk           = src_trans + src_trans_size;

    int64_t src_h_stride    = atom_ic * src_w;
    int64_t filter_m_stride = tile_gemm_m_blk * total_k;

    // blk loops, TODO: k_blk
    for (int64_t dst_h_beg = 0; dst_h_beg < dst_h; dst_h_beg += tile_gemm_dst_h_blk) {
        int64_t real_dst_h_blk = min(tile_gemm_dst_h_blk, dst_h - dst_h_beg);
        for (int64_t dst_w_beg = 0; dst_w_beg < dst_w; dst_w_beg += tile_gemm_dst_w_blk) {
            int64_t real_dst_w_blk = min(tile_gemm_dst_w_blk, dst_w - dst_w_beg);
            int64_t real_n_blk     = real_dst_h_blk * real_dst_w_blk;

            auto filter_temp = filter;

            conv2d_nxcx_tile_gemm_src_blk_im2col_fp32_vec128<atom_ic>(
                src,
                src_h,
                src_w,
                dst_h,
                dst_w,
                flt_h,
                flt_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                hole_h,
                hole_w,
                ic,
                dst_h_beg,
                real_dst_h_blk,
                dst_w_beg,
                real_dst_w_blk,
                src_trans);

            for (int64_t m_beg = 0; m_beg < total_m; m_beg += tile_gemm_m_blk) {
                int64_t real_m_blk     = min(tile_gemm_m_blk, total_m - m_beg);
                int64_t real_pad_m_blk = round_up(real_m_blk, atom_oc);
                auto gemm_func =
                    conv2d_gemm_select_xcto4c_kernel_fp32_vec128<atom_ic, true>(real_pad_m_blk, real_n_blk);

                gemm_func(filter_temp, src_trans, dst_blk, real_pad_m_blk, real_n_blk, total_k);

                auto dst_ptr  = dst + m_beg * (dst_h * dst_w) + dst_h_beg * dst_w * atom_oc + dst_w_beg * atom_oc;
                auto bias_ptr = bias + m_beg;

                conv2d_n4cx_mem_dst_blk_trans_fp32_vec128<false>(
                    dst_blk,
                    real_dst_h_blk,
                    real_dst_w_blk,
                    dst_ptr,
                    dst_h,
                    dst_w,
                    real_m_blk,
                    real_dst_h_blk,
                    real_dst_w_blk,
                    bias_ptr);

                filter_temp += real_m_blk * total_k;
            }
        }
    }
}

template <int64_t atom_ic>
size_t conv2d_nxcx_conv_tile_gemm_get_cvt_filter_size_fp32_vec128(
    int64_t flt_h,
    int64_t flt_w,
    int64_t channels,
    int64_t num_outs,
    int64_t group)
{
    const int64_t atom_oc = 4;

    int64_t num_outs_per_group     = num_outs / group;
    int64_t channels_per_group     = channels / group;
    int64_t pad_num_outs_per_group = round_up(num_outs_per_group, atom_oc);
    int64_t pad_channels_per_group = round_up(channels_per_group, atom_ic);

    int64_t num_cvt_filter_elem = pad_num_outs_per_group * pad_channels_per_group * group * flt_h * flt_w;
    auto cvt_filter_size        = num_cvt_filter_elem * sizeof(float);
    return cvt_filter_size;
}

template <int64_t atom_ic>
static void conv2d_nxcx_conv_tile_gemm_cvt_filter_kernel_fp32_vec128(
    const float* filter,
    int64_t flt_h,
    int64_t flt_w,
    int64_t num_outs,
    int64_t channels,
    int64_t tile_gemm_m_blk,
    int64_t tile_gemm_k_blk,
    float* filter_cvt)
{
    const int64_t atom_oc = 4;

    tile_gemm_m_blk = round_up(tile_gemm_m_blk, atom_oc);
    tile_gemm_k_blk = round_up(tile_gemm_k_blk, atom_ic);
    tile_gemm_k_blk = tile_gemm_k_blk / (flt_h * flt_w);

    int64_t n;
    int64_t pad_channels         = round_up(channels, atom_ic);
    int64_t pad_num_outs         = round_up(num_outs, atom_oc);
    int64_t flt_size             = flt_h * flt_w;
    int64_t tile_gemm_k_blk_left = round_up(pad_channels, tile_gemm_k_blk) - pad_channels;
    memset(filter_cvt, 0, pad_channels * pad_num_outs * flt_size * sizeof(float));

    // handle tile_gemm_m_blk
    for (n = 0; n < num_outs; n++) {
        for (int64_t c = 0; c < channels; c++) {
            int64_t ch_blk = tile_gemm_k_blk;
            if (pad_channels - c <= tile_gemm_k_blk_left) {
                ch_blk = tile_gemm_k_blk_left;
            }

            // m_blks -> k_blks -> blk m / 8 -> blk k / atom_ic -> flt_size -> atom_ic(k) -> 8(m)
            for (int64_t i = 0; i < flt_size; i++) {
                int64_t filter_cvt_loc = 0;
                filter_cvt_loc += (n / tile_gemm_m_blk) * tile_gemm_m_blk * pad_channels * flt_size; // which m_blk
                filter_cvt_loc += (c / tile_gemm_k_blk) * tile_gemm_m_blk * tile_gemm_k_blk * flt_size; // which k_blk
                filter_cvt_loc += ((n % tile_gemm_m_blk) / atom_oc) * ch_blk * flt_size * atom_oc;
                filter_cvt_loc += ((c % tile_gemm_k_blk) / atom_ic) * flt_size * atom_ic * atom_oc;
                filter_cvt_loc += i * atom_ic * atom_oc;
                filter_cvt_loc += ((c % tile_gemm_k_blk) % atom_ic) * atom_oc;
                filter_cvt_loc += ((n % tile_gemm_m_blk) % atom_oc);
                filter_cvt[filter_cvt_loc] = filter[n * channels * flt_size + c * flt_size + i];
            }
        }
    }
}

template <int64_t atom_ic>
void conv2d_nxcx_conv_tile_gemm_cvt_filter_fp32_vec128(
    const float* filter,
    int64_t flt_h,
    int64_t flt_w,
    int64_t num_outs,
    int64_t channels,
    int64_t group,
    int64_t tile_gemm_m_blk,
    int64_t tile_gemm_k_blk,
    float* filter_cvt)
{
    const int64_t atom_oc = 4;

    tile_gemm_m_blk = round_up(tile_gemm_m_blk, atom_oc);
    tile_gemm_k_blk = round_up(tile_gemm_k_blk, atom_ic);

    int64_t num_outs_per_group     = num_outs / group;
    int64_t channels_per_group     = channels / group;
    int64_t pad_num_outs_per_group = round_up(num_outs_per_group, atom_oc);
    int64_t pad_channels_per_group = round_up(channels_per_group, atom_ic);

    auto filter_per_group           = filter;
    auto filter_cvt_per_group       = filter_cvt;
    int64_t filter_group_stride     = flt_h * flt_w * num_outs_per_group * channels_per_group;
    int64_t filter_cvt_group_stride = flt_h * flt_w * pad_num_outs_per_group * pad_channels_per_group;

    for (int64_t g = 0; g < group; g += 1) {
        conv2d_nxcx_conv_tile_gemm_cvt_filter_kernel_fp32_vec128<atom_ic>(
            filter_per_group, flt_h, flt_w, num_outs_per_group, channels_per_group, tile_gemm_m_blk, tile_gemm_k_blk, filter_cvt_per_group);

        filter_per_group += filter_group_stride;
        filter_cvt_per_group += filter_cvt_group_stride;
    }
}

template <int64_t atom_ic>
size_t conv2d_nxcx_tile_gemm_get_temp_buffer_size_fp32_vec128(
    int64_t src_h,
    int64_t src_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t flt_h,
    int64_t flt_w,
    int64_t hole_h,
    int64_t hole_w,
    int64_t channels,
    int64_t group,
    int64_t num_outs,

    int64_t tile_gemm_m_blk,
    int64_t tile_gemm_dst_h_blk,
    int64_t tile_gemm_dst_w_blk,
    int64_t num_threads)
{
    const int64_t atom_oc = 4;

    int64_t channels_per_group     = channels / group;
    int64_t num_outs_per_group     = num_outs / group;
    int64_t pad_channels_per_group = round_up(channels_per_group, atom_ic);
    int64_t pad_num_outs_per_group = round_up(num_outs_per_group, atom_oc);
    int64_t flt_h_with_hole        = hole_h * (flt_h - 1) + 1;
    int64_t flt_w_with_hole        = hole_w * (flt_w - 1) + 1;
    int64_t src_pad_h              = src_h + 2 * padding_h;
    int64_t src_pad_w              = src_w + 2 * padding_w;
    int64_t dst_h                  = (src_pad_h - flt_h_with_hole + stride_h) / stride_h;
    int64_t dst_w                  = (src_pad_w - flt_w_with_hole + stride_w) / stride_w;

    tile_gemm_dst_h_blk = min(tile_gemm_dst_h_blk, dst_h);
    tile_gemm_dst_w_blk = min(tile_gemm_dst_w_blk, dst_w);

    size_t src_pad_size_for_group = 0;
    size_t dst_pad_size_for_group = 0;

    if (channels_per_group % atom_ic != 0 && group != 1) {
        src_pad_size_for_group = group * pad_channels_per_group * src_h * src_w * sizeof(float);
    }

    if (num_outs_per_group % atom_oc != 0 && group != 1) {
        dst_pad_size_for_group = group * pad_num_outs_per_group * dst_h * dst_w * sizeof(float);
    }

    const int64_t tile_gemm_k_blk = flt_h * flt_w * pad_channels_per_group;
    size_t src_trans_size         = tile_gemm_k_blk * tile_gemm_dst_h_blk * tile_gemm_dst_w_blk * num_threads * sizeof(float);
    size_t dst_blocking_size =
        tile_gemm_m_blk * tile_gemm_dst_h_blk * tile_gemm_dst_w_blk * num_threads * sizeof(float);

    return src_pad_size_for_group + dst_pad_size_for_group + src_trans_size + dst_blocking_size;
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP32_CONV2D_TILE_GEMM_CONV2D_GENERIC_TILE_GEMM_FP32_VEC128_H_
