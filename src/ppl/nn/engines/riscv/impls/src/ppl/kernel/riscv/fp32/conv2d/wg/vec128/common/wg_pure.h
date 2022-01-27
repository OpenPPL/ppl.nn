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

#ifndef __ST_PPL_KERNEL_RISCV_FP32_CONV2D_WG_VEC128_COMMON_WG_PURE_H_
#define __ST_PPL_KERNEL_RISCV_FP32_CONV2D_WG_VEC128_COMMON_WG_PURE_H_

#include <cstring>
#include "ppl/kernel/riscv/fp32/conv2d/common/conv2d_gemm_kernel_fp32.h"
#include "ppl/kernel/riscv/common/math.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() ((int64_t)4)

struct conv2d_n4cx_wg_bxfxs1_fp32_vec128_tunning_param {
    int64_t oc_blk;
    int64_t ic_blk;
    int64_t oh_blk;
    int64_t ow_blk;
};

struct conv2d_n4cx_wg_bxfxs1_fp32_vec128_extra_param {
    int64_t oc_blk;
    int64_t ic_blk;
    int64_t oh_blk;
    int64_t ow_blk;

    const float* trans_mat_src;
    const float* trans_mat_dst;
};
typedef void (*conv_wg_riscv_fp32_n4chw_src_trans_kernel_func_t)(
    const float* src_pad,
    const float* trans_mat, // TODO: should be removed
    int64_t src_pad_h_stride,

    float* src_trans_d,
    int64_t src_trans_wg_tile_stride);

typedef void (*conv_wg_riscv_fp32_n4chw_dst_trans_kernel_func_t)(
    const float* dst_trans,
    const float* bias,
    const float* trans_mat, // TODO: should be removed
    int64_t dst_trans_wg_tile_stride,

    float* dst,
    int64_t dst_h_stride,
    int64_t dst_h_offset,
    int64_t dst_w_offset,
    int64_t dst_trans_h,
    int64_t dst_trans_w);

template <int64_t wgb,
          int64_t wgf,
          conv_wg_riscv_fp32_n4chw_src_trans_kernel_func_t src_trans_kernel_func>
void conv2d_conv_wg_bxfxs1_src_blk_trans_fp32(
    const float* src_pad,
    const float* trans_mat,
    int64_t src_trans_pad_h,
    int64_t src_trans_pad_w,
    int64_t pad_channels,
    int64_t num_h_tile,
    int64_t num_w_tile,
    float* src_trans_d)
{
    int64_t tile_len              = wgb + wgf - 1;
    int64_t tile_size             = tile_len * tile_len;
    int64_t pad_channels_div4     = pad_channels / C_BLK();
    int64_t src_trans_tile_stride = num_h_tile * num_w_tile * pad_channels;

    for (int64_t c = 0; c < pad_channels_div4; c++) {
        for (int64_t ht = 0; ht < num_h_tile; ht++) {
            for (int64_t wt = 0; wt < num_w_tile; wt++) {
                int64_t src_offset       = c * src_trans_pad_h * src_trans_pad_w + ht * wgb * src_trans_pad_w + wt * wgb;
                int64_t src_trans_offset = c * num_h_tile * num_w_tile + ht * num_w_tile + wt;
                src_trans_kernel_func(
                    src_pad + src_offset * C_BLK(),
                    trans_mat,
                    src_trans_pad_w * C_BLK(),
                    src_trans_d + src_trans_offset * C_BLK(),
                    src_trans_tile_stride);
            }
        }
    }
}

template <int64_t wgb,
          int64_t wgf,
          conv_wg_riscv_fp32_n4chw_dst_trans_kernel_func_t dst_trans_kernel_func>
void conv2d_conv_wg_bxfxs1_dst_blk_trans_fp32(
    const float* dst_trans,
    const float* trans_mat,
    const float* bias,
    int64_t dst_trans_h,
    int64_t dst_trans_w,
    int64_t dst_trans_pad_h,
    int64_t dst_trans_pad_w,
    int64_t pad_num_outs,
    int64_t num_h_tile,
    int64_t num_w_tile,

    float* dst,
    int64_t dst_h,
    int64_t dst_w)
{
    int64_t tile_len              = wgb + wgf - 1;
    int64_t pad_num_outs_div4     = pad_num_outs / C_BLK();
    int64_t dst_trans_tile_stride = num_h_tile * num_w_tile * pad_num_outs;
    int64_t dst_h_stride          = dst_w * C_BLK();
    for (int64_t c = 0; c < pad_num_outs_div4; c++) {
        for (int64_t ht = 0; ht < num_h_tile; ht++) {
            for (int64_t wt = 0; wt < num_w_tile; wt++) {
                int64_t dst_trans_offset = c * num_h_tile * num_w_tile + ht * num_w_tile + wt;
                int64_t dst_offset       = wt * wgb + ht * wgb * dst_w + c * dst_h * dst_w;
                int64_t dst_w_offset     = wt * wgb;
                int64_t dst_h_offset     = ht * wgb;
                dst_trans_kernel_func(
                    dst_trans + dst_trans_offset * C_BLK(),
                    bias + c * C_BLK(),
                    trans_mat,
                    dst_trans_tile_stride,
                    dst + dst_offset * C_BLK(),
                    dst_h_stride,
                    dst_h_offset,
                    dst_w_offset,
                    dst_trans_h,
                    dst_trans_w);
            }
        }
    }
}

static void conv2d_conv_wg_s1_blocking_src_fp32(
    const float* src,
    int64_t src_h,
    int64_t src_w,
    int64_t pad_channels,
    int64_t flt_h,
    int64_t flt_w,

    int64_t src_h_beg,
    int64_t src_w_beg,
    int64_t blk_src_h,
    int64_t blk_src_w,

    float* src_pad)
{
    int64_t src_h_idx = 0, src_w_idx = 0;
    int64_t src_h_pad_end      = src_h_beg + blk_src_h;
    int64_t src_w_pad_end      = src_w_beg + blk_src_w;
    int64_t src_h_end          = min(src_h, src_h_pad_end);
    int64_t src_w_end          = min(src_w, src_w_pad_end);
    int64_t src_h_stride       = src_w * C_BLK();
    int64_t blk_h_stride       = blk_src_w * C_BLK();
    int64_t src_channel_stride = src_h * src_h_stride;

    auto src_per_channels = src;
    auto src_pad_d        = src_pad;

    for (int64_t i = 0; i < pad_channels; i += C_BLK()) {
        // top pad
        for (src_h_idx = src_h_beg; src_h_idx < 0; src_h_idx += 1) {
            memset(src_pad_d, 0.f, blk_h_stride * sizeof(float));
            src_pad_d += blk_h_stride;
        }

        for (; src_h_idx < src_h_end; src_h_idx += 1) {
            // left pad
            src_w_idx = src_w_beg;
            if (src_w_idx < 0) {
                int64_t num_pad = (0 - src_w_idx) * C_BLK();
                memset(src_pad_d, 0.f, num_pad * sizeof(float));
                src_pad_d += num_pad;
                src_w_idx = 0;
            }

            if (src_w_idx < src_w_end) {
                int64_t num_cpy = (src_w_end - src_w_idx) * C_BLK();
                memcpy(src_pad_d, src_per_channels + src_h_idx * src_h_stride + src_w_idx * C_BLK(), num_cpy * sizeof(float));
                src_pad_d += num_cpy;
                src_w_idx = src_w_end;
            }

            // right pad
            if (src_w_idx < src_w_pad_end) {
                int64_t num_pad = (src_w_pad_end - src_w_idx) * C_BLK();
                memset(src_pad_d, 0.f, num_pad * sizeof(float));
                src_pad_d += num_pad;
            }
        }

        // bottom pad
        for (; src_h_idx < src_h_pad_end; src_h_idx += 1) {
            memset(src_pad_d, 0.f, blk_h_stride * sizeof(float));
            src_pad_d += blk_h_stride;
        }

        src_per_channels += src_channel_stride;
    }
}

template <int64_t wgb,
          int64_t wgf,
          conv_wg_riscv_fp32_n4chw_src_trans_kernel_func_t src_trans_kernel_func,
          conv_wg_riscv_fp32_n4chw_dst_trans_kernel_func_t dst_trans_kernel_func>
void conv2d_conv_wg_bxfxs1_pure_fp32(
    const float* src,
    const float* trans_mat_src,
    const float* trans_mat_dst,
    int64_t src_h,
    int64_t src_w,
    int64_t channels,
    int64_t num_outs,
    int64_t padding_h,
    int64_t padding_w,

    int64_t blk_dst_h,
    int64_t blk_dst_w,
    int64_t blk_channels,
    int64_t blk_num_outs,

    const float* filter,
    const float* bias,
    float* temp_buffer,
    float* dst)
{
    int64_t pad_channels = round_up(channels, C_BLK());
    int64_t pad_num_outs = round_up(num_outs, C_BLK());

    int64_t dst_h = (src_h + 2 * padding_h) - wgf + 1;
    int64_t dst_w = (src_w + 2 * padding_w) - wgf + 1;

    int64_t dst_pad_h = round_up(dst_h, wgb);
    int64_t dst_pad_w = round_up(dst_w, wgb);
    int64_t src_pad_h = dst_pad_h + wgf - 1;
    int64_t src_pad_w = dst_pad_w + wgf - 1;

    int64_t blk_dst_pad_h = min(dst_pad_h, blk_dst_h);
    int64_t blk_dst_pad_w = min(dst_pad_w, blk_dst_w);
    int64_t blk_src_pad_h = blk_dst_pad_h + wgf - 1;
    int64_t blk_src_pad_w = blk_dst_pad_w + wgf - 1;

    int64_t blk_num_wg_tile = (blk_dst_pad_h / wgb) * (blk_dst_pad_w / wgb);

    int64_t vert_beg = -padding_h;
    int64_t vert_end = vert_beg + src_pad_h;
    int64_t hori_beg = -padding_w;
    int64_t hori_end = hori_beg + src_pad_w;

    const int64_t src_tile_h    = wgb + wgf - 1;
    const int64_t src_tile_w    = wgb + wgf - 1;
    const int64_t src_tile_size = src_tile_h * src_tile_w;

    auto src_pad_blk = temp_buffer;
    auto src_trans   = src_pad_blk + pad_channels * blk_src_pad_h * blk_src_pad_w;
    auto dst_trans   = src_trans + src_tile_size * pad_channels * blk_num_wg_tile;

    for (int64_t h_dst_idx = 0; h_dst_idx < dst_pad_h; h_dst_idx += blk_dst_pad_h) {
        int64_t real_blk_dst_h     = min(dst_h - h_dst_idx, blk_dst_pad_h);
        int64_t real_blk_dst_pad_h = min(dst_pad_h - h_dst_idx, blk_dst_pad_h);

        int64_t h_src_idx          = h_dst_idx + vert_beg;
        int64_t real_blk_src_pad_h = real_blk_dst_pad_h + wgf - 1;

        int64_t real_blk_num_h_tile = real_blk_dst_pad_h / wgb;

        for (int64_t w_dst_idx = 0; w_dst_idx < dst_pad_w; w_dst_idx += blk_dst_pad_w) {
            int64_t real_blk_dst_w     = min(dst_w - w_dst_idx, blk_dst_pad_w);
            int64_t real_blk_dst_pad_w = min(dst_pad_w - w_dst_idx, blk_dst_pad_w);

            int64_t w_src_idx          = w_dst_idx + hori_beg;
            int64_t real_blk_src_pad_w = real_blk_dst_pad_w + wgf - 1;

            int64_t real_blk_num_w_tile = real_blk_dst_pad_w / wgb;
            int64_t real_blk_num_tile   = real_blk_num_w_tile * real_blk_num_h_tile;

            // get blk src from src
            conv2d_conv_wg_s1_blocking_src_fp32(src, src_h, src_w, pad_channels, wgf, wgf, h_src_idx, w_src_idx, real_blk_src_pad_h, real_blk_src_pad_w, src_pad_blk);

            // src trans
            {
                auto src_pad_blk_d                 = src_pad_blk;
                auto src_trans_d                   = src_trans;
                int64_t src_pad_blk_channel_stride = real_blk_src_pad_h * real_blk_src_pad_w;
                int64_t src_trans_channels_stride  = real_blk_num_tile * src_tile_h * src_tile_w;

                for (int64_t i = 0; i < pad_channels; i += blk_channels) {
                    int64_t real_blk_channels = min(pad_channels - i, blk_channels);

                    conv2d_conv_wg_bxfxs1_src_blk_trans_fp32<wgb, wgf, src_trans_kernel_func>(
                        src_pad_blk_d,
                        trans_mat_src,
                        real_blk_src_pad_h,
                        real_blk_src_pad_w,
                        real_blk_channels,
                        real_blk_num_h_tile,
                        real_blk_num_w_tile,
                        src_trans_d);

                    src_pad_blk_d += real_blk_channels * src_pad_blk_channel_stride;
                    src_trans_d += real_blk_channels * src_trans_channels_stride;
                }
            }

            // gemm + dst trans
            {
                auto dst_d    = dst + h_dst_idx * dst_w * C_BLK() + w_dst_idx * C_BLK();
                auto filter_d = filter;

                for (int64_t i = 0; i < pad_num_outs; i += blk_num_outs) {
                    int64_t real_blk_num_outs = min(pad_num_outs - i, blk_num_outs);
                    auto gemm_first_func =
                        conv2d_gemm_select_4cto4c_kernel_fp32_vec128<true>(real_blk_num_outs, real_blk_num_tile);
                    auto gemm_func =
                        conv2d_gemm_select_4cto4c_kernel_fp32_vec128<false>(real_blk_num_outs, real_blk_num_tile);
                    auto src_trans_d = src_trans;

                    int64_t j = 0;
                    {
                        int64_t real_blk_channels = min(pad_channels - j, blk_channels);
                        auto dst_trans_d          = dst_trans;

                        for (int64_t k = 0; k < src_tile_size; k += 1) {
                            gemm_first_func(
                                filter_d,
                                src_trans_d,
                                dst_trans_d,
                                real_blk_num_outs,
                                real_blk_num_tile,
                                real_blk_channels);
                            src_trans_d += real_blk_channels * real_blk_num_tile; // k * n
                            dst_trans_d += real_blk_num_outs * real_blk_num_tile; // m * n
                            filter_d += real_blk_num_outs * real_blk_channels; // m * k
                        }
                        j += blk_channels;
                    }

                    for (; j < pad_channels; j += blk_channels) {
                        int64_t real_blk_channels = min(pad_channels - j, blk_channels);
                        auto dst_trans_d          = dst_trans;

                        for (int64_t k = 0; k < src_tile_size; k += 1) {
                            gemm_func(
                                filter_d,
                                src_trans_d,
                                dst_trans_d,
                                real_blk_num_outs,
                                real_blk_num_tile,
                                real_blk_channels);
                            src_trans_d += real_blk_channels * real_blk_num_tile; // k * n
                            dst_trans_d += real_blk_num_outs * real_blk_num_tile; // m * n
                            filter_d += real_blk_num_outs * real_blk_channels; // m * k
                        }
                    }

                    conv2d_conv_wg_bxfxs1_dst_blk_trans_fp32<wgb, wgf, dst_trans_kernel_func>(
                        dst_trans,
                        trans_mat_dst,
                        bias + i,
                        real_blk_dst_h,
                        real_blk_dst_w,
                        real_blk_dst_pad_h,
                        real_blk_dst_pad_w,
                        real_blk_num_outs,
                        real_blk_num_h_tile,
                        real_blk_num_w_tile,

                        dst_d,
                        dst_h,
                        dst_w);

                    dst_d += real_blk_num_outs * dst_h * dst_w;
                }
            }
        }
    }
}

template <int64_t wgb,
          int64_t wgf,
          conv_wg_riscv_fp32_n4chw_src_trans_kernel_func_t src_trans_kernel_func,
          conv_wg_riscv_fp32_n4chw_dst_trans_kernel_func_t dst_trans_kernel_func>
void conv2d_conv_wg_bxfxs1_riscv_per_group_fp32(
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

    conv2d_n4cx_wg_bxfxs1_fp32_vec128_extra_param extra_info)
{
    int64_t real_blk_channels = round_up(extra_info.ic_blk, C_BLK());
    int64_t real_blk_num_outs = round_up(extra_info.oc_blk, C_BLK());
    int64_t real_blk_dst_h    = round_up(extra_info.oh_blk, wgb);
    int64_t real_blk_dst_w    = round_up(extra_info.ow_blk, wgb);
    conv2d_conv_wg_bxfxs1_pure_fp32<wgb, wgf, src_trans_kernel_func, dst_trans_kernel_func>(
        src,
        extra_info.trans_mat_src,
        extra_info.trans_mat_dst,

        src_h,
        src_w,
        ic,
        oc,
        pad_h,
        pad_w,

        real_blk_dst_h,
        real_blk_dst_w,
        real_blk_channels,
        real_blk_num_outs,

        filter,
        bias,
        temp_buffer,
        dst);
}

template <int64_t wgb, int64_t wgf>
int64_t conv2d_get_real_filter_size(const int64_t flt)
{
    return wgb + wgf - 1;
}

}}}; //  namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP32_CONV2D_WG_VEC128_COMMON_WG_PURE_H_
