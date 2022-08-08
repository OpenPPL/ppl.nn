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

#include <float.h>
#include <string.h>
#include <nmmintrin.h>
#include <string.h>

#include "ppl/kernel/x86/common/sse_tools.h"
#include "ppl/kernel/x86/fp32/transpose/transpose_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/sse/conv2d_winograd_b6f3_fp32_sse.h"
#include "ppl/kernel/x86/fp32/conv2d/sse/conv2d_n8cx_gemm_direct_kernel_fp32_sse.h"

#define CH_DT_BLK()  8
#define TILE_OC_RF() 6
#define CH_RF_BLK()  4

#define TILE_IN_H()  8
#define TILE_IN_W()  8
#define TILE_OUT_H() 6
#define TILE_OUT_W() 6
#define KERNEL_H()   3
#define KERNEL_W()   3
#define STRIDE_H()   1
#define STRIDE_W()   1

#define IC_L2_BLK_MAX_L()    (12 * CH_DT_BLK())
#define IC_L2_BLK_MAX_L_LC() (16 * CH_DT_BLK())
#define IC_L2_BLK_MAX_S()    (8  * CH_DT_BLK()) 
#define OC_L2_BLK_MAX()      (48 * CH_DT_BLK())
#define OC_L2_BLK_MAX_L()    (64 * CH_DT_BLK())
#define TILE_L2_BLK_MAX_L()  (18 * CH_DT_BLK())

#define PARALLEL_INNER() 1

#define TIMER_COUNT() 3
#define SRCTR_TIMER() 0
#define GEMM_TIMER()  1
#define DSTTR_TIMER() 2

namespace ppl { namespace kernel { namespace x86 {

bool conv2d_winograd_b6f3_fp32_sse_executor::init_profiler()
{
#ifdef PPL_X86_KERNEL_TIMING
    profiler_.init(TIMER_COUNT());
    return true;
#else
    return false;
#endif
}

void conv2d_winograd_b6f3_fp32_sse_executor::clear_profiler()
{
#ifdef PPL_X86_KERNEL_TIMING
    profiler_.clear();
#endif
}

std::string conv2d_winograd_b6f3_fp32_sse_executor::export_profiler()
{
#ifdef PPL_X86_KERENL_TIMING
    static const char *time_name[] = {
        "src_trans",
        "gemm",
        "dst_trans"};
    return profiler_export_csv(time_name, false);
#else
    return "";
#endif
}

static int64_t get_ic_l2_blk(
    const int64_t channels,
    const int64_t num_output)
{
    int64_t ic_l2 = channels <= 1024 ? IC_L2_BLK_MAX_L_LC() : IC_L2_BLK_MAX_L();
    int64_t rst = ic_l2;
    if (channels <= num_output && channels <= ic_l2) {
        rst = IC_L2_BLK_MAX_S();
    }
    if (rst > round_up(channels, CH_DT_BLK())) {
        rst = round_up(channels, CH_DT_BLK());
    }
    return rst;
}

static int64_t get_oc_l2_blk(
    const int64_t channels,
    const int64_t num_output)
{
    int64_t oc_l2 = (channels >= 512) ? OC_L2_BLK_MAX_L() : OC_L2_BLK_MAX();
    int64_t rst = oc_l2;
    if (rst > round_up(num_output, CH_DT_BLK())) {
        rst = round_up(num_output, CH_DT_BLK());
    }
    return rst;
}

static int64_t get_tiles_l2_blk(
    const int64_t batch,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t channels,
    const int64_t num_output,
    const int32_t mode)
{
    const int64_t dst_h       = src_h + 2 * pad_h - KERNEL_H() + 1;
    const int64_t dst_w       = src_w + 2 * pad_w - KERNEL_W() + 1;
    const int64_t num_tiles_h = div_up(dst_h, TILE_OUT_H());
    const int64_t num_tiles_w = div_up(dst_w, TILE_OUT_W());
    const int64_t num_tiles_b = num_tiles_h * num_tiles_w;
    const int64_t num_tiles   = num_tiles_b * batch;
    return min<int64_t>(TILE_L2_BLK_MAX_L(), num_tiles);
}

void conv2d_winograd_b6f3_fp32_sse_executor::init_preproc_param()
{
    kernel_schedule_param &sp   = schedule_param_;
    const conv2d_param &cp = *conv_param_;

    sp.ic_per_gp = cp.channels / cp.group;
    sp.oc_per_gp = cp.num_output / cp.group;
    sp.padded_ic = round_up(sp.ic_per_gp, CH_DT_BLK());
    sp.padded_oc = round_up(sp.oc_per_gp, CH_DT_BLK());

    const int64_t batch = src_shape_->GetDim(0);
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);

    sp.num_tiles_h   = div_up(dst_h, TILE_OUT_H());
    sp.num_tiles_w   = div_up(dst_w, TILE_OUT_W());
    sp.num_tiles_b   = sp.num_tiles_h * sp.num_tiles_w;
    sp.num_tiles     = sp.num_tiles_b * batch;
    sp.ic_l2_blk     = get_ic_l2_blk(sp.ic_per_gp, sp.oc_per_gp);
    sp.override_only = sp.ic_l2_blk >= sp.ic_per_gp;

    sp.parallel_mode = PARALLEL_INNER();
    sp.tiles_l2_blk  = get_tiles_l2_blk(batch, src_shape_->GetDim(2), src_shape_->GetDim(3), cp.pad_h, cp.pad_w, src_shape_->GetDim(1), dst_shape_->GetDim(1), sp.parallel_mode);

    sp.oc_l2_blk = get_oc_l2_blk(sp.ic_per_gp, sp.oc_per_gp);

    sp.thread_tile_in_len   = round_up(CH_DT_BLK() * TILE_IN_H() * TILE_IN_W(), PPL_X86_CACHELINE_BYTES() / sizeof(float));
    sp.thread_matmul_in_len = round_up(CH_DT_BLK() * TILE_IN_H() * TILE_IN_W(), PPL_X86_CACHELINE_BYTES() / sizeof(float));

    sp.src_trans_len = round_up(sp.ic_l2_blk * TILE_IN_H() * TILE_IN_W() * sp.tiles_l2_blk, PPL_X86_CACHELINE_BYTES() / sizeof(float));
    sp.gemm_out_len  = round_up(sp.padded_oc * TILE_IN_H() * TILE_IN_W() * sp.tiles_l2_blk, PPL_X86_CACHELINE_BYTES() / sizeof(float));
    if (sp.override_only) {
        sp.gemm_out_len = round_up(sp.oc_l2_blk * TILE_IN_H() * TILE_IN_W() * sp.tiles_l2_blk, PPL_X86_CACHELINE_BYTES() / sizeof(float));
    }

    sp.thread_matmul_out_len    = round_up(TILE_IN_H() * TILE_IN_W() * CH_DT_BLK(), PPL_X86_CACHELINE_BYTES() / sizeof(float));
    sp.thread_postprocess_len   = 2 * sp.thread_matmul_out_len;
    sp.thread_src_dst_trans_len = max<int64_t>(sp.thread_tile_in_len + sp.thread_matmul_in_len, sp.thread_postprocess_len);
    sp.thread_workspace_len     = sp.thread_src_dst_trans_len;
    sp.use_nt_store             = 0;
}

uint64_t conv2d_winograd_b6f3_fp32_sse_executor::cal_temp_buffer_size()
{
    const kernel_schedule_param &sp = schedule_param_;
    const int64_t num_thread        = PPL_OMP_MAX_THREADS();

    // PARALLEL_INNER
    return sp.src_trans_len * sizeof(float) +
           sp.gemm_out_len * sizeof(float) +
           sp.thread_workspace_len * num_thread * sizeof(float);
}

ppl::common::RetCode conv2d_winograd_b6f3_fp32_sse_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::SUM) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();

    return ppl::common::RC_SUCCESS;
}

template <bool first, int64_t channel, int64_t dst_stride>
static inline void wingorad_b6f3_transpose_4x4_fp32_sse(
    const float *src,
    const int64_t src_stride,
    float *dst)
{
    __m128 xmm0 = _mm_setzero_ps();
    __m128 xmm1 = xmm0, xmm2 = xmm0, xmm3 = xmm0;
    __m128 xmm4, xmm5, xmm6, xmm7;
    if (channel >= 1) xmm0 = _mm_loadu_ps(src + 0 * src_stride);
    if (channel >= 2) xmm1 = _mm_loadu_ps(src + 1 * src_stride);
    if (channel >= 3) xmm2 = _mm_loadu_ps(src + 2 * src_stride);
    if (channel >= 4) xmm3 = _mm_loadu_ps(src + 3 * src_stride);

    if (channel >= 1) TRANSPOSE_4X4_FP32_SSE_MACRO(xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7);

    _mm_storeu_ps(dst + 0 * dst_stride + first * CH_RF_BLK(), xmm0);
    _mm_storeu_ps(dst + 1 * dst_stride + first * CH_RF_BLK(), xmm1);
    _mm_storeu_ps(dst + 2 * dst_stride + first * CH_RF_BLK(), xmm2);
    _mm_storeu_ps(dst + 3 * dst_stride + first * CH_RF_BLK(), xmm3);
}

typedef void (*winograd_b6f3_kernel_fp32_sse_func_t)(const float *, const int64_t, float *);
static const winograd_b6f3_kernel_fp32_sse_func_t winograd_b6f3_transpose_4x4_fp32_sse_func_table[2][CH_RF_BLK() + 1]{
    {
        nullptr,
        wingorad_b6f3_transpose_4x4_fp32_sse<false, 1, CH_DT_BLK()>,
        wingorad_b6f3_transpose_4x4_fp32_sse<false, 2, CH_DT_BLK()>,
        wingorad_b6f3_transpose_4x4_fp32_sse<false, 3, CH_DT_BLK()>,
        wingorad_b6f3_transpose_4x4_fp32_sse<false, 4, CH_DT_BLK()>,
    },
    {
        nullptr,
        wingorad_b6f3_transpose_4x4_fp32_sse<true, 1, CH_DT_BLK()>,
        wingorad_b6f3_transpose_4x4_fp32_sse<true, 2, CH_DT_BLK()>,
        wingorad_b6f3_transpose_4x4_fp32_sse<true, 3, CH_DT_BLK()>,
        wingorad_b6f3_transpose_4x4_fp32_sse<true, 4, CH_DT_BLK()>,
    },
};

template <int64_t channel>
static inline void winograd_b6f3_src_trans_sse(
    const float *l_base_src,
    float *l_base_dst,
    const int64_t src_hw,
    const int64_t src_stride,
    const int64_t dst_stride)
{
    // trans
    const int64_t channel1 = min<const int64_t>(CH_RF_BLK(), channel);
    const int64_t channel2 = max<const int64_t>(0, channel - CH_RF_BLK());

    winograd_b6f3_transpose_4x4_fp32_sse_func_table[0][channel1](l_base_src + 0 * src_stride + 0 * CH_RF_BLK(), src_hw, l_base_dst + 0 * dst_stride);
    winograd_b6f3_transpose_4x4_fp32_sse_func_table[0][channel1](l_base_src + 0 * src_stride + 1 * CH_RF_BLK(), src_hw, l_base_dst + 1 * dst_stride);

    if (channel > CH_RF_BLK()) {
        winograd_b6f3_transpose_4x4_fp32_sse_func_table[1][channel2](l_base_src + 1 * src_stride + 0 * CH_RF_BLK(), src_hw, l_base_dst + 0 * dst_stride);
        winograd_b6f3_transpose_4x4_fp32_sse_func_table[1][channel2](l_base_src + 1 * src_stride + 1 * CH_RF_BLK(), src_hw, l_base_dst + 1 * dst_stride);
    } else {
        winograd_b6f3_transpose_4x4_fp32_sse_func_table[1][channel2](l_base_src + 0 * src_stride + 0 * CH_RF_BLK(), src_hw, l_base_dst + 0 * dst_stride);
        winograd_b6f3_transpose_4x4_fp32_sse_func_table[1][channel2](l_base_src + 0 * src_stride + 1 * CH_RF_BLK(), src_hw, l_base_dst + 1 * dst_stride);
    }
}

template <int64_t channel>
static inline void wingorad_b6f3_memcpy_sse(
    const float *l_base_src,
    float *l_base_dst,
    const int64_t tw_len,
    const int64_t src_hw,
    const int64_t src_stride,
    const int64_t dst_stride)
{
    const int64_t channel1 = min<const int64_t>(CH_RF_BLK(), channel);
    const int64_t channel2 = max<const int64_t>(0, channel - CH_RF_BLK());
    const int64_t tw_len1  = min<const int64_t>(CH_RF_BLK(), tw_len);
    const int64_t tw_len2  = max<const int64_t>(0, tw_len - CH_RF_BLK());

    if (tw_len1 == CH_RF_BLK()) {
        winograd_b6f3_transpose_4x4_fp32_sse_func_table[0][channel1](l_base_src + 0 * src_stride + 0 * CH_RF_BLK(), src_hw, l_base_dst + 0 * dst_stride);
        if (channel > CH_RF_BLK()) {
            winograd_b6f3_transpose_4x4_fp32_sse_func_table[1][channel2](l_base_src + 1 * src_stride + 0 * CH_RF_BLK(), src_hw, l_base_dst + 0 * dst_stride);
        } else {
            winograd_b6f3_transpose_4x4_fp32_sse_func_table[1][channel2](l_base_src + 0 * src_stride + 0 * CH_RF_BLK(), src_hw, l_base_dst + 0 * dst_stride);
        }
    } else {
        for (int64_t il = 0; il < tw_len1; ++il) {
            int64_t ic = 0;
            for (; ic < channel; ++ic) {
                l_base_dst[ic] = l_base_src[ic * src_hw];
            }
            for (; ic < CH_DT_BLK(); ++ic) {
                l_base_dst[ic] = 0.0f;
            }
            l_base_src += 1;
            l_base_dst += CH_DT_BLK();
        }
    }

    if (tw_len2 == CH_RF_BLK()) {
        winograd_b6f3_transpose_4x4_fp32_sse_func_table[0][channel1](l_base_src + 0 * src_stride + 1 * CH_RF_BLK(), src_hw, l_base_dst + 1 * dst_stride);
        if (channel > CH_RF_BLK()) {
            winograd_b6f3_transpose_4x4_fp32_sse_func_table[1][channel2](l_base_src + 1 * src_stride + 1 * CH_RF_BLK(), src_hw, l_base_dst + 1 * dst_stride);
        } else {
            winograd_b6f3_transpose_4x4_fp32_sse_func_table[1][channel2](l_base_src + 0 * src_stride + 1 * CH_RF_BLK(), src_hw, l_base_dst + 1 * dst_stride);
        }
    } else {
        if (tw_len1 == CH_RF_BLK()) {
            l_base_src += CH_RF_BLK();
            l_base_dst += CH_RF_BLK() * CH_DT_BLK();
        }
        for (int64_t il = 0; il < tw_len2; ++il) {
            int64_t ic = 0;
            for (; ic < channel; ++ic) {
                l_base_dst[ic] = l_base_src[ic * src_hw];
            }
            for (; ic < CH_DT_BLK(); ++ic) {
                l_base_dst[ic] = 0.0f;
            }
            l_base_src += 1;
            l_base_dst += CH_DT_BLK();
        }
    }
}

template <int64_t channel>
static inline void winograd_b6f3_dst_trans_fp32_sse(
    const float *src,
    const float *sum_src,
    const uint64_t fuse_flag,
    const int64_t sum_src_stride,
    const int64_t dst_stride, // dst_h * dst_w
    float *dst)
{
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;
    if (channel >= 1) xmm0 = _mm_loadu_ps(src + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
    if (channel >= 5) xmm8 = _mm_loadu_ps(src + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
    if (channel >= 1) xmm1 = _mm_loadu_ps(src + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
    if (channel >= 5) xmm9 = _mm_loadu_ps(src + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
    if (channel >= 1) xmm2 = _mm_loadu_ps(src + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
    if (channel >= 5) xmm10 = _mm_loadu_ps(src + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
    if (channel >= 1) xmm3 = _mm_loadu_ps(src + 3 * CH_DT_BLK() + 0 * CH_RF_BLK());
    if (channel >= 5) xmm11 = _mm_loadu_ps(src + 3 * CH_DT_BLK() + 1 * CH_RF_BLK());

    if (channel >= 1) TRANSPOSE_4X4_FP32_SSE_MACRO(xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7);
    if (channel >= 5) TRANSPOSE_4X4_FP32_SSE_MACRO(xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15);

    if (fuse_flag & conv_fuse_flag::SUM) {
        if (channel >= 1) xmm0 = _mm_add_ps(xmm0, _mm_loadu_ps(sum_src + 0 * sum_src_stride));
        if (channel >= 2) xmm1 = _mm_add_ps(xmm1, _mm_loadu_ps(sum_src + 1 * sum_src_stride));
        if (channel >= 3) xmm2 = _mm_add_ps(xmm2, _mm_loadu_ps(sum_src + 2 * sum_src_stride));
        if (channel >= 4) xmm3 = _mm_add_ps(xmm3, _mm_loadu_ps(sum_src + 3 * sum_src_stride));
        if (channel >= 5) xmm8 = _mm_add_ps(xmm8, _mm_loadu_ps(sum_src + 4 * sum_src_stride));
        if (channel >= 6) xmm9 = _mm_add_ps(xmm9, _mm_loadu_ps(sum_src + 5 * sum_src_stride));
        if (channel >= 7) xmm10 = _mm_add_ps(xmm10, _mm_loadu_ps(sum_src + 6 * sum_src_stride));
        if (channel >= 8) xmm11 = _mm_add_ps(xmm11, _mm_loadu_ps(sum_src + 7 * sum_src_stride));
    }

    if (fuse_flag & (conv_fuse_flag::RELU | conv_fuse_flag::RELU6)) {
        if (channel >= 1) xmm7 = _mm_setzero_ps();
        if (channel >= 1) xmm0 = _mm_max_ps(xmm0, xmm7);
        if (channel >= 2) xmm1 = _mm_max_ps(xmm1, xmm7);
        if (channel >= 3) xmm2 = _mm_max_ps(xmm2, xmm7);
        if (channel >= 4) xmm3 = _mm_max_ps(xmm3, xmm7);
        if (channel >= 5) xmm8 = _mm_max_ps(xmm8, xmm7);
        if (channel >= 6) xmm9 = _mm_max_ps(xmm9, xmm7);
        if (channel >= 7) xmm10 = _mm_max_ps(xmm10, xmm7);
        if (channel >= 8) xmm11 = _mm_max_ps(xmm11, xmm7);
    }

    if (fuse_flag & conv_fuse_flag::RELU6) {
        if (channel >= 1) xmm6 = _mm_set1_ps(6.0f);
        if (channel >= 1) xmm0 = _mm_min_ps(xmm0, xmm6);
        if (channel >= 2) xmm1 = _mm_min_ps(xmm1, xmm6);
        if (channel >= 3) xmm2 = _mm_min_ps(xmm2, xmm6);
        if (channel >= 4) xmm3 = _mm_min_ps(xmm3, xmm6);
        if (channel >= 5) xmm8 = _mm_min_ps(xmm8, xmm6);
        if (channel >= 6) xmm9 = _mm_min_ps(xmm9, xmm6);
        if (channel >= 7) xmm10 = _mm_min_ps(xmm10, xmm6);
        if (channel >= 8) xmm11 = _mm_min_ps(xmm11, xmm6);
    }

    if (channel >= 1) _mm_storeu_ps(dst + 0 * dst_stride, xmm0);
    if (channel >= 2) _mm_storeu_ps(dst + 1 * dst_stride, xmm1);
    if (channel >= 3) _mm_storeu_ps(dst + 2 * dst_stride, xmm2);
    if (channel >= 4) _mm_storeu_ps(dst + 3 * dst_stride, xmm3);
    if (channel >= 5) _mm_storeu_ps(dst + 4 * dst_stride, xmm8);
    if (channel >= 6) _mm_storeu_ps(dst + 5 * dst_stride, xmm9);
    if (channel >= 7) _mm_storeu_ps(dst + 6 * dst_stride, xmm10);
    if (channel >= 8) _mm_storeu_ps(dst + 7 * dst_stride, xmm11);

    if (channel >= 1) xmm0 = _mm_loadu_ps(src + 4 * CH_DT_BLK() + 0 * CH_RF_BLK());
    if (channel >= 5) xmm8 = _mm_loadu_ps(src + 4 * CH_DT_BLK() + 1 * CH_RF_BLK());
    if (channel >= 1) xmm1 = _mm_loadu_ps(src + 5 * CH_DT_BLK() + 0 * CH_RF_BLK());
    if (channel >= 5) xmm9 = _mm_loadu_ps(src + 5 * CH_DT_BLK() + 1 * CH_RF_BLK());
    if (channel >= 1) xmm2 = _mm_setzero_ps(), xmm3 = xmm2;
    if (channel >= 5) xmm10 = _mm_setzero_ps(), xmm11 = xmm10;

    if (channel >= 1) TRANSPOSE_4X4_FP32_SSE_MACRO(xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7);
    if (channel >= 5) TRANSPOSE_4X4_FP32_SSE_MACRO(xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15);

    if (fuse_flag & conv_fuse_flag::SUM) {
        if (channel >= 1) xmm0 = _mm_add_ps(xmm0, _mm_loadl_pi(xmm7, (__m64 *)(sum_src + 0 * sum_src_stride + CH_RF_BLK())));
        if (channel >= 2) xmm1 = _mm_add_ps(xmm1, _mm_loadl_pi(xmm7, (__m64 *)(sum_src + 1 * sum_src_stride + CH_RF_BLK())));
        if (channel >= 3) xmm2 = _mm_add_ps(xmm2, _mm_loadl_pi(xmm7, (__m64 *)(sum_src + 2 * sum_src_stride + CH_RF_BLK())));
        if (channel >= 4) xmm3 = _mm_add_ps(xmm3, _mm_loadl_pi(xmm7, (__m64 *)(sum_src + 3 * sum_src_stride + CH_RF_BLK())));
        if (channel >= 5) xmm8 = _mm_add_ps(xmm8, _mm_loadl_pi(xmm7, (__m64 *)(sum_src + 4 * sum_src_stride + CH_RF_BLK())));
        if (channel >= 6) xmm9 = _mm_add_ps(xmm9, _mm_loadl_pi(xmm7, (__m64 *)(sum_src + 5 * sum_src_stride + CH_RF_BLK())));
        if (channel >= 7) xmm10 = _mm_add_ps(xmm10, _mm_loadl_pi(xmm7, (__m64 *)(sum_src + 6 * sum_src_stride + CH_RF_BLK())));
        if (channel >= 8) xmm11 = _mm_add_ps(xmm11, _mm_loadl_pi(xmm7, (__m64 *)(sum_src + 7 * sum_src_stride + CH_RF_BLK())));
    }

    if (fuse_flag & (conv_fuse_flag::RELU | conv_fuse_flag::RELU6)) {
        // xmm7 sets to zero in transpose
        if (channel >= 1) xmm0 = _mm_max_ps(xmm0, xmm7);
        if (channel >= 2) xmm1 = _mm_max_ps(xmm1, xmm7);
        if (channel >= 3) xmm2 = _mm_max_ps(xmm2, xmm7);
        if (channel >= 4) xmm3 = _mm_max_ps(xmm3, xmm7);
        if (channel >= 5) xmm8 = _mm_max_ps(xmm8, xmm7);
        if (channel >= 6) xmm9 = _mm_max_ps(xmm9, xmm7);
        if (channel >= 7) xmm10 = _mm_max_ps(xmm10, xmm7);
        if (channel >= 8) xmm11 = _mm_max_ps(xmm11, xmm7);
    }

    if (fuse_flag & conv_fuse_flag::RELU6) {
        if (channel >= 1) xmm6 = _mm_set1_ps(6.0f);
        if (channel >= 1) xmm0 = _mm_min_ps(xmm0, xmm6);
        if (channel >= 2) xmm1 = _mm_min_ps(xmm1, xmm6);
        if (channel >= 3) xmm2 = _mm_min_ps(xmm2, xmm6);
        if (channel >= 4) xmm3 = _mm_min_ps(xmm3, xmm6);
        if (channel >= 5) xmm8 = _mm_min_ps(xmm8, xmm6);
        if (channel >= 6) xmm9 = _mm_min_ps(xmm9, xmm6);
        if (channel >= 7) xmm10 = _mm_min_ps(xmm10, xmm6);
        if (channel >= 8) xmm11 = _mm_min_ps(xmm11, xmm6);
    }

    if (channel >= 1) _mm_storel_pi((__m64 *)(dst + 0 * dst_stride + CH_RF_BLK()), xmm0);
    if (channel >= 2) _mm_storel_pi((__m64 *)(dst + 1 * dst_stride + CH_RF_BLK()), xmm1);
    if (channel >= 3) _mm_storel_pi((__m64 *)(dst + 2 * dst_stride + CH_RF_BLK()), xmm2);
    if (channel >= 4) _mm_storel_pi((__m64 *)(dst + 3 * dst_stride + CH_RF_BLK()), xmm3);
    if (channel >= 5) _mm_storel_pi((__m64 *)(dst + 4 * dst_stride + CH_RF_BLK()), xmm8);
    if (channel >= 6) _mm_storel_pi((__m64 *)(dst + 5 * dst_stride + CH_RF_BLK()), xmm9);
    if (channel >= 7) _mm_storel_pi((__m64 *)(dst + 6 * dst_stride + CH_RF_BLK()), xmm10);
    if (channel >= 8) _mm_storel_pi((__m64 *)(dst + 7 * dst_stride + CH_RF_BLK()), xmm11);
}

template <int64_t channel>
static inline void winograd_b6f3_preprocess_fp32_sse(
    const float *base_src,
    const int64_t ih,
    const int64_t iw,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t src_trans_ti_stride,
    float *tile_buffer,
    float *matmul_buffer,
    float *src_trans)
{
    const int64_t tile_h_stride = TILE_IN_W() * CH_DT_BLK();
    float *l_base_dst           = tile_buffer;
    const int64_t src_hw        = src_h * src_w;
    const int64_t src_stride    = src_hw * CH_RF_BLK();
    const int64_t dst_stride    = CH_RF_BLK() * CH_DT_BLK();

    if (ih >= 0 && ih + TILE_IN_H() <= src_h && iw >= 0 && iw + TILE_IN_W() <= src_w) {
        winograd_b6f3_src_trans_sse<channel>(base_src + (ih + 0) * src_w + iw, l_base_dst + 0 * tile_h_stride, src_hw, src_stride, dst_stride);
        winograd_b6f3_src_trans_sse<channel>(base_src + (ih + 1) * src_w + iw, l_base_dst + 1 * tile_h_stride, src_hw, src_stride, dst_stride);
        winograd_b6f3_src_trans_sse<channel>(base_src + (ih + 2) * src_w + iw, l_base_dst + 2 * tile_h_stride, src_hw, src_stride, dst_stride);
        winograd_b6f3_src_trans_sse<channel>(base_src + (ih + 3) * src_w + iw, l_base_dst + 3 * tile_h_stride, src_hw, src_stride, dst_stride);
        winograd_b6f3_src_trans_sse<channel>(base_src + (ih + 4) * src_w + iw, l_base_dst + 4 * tile_h_stride, src_hw, src_stride, dst_stride);
        winograd_b6f3_src_trans_sse<channel>(base_src + (ih + 5) * src_w + iw, l_base_dst + 5 * tile_h_stride, src_hw, src_stride, dst_stride);
        winograd_b6f3_src_trans_sse<channel>(base_src + (ih + 6) * src_w + iw, l_base_dst + 6 * tile_h_stride, src_hw, src_stride, dst_stride);
        winograd_b6f3_src_trans_sse<channel>(base_src + (ih + 7) * src_w + iw, l_base_dst + 7 * tile_h_stride, src_hw, src_stride, dst_stride);

    } else {
        int64_t tl_pad       = max<int64_t>(0 - iw, 0);
        int64_t tw_start     = max<int64_t>(iw, 0);
        int64_t tw_len       = max<int64_t>(min<int64_t>(src_w, iw + TILE_IN_W()) - tw_start, 0);
        int64_t tr_pad       = max<int64_t>(iw + TILE_IN_W() - src_w, 0);
        float *l_tile_buffer = tile_buffer;

        for (int64_t h = ih; h < ih + TILE_IN_H(); ++h) {
            if (h < 0 || h >= src_h) {
                memset32_sse(l_tile_buffer, 0, tile_h_stride);
            } else {
                int64_t w = 0;
                memset32_sse(l_tile_buffer + w * CH_DT_BLK(), 0, tl_pad * CH_DT_BLK());
                w += tl_pad;
                wingorad_b6f3_memcpy_sse<channel>(base_src + h * src_w + tw_start, l_tile_buffer + w * CH_DT_BLK(), tw_len, src_hw, src_stride, dst_stride);
                w += tw_len;
                memset32_sse(l_tile_buffer + w * CH_DT_BLK(), 0, tr_pad * CH_DT_BLK());
                w += tr_pad;
            }
            l_tile_buffer += tile_h_stride;
        }
    }

    // B
    __m128 xmm12 = _mm_set1_ps(5.25f);
    __m128 xmm13 = _mm_set1_ps(4.25f);
    __m128 xmm14 = _mm_set1_ps(2.5f);
    __m128 xmm15 = _mm_set1_ps(1.25f);
    for (int64_t th = 0; th < TILE_IN_H(); ++th) {
        const float *l_tile = tile_buffer + th * CH_DT_BLK();
        float *l_temp       = matmul_buffer + th * CH_DT_BLK();

        __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9;
        __m128 xmm10, xmm11;

        xmm0 = _mm_loadu_ps(l_tile + 0 * tile_h_stride + 0 * CH_RF_BLK());
        xmm1 = _mm_loadu_ps(l_tile + 1 * tile_h_stride + 0 * CH_RF_BLK());
        xmm2 = _mm_loadu_ps(l_tile + 2 * tile_h_stride + 0 * CH_RF_BLK());
        xmm3 = _mm_loadu_ps(l_tile + 3 * tile_h_stride + 0 * CH_RF_BLK());
        xmm4 = _mm_loadu_ps(l_tile + 4 * tile_h_stride + 0 * CH_RF_BLK());
        xmm5 = _mm_loadu_ps(l_tile + 5 * tile_h_stride + 0 * CH_RF_BLK());
        xmm6 = _mm_loadu_ps(l_tile + 6 * tile_h_stride + 0 * CH_RF_BLK());
        xmm7 = _mm_loadu_ps(l_tile + 7 * tile_h_stride + 0 * CH_RF_BLK());

        xmm12 = _mm_set1_ps(5.25f);
        xmm8  = _mm_sub_ps(xmm0, xmm6);
        xmm9  = _mm_sub_ps(xmm4, xmm2);
        xmm10 = _mm_mul_ps(xmm9, xmm12);
        xmm11 = _mm_add_ps(xmm8, xmm10);

        _mm_storeu_ps(l_temp + 0 * tile_h_stride + 0 * CH_RF_BLK(), xmm11);

        xmm8  = _mm_sub_ps(xmm7, xmm1);
        xmm9  = _mm_sub_ps(xmm3, xmm5);
        xmm10 = _mm_mul_ps(xmm9, xmm12);
        xmm11 = _mm_add_ps(xmm8, xmm10);

        _mm_storeu_ps(l_temp + 7 * tile_h_stride + 0 * CH_RF_BLK(), xmm11);

        xmm8 = _mm_mul_ps(xmm4, xmm13);
        xmm8 = _mm_sub_ps(xmm6, xmm8);
        xmm8 = _mm_add_ps(xmm2, xmm8);

        xmm9 = _mm_mul_ps(xmm3, xmm13);
        xmm9 = _mm_sub_ps(xmm1, xmm9);
        xmm9 = _mm_add_ps(xmm5, xmm9);

        xmm10 = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_temp + 1 * tile_h_stride + 0 * CH_RF_BLK(), xmm10);

        xmm11 = _mm_sub_ps(xmm8, xmm9);
        _mm_storeu_ps(l_temp + 2 * tile_h_stride + 0 * CH_RF_BLK(), xmm11);

        xmm12 = _mm_set1_ps(0.25f);
        xmm8  = _mm_mul_ps(xmm2, xmm12);
        xmm9  = _mm_mul_ps(xmm4, xmm15);
        xmm8  = _mm_sub_ps(xmm8, xmm9);
        xmm8  = _mm_add_ps(xmm6, xmm8);

        xmm12 = _mm_set1_ps(0.5f);
        xmm9  = _mm_mul_ps(xmm1, xmm12);
        xmm10 = _mm_mul_ps(xmm3, xmm14);
        xmm9  = _mm_sub_ps(xmm9, xmm10);
        xmm12 = _mm_set1_ps(2.0f);
        xmm10 = _mm_mul_ps(xmm5, xmm12);
        xmm10 = _mm_add_ps(xmm9, xmm10);

        xmm11 = _mm_add_ps(xmm8, xmm10);
        _mm_storeu_ps(l_temp + 3 * tile_h_stride + 0 * CH_RF_BLK(), xmm11);

        xmm11 = _mm_sub_ps(xmm8, xmm10);
        _mm_storeu_ps(l_temp + 4 * tile_h_stride + 0 * CH_RF_BLK(), xmm11);

        xmm8 = _mm_mul_ps(xmm4, xmm15);
        xmm8 = _mm_sub_ps(xmm2, xmm8);
        xmm9 = _mm_set1_ps(4.0f);
        xmm8 = _mm_mul_ps(xmm8, xmm9);
        xmm8 = _mm_add_ps(xmm6, xmm8);

        xmm9  = _mm_mul_ps(xmm1, xmm12);
        xmm10 = _mm_mul_ps(xmm3, xmm14);
        xmm9  = _mm_sub_ps(xmm9, xmm10);
        xmm12 = _mm_set1_ps(0.5f);
        xmm10 = _mm_mul_ps(xmm5, xmm12);
        xmm10 = _mm_add_ps(xmm9, xmm10);

        xmm11 = _mm_add_ps(xmm8, xmm10);
        _mm_storeu_ps(l_temp + 5 * tile_h_stride + 0 * CH_RF_BLK(), xmm11);

        xmm11 = _mm_sub_ps(xmm8, xmm10);
        _mm_storeu_ps(l_temp + 6 * tile_h_stride + 0 * CH_RF_BLK(), xmm11);

        xmm0 = _mm_loadu_ps(l_tile + 0 * tile_h_stride + 1 * CH_RF_BLK());
        xmm1 = _mm_loadu_ps(l_tile + 1 * tile_h_stride + 1 * CH_RF_BLK());
        xmm2 = _mm_loadu_ps(l_tile + 2 * tile_h_stride + 1 * CH_RF_BLK());
        xmm3 = _mm_loadu_ps(l_tile + 3 * tile_h_stride + 1 * CH_RF_BLK());
        xmm4 = _mm_loadu_ps(l_tile + 4 * tile_h_stride + 1 * CH_RF_BLK());
        xmm5 = _mm_loadu_ps(l_tile + 5 * tile_h_stride + 1 * CH_RF_BLK());
        xmm6 = _mm_loadu_ps(l_tile + 6 * tile_h_stride + 1 * CH_RF_BLK());
        xmm7 = _mm_loadu_ps(l_tile + 7 * tile_h_stride + 1 * CH_RF_BLK());

        xmm12 = _mm_set1_ps(5.25f);
        xmm8  = _mm_sub_ps(xmm0, xmm6);
        xmm9  = _mm_sub_ps(xmm4, xmm2);
        xmm10 = _mm_mul_ps(xmm9, xmm12);
        xmm11 = _mm_add_ps(xmm8, xmm10);

        _mm_storeu_ps(l_temp + 0 * tile_h_stride + 1 * CH_RF_BLK(), xmm11);

        xmm8  = _mm_sub_ps(xmm7, xmm1);
        xmm9  = _mm_sub_ps(xmm3, xmm5);
        xmm10 = _mm_mul_ps(xmm9, xmm12);
        xmm11 = _mm_add_ps(xmm8, xmm10);

        _mm_storeu_ps(l_temp + 7 * tile_h_stride + 1 * CH_RF_BLK(), xmm11);

        xmm8 = _mm_mul_ps(xmm4, xmm13);
        xmm8 = _mm_sub_ps(xmm6, xmm8);
        xmm8 = _mm_add_ps(xmm2, xmm8);

        xmm9 = _mm_mul_ps(xmm3, xmm13);
        xmm9 = _mm_sub_ps(xmm1, xmm9);
        xmm9 = _mm_add_ps(xmm5, xmm9);

        xmm10 = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_temp + 1 * tile_h_stride + 1 * CH_RF_BLK(), xmm10);

        xmm11 = _mm_sub_ps(xmm8, xmm9);
        _mm_storeu_ps(l_temp + 2 * tile_h_stride + 1 * CH_RF_BLK(), xmm11);

        xmm12 = _mm_set1_ps(0.25f);
        xmm8  = _mm_mul_ps(xmm2, xmm12);
        xmm9  = _mm_mul_ps(xmm4, xmm15);
        xmm8  = _mm_sub_ps(xmm8, xmm9);
        xmm8  = _mm_add_ps(xmm6, xmm8);

        xmm12 = _mm_set1_ps(0.5f);
        xmm9  = _mm_mul_ps(xmm1, xmm12);
        xmm10 = _mm_mul_ps(xmm3, xmm14);
        xmm9  = _mm_sub_ps(xmm9, xmm10);
        xmm12 = _mm_set1_ps(2.0f);
        xmm10 = _mm_mul_ps(xmm5, xmm12);
        xmm10 = _mm_add_ps(xmm9, xmm10);

        xmm11 = _mm_add_ps(xmm8, xmm10);
        _mm_storeu_ps(l_temp + 3 * tile_h_stride + 1 * CH_RF_BLK(), xmm11);

        xmm11 = _mm_sub_ps(xmm8, xmm10);
        _mm_storeu_ps(l_temp + 4 * tile_h_stride + 1 * CH_RF_BLK(), xmm11);

        xmm8 = _mm_mul_ps(xmm4, xmm15);
        xmm8 = _mm_sub_ps(xmm2, xmm8);
        xmm9 = _mm_set1_ps(4.0f);
        xmm8 = _mm_mul_ps(xmm8, xmm9);
        xmm8 = _mm_add_ps(xmm6, xmm8);

        xmm9  = _mm_mul_ps(xmm1, xmm12);
        xmm10 = _mm_mul_ps(xmm3, xmm14);
        xmm9  = _mm_sub_ps(xmm9, xmm10);
        xmm12 = _mm_set1_ps(0.5f);
        xmm10 = _mm_mul_ps(xmm5, xmm12);
        xmm10 = _mm_add_ps(xmm9, xmm10);

        xmm11 = _mm_add_ps(xmm8, xmm10);
        _mm_storeu_ps(l_temp + 5 * tile_h_stride + 1 * CH_RF_BLK(), xmm11);

        xmm11 = _mm_sub_ps(xmm8, xmm10);
        _mm_storeu_ps(l_temp + 6 * tile_h_stride + 1 * CH_RF_BLK(), xmm11);
    }

    for (int64_t tw = 0; tw < TILE_IN_W(); ++tw) {
        const float *l_temp = matmul_buffer + tw * tile_h_stride;
        float *l_dst        = src_trans + tw * TILE_IN_W() * src_trans_ti_stride;

        __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9;
        __m128 xmm10, xmm11;

        xmm0 = _mm_loadu_ps(l_temp + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm1 = _mm_loadu_ps(l_temp + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm2 = _mm_loadu_ps(l_temp + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm3 = _mm_loadu_ps(l_temp + 3 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm4 = _mm_loadu_ps(l_temp + 4 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm5 = _mm_loadu_ps(l_temp + 5 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm6 = _mm_loadu_ps(l_temp + 6 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm7 = _mm_loadu_ps(l_temp + 7 * CH_DT_BLK() + 0 * CH_RF_BLK());

        xmm12 = _mm_set1_ps(5.25f);
        xmm8  = _mm_sub_ps(xmm0, xmm6);
        xmm9  = _mm_sub_ps(xmm4, xmm2);
        xmm10 = _mm_mul_ps(xmm9, xmm12);
        xmm11 = _mm_add_ps(xmm8, xmm10);

        _mm_storeu_ps(l_dst + 0 * src_trans_ti_stride + 0 * CH_RF_BLK(), xmm11);

        xmm8  = _mm_sub_ps(xmm7, xmm1);
        xmm9  = _mm_sub_ps(xmm3, xmm5);
        xmm10 = _mm_mul_ps(xmm9, xmm12);
        xmm11 = _mm_add_ps(xmm8, xmm10);

        _mm_storeu_ps(l_dst + 7 * src_trans_ti_stride + 0 * CH_RF_BLK(), xmm11);

        xmm8 = _mm_mul_ps(xmm4, xmm13);
        xmm8 = _mm_sub_ps(xmm6, xmm8);
        xmm8 = _mm_add_ps(xmm2, xmm8);

        xmm9 = _mm_mul_ps(xmm3, xmm13);
        xmm9 = _mm_sub_ps(xmm1, xmm9);
        xmm9 = _mm_add_ps(xmm5, xmm9);

        xmm10 = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_dst + 1 * src_trans_ti_stride + 0 * CH_RF_BLK(), xmm10);

        xmm11 = _mm_sub_ps(xmm8, xmm9);
        _mm_storeu_ps(l_dst + 2 * src_trans_ti_stride + 0 * CH_RF_BLK(), xmm11);

        xmm12 = _mm_set1_ps(0.25f);
        xmm8  = _mm_mul_ps(xmm2, xmm12);
        xmm9  = _mm_mul_ps(xmm4, xmm15);
        xmm8  = _mm_sub_ps(xmm8, xmm9);
        xmm8  = _mm_add_ps(xmm6, xmm8);

        xmm12 = _mm_set1_ps(0.5f);
        xmm9  = _mm_mul_ps(xmm1, xmm12);
        xmm10 = _mm_mul_ps(xmm3, xmm14);
        xmm9  = _mm_sub_ps(xmm9, xmm10);
        xmm12 = _mm_set1_ps(2.0f);
        xmm10 = _mm_mul_ps(xmm5, xmm12);
        xmm10 = _mm_add_ps(xmm9, xmm10);

        xmm11 = _mm_add_ps(xmm8, xmm10);
        _mm_storeu_ps(l_dst + 3 * src_trans_ti_stride + 0 * CH_RF_BLK(), xmm11);

        xmm11 = _mm_sub_ps(xmm8, xmm10);
        _mm_storeu_ps(l_dst + 4 * src_trans_ti_stride + 0 * CH_RF_BLK(), xmm11);

        xmm8 = _mm_mul_ps(xmm4, xmm15);
        xmm8 = _mm_sub_ps(xmm2, xmm8);
        xmm9 = _mm_set1_ps(4.0f);
        xmm8 = _mm_mul_ps(xmm8, xmm9);
        xmm8 = _mm_add_ps(xmm6, xmm8);

        xmm9  = _mm_mul_ps(xmm1, xmm12);
        xmm10 = _mm_mul_ps(xmm3, xmm14);
        xmm9  = _mm_sub_ps(xmm9, xmm10);
        xmm12 = _mm_set1_ps(0.5f);
        xmm10 = _mm_mul_ps(xmm5, xmm12);
        xmm10 = _mm_add_ps(xmm9, xmm10);

        xmm11 = _mm_add_ps(xmm8, xmm10);
        _mm_storeu_ps(l_dst + 5 * src_trans_ti_stride + 0 * CH_RF_BLK(), xmm11);

        xmm11 = _mm_sub_ps(xmm8, xmm10);
        _mm_storeu_ps(l_dst + 6 * src_trans_ti_stride + 0 * CH_RF_BLK(), xmm11);

        xmm0 = _mm_loadu_ps(l_temp + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm1 = _mm_loadu_ps(l_temp + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm2 = _mm_loadu_ps(l_temp + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm3 = _mm_loadu_ps(l_temp + 3 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm4 = _mm_loadu_ps(l_temp + 4 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm5 = _mm_loadu_ps(l_temp + 5 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm6 = _mm_loadu_ps(l_temp + 6 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm7 = _mm_loadu_ps(l_temp + 7 * CH_DT_BLK() + 1 * CH_RF_BLK());

        xmm12 = _mm_set1_ps(5.25f);
        xmm8  = _mm_sub_ps(xmm0, xmm6);
        xmm9  = _mm_sub_ps(xmm4, xmm2);
        xmm10 = _mm_mul_ps(xmm9, xmm12);
        xmm11 = _mm_add_ps(xmm8, xmm10);

        _mm_storeu_ps(l_dst + 0 * src_trans_ti_stride + 1 * CH_RF_BLK(), xmm11);

        xmm8  = _mm_sub_ps(xmm7, xmm1);
        xmm9  = _mm_sub_ps(xmm3, xmm5);
        xmm10 = _mm_mul_ps(xmm9, xmm12);
        xmm11 = _mm_add_ps(xmm8, xmm10);

        _mm_storeu_ps(l_dst + 7 * src_trans_ti_stride + 1 * CH_RF_BLK(), xmm11);

        xmm8 = _mm_mul_ps(xmm4, xmm13);
        xmm8 = _mm_sub_ps(xmm6, xmm8);
        xmm8 = _mm_add_ps(xmm2, xmm8);

        xmm9 = _mm_mul_ps(xmm3, xmm13);
        xmm9 = _mm_sub_ps(xmm1, xmm9);
        xmm9 = _mm_add_ps(xmm5, xmm9);

        xmm10 = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_dst + 1 * src_trans_ti_stride + 1 * CH_RF_BLK(), xmm10);

        xmm11 = _mm_sub_ps(xmm8, xmm9);
        _mm_storeu_ps(l_dst + 2 * src_trans_ti_stride + 1 * CH_RF_BLK(), xmm11);

        xmm12 = _mm_set1_ps(0.25f);
        xmm8  = _mm_mul_ps(xmm2, xmm12);
        xmm9  = _mm_mul_ps(xmm4, xmm15);
        xmm8  = _mm_sub_ps(xmm8, xmm9);
        xmm8  = _mm_add_ps(xmm6, xmm8);

        xmm12 = _mm_set1_ps(0.5f);
        xmm9  = _mm_mul_ps(xmm1, xmm12);
        xmm10 = _mm_mul_ps(xmm3, xmm14);
        xmm9  = _mm_sub_ps(xmm9, xmm10);
        xmm12 = _mm_set1_ps(2.0f);
        xmm10 = _mm_mul_ps(xmm5, xmm12);
        xmm10 = _mm_add_ps(xmm9, xmm10);

        xmm11 = _mm_add_ps(xmm8, xmm10);
        _mm_storeu_ps(l_dst + 3 * src_trans_ti_stride + 1 * CH_RF_BLK(), xmm11);

        xmm11 = _mm_sub_ps(xmm8, xmm10);
        _mm_storeu_ps(l_dst + 4 * src_trans_ti_stride + 1 * CH_RF_BLK(), xmm11);

        xmm8 = _mm_mul_ps(xmm4, xmm15);
        xmm8 = _mm_sub_ps(xmm2, xmm8);
        xmm9 = _mm_set1_ps(4.0f);
        xmm8 = _mm_mul_ps(xmm8, xmm9);
        xmm8 = _mm_add_ps(xmm6, xmm8);

        xmm9  = _mm_mul_ps(xmm1, xmm12);
        xmm10 = _mm_mul_ps(xmm3, xmm14);
        xmm9  = _mm_sub_ps(xmm9, xmm10);
        xmm12 = _mm_set1_ps(0.5f);
        xmm10 = _mm_mul_ps(xmm5, xmm12);
        xmm10 = _mm_add_ps(xmm9, xmm10);

        xmm11 = _mm_add_ps(xmm8, xmm10);
        _mm_storeu_ps(l_dst + 5 * src_trans_ti_stride + 1 * CH_RF_BLK(), xmm11);

        xmm11 = _mm_sub_ps(xmm8, xmm10);
        _mm_storeu_ps(l_dst + 6 * src_trans_ti_stride + 1 * CH_RF_BLK(), xmm11);
    }
}

template <int64_t channel>
static inline void winograd_b6f3_postprocess_fp32_sse(
    const float *dst_trans,
    const float *sum_src,
    const float *bias,
    const int64_t dst_trans_ti_stride,
    const int64_t dst_h_stride,
    const int64_t dst_hw,
    const uint64_t fuse_flag,
    const bool flag,
    float *matmul_buffer,
    float *dst_buf,
    float *dst)
{
    int64_t matmul_h_stride = TILE_IN_W() * CH_DT_BLK();

    __m128 xmm11, xmm12, xmm13, xmm14, xmm15;
    xmm11 = _mm_set1_ps(2.0f);
    xmm12 = _mm_set1_ps(4.0f);
    xmm13 = _mm_set1_ps(8.0f);
    xmm14 = _mm_set1_ps(16.0f);
    xmm15 = _mm_set1_ps(32.0f);

    for (int64_t th = 0; th < TILE_IN_W(); ++th) {
        const float *l_dst_trans = dst_trans + th * dst_trans_ti_stride;
        float *l_temp            = dst_buf + th * CH_DT_BLK();

        __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
        __m128 xmm8, xmm9, xmm10;

        xmm0 = _mm_loadu_ps(l_dst_trans + 0 * TILE_IN_W() * dst_trans_ti_stride + 0 * CH_RF_BLK());
        xmm1 = _mm_loadu_ps(l_dst_trans + 1 * TILE_IN_W() * dst_trans_ti_stride + 0 * CH_RF_BLK());
        xmm2 = _mm_loadu_ps(l_dst_trans + 2 * TILE_IN_W() * dst_trans_ti_stride + 0 * CH_RF_BLK());
        xmm3 = _mm_loadu_ps(l_dst_trans + 3 * TILE_IN_W() * dst_trans_ti_stride + 0 * CH_RF_BLK());
        xmm4 = _mm_loadu_ps(l_dst_trans + 4 * TILE_IN_W() * dst_trans_ti_stride + 0 * CH_RF_BLK());
        xmm5 = _mm_loadu_ps(l_dst_trans + 5 * TILE_IN_W() * dst_trans_ti_stride + 0 * CH_RF_BLK());
        xmm6 = _mm_loadu_ps(l_dst_trans + 6 * TILE_IN_W() * dst_trans_ti_stride + 0 * CH_RF_BLK());
        xmm7 = _mm_loadu_ps(l_dst_trans + 7 * TILE_IN_W() * dst_trans_ti_stride + 0 * CH_RF_BLK());

        xmm10 = _mm_add_ps(xmm5, xmm6);
        xmm8  = _mm_mul_ps(xmm10, xmm15);
        xmm8  = _mm_add_ps(xmm0, xmm8);
        xmm0  = _mm_add_ps(xmm1, xmm2);
        xmm8  = _mm_add_ps(xmm8, xmm0);
        xmm9  = _mm_add_ps(xmm3, xmm4);
        xmm8  = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_temp + 0 * matmul_h_stride + 0 * CH_RF_BLK(), xmm8);

        xmm9 = _mm_mul_ps(xmm9, xmm12);
        xmm8 = _mm_mul_ps(xmm10, xmm13);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        xmm8 = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_temp + 2 * matmul_h_stride + 0 * CH_RF_BLK(), xmm8);

        xmm9 = _mm_mul_ps(xmm9, xmm12);
        xmm8 = _mm_mul_ps(xmm10, xmm11);
        xmm8 = _mm_add_ps(xmm8, xmm9);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        _mm_storeu_ps(l_temp + 4 * matmul_h_stride + 0 * CH_RF_BLK(), xmm8);

        xmm0 = _mm_sub_ps(xmm1, xmm2);
        xmm1 = _mm_sub_ps(xmm3, xmm4);
        xmm2 = _mm_sub_ps(xmm5, xmm6);

        xmm1 = _mm_mul_ps(xmm1, xmm11);
        xmm8 = _mm_mul_ps(xmm2, xmm14);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        xmm8 = _mm_add_ps(xmm8, xmm1);
        _mm_storeu_ps(l_temp + 1 * matmul_h_stride + 0 * CH_RF_BLK(), xmm8);

        xmm1 = _mm_mul_ps(xmm1, xmm12);
        xmm8 = _mm_mul_ps(xmm2, xmm12);
        xmm8 = _mm_add_ps(xmm8, xmm1);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        _mm_storeu_ps(l_temp + 3 * matmul_h_stride + 0 * CH_RF_BLK(), xmm8);

        xmm1 = _mm_mul_ps(xmm1, xmm12);
        xmm0 = _mm_add_ps(xmm0, xmm7);
        xmm8 = _mm_add_ps(xmm0, xmm1);
        xmm8 = _mm_add_ps(xmm8, xmm2);
        _mm_storeu_ps(l_temp + 5 * matmul_h_stride + 0 * CH_RF_BLK(), xmm8);

        xmm0 = _mm_loadu_ps(l_dst_trans + 0 * TILE_IN_W() * dst_trans_ti_stride + 1 * CH_RF_BLK());
        xmm1 = _mm_loadu_ps(l_dst_trans + 1 * TILE_IN_W() * dst_trans_ti_stride + 1 * CH_RF_BLK());
        xmm2 = _mm_loadu_ps(l_dst_trans + 2 * TILE_IN_W() * dst_trans_ti_stride + 1 * CH_RF_BLK());
        xmm3 = _mm_loadu_ps(l_dst_trans + 3 * TILE_IN_W() * dst_trans_ti_stride + 1 * CH_RF_BLK());
        xmm4 = _mm_loadu_ps(l_dst_trans + 4 * TILE_IN_W() * dst_trans_ti_stride + 1 * CH_RF_BLK());
        xmm5 = _mm_loadu_ps(l_dst_trans + 5 * TILE_IN_W() * dst_trans_ti_stride + 1 * CH_RF_BLK());
        xmm6 = _mm_loadu_ps(l_dst_trans + 6 * TILE_IN_W() * dst_trans_ti_stride + 1 * CH_RF_BLK());
        xmm7 = _mm_loadu_ps(l_dst_trans + 7 * TILE_IN_W() * dst_trans_ti_stride + 1 * CH_RF_BLK());

        xmm10 = _mm_add_ps(xmm5, xmm6);
        xmm8  = _mm_mul_ps(xmm10, xmm15);
        xmm8  = _mm_add_ps(xmm0, xmm8);
        xmm0  = _mm_add_ps(xmm1, xmm2);
        xmm8  = _mm_add_ps(xmm8, xmm0);
        xmm9  = _mm_add_ps(xmm3, xmm4);
        xmm8  = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_temp + 0 * matmul_h_stride + 1 * CH_RF_BLK(), xmm8);

        xmm9 = _mm_mul_ps(xmm9, xmm12);
        xmm8 = _mm_mul_ps(xmm10, xmm13);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        xmm8 = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_temp + 2 * matmul_h_stride + 1 * CH_RF_BLK(), xmm8);

        xmm9 = _mm_mul_ps(xmm9, xmm12);
        xmm8 = _mm_mul_ps(xmm10, xmm11);
        xmm8 = _mm_add_ps(xmm8, xmm9);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        _mm_storeu_ps(l_temp + 4 * matmul_h_stride + 1 * CH_RF_BLK(), xmm8);

        xmm0 = _mm_sub_ps(xmm1, xmm2);
        xmm1 = _mm_sub_ps(xmm3, xmm4);
        xmm2 = _mm_sub_ps(xmm5, xmm6);

        xmm1 = _mm_mul_ps(xmm1, xmm11);
        xmm8 = _mm_mul_ps(xmm2, xmm14);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        xmm8 = _mm_add_ps(xmm8, xmm1);
        _mm_storeu_ps(l_temp + 1 * matmul_h_stride + 1 * CH_RF_BLK(), xmm8);

        xmm1 = _mm_mul_ps(xmm1, xmm12);
        xmm8 = _mm_mul_ps(xmm2, xmm12);
        xmm8 = _mm_add_ps(xmm8, xmm1);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        _mm_storeu_ps(l_temp + 3 * matmul_h_stride + 1 * CH_RF_BLK(), xmm8);

        xmm1 = _mm_mul_ps(xmm1, xmm12);
        xmm0 = _mm_add_ps(xmm0, xmm7);
        xmm8 = _mm_add_ps(xmm0, xmm1);
        xmm8 = _mm_add_ps(xmm8, xmm2);
        _mm_storeu_ps(l_temp + 5 * matmul_h_stride + 1 * CH_RF_BLK(), xmm8);
    }

    for (int64_t tw = 0; tw < TILE_OUT_H(); ++tw) {
        float *l_dst  = matmul_buffer + tw * TILE_OUT_W() * CH_DT_BLK();
        float *l_temp = dst_buf + tw * matmul_h_stride;

        __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
        __m128 xmm8, xmm9, xmm10;

        xmm0 = _mm_loadu_ps(l_temp + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm1 = _mm_loadu_ps(l_temp + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm2 = _mm_loadu_ps(l_temp + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm3 = _mm_loadu_ps(l_temp + 3 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm4 = _mm_loadu_ps(l_temp + 4 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm5 = _mm_loadu_ps(l_temp + 5 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm6 = _mm_loadu_ps(l_temp + 6 * CH_DT_BLK() + 0 * CH_RF_BLK());
        xmm7 = _mm_loadu_ps(l_temp + 7 * CH_DT_BLK() + 0 * CH_RF_BLK());

        xmm10 = _mm_add_ps(xmm5, xmm6);
        xmm8  = _mm_mul_ps(xmm10, xmm15);
        xmm8  = _mm_add_ps(xmm0, xmm8);
        // add bias
        xmm0  = _mm_loadu_ps(bias + 0 * CH_RF_BLK());
        xmm0  = _mm_add_ps(xmm0, xmm2);
        xmm0  = _mm_add_ps(xmm0, xmm1);

        xmm8 = _mm_add_ps(xmm8, xmm0);
        xmm9 = _mm_add_ps(xmm3, xmm4);
        xmm8 = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);

        xmm9 = _mm_mul_ps(xmm9, xmm12);
        xmm8 = _mm_mul_ps(xmm10, xmm13);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        xmm8 = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);

        xmm9 = _mm_mul_ps(xmm9, xmm12);
        xmm8 = _mm_mul_ps(xmm10, xmm11);
        xmm8 = _mm_add_ps(xmm8, xmm9);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        _mm_storeu_ps(l_dst + 4 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);

        // add bias
        xmm0 = _mm_loadu_ps(bias + 0 * CH_RF_BLK());
        xmm0 = _mm_add_ps(xmm0, xmm1);
        xmm0 = _mm_sub_ps(xmm0, xmm2);

        xmm1 = _mm_sub_ps(xmm3, xmm4);
        xmm2 = _mm_sub_ps(xmm5, xmm6);

        xmm1 = _mm_mul_ps(xmm1, xmm11);
        xmm8 = _mm_mul_ps(xmm2, xmm14);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        xmm8 = _mm_add_ps(xmm8, xmm1);
        _mm_storeu_ps(l_dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);

        xmm1 = _mm_mul_ps(xmm1, xmm12);
        xmm8 = _mm_mul_ps(xmm2, xmm12);
        xmm8 = _mm_add_ps(xmm8, xmm1);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        _mm_storeu_ps(l_dst + 3 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);

        xmm1 = _mm_mul_ps(xmm1, xmm12);
        xmm0 = _mm_add_ps(xmm0, xmm7);
        xmm8 = _mm_add_ps(xmm0, xmm1);
        xmm8 = _mm_add_ps(xmm8, xmm2);
        _mm_storeu_ps(l_dst + 5 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);

        xmm0 = _mm_loadu_ps(l_temp + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm1 = _mm_loadu_ps(l_temp + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm2 = _mm_loadu_ps(l_temp + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm3 = _mm_loadu_ps(l_temp + 3 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm4 = _mm_loadu_ps(l_temp + 4 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm5 = _mm_loadu_ps(l_temp + 5 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm6 = _mm_loadu_ps(l_temp + 6 * CH_DT_BLK() + 1 * CH_RF_BLK());
        xmm7 = _mm_loadu_ps(l_temp + 7 * CH_DT_BLK() + 1 * CH_RF_BLK());

        xmm10 = _mm_add_ps(xmm5, xmm6);
        xmm8  = _mm_mul_ps(xmm10, xmm15);
        xmm8  = _mm_add_ps(xmm0, xmm8);
        // add bias
        xmm0  = _mm_loadu_ps(bias + 1 * CH_RF_BLK());
        xmm0  = _mm_add_ps(xmm0, xmm2);
        xmm0  = _mm_add_ps(xmm0, xmm1);

        xmm8 = _mm_add_ps(xmm8, xmm0);
        xmm9 = _mm_add_ps(xmm3, xmm4);
        xmm8 = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm8);

        xmm9 = _mm_mul_ps(xmm9, xmm12);
        xmm8 = _mm_mul_ps(xmm10, xmm13);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        xmm8 = _mm_add_ps(xmm8, xmm9);
        _mm_storeu_ps(l_dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm8);

        xmm9 = _mm_mul_ps(xmm9, xmm12);
        xmm8 = _mm_mul_ps(xmm10, xmm11);
        xmm8 = _mm_add_ps(xmm8, xmm9);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        _mm_storeu_ps(l_dst + 4 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm8);

        // add bias
        xmm0 = _mm_loadu_ps(bias + 1 * CH_RF_BLK());
        xmm0 = _mm_add_ps(xmm0, xmm1);
        xmm0 = _mm_sub_ps(xmm0, xmm2);

        xmm1 = _mm_sub_ps(xmm3, xmm4);
        xmm2 = _mm_sub_ps(xmm5, xmm6);

        xmm1 = _mm_mul_ps(xmm1, xmm11);
        xmm8 = _mm_mul_ps(xmm2, xmm14);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        xmm8 = _mm_add_ps(xmm8, xmm1);
        _mm_storeu_ps(l_dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm8);

        xmm1 = _mm_mul_ps(xmm1, xmm12);
        xmm8 = _mm_mul_ps(xmm2, xmm12);
        xmm8 = _mm_add_ps(xmm8, xmm1);
        xmm8 = _mm_add_ps(xmm8, xmm0);
        _mm_storeu_ps(l_dst + 3 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm8);

        xmm1 = _mm_mul_ps(xmm1, xmm12);
        xmm0 = _mm_add_ps(xmm0, xmm7);
        xmm8 = _mm_add_ps(xmm0, xmm1);
        xmm8 = _mm_add_ps(xmm8, xmm2);
        _mm_storeu_ps(l_dst + 5 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm8);
    }

    matmul_h_stride            = TILE_OUT_W() * CH_DT_BLK();
    const int64_t tmp_h_stride = flag ? dst_h_stride : TILE_IN_W();
    const int64_t tmp_hw       = flag ? dst_hw : TILE_IN_H() * TILE_IN_W();

    // dst trans: trans data from matmul to dst
    winograd_b6f3_dst_trans_fp32_sse<channel>(matmul_buffer + 0 * matmul_h_stride, sum_src + 0 * dst_h_stride, fuse_flag, dst_hw, tmp_hw, dst + 0 * tmp_h_stride);
    winograd_b6f3_dst_trans_fp32_sse<channel>(matmul_buffer + 1 * matmul_h_stride, sum_src + 1 * dst_h_stride, fuse_flag, dst_hw, tmp_hw, dst + 1 * tmp_h_stride);
    winograd_b6f3_dst_trans_fp32_sse<channel>(matmul_buffer + 2 * matmul_h_stride, sum_src + 2 * dst_h_stride, fuse_flag, dst_hw, tmp_hw, dst + 2 * tmp_h_stride);
    winograd_b6f3_dst_trans_fp32_sse<channel>(matmul_buffer + 3 * matmul_h_stride, sum_src + 3 * dst_h_stride, fuse_flag, dst_hw, tmp_hw, dst + 3 * tmp_h_stride);
    winograd_b6f3_dst_trans_fp32_sse<channel>(matmul_buffer + 4 * matmul_h_stride, sum_src + 4 * dst_h_stride, fuse_flag, dst_hw, tmp_hw, dst + 4 * tmp_h_stride);
    winograd_b6f3_dst_trans_fp32_sse<channel>(matmul_buffer + 5 * matmul_h_stride, sum_src + 5 * dst_h_stride, fuse_flag, dst_hw, tmp_hw, dst + 5 * tmp_h_stride);
}

template <int64_t channel>
void winograd_b6f3_store_dst_fp32_sse(
    const float *src,
    const int64_t oh_len,
    const int64_t ow_len,
    const int64_t dst_h_stride,
    const int64_t dst_hw,
    float *dst)
{
    for (int64_t oh = 0; oh < oh_len; ++oh) {
        const float *l_src = src + oh * TILE_IN_W();
        float *l_dst       = dst + oh * dst_h_stride;
        if (channel >= 1) memcpy32_sse(l_dst + 0 * dst_hw, l_src + 0 * TILE_IN_H() * TILE_IN_W(), ow_len);
        if (channel >= 2) memcpy32_sse(l_dst + 1 * dst_hw, l_src + 1 * TILE_IN_H() * TILE_IN_W(), ow_len);
        if (channel >= 3) memcpy32_sse(l_dst + 2 * dst_hw, l_src + 2 * TILE_IN_H() * TILE_IN_W(), ow_len);
        if (channel >= 4) memcpy32_sse(l_dst + 3 * dst_hw, l_src + 3 * TILE_IN_H() * TILE_IN_W(), ow_len);
        if (channel >= 5) memcpy32_sse(l_dst + 4 * dst_hw, l_src + 4 * TILE_IN_H() * TILE_IN_W(), ow_len);
        if (channel >= 6) memcpy32_sse(l_dst + 5 * dst_hw, l_src + 5 * TILE_IN_H() * TILE_IN_W(), ow_len);
        if (channel >= 7) memcpy32_sse(l_dst + 6 * dst_hw, l_src + 6 * TILE_IN_H() * TILE_IN_W(), ow_len);
        if (channel >= 8) memcpy32_sse(l_dst + 7 * dst_hw, l_src + 7 * TILE_IN_H() * TILE_IN_W(), ow_len);
    }
}

typedef void (*winograd_b6f3_preprocess_fp32_sse_func_t)(const float *, const int64_t, const int64_t, const int64_t, const int64_t, const int64_t, float *, float *, float *);
static const winograd_b6f3_preprocess_fp32_sse_func_t winograd_b6f3_preprocess_fp32_sse_func_table[CH_DT_BLK() + 1]
{
    nullptr,
    winograd_b6f3_preprocess_fp32_sse<1>,
    winograd_b6f3_preprocess_fp32_sse<2>,
    winograd_b6f3_preprocess_fp32_sse<3>,
    winograd_b6f3_preprocess_fp32_sse<4>,
    winograd_b6f3_preprocess_fp32_sse<5>,
    winograd_b6f3_preprocess_fp32_sse<6>,
    winograd_b6f3_preprocess_fp32_sse<7>,
    winograd_b6f3_preprocess_fp32_sse<8>,
};

typedef void (*winograd_b6f3_postprocess_fp32_sse_func_t)(const float *, const float *, const float *, const int64_t, const int64_t, const int64_t, const uint64_t, const bool, float *, float *, float *);
static const winograd_b6f3_postprocess_fp32_sse_func_t winograd_b6f3_postprocess_fp32_sse_func_table[CH_DT_BLK() + 1]
{
    nullptr,
    winograd_b6f3_postprocess_fp32_sse<1>,
    winograd_b6f3_postprocess_fp32_sse<2>,
    winograd_b6f3_postprocess_fp32_sse<3>,
    winograd_b6f3_postprocess_fp32_sse<4>,
    winograd_b6f3_postprocess_fp32_sse<5>,
    winograd_b6f3_postprocess_fp32_sse<6>,
    winograd_b6f3_postprocess_fp32_sse<7>,
    winograd_b6f3_postprocess_fp32_sse<8>,
};

typedef void (*winograd_b6f3_store_dst_fp32_sse_func_t)(const float *, const int64_t, const int64_t, const int64_t, const int64_t, float *);
static const winograd_b6f3_store_dst_fp32_sse_func_t winograd_b6f3_store_dst_fp32_sse_func_table[CH_DT_BLK() + 1]
{
    nullptr,
    winograd_b6f3_store_dst_fp32_sse<1>,
    winograd_b6f3_store_dst_fp32_sse<2>,
    winograd_b6f3_store_dst_fp32_sse<3>,
    winograd_b6f3_store_dst_fp32_sse<4>,
    winograd_b6f3_store_dst_fp32_sse<5>,
    winograd_b6f3_store_dst_fp32_sse<6>,
    winograd_b6f3_store_dst_fp32_sse<7>,
    winograd_b6f3_store_dst_fp32_sse<8>,
};

ppl::common::RetCode conv2d_winograd_b6f3_fp32_sse_executor::execute()
{
    if (!conv_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || ((conv_param_->fuse_flag & conv_fuse_flag::SUM) && !sum_src_) || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const conv2d_param &cp     = *conv_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int64_t src_h = src_shape_->GetDim(2);
    const int64_t src_w = src_shape_->GetDim(3);
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);

    const int64_t src_c  = src_shape_->GetDim(1);
    const int64_t dst_c  = dst_shape_->GetDim(1);
    const int64_t dst_hw = dst_h * dst_w;

    const int64_t src_g_stride     = sp.ic_per_gp * src_h * src_w;
    const int64_t src_b_stride     = src_c * src_h * src_w;
    const int64_t dst_g_stride     = sp.oc_per_gp * dst_hw;
    const int64_t dst_b_stride     = dst_c * dst_hw;
    const int64_t bias_g_stride    = sp.padded_oc;
    const int64_t cvt_flt_g_stride = sp.ic_per_gp * sp.padded_oc * TILE_IN_H() * TILE_IN_W();

    int64_t sum_src_b_stride = 0;
    if (cp.fuse_flag & conv_fuse_flag::SUM) {
        sum_src_b_stride = sum_src_shape_->GetDim(1) * dst_hw;
    }

    // cvt_flt:   [group, ic_l2_cnt, 8h, 8w, oc/8o, icl2_eff, 8o]
    // src_trans: [8h, 8w, tile_l2_blk/8t, icl2_eff/8i, tile_kr_eff, 8i]
    // gemm_out:  [8h, 8w, (oc_l2_blk/8, )tile_l2_eff, 8o]
    PRAGMA_OMP_PARALLEL()
    for (int64_t g = 0; g < cp.group; ++g) {
        for (int64_t tl2 = 0; tl2 < sp.num_tiles; tl2 += sp.tiles_l2_blk) {
            const int64_t tl2_eff = min<int64_t>(sp.tiles_l2_blk, (sp.num_tiles - tl2));

            float *src_trans      = (float *)temp_buffer_;
            float *gemm_out_buf   = src_trans + sp.src_trans_len;
            float *base_workspace = gemm_out_buf + sp.gemm_out_len;

            for (int64_t icl2 = 0; icl2 < sp.ic_per_gp; icl2 += sp.ic_l2_blk) {
                const int64_t icl2_eff        = min<int64_t>(sp.ic_l2_blk, sp.ic_per_gp - icl2);
                const int64_t icl2_eff_padded = round_up(icl2_eff, CH_DT_BLK());
                const bool is_first_ic        = icl2 == 0;
                const bool is_last_ic         = icl2 + sp.ic_l2_blk >= sp.ic_per_gp;
                uint64_t kernel_flags         = 0;
                float *l_bias                 = nullptr;
                float tmp_bias[TILE_OC_RF() * CH_DT_BLK()] = {0};

                if (is_first_ic) {
                    kernel_flags |= KERNEL_FLAG_LD_BIAS();
                    l_bias = tmp_bias;
                }

#ifdef PPL_USE_X86_OMP_COLLAPSE
                PRAGMA_OMP_FOR_COLLAPSE(2)
#endif
                for (int64_t icb = icl2; icb < icl2 + icl2_eff_padded; icb += CH_DT_BLK()) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                    PRAGMA_OMP_FOR()
#endif
                    for (int64_t tk = tl2; tk < tl2 + tl2_eff; ++tk) {
#ifdef PPL_X86_KERNEL_TIMING
                        profiler_.tic(SRCTR_TIMER());
#endif
                        float *thread_workspace = base_workspace + PPL_OMP_THREAD_ID() * sp.thread_workspace_len;
                        float *tile_in_buf      = thread_workspace;
                        float *matmul_in_buf    = tile_in_buf + sp.thread_tile_in_len;

                        tile_corr tc     = cal_tile_corr(sp, tk);
                        const int64_t b  = tc.b;
                        const int64_t oh = tc.th * TILE_OUT_H();
                        const int64_t ow = tc.tw * TILE_OUT_W();
                        const int64_t ih = oh * STRIDE_H() - cp.pad_h;
                        const int64_t iw = ow * STRIDE_W() - cp.pad_w;

                        float *l_src_trans = src_trans 
                            + (icb - icl2) * tl2_eff 
                            + (tk - tl2) * CH_DT_BLK();

                        const float *base_src = src_ 
                            + b * src_b_stride 
                            + g * src_g_stride 
                            + icb * src_h * src_w;

                        const int64_t channel = min<int64_t>(CH_DT_BLK(), icl2_eff - icb + icl2);

                        winograd_b6f3_preprocess_fp32_sse_func_table[channel](
                            base_src, ih, iw, src_h, src_w, tl2_eff * icl2_eff_padded, tile_in_buf, matmul_in_buf, l_src_trans);

#ifdef PPL_X86_KERNEL_TIMING
                        profiler_.toc(SRCTR_TIMER());
#endif
                    }
                }

                for (int64_t ocl2 = 0; ocl2 < sp.padded_oc; ocl2 += sp.oc_l2_blk) {
                    const int64_t ocl2_eff = min<int64_t>(sp.oc_l2_blk, sp.padded_oc - ocl2);

#ifdef PPL_USE_X86_OMP_COLLAPSE
                    PRAGMA_OMP_FOR_COLLAPSE(2)
#else
                    PRAGMA_OMP_FOR()
#endif
                    for (int64_t ti = 0; ti < TILE_IN_H() * TILE_IN_W(); ++ti) {
                        for (int64_t ocb = ocl2; ocb < ocl2 + ocl2_eff; ocb += CH_DT_BLK() * TILE_OC_RF()) {
#ifdef PPL_X86_KERNEL_TIMING
                            profiler_.tic(GEMM_TIMER());
#endif
                            float *l_src_trans = src_trans + ti * tl2_eff * icl2_eff_padded;

                            const float *l_cvt_flt = cvt_filter_ 
                                + g * cvt_flt_g_stride 
                                + icl2 * TILE_IN_H() * TILE_IN_W() * sp.padded_oc 
                                + ti * sp.padded_oc * icl2_eff 
                                + ocb * icl2_eff;

                            float *l_gemm_out;
                            if (sp.override_only) {
                                l_gemm_out = gemm_out_buf 
                                            + ti * ocl2_eff * tl2_eff 
                                            + (ocb - ocl2) * tl2_eff;
                            } else {
                                l_gemm_out = gemm_out_buf 
                                            + ti * sp.padded_oc * tl2_eff 
                                            + ocb * tl2_eff;
                            }

                            int64_t share_param[SHAR_PARAM_LEN()];
                            share_param[SRC_ICB_STRIDE_IDX()] = tl2_eff * CH_DT_BLK();
                            share_param[FLT_OCB_STRIDE_IDX()] = icl2_eff * CH_DT_BLK();
                            share_param[HIS_OCB_STRIDE_IDX()] = tl2_eff * CH_DT_BLK();
                            share_param[DST_OCB_STRIDE_IDX()] = tl2_eff * CH_DT_BLK();
                            share_param[FLAGS_IDX()]          = kernel_flags;
                            share_param[CHANNELS_IDX()]       = icl2_eff;
                            const int64_t oc_sel              = min<int64_t>(TILE_OC_RF(), div_up(ocl2_eff - ocb + ocl2, CH_DT_BLK())) - 1;
                            int64_t private_param[PRIV_PARAM_LEN()];
                            PICK_PARAM(const float *, private_param, FLT_IDX())  = l_cvt_flt;
                            PICK_PARAM(const float *, private_param, BIAS_IDX()) = l_bias;
                            PICK_PARAM(const float *, private_param, SRC_IDX())  = l_src_trans;
                            PICK_PARAM(const float *, private_param, HIS_IDX())  = l_gemm_out;
                            PICK_PARAM(float *, private_param, DST_IDX())        = l_gemm_out;
                            private_param[HW_IDX()]                              = tl2_eff;
                            conv2d_n8cx_gemm_direct_kernel_fp32_sse_hw1_table[0][oc_sel](private_param, share_param);

#ifdef PPL_X86_KERNEL_TIMING
                            profiler_.toc(GEMM_TIMER());
#endif
                        }
                    }

                    if (is_last_ic) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
                        PRAGMA_OMP_FOR_COLLAPSE(2)
#else
                        PRAGMA_OMP_FOR()
#endif
                        for (int64_t ocb = ocl2; ocb < ocl2 + ocl2_eff; ocb += CH_DT_BLK()) {
                            for (int64_t tk = tl2; tk < tl2 + tl2_eff; ++tk) {
#ifdef PPL_X86_KERNEL_TIMING
                                profiler_.tic(DSTTR_TIMER());
#endif
                                float *thread_workspace = base_workspace + PPL_OMP_THREAD_ID() * sp.thread_workspace_len;
                                float *postprocess_buf  = thread_workspace;

                                tile_corr tc         = cal_tile_corr(sp, tk);
                                const int64_t b      = tc.b;
                                const int64_t oh     = tc.th * TILE_OUT_H();
                                const int64_t ow     = tc.tw * TILE_OUT_W();
                                const int64_t oh_len = min<int64_t>(dst_h - oh, TILE_OUT_H());
                                const int64_t ow_len = min<int64_t>(dst_w - ow, TILE_OUT_W());

                                float *l_dst = dst_ 
                                    + b * dst_b_stride 
                                    + g * dst_g_stride 
                                    + ocb * dst_hw 
                                    + oh * dst_w + ow;

                                const float *l_sum_src = sum_src_ 
                                    + b * sum_src_b_stride 
                                    + g * dst_g_stride 
                                    + ocb * dst_hw 
                                    + oh * dst_w + ow;

                                const float *l_gemm_out = gemm_out_buf 
                                    + ocb * tl2_eff 
                                    + (tk - tl2) * CH_DT_BLK();

                                int64_t gemm_out_ti_stride = tl2_eff * sp.padded_oc;
                                if (sp.override_only) {
                                    l_gemm_out         = gemm_out_buf + (ocb - ocl2) * tl2_eff + (tk - tl2) * CH_DT_BLK();
                                    gemm_out_ti_stride = tl2_eff * ocl2_eff;
                                }

                                const int64_t channel = min<int64_t>(CH_DT_BLK(), sp.oc_per_gp - ocb);
                                float *dst_buf        = postprocess_buf + sp.thread_matmul_out_len;

                                if (oh_len == TILE_OUT_H() && ow_len == TILE_OUT_W()) {
                                    winograd_b6f3_postprocess_fp32_sse_func_table[channel](
                                        l_gemm_out, l_sum_src, cvt_bias_ + g * bias_g_stride + ocb, 
                                        gemm_out_ti_stride, dst_w, dst_hw, cp.fuse_flag, true, postprocess_buf, dst_buf, l_dst);

                                } else {
                                    winograd_b6f3_postprocess_fp32_sse_func_table[channel](
                                        l_gemm_out, l_sum_src, cvt_bias_ + g * bias_g_stride + ocb, 
                                        gemm_out_ti_stride, dst_w, dst_hw, cp.fuse_flag, false, postprocess_buf, dst_buf, dst_buf);

                                    winograd_b6f3_store_dst_fp32_sse_func_table[channel](
                                        dst_buf, oh_len, ow_len, dst_w, dst_hw, l_dst);
                                }
#ifdef PPL_X86_KERNEL_TIMING
                                profiler_.toc(DSTTR_TIMER());
#endif
                            }
                        }
                    }
                }
            }
        }
    } // OMP_PARALLEL

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_winograd_b6f3_fp32_sse_manager::gen_cvt_weights(
    const float *filter,
    const float *bias)
{
    const int64_t ic_per_gp = param_.channels / param_.group;
    const int64_t oc_per_gp = param_.num_output / param_.group;
    const int64_t padded_oc = round_up(oc_per_gp, CH_DT_BLK());
    const int64_t padded_ic = round_up(ic_per_gp, CH_DT_BLK());

    const int64_t ic_l2_blk = get_ic_l2_blk(ic_per_gp, oc_per_gp);

    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }
    cvt_bias_size_ = param_.group * padded_oc;
    cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
    if (cvt_bias_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }
    for (int64_t g = 0; g < param_.group; ++g) {
        memcpy(cvt_bias_ + g * padded_oc, bias + g * oc_per_gp, oc_per_gp * sizeof(float));
        memset(cvt_bias_ + g * padded_oc + oc_per_gp, 0, (padded_oc - oc_per_gp) * sizeof(float));
    }

    const int64_t cvt_flt_g_stride = TILE_IN_H() * TILE_IN_W() * padded_oc * ic_per_gp;
    cvt_filter_size_               = cvt_flt_g_stride * param_.group;
    cvt_filter_                    = (float *)allocator_->Alloc(cvt_filter_size_ * sizeof(float));
    if (cvt_filter_ == nullptr) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    const float mat_g[TILE_IN_H()][KERNEL_H()] = {
        {1.0f, 0.0f, 0.0f},
        {-2.0f / 9, -2.0f / 9, -2.0f / 9},
        {-2.0f / 9, 2.0f / 9, -2.0f / 9},
        {1.0f / 90, 1.0f / 45, 2.0f / 45},
        {1.0f / 90, -1.0f / 45, 2.0f / 45},
        {1.0f / 45, 1.0f / 90, 1.0f / 180},
        {1.0f / 45, -1.0f / 90, 1.0f / 180},
        {0.0f, 0.0f, 1.0f},
    };

    // goihw trans goithtw -> gIthtwOi8o
    for (int64_t g = 0; g < param_.group; ++g) {
        for (int64_t icl2 = 0; icl2 < padded_ic; icl2 += ic_l2_blk) {
            for (int64_t ocb = 0; ocb < padded_oc; ocb += CH_DT_BLK()) {
                const int64_t icl2_eff = min<int64_t>(ic_per_gp - icl2, ic_l2_blk);
                const int64_t ocb_eff  = min<int64_t>(oc_per_gp - ocb, CH_DT_BLK());
                float mat_T[TILE_IN_H()][KERNEL_W()];
                for (int64_t ic = icl2; ic < icl2 + icl2_eff; ++ic) {

                    const float *l_flt = filter 
                        + g * oc_per_gp * ic_per_gp * KERNEL_H() * KERNEL_W() 
                        + ocb * ic_per_gp * KERNEL_H() * KERNEL_W() 
                        + ic * KERNEL_H() * KERNEL_W();
                    
                    float *l_cvt_flt   = cvt_filter_ 
                        + g * cvt_flt_g_stride 
                        + icl2 * TILE_IN_H() * TILE_IN_W() * padded_oc 
                        + ocb * icl2_eff 
                        + (ic - icl2) * CH_DT_BLK();

                    for (int64_t oc = 0; oc < ocb_eff; ++oc) {
                        // G * filter;
                        for (int64_t i = 0; i < TILE_IN_H(); ++i) {
                            for (int64_t j = 0; j < KERNEL_W(); ++j) {
                                float sum = 0.0f;
                                for (int64_t k = 0; k < KERNEL_H(); ++k) {
                                    sum += mat_g[i][k] * l_flt[oc * ic_per_gp * KERNEL_H() * KERNEL_W() + k * KERNEL_W() + j];
                                }
                                mat_T[i][j] = sum;
                            }
                        }
                        // (G * filter) * GT
                        for (int64_t i = 0; i < TILE_IN_H(); ++i) {
                            for (int64_t j = 0; j < TILE_IN_W(); ++j) {
                                float sum = 0.0f;
                                for (int64_t k = 0; k < KERNEL_W(); ++k) {
                                    sum += mat_T[i][k] * mat_g[j][k];
                                }
                                l_cvt_flt[(i * TILE_IN_W() + j) * padded_oc * icl2_eff + oc] = sum;
                            }
                        }
                    }
                    if (ocb_eff < CH_DT_BLK()) {
                        for (int64_t i = 0; i < TILE_IN_H(); ++i) {
                            for (int64_t j = 0; j < TILE_IN_W(); ++j) {
                                for (int64_t oc = ocb_eff; oc < CH_DT_BLK(); ++oc) {
                                    l_cvt_flt[(i * TILE_IN_W() + j) * padded_oc * icl2_eff + oc] = 0.0f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

bool conv2d_winograd_b6f3_fp32_sse_manager::is_supported()
{
    if (param_.is_pointwise()) {
        return false;
    }
    if (param_.channels / param_.group <= CH_RF_BLK()) {
        return false;
    }

    return param_.kernel_h == KERNEL_H() &&
           param_.kernel_w == KERNEL_W() &&
           param_.stride_h == STRIDE_H() &&
           param_.stride_w == STRIDE_W() &&
           param_.dilation_h == 1 &&
           param_.dilation_w == 1;
}

conv2d_fp32_executor *conv2d_winograd_b6f3_fp32_sse_manager::gen_executor()
{
    return new conv2d_winograd_b6f3_fp32_sse_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86