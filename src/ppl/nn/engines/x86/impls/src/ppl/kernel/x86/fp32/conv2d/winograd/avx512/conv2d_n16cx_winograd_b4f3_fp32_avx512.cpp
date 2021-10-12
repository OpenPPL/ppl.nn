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
#include <limits.h>
#include <string.h>

#include "ppl/kernel/x86/fp32/conv2d/winograd/avx512/conv2d_n16cx_winograd_b4f3_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/winograd/avx512/conv2d_n16cx_winograd_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/common/avx512_tools.h"
#include "ppl/common/sys.h"

#define ASSUME_L2_BYTES() (256 * 1024)
#define ASSUME_L2_WAYS()  4
#define ASSUME_L3_BYTES() (2048 * 1024)
#define L2_RATIO()        0.251
#define L3_RATIO()        0.501

#define TILE_KR_BLK() T14_TILES_RF()
#define TILE_IN_H()   6
#define TILE_IN_W()   6
#define TILE_OUT_H()  4
#define TILE_OUT_W()  4
#define KERNEL_H()    3
#define KERNEL_W()    3
#define STRIDE_H()    1
#define STRIDE_W()    1

#define IC_L2_BLK_MAX_L()   (16 * CH_DT_BLK())
#define IC_L2_BLK_MAX_S()   (8 * CH_DT_BLK())
#define OC_KR_BLK()         (T14_OC_RF() * CH_DT_BLK())
#define OC_L2_BLK_MAX()     (16 * OC_KR_BLK())
#define TILE_L2_BLK_MIN()   (1 * TILE_KR_BLK())
#define TILE_L2_BLK_MAX_S() (2 * TILE_KR_BLK())
#define TILE_L2_BLK_MAX_L() (8 * TILE_KR_BLK())

#define PARALLEL_OUTER() 0
#define PARALLEL_INNER() 1

#define PARALLEL_TILE_COEF() 0.1
#define PARALLEL_SEL_COEF()  256

#define TIMER_COUNT() 3
#define SRCTR_TIMER() 0
#define GEMM_TIMER()  1
#define DSTTR_TIMER() 2

namespace ppl { namespace kernel { namespace x86 {

bool conv2d_n16cx_winograd_b4f3_fp32_avx512_executor::init_profiler()
{
#ifdef PPL_X86_KERNEL_TIMING
    profiler_.init(TIMER_COUNT());
    return true;
#else
    return false;
#endif
}

void conv2d_n16cx_winograd_b4f3_fp32_avx512_executor::clear_profiler()
{
#ifdef PPL_X86_KERNEL_TIMING
    profiler_.clear();
#endif
}

std::string conv2d_n16cx_winograd_b4f3_fp32_avx512_executor::export_profiler()
{
#ifdef PPL_X86_KERNEL_TIMING
    static const char *timer_name[] = {
        "src_trans",
        "gemm",
        "dst_trans"};
    return profiler_.export_csv(timer_name, false);
#else
    return "";
#endif
}

static int64_t get_ic_l2_blk(
    const int64_t channels,
    const int64_t num_output)
{
    int64_t rst = IC_L2_BLK_MAX_L();
    if (channels <= num_output && channels <= IC_L2_BLK_MAX_L()) {
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
    int64_t rst = OC_L2_BLK_MAX();
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
    const int64_t num_threads = PPL_OMP_MAX_THREADS();
    const int64_t dst_h       = src_h + 2 * pad_h - KERNEL_H() + 1;
    const int64_t dst_w       = src_w + 2 * pad_w - KERNEL_W() + 1;
    const int64_t num_tiles_h = div_up(dst_h, TILE_OUT_H());
    const int64_t num_tiles_w = div_up(dst_w, TILE_OUT_W());
    const int64_t num_tiles_b = num_tiles_h * num_tiles_w;
    const int64_t num_tiles   = num_tiles_b * batch;

    int64_t tiles_l2_blk = TILE_L2_BLK_MAX_S();
    if (mode == PARALLEL_OUTER()) {
        float min_cost = FLT_MAX;
        for (int64_t tl2 = TILE_L2_BLK_MIN(); tl2 <= TILE_L2_BLK_MAX_S(); tl2 += TILE_KR_BLK()) {
            const int64_t num_tasks = div_up(div_up(num_tiles, tl2), num_threads);
            const float factor = PARALLEL_TILE_COEF() * (TILE_L2_BLK_MAX_S() - tl2) / TILE_L2_BLK_MAX_S();
            const float cost_estimate = num_tasks * tl2 * (1 + factor);
            if (cost_estimate < min_cost) {
                min_cost = cost_estimate;
                tiles_l2_blk = tl2;
            }
        }
    } else {
        tiles_l2_blk = TILE_L2_BLK_MAX_L();
    }

    tiles_l2_blk = round_up(min(tiles_l2_blk, num_tiles), TILE_KR_BLK());

    return tiles_l2_blk;
}

void conv2d_n16cx_winograd_b4f3_fp32_avx512_executor::init_preproc_param()
{
    kernel_schedule_param &sp   = schedule_param_;
    const conv2d_fp32_param &cp = *conv_param_;

    const int64_t num_thread = PPL_OMP_MAX_THREADS();

    sp.ic_per_gp = cp.channels / cp.group;
    sp.oc_per_gp = cp.num_output / cp.group;
    sp.padded_ic = round_up(sp.ic_per_gp, CH_DT_BLK());
    sp.padded_oc = round_up(sp.oc_per_gp, CH_DT_BLK());

    const int64_t batch = src_shape_->GetDim(0);
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);

    sp.num_tiles_h      = div_up(dst_h, TILE_OUT_H());
    sp.num_tiles_w      = div_up(dst_w, TILE_OUT_W());
    sp.num_tiles_b      = sp.num_tiles_h * sp.num_tiles_w;
    sp.num_tiles        = sp.num_tiles_b * batch;
    sp.ic_l2_blk        = get_ic_l2_blk(sp.ic_per_gp, sp.oc_per_gp);
    sp.override_only    = sp.ic_l2_blk >= sp.ic_per_gp;

    const float l3_cap_all_core = (ppl::common::GetCpuCacheL3() == 0 ? (ASSUME_L3_BYTES() * num_thread) : ppl::common::GetCpuCacheL3()) * L3_RATIO() / sizeof(float);

    if (sp.num_tiles > PARALLEL_SEL_COEF() * num_thread) {
        sp.parallel_mode = PARALLEL_OUTER();
    } else {
        sp.parallel_mode = PARALLEL_INNER();
    }

    sp.tiles_l2_blk = get_tiles_l2_blk(batch, src_shape_->GetDim(2), src_shape_->GetDim(3), cp.pad_h, cp.pad_w, src_shape_->GetDim(1), dst_shape_->GetDim(1), sp.parallel_mode);

    if (sp.parallel_mode == PARALLEL_OUTER()) {
        const int64_t tiles_all_threads = num_thread * sp.tiles_l2_blk;
        const int64_t oc_l2_cnt         = max<int64_t>(tiles_all_threads / sp.num_tiles, 1);

        sp.oc_l2_blk = round_up(max<int64_t>(sp.oc_per_gp / oc_l2_cnt, 1), OC_KR_BLK());
        
        sp.thread_tile_in_len   = round_up(CH_DT_BLK() * TILE_IN_H() * TILE_IN_W(), PPL_X86_CACHELINE_BYTES() / sizeof(float));
        sp.thread_matmul_in_len = round_up(CH_DT_BLK() * TILE_IN_H() * TILE_IN_W(), PPL_X86_CACHELINE_BYTES() / sizeof(float));

        sp.thread_src_trans_len = round_up(sp.ic_l2_blk * TILE_IN_H() * TILE_IN_W() * sp.tiles_l2_blk, PPL_X86_CACHELINE_BYTES() / sizeof(float));
        sp.thread_gemm_out_len  = round_up(sp.oc_l2_blk * TILE_IN_H() * TILE_IN_W() * sp.tiles_l2_blk, PPL_X86_CACHELINE_BYTES() / sizeof(float));
        if (sp.override_only) {
            sp.thread_gemm_out_len = round_up(OC_KR_BLK() * TILE_IN_H() * TILE_IN_W() * sp.tiles_l2_blk, PPL_X86_CACHELINE_BYTES() / sizeof(float));
        }
        sp.thread_matmul_out_len    = round_up(TILE_IN_H() * TILE_IN_W() * CH_DT_BLK(), PPL_X86_CACHELINE_BYTES() / sizeof(float));
        sp.thread_postprocess_len   = 2 * sp.thread_matmul_out_len;
        sp.thread_src_dst_trans_len = max<int64_t>(sp.thread_tile_in_len + sp.thread_matmul_in_len + sp.thread_src_trans_len, sp.thread_postprocess_len);

        sp.thread_workspace_len = sp.thread_src_dst_trans_len + sp.thread_gemm_out_len;
        sp.gemm_out_len         = sp.thread_gemm_out_len * num_thread;
    } else {
        sp.oc_l2_blk = get_oc_l2_blk(sp.ic_per_gp, sp.oc_per_gp);

        sp.thread_tile_in_len   = round_up(CH_DT_BLK() * TILE_IN_H() * TILE_IN_W(), PPL_X86_CACHELINE_BYTES() / sizeof(float));
        sp.thread_matmul_in_len = round_up(CH_DT_BLK() * TILE_IN_H() * TILE_IN_W(), PPL_X86_CACHELINE_BYTES() / sizeof(float));

        sp.src_trans_len        = round_up(sp.ic_l2_blk * TILE_IN_H() * TILE_IN_W() * sp.tiles_l2_blk, PPL_X86_CACHELINE_BYTES() / sizeof(float));
        sp.gemm_out_len         = round_up(sp.padded_oc * TILE_IN_H() * TILE_IN_W() * sp.tiles_l2_blk, PPL_X86_CACHELINE_BYTES() / sizeof(float));
        if (sp.override_only) {
            sp.gemm_out_len = round_up(sp.oc_l2_blk * TILE_IN_H() * TILE_IN_W() * sp.tiles_l2_blk, PPL_X86_CACHELINE_BYTES() / sizeof(float));
        }

        sp.thread_matmul_out_len    = round_up(TILE_IN_H() * TILE_IN_W() * CH_DT_BLK(), PPL_X86_CACHELINE_BYTES() / sizeof(float));
        sp.thread_postprocess_len   = 2 * sp.thread_matmul_out_len;
        sp.thread_src_dst_trans_len = max<int64_t>(sp.thread_tile_in_len + sp.thread_matmul_in_len, sp.thread_postprocess_len);
        sp.thread_workspace_len     = sp.thread_src_dst_trans_len;
    }

    sp.use_nt_store = 0;
    const int64_t dst_element_num = batch * cp.group * sp.padded_oc * dst_shape_->GetDim(2) * dst_shape_->GetDim(3);
    if (dst_element_num + sp.gemm_out_len > l3_cap_all_core * 2) {
        sp.use_nt_store = 1;
    }
}

uint64_t conv2d_n16cx_winograd_b4f3_fp32_avx512_executor::cal_temp_buffer_size()
{
    const kernel_schedule_param &sp = schedule_param_;
    const int64_t num_thread        = PPL_OMP_MAX_THREADS();

    if (sp.parallel_mode == PARALLEL_OUTER()) {
        return sp.thread_workspace_len * num_thread * sizeof(float);
    } else { // PARALLEL_INNER
        return sp.src_trans_len * sizeof(float) +
               sp.gemm_out_len * sizeof(float) +
               sp.thread_workspace_len * num_thread * sizeof(float);
    }
}

ppl::common::RetCode conv2d_n16cx_winograd_b4f3_fp32_avx512_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_shape_)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    init_preproc_param();

    return ppl::common::RC_SUCCESS;
}

static inline void winograd_b4f3_preprocess_fp32_avx512(
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
    const float *tile_src;
    int64_t tile_src_h_stride;
    if (ih >= 0 && ih + TILE_IN_H() <= src_h && iw >= 0 && iw + TILE_IN_W() <= src_w) {
        tile_src = base_src + ih * src_w * CH_DT_BLK() + iw * CH_DT_BLK();
        tile_src_h_stride = src_w * CH_DT_BLK();
    } else {
        tile_src = tile_buffer;
        tile_src_h_stride = tile_h_stride;
        int64_t tl_pad   = max<int64_t>(0 - iw, 0);
        int64_t tw_start = max<int64_t>(iw, 0);
        int64_t tw_len = max<int64_t>(min<int64_t>(src_w, iw + TILE_IN_W()) - tw_start, 0);
        int64_t tr_pad = max<int64_t>(iw + TILE_IN_W() - src_w, 0);
        float *l_tile_buffer = tile_buffer;
        for (int64_t h = ih; h < ih + TILE_IN_H(); ++h) {
            if (h < 0 || h >= src_h) {
                memset32_avx(l_tile_buffer, 0, tile_h_stride);
            } else {
                int64_t w = 0;
                memset32_avx(l_tile_buffer + w * CH_DT_BLK(), 0, tl_pad * CH_DT_BLK());
                w += tl_pad;
                memcpy32_avx(l_tile_buffer + w * CH_DT_BLK(), base_src + (h * src_w + tw_start) * CH_DT_BLK(), tw_len * CH_DT_BLK());
                w += tw_len;
                memset32_avx(l_tile_buffer + w * CH_DT_BLK(), 0, tr_pad * CH_DT_BLK());
                w += tr_pad;
            }
            l_tile_buffer += tile_h_stride;
        }
    }

    __m512 zmm12, zmm13, zmm14;
    zmm12 = _mm512_set1_ps(2.0f);
    zmm13 = _mm512_set1_ps(4.0f);
    zmm14 = _mm512_set1_ps(5.0f);
    for (int64_t th = 0; th < TILE_IN_H(); ++th) {
        const float *l_tile = tile_src + th * tile_src_h_stride;
        float *l_temp = matmul_buffer + th * tile_h_stride;
        
        __m512 zmm0, zmm2, zmm4, zmm6;
        __m512 zmm8, zmm10, zmm15;

        zmm0  = _mm512_loadu_ps(l_tile + 0 * CH_DT_BLK()) * zmm13;

        zmm15 = _mm512_loadu_ps(l_tile + 1 * CH_DT_BLK());
        zmm4  = zmm15 * zmm13;
        zmm2  = -zmm4;
        zmm8  = zmm15 * zmm12;
        zmm6  = -zmm8;
        zmm10 = zmm4;

        zmm15 = _mm512_loadu_ps(l_tile + 2 * CH_DT_BLK());
        zmm0 -= zmm15 * zmm14;
        zmm2 -= zmm15 * zmm13;
        zmm4 -= zmm15 * zmm13;
        zmm6 -= zmm15;
        zmm8 -= zmm15;

        zmm15 = _mm512_loadu_ps(l_tile + 3 * CH_DT_BLK());
        zmm2 += zmm15;
        zmm4 -= zmm15;
        zmm6 += zmm15 * zmm12;
        zmm8 -= zmm15 * zmm12;
        zmm10 -= zmm15 * zmm14;

        zmm15 = _mm512_loadu_ps(l_tile + 4 * CH_DT_BLK());
        zmm0 += zmm15;
        zmm2 += zmm15;
        zmm4 += zmm15;
        zmm6 += zmm15;
        zmm8 += zmm15;

        zmm15 = _mm512_loadu_ps(l_tile + 5 * CH_DT_BLK());
        zmm10 += zmm15;

        _mm512_storeu_ps(l_temp + 0 * CH_DT_BLK(), zmm0);
        _mm512_storeu_ps(l_temp + 1 * CH_DT_BLK(), zmm2);
        _mm512_storeu_ps(l_temp + 2 * CH_DT_BLK(), zmm4);
        _mm512_storeu_ps(l_temp + 3 * CH_DT_BLK(), zmm6);
        _mm512_storeu_ps(l_temp + 4 * CH_DT_BLK(), zmm8);
        _mm512_storeu_ps(l_temp + 5 * CH_DT_BLK(), zmm10);
    }

    for (int64_t tw = 0; tw < TILE_IN_W(); ++tw) {
        const float *l_temp = matmul_buffer + tw * CH_DT_BLK();
        float *l_dst        = src_trans + tw * src_trans_ti_stride;

        __m512 zmm0, zmm2, zmm4, zmm6;
        __m512 zmm8, zmm10, zmm15;

        zmm0 = _mm512_loadu_ps(l_temp + 0 * tile_h_stride) * zmm13;

        zmm15 = _mm512_loadu_ps(l_temp + 1 * tile_h_stride);
        zmm4  = zmm15 * zmm13;
        zmm2  = -zmm4;
        zmm8  = zmm15 * zmm12;
        zmm6  = -zmm8;
        zmm10 = zmm4;

        zmm15 = _mm512_loadu_ps(l_temp + 2 * tile_h_stride);
        zmm0 -= zmm15 * zmm14;
        zmm2 -= zmm15 * zmm13;
        zmm4 -= zmm15 * zmm13;
        zmm6 -= zmm15;
        zmm8 -= zmm15;

        zmm15 = _mm512_loadu_ps(l_temp + 3 * tile_h_stride);
        zmm2 += zmm15;
        zmm4 -= zmm15;
        zmm6 += zmm15 * zmm12;
        zmm8 -= zmm15 * zmm12;
        zmm10 -= zmm15 * zmm14;

        zmm15 = _mm512_loadu_ps(l_temp + 4 * tile_h_stride);
        zmm0 += zmm15;
        zmm2 += zmm15;
        zmm4 += zmm15;
        zmm6 += zmm15;
        zmm8 += zmm15;

        zmm15 = _mm512_loadu_ps(l_temp + 5 * tile_h_stride);
        zmm10 += zmm15;

        _mm512_storeu_ps(l_dst + 0 * TILE_IN_W() * src_trans_ti_stride, zmm0);
        _mm512_storeu_ps(l_dst + 1 * TILE_IN_W() * src_trans_ti_stride, zmm2);
        _mm512_storeu_ps(l_dst + 2 * TILE_IN_W() * src_trans_ti_stride, zmm4);
        _mm512_storeu_ps(l_dst + 3 * TILE_IN_W() * src_trans_ti_stride, zmm6);
        _mm512_storeu_ps(l_dst + 4 * TILE_IN_W() * src_trans_ti_stride, zmm8);
        _mm512_storeu_ps(l_dst + 5 * TILE_IN_W() * src_trans_ti_stride, zmm10);
    }
}

template <bool nt_store>
static inline void winograd_b4f3_dst_trans_fp32_avx512(
    const float *dst_trans,
    const float *sum_src,
    const float *bias,
    const int64_t dst_trans_ti_stride,
    const int64_t sum_src_h_stride,
    const int64_t dst_h_stride,
    const uint64_t fuse_flag,
    float *matmul_buffer,
    float *dst)
{
    const int64_t matmul_h_stride = TILE_OUT_W() * CH_DT_BLK();

    __m512 zmm13, zmm14, zmm15;
    zmm13 = _mm512_set1_ps(2.0f);
    zmm14 = _mm512_set1_ps(4.0f);
    zmm15 = _mm512_set1_ps(8.0f);
    for (int64_t th = 0; th < TILE_IN_H(); ++th) {
        const float *l_dst_trans = dst_trans + th * TILE_IN_W() * dst_trans_ti_stride;
        float *l_temp = matmul_buffer + th * matmul_h_stride;
        __m512 zmm0, zmm2, zmm4, zmm6;
        __m512 zmm8, zmm10;

        zmm8  = _mm512_loadu_ps(l_dst_trans + 0 * dst_trans_ti_stride);
        zmm10 = _mm512_loadu_ps(l_dst_trans + 1 * dst_trans_ti_stride);
        zmm0  = zmm8 + zmm10;
        zmm2  = zmm10;
        zmm4  = zmm10;
        zmm6  = zmm10;

        zmm8 = _mm512_loadu_ps(l_dst_trans + 2 * dst_trans_ti_stride);
        zmm0 += zmm8;
        zmm2 -= zmm8;
        zmm4 += zmm8;
        zmm6 -= zmm8;

        zmm8 = _mm512_loadu_ps(l_dst_trans + 3 * dst_trans_ti_stride);
        zmm0 += zmm8;
        zmm2 += zmm8 * zmm13;
        zmm4 += zmm8 * zmm14;
        zmm6 += zmm8 * zmm15;

        zmm8 = _mm512_loadu_ps(l_dst_trans + 4 * dst_trans_ti_stride);
        zmm0 += zmm8;
        zmm2 -= zmm8 * zmm13;
        zmm4 += zmm8 * zmm14;
        zmm6 -= zmm8 * zmm15;

        zmm8 = _mm512_loadu_ps(l_dst_trans + 5 * dst_trans_ti_stride);
        zmm6 += zmm8;

        _mm512_storeu_ps(l_temp + 0 * CH_DT_BLK(), zmm0);
        _mm512_storeu_ps(l_temp + 1 * CH_DT_BLK(), zmm2);
        _mm512_storeu_ps(l_temp + 2 * CH_DT_BLK(), zmm4);
        _mm512_storeu_ps(l_temp + 3 * CH_DT_BLK(), zmm6);
    }
    for (int64_t tw = 0; tw < TILE_OUT_W(); ++tw) {
        float *l_dst           = dst + tw * CH_DT_BLK();
        const float *l_sum_src = sum_src + tw * CH_DT_BLK();
        float *l_temp = matmul_buffer + tw * CH_DT_BLK();

        __m512 zmm0, zmm2, zmm4, zmm6;
        __m512 zmm8, zmm10, zmm11;

        zmm8  = _mm512_loadu_ps(l_temp + 0 * matmul_h_stride);
        zmm10 = _mm512_loadu_ps(l_temp + 1 * matmul_h_stride);
        zmm0  = zmm8 + zmm10;
        zmm2  = zmm10;
        zmm4  = zmm10;
        zmm6  = zmm10;

        zmm8 = _mm512_loadu_ps(l_temp + 2 * matmul_h_stride);
        zmm0 += zmm8;
        zmm2 -= zmm8;
        zmm4 += zmm8;
        zmm6 -= zmm8;

        zmm8 = _mm512_loadu_ps(l_temp + 3 * matmul_h_stride);
        zmm0 += zmm8;
        zmm2 += zmm8 * zmm13;
        zmm4 += zmm8 * zmm14;
        zmm6 += zmm8 * zmm15;

        zmm8 = _mm512_loadu_ps(l_temp + 4 * matmul_h_stride);
        zmm0 += zmm8;
        zmm2 -= zmm8 * zmm13;
        zmm4 += zmm8 * zmm14;
        zmm6 -= zmm8 * zmm15;

        zmm8 = _mm512_loadu_ps(l_temp + 5 * matmul_h_stride);
        zmm6 += zmm8;

        zmm8 = _mm512_loadu_ps(bias);

        zmm0 += zmm8;
        zmm2 += zmm8;
        zmm4 += zmm8;
        zmm6 += zmm8;

        if (fuse_flag & conv_fuse_flag::sum) {
            zmm0 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 0 * sum_src_h_stride), zmm0);
            zmm2 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 1 * sum_src_h_stride), zmm2);
            zmm4 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 2 * sum_src_h_stride), zmm4);
            zmm6 = _mm512_add_ps(_mm512_loadu_ps(l_sum_src + 3 * sum_src_h_stride), zmm6);
        }

        if (fuse_flag & (conv_fuse_flag::relu | conv_fuse_flag::relu6)) {
            zmm10 = _mm512_setzero_ps();
            zmm0  = _mm512_max_ps(zmm10, zmm0);
            zmm2  = _mm512_max_ps(zmm10, zmm2);
            zmm4  = _mm512_max_ps(zmm10, zmm4);
            zmm6  = _mm512_max_ps(zmm10, zmm6);
        }

        if (fuse_flag & conv_fuse_flag::relu6) {
            zmm11 = _mm512_set1_ps(6.0f);
            zmm0  = _mm512_min_ps(zmm11, zmm0);
            zmm2  = _mm512_min_ps(zmm11, zmm2);
            zmm4  = _mm512_min_ps(zmm11, zmm4);
            zmm6  = _mm512_min_ps(zmm11, zmm6);
        }

        if (nt_store) {
            _mm512_stream_ps(l_dst + 0 * dst_h_stride, zmm0);
            _mm512_stream_ps(l_dst + 1 * dst_h_stride, zmm2);
            _mm512_stream_ps(l_dst + 2 * dst_h_stride, zmm4);
            _mm512_stream_ps(l_dst + 3 * dst_h_stride, zmm6);
        } else {
            _mm512_storeu_ps(l_dst + 0 * dst_h_stride, zmm0);
            _mm512_storeu_ps(l_dst + 1 * dst_h_stride, zmm2);
            _mm512_storeu_ps(l_dst + 2 * dst_h_stride, zmm4);
            _mm512_storeu_ps(l_dst + 3 * dst_h_stride, zmm6);
        }
    }
}

template <bool nt_store>
void winograd_b4f3_store_dst_fp32_avx512(
    const float *src,
    const float *sum_src,
    const int64_t oh_len,
    const int64_t ow_len,
    const int64_t dst_h_stride,
    const uint64_t fuse_flag,
    float *dst)
{
    __m512 vmin, vmax;
    if (fuse_flag & (conv_fuse_flag::relu | conv_fuse_flag::relu6)) {
        vmin = _mm512_setzero_ps();
    } else {
        vmin = _mm512_set1_ps(-FLT_MAX);
    }

    if (fuse_flag & conv_fuse_flag::relu6) {
        vmax = _mm512_set1_ps(6.0f);
    } else {
        vmax = _mm512_set1_ps(FLT_MAX);
    }

    if (fuse_flag & conv_fuse_flag::sum) {
        for (int64_t oh = 0; oh < oh_len; ++oh) {
            const float *l_src = src + oh * TILE_OUT_W() * CH_DT_BLK();
            const float *l_sum_src = sum_src + oh * dst_h_stride;
            float *l_dst = dst + oh * dst_h_stride;
            for (int64_t ow = 0; ow < ow_len; ++ow) {
                __m512 vres = _mm512_add_ps(_mm512_loadu_ps(l_sum_src), _mm512_loadu_ps(l_src));
                vres        = _mm512_min_ps(_mm512_max_ps(vres, vmin), vmax);
                if (nt_store) {
                    _mm512_stream_ps(l_dst, vres);
                } else {
                    _mm512_storeu_ps(l_dst, vres);
                }
                l_dst += CH_DT_BLK();
                l_sum_src += CH_DT_BLK();
                l_src += CH_DT_BLK();
            }
        }
    } else {
        for (int64_t oh = 0; oh < oh_len; ++oh) {
            const float *l_src = src + oh * TILE_OUT_W() * CH_DT_BLK();
            float *l_dst = dst + oh * dst_h_stride;
            for (int64_t ow = 0; ow < ow_len; ++ow) {
                __m512 vres = _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(l_src), vmin), vmax);
                if (nt_store) {
                    _mm512_stream_ps(l_dst, vres);
                } else {
                    _mm512_storeu_ps(l_dst, vres);
                }
                l_dst += CH_DT_BLK();
                l_src += CH_DT_BLK();
            }
        }
    }
}

ppl::common::RetCode conv2d_n16cx_winograd_b4f3_fp32_avx512_executor::execute()
{
    if (!conv_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || ((conv_param_->fuse_flag & conv_fuse_flag::sum) && !sum_src_) || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const conv2d_fp32_param &cp     = *conv_param_;
    const kernel_schedule_param &sp = schedule_param_;

    const int64_t src_h = src_shape_->GetDim(2);
    const int64_t src_w = src_shape_->GetDim(3);
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);

    const int64_t padded_src_c = round_up(src_shape_->GetDim(1), CH_DT_BLK());
    const int64_t padded_dst_c = round_up(dst_shape_->GetDim(1), CH_DT_BLK());

    const int64_t src_g_stride     = sp.padded_ic * src_h * src_w;
    const int64_t src_b_stride     = padded_src_c * src_h * src_w;
    const int64_t dst_g_stride     = sp.padded_oc * dst_h * dst_w;
    const int64_t dst_b_stride     = padded_dst_c * dst_h * dst_w;
    const int64_t bias_g_stride    = sp.padded_oc;
    const int64_t cvt_flt_g_stride = sp.padded_ic * sp.padded_oc * TILE_IN_H() * TILE_IN_W();
    int64_t sum_src_b_stride       = 0;
    if (conv_param_->fuse_flag & conv_fuse_flag::sum) {
        sum_src_b_stride = int64_t(round_up(sum_src_shape_->GetDim(1), CH_DT_BLK())) * dst_h * dst_w;
    }

    // cvt_flt:   [group, ic_l2_cnt, 6h, 6w, oc/16o, icl2_eff, 16o]
    // src_trans: [6h, 6w, tile_l2_blk/6t, icl2_eff/16o, tile_kr_eff, 16i]
    // gemm_out:  [6h, 6w, (oc_l2_blk/16, )tile_l2_eff, 16o]
    if (sp.parallel_mode == PARALLEL_OUTER()) {
        float *base_workspace = (float *)temp_buffer_;
#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
#endif
        for (int64_t g = 0; g < cp.group; ++g) {
            for (int64_t ocl2 = 0; ocl2 < sp.oc_per_gp; ocl2 += sp.oc_l2_blk) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                PRAGMA_OMP_PARALLEL_FOR()
#endif
                for (int64_t tl2 = 0; tl2 < sp.num_tiles; tl2 += sp.tiles_l2_blk) {
                    int64_t kernel_param[KERNEL_PARAM_LEN()];
                    const int64_t ocl2_eff = min<int64_t>(sp.oc_l2_blk, sp.padded_oc - ocl2);
                    const int64_t tl2_eff = min<int64_t>(sp.tiles_l2_blk, (sp.num_tiles - tl2));
                    const int64_t t_body = round(tl2_eff, TILE_KR_BLK());
                    const int64_t t_tail = tl2_eff - t_body;

                    float *thread_workspace = base_workspace + PPL_OMP_THREAD_ID() * sp.thread_workspace_len;
                    float *tile_in_buf      = thread_workspace;
                    float *matmul_in_buf    = tile_in_buf + sp.thread_tile_in_len;
                    float *src_trans        = matmul_in_buf + sp.thread_matmul_in_len;
                    float *postprocess_buf  = thread_workspace;
                    float *gemm_out_buf     = thread_workspace + sp.thread_src_dst_trans_len;

                    for (int64_t icl2 = 0; icl2 < sp.ic_per_gp; icl2 += sp.ic_l2_blk) {
#ifdef PPL_X86_KERNEL_TIMING
                        profiler_.tic(SRCTR_TIMER());
#endif
                        const int64_t icl2_eff        = min<int64_t>(sp.ic_l2_blk, sp.ic_per_gp - icl2);
                        const int64_t icl2_eff_padded = round_up(icl2_eff, CH_DT_BLK());
                        const int64_t is_first_ic = icl2 == 0;
                        const int64_t is_last_ic = icl2 + sp.ic_l2_blk >= sp.ic_per_gp;
                        kernel_param[CHANNELS_IDX()] = icl2_eff;
                        kernel_param[LOAD_DST_IDX()] = !is_first_ic;
                        kernel_param[FLT_OCB_STRIDE_IDX()] = icl2_eff * CH_DT_BLK();
                        kernel_param[DST_OCB_STRIDE_IDX()] = tl2_eff * CH_DT_BLK();
                        for (int64_t tk = tl2; tk < tl2 + tl2_eff; tk += TILE_KR_BLK()) {
                            const int64_t tk_eff = min<int64_t>(tl2 + tl2_eff - tk, TILE_KR_BLK());
                            for (int64_t icb = icl2; icb < icl2 + icl2_eff_padded; icb += CH_DT_BLK()) {
                                for (int64_t t = 0; t < tk_eff; ++t) {
                                    tile_corr tc = cal_tile_corr(sp, tk + t);
                                    const int64_t b  = tc.b;
                                    const int64_t oh = tc.th * TILE_OUT_H();
                                    const int64_t ow = tc.tw * TILE_OUT_W();
                                    const int64_t ih = oh * STRIDE_H() - cp.pad_h;
                                    const int64_t iw = ow * STRIDE_W() - cp.pad_w;

                                    float *l_src_trans = src_trans
                                        + (tk - tl2) * icl2_eff_padded
                                        + (icb - icl2) * tk_eff
                                        + t * CH_DT_BLK();
                                    const float *base_src = src_
                                        + b * src_b_stride
                                        + g * src_g_stride
                                        + icb * src_h * src_w;

                                    winograd_b4f3_preprocess_fp32_avx512(
                                        base_src, ih, iw, src_h, src_w,
                                        tl2_eff * icl2_eff_padded,
                                        tile_in_buf,
                                        matmul_in_buf,
                                        l_src_trans);
                                }
                            }
                        }

#ifdef PPL_X86_KERNEL_TIMING
                        profiler_.toc(SRCTR_TIMER());
#endif

                        for (int64_t ock = ocl2; ock < ocl2 + ocl2_eff; ock += OC_KR_BLK()) {
#ifdef PPL_X86_KERNEL_TIMING
                            profiler_.tic(GEMM_TIMER());
#endif
                            const int64_t ock_eff = min<int64_t>(OC_KR_BLK(), sp.padded_oc - ock);
                            const int64_t ock_sel = div_up(ock_eff, CH_DT_BLK()) - 1;
                            for (int64_t ti = 0; ti < TILE_IN_H() * TILE_IN_W(); ++ti) {
                                float *l_src_trans = src_trans
                                                + ti * tl2_eff * icl2_eff_padded;
                                const float *l_cvt_flt = cvt_filter_
                                                + g * cvt_flt_g_stride
                                                + icl2 * TILE_IN_H() * TILE_IN_W() * sp.padded_oc
                                                + ti * sp.padded_oc * icl2_eff
                                                + ock * icl2_eff;
                                float *l_gemm_out;
                                if (sp.override_only) {
                                    l_gemm_out = gemm_out_buf
                                                + ti * ock_eff * tl2_eff;
                                } else {
                                    l_gemm_out = gemm_out_buf
                                                + ti * ocl2_eff * tl2_eff
                                                + (ock - ocl2) * tl2_eff;
                                }
                                
                                PICK_PARAM(const float*, kernel_param, SRC_IDX()) = l_src_trans;
                                PICK_PARAM(const float*, kernel_param, FLT_IDX()) = l_cvt_flt;
                                PICK_PARAM(float *, kernel_param, DST_IDX()) = l_gemm_out;
                                if (t_body) {
                                    kernel_param[TILES_IDX()] = t_body;
                                    kernel_param[SRC_TKB_STRIDE_IDX()] = TILE_KR_BLK() * icl2_eff_padded;
                                    switch (ock_sel) {
                                        case 0: conv2d_n16cx_winograd_kernel_fp32_avx512_o16_table[TILE_KR_BLK() - 1](kernel_param); break;
                                        case 1: conv2d_n16cx_winograd_kernel_fp32_avx512_o32_table[TILE_KR_BLK() - 1](kernel_param); break;
                                        // case 2: conv2d_n16cx_winograd_kernel_fp32_avx512_o48_table[TILE_KR_BLK() - 1](kernel_param); break;
                                        // case 3: conv2d_n16cx_winograd_kernel_fp32_avx512_o64_table[TILE_KR_BLK() - 1](kernel_param); break;
                                    }
                                    PICK_PARAM(const float*, kernel_param, SRC_IDX()) += t_body * icl2_eff_padded;
                                    PICK_PARAM(float *, kernel_param, DST_IDX()) += t_body * CH_DT_BLK();
                                }
                                if (t_tail) {
                                    kernel_param[TILES_IDX()] = t_tail;
                                    kernel_param[SRC_TKB_STRIDE_IDX()] = t_tail * icl2_eff_padded;
                                    switch (ock_sel) {
                                        case 0: conv2d_n16cx_winograd_kernel_fp32_avx512_o16_table[t_tail - 1](kernel_param); break;
                                        case 1: conv2d_n16cx_winograd_kernel_fp32_avx512_o32_table[t_tail - 1](kernel_param); break;
                                        // case 2: conv2d_n16cx_winograd_kernel_fp32_avx512_o48_table[t_tail - 1](kernel_param); break;
                                        // case 3: conv2d_n16cx_winograd_kernel_fp32_avx512_o64_table[t_tail - 1](kernel_param); break;
                                    }
                                }
                            }
#ifdef PPL_X86_KERNEL_TIMING
                            profiler_.toc(GEMM_TIMER());
#endif
#ifdef PPL_X86_KERNEL_TIMING
                            profiler_.tic(DSTTR_TIMER());
#endif

                            if (is_last_ic) {
                                for (int64_t ocb = ock; ocb < ock + ock_eff; ocb += CH_DT_BLK()) {
                                    for (int64_t tk = tl2; tk < tl2 + tl2_eff; tk += TILE_KR_BLK()) {
                                        const int64_t tk_eff = min<int64_t>(tl2 + tl2_eff - tk, TILE_KR_BLK());
                                        for (int64_t t = 0; t < tk_eff; ++t) {
                                            tile_corr tc     = cal_tile_corr(sp, tk + t);
                                            const int64_t b  = tc.b;
                                            const int64_t oh = tc.th * TILE_OUT_H();
                                            const int64_t ow = tc.tw * TILE_OUT_W();
                                            const int64_t oh_len = min<int64_t>(dst_h - oh, TILE_OUT_H());
                                            const int64_t ow_len = min<int64_t>(dst_w - ow, TILE_OUT_W());

                                            float *l_dst = dst_
                                                        + b * dst_b_stride
                                                        + g * dst_g_stride
                                                        + ocb * (dst_h * dst_w)
                                                        + (oh * dst_w + ow) * CH_DT_BLK();
                                            const float *l_sum_src = sum_src_
                                                        + b * sum_src_b_stride
                                                        + g * dst_g_stride
                                                        + ocb * (dst_h * dst_w)
                                                        + (oh * dst_w + ow) * CH_DT_BLK();
                                            float *l_gemm_out = gemm_out_buf
                                                        + (ocb - ocl2) * tl2_eff
                                                        + (tk - tl2 + t) * CH_DT_BLK();

                                            int64_t gemm_out_ti_stride = tl2_eff * ocl2_eff;
                                            if (sp.override_only) {
                                                l_gemm_out         = gemm_out_buf + (ocb - ock) * tl2_eff + (tk - tl2 + t) * CH_DT_BLK();
                                                gemm_out_ti_stride = tl2_eff * ock_eff;
                                            }

                                            if (oh_len == TILE_OUT_H() && ow_len == TILE_OUT_W()) {
                                                if (sp.use_nt_store) {
                                                    winograd_b4f3_dst_trans_fp32_avx512<true>(
                                                        l_gemm_out, l_sum_src,
                                                        cvt_bias_ + g * bias_g_stride + ocb,
                                                        gemm_out_ti_stride, dst_w * CH_DT_BLK(),
                                                        dst_w * CH_DT_BLK(), cp.fuse_flag,
                                                        postprocess_buf, l_dst);
                                                } else {
                                                    winograd_b4f3_dst_trans_fp32_avx512<false>(
                                                        l_gemm_out, l_sum_src,
                                                        cvt_bias_ + g * bias_g_stride + ocb,
                                                        gemm_out_ti_stride, dst_w * CH_DT_BLK(),
                                                        dst_w * CH_DT_BLK(), cp.fuse_flag,
                                                        postprocess_buf, l_dst);
                                                }
                                            } else {
                                                float *dst_buf = postprocess_buf + sp.thread_matmul_out_len;
                                                winograd_b4f3_dst_trans_fp32_avx512<false>(
                                                    l_gemm_out, l_sum_src,
                                                    cvt_bias_ + g * bias_g_stride + ocb,
                                                    gemm_out_ti_stride, dst_w * CH_DT_BLK(),
                                                    TILE_OUT_W() * CH_DT_BLK(), conv_fuse_flag::none,
                                                    postprocess_buf, dst_buf);
                                                if (sp.use_nt_store) {
                                                    winograd_b4f3_store_dst_fp32_avx512<true>(
                                                        dst_buf, l_sum_src,
                                                        oh_len,  ow_len,
                                                        dst_w * CH_DT_BLK(),
                                                        cp.fuse_flag, l_dst);
                                                } else {
                                                    winograd_b4f3_store_dst_fp32_avx512<false>(
                                                        dst_buf, l_sum_src,
                                                        oh_len, ow_len,
                                                        dst_w * CH_DT_BLK(),
                                                        cp.fuse_flag, l_dst);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
#ifdef PPL_X86_KERNEL_TIMING
                            profiler_.toc(DSTTR_TIMER());
#endif
                        }
                    }
                }
            }
        }
    } else { // PARALLEL_INNER
        PRAGMA_OMP_PARALLEL()
        {
        int64_t kernel_param[KERNEL_PARAM_LEN()];
        for (int64_t g = 0; g < cp.group; ++g) {
            for (int64_t tl2 = 0; tl2 < sp.num_tiles; tl2 += sp.tiles_l2_blk) {
                const int64_t tl2_eff = min<int64_t>(sp.tiles_l2_blk, (sp.num_tiles - tl2));
                const int64_t t_body = round(tl2_eff, TILE_KR_BLK());
                const int64_t t_tail = tl2_eff - t_body;

                float *src_trans      = (float *)temp_buffer_;
                float *gemm_out_buf   = src_trans + sp.src_trans_len;
                float *base_workspace = gemm_out_buf + sp.gemm_out_len;

                for (int64_t icl2 = 0; icl2 < sp.ic_per_gp; icl2 += sp.ic_l2_blk) {
                    const int64_t icl2_eff        = min<int64_t>(sp.ic_l2_blk, sp.ic_per_gp - icl2);
                    const int64_t icl2_eff_padded = round_up(icl2_eff, CH_DT_BLK());
                    const int64_t is_first_ic = icl2 == 0;
                    const int64_t is_last_ic = icl2 + sp.ic_l2_blk >= sp.ic_per_gp;
                    kernel_param[CHANNELS_IDX()] = icl2_eff;
                    kernel_param[LOAD_DST_IDX()] = !is_first_ic;
                    kernel_param[FLT_OCB_STRIDE_IDX()] = icl2_eff * CH_DT_BLK();
                    kernel_param[DST_OCB_STRIDE_IDX()] = tl2_eff * CH_DT_BLK();
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

                            tile_corr tc = cal_tile_corr(sp, tk);
                            const int64_t b  = tc.b;
                            const int64_t oh = tc.th * TILE_OUT_H();
                            const int64_t ow = tc.tw * TILE_OUT_W();
                            const int64_t ih = oh * STRIDE_H() - cp.pad_h;
                            const int64_t iw = ow * STRIDE_W() - cp.pad_w;
                            const int64_t t  = tk % TILE_KR_BLK();

                            const int64_t tk_eff = min<int64_t>(tl2 + tl2_eff - (tk - t), TILE_KR_BLK());
                            float *l_src_trans = src_trans
                                + (tk - tl2 - t) * icl2_eff_padded
                                + (icb - icl2) * tk_eff
                                + t * CH_DT_BLK();
                            const float *base_src  = src_
                                + b * src_b_stride
                                + g * src_g_stride
                                + icb * src_h * src_w;

                            winograd_b4f3_preprocess_fp32_avx512(
                                base_src, ih, iw, src_h, src_w,
                                tl2_eff * icl2_eff_padded,
                                tile_in_buf,
                                matmul_in_buf,
                                l_src_trans);
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
                            for (int64_t ock = ocl2; ock < ocl2 + ocl2_eff; ock += OC_KR_BLK()) {
#ifdef PPL_X86_KERNEL_TIMING
                                profiler_.tic(GEMM_TIMER());
#endif
                                const int64_t ock_eff = min<int64_t>(OC_KR_BLK(), sp.padded_oc - ock);
                                const int64_t ock_sel = div_up(ock_eff, CH_DT_BLK()) - 1;
                                float *l_src_trans = src_trans
                                                + ti * tl2_eff * icl2_eff_padded;
                                const float *l_cvt_flt = cvt_filter_
                                                + g * cvt_flt_g_stride
                                                + icl2 * TILE_IN_H() * TILE_IN_W() * sp.padded_oc
                                                + ti * sp.padded_oc * icl2_eff
                                                + ock * icl2_eff;

                                float *l_gemm_out;
                                if (sp.override_only) {
                                    l_gemm_out = gemm_out_buf
                                                + ti * ocl2_eff * tl2_eff
                                                + (ock - ocl2) * tl2_eff;
                                } else {
                                    l_gemm_out = gemm_out_buf
                                                + ti * sp.padded_oc * tl2_eff
                                                + ock * tl2_eff;
                                }
                                PICK_PARAM(const float*, kernel_param, SRC_IDX()) = l_src_trans;
                                PICK_PARAM(const float*, kernel_param, FLT_IDX()) = l_cvt_flt;
                                PICK_PARAM(float *, kernel_param, DST_IDX()) = l_gemm_out;
                                if (t_body) {
                                    kernel_param[TILES_IDX()] = t_body;
                                    kernel_param[SRC_TKB_STRIDE_IDX()] = TILE_KR_BLK() * icl2_eff_padded;
                                    switch (ock_sel) {
                                        case 0: conv2d_n16cx_winograd_kernel_fp32_avx512_o16_table[TILE_KR_BLK() - 1](kernel_param); break;
                                        case 1: conv2d_n16cx_winograd_kernel_fp32_avx512_o32_table[TILE_KR_BLK() - 1](kernel_param); break;
                                        // case 2: conv2d_n16cx_winograd_kernel_fp32_avx512_o48_table[TILE_KR_BLK() - 1](kernel_param); break;
                                        // case 3: conv2d_n16cx_winograd_kernel_fp32_avx512_o64_table[TILE_KR_BLK() - 1](kernel_param); break;
                                    }
                                    PICK_PARAM(const float*, kernel_param, SRC_IDX()) += t_body * icl2_eff_padded;
                                    PICK_PARAM(float *, kernel_param, DST_IDX()) += t_body * CH_DT_BLK();
                                }
                                if (t_tail) {
                                    kernel_param[TILES_IDX()] = t_tail;
                                    kernel_param[SRC_TKB_STRIDE_IDX()] = t_tail * icl2_eff_padded;
                                    switch (ock_sel) {
                                        case 0: conv2d_n16cx_winograd_kernel_fp32_avx512_o16_table[t_tail - 1](kernel_param); break;
                                        case 1: conv2d_n16cx_winograd_kernel_fp32_avx512_o32_table[t_tail - 1](kernel_param); break;
                                        // case 2: conv2d_n16cx_winograd_kernel_fp32_avx512_o48_table[t_tail - 1](kernel_param); break;
                                        // case 3: conv2d_n16cx_winograd_kernel_fp32_avx512_o64_table[t_tail - 1](kernel_param); break;
                                    }
                                }
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

                                    tile_corr tc = cal_tile_corr(sp, tk);
                                    const int64_t b = tc.b;
                                    const int64_t oh = tc.th * TILE_OUT_H();
                                    const int64_t ow = tc.tw * TILE_OUT_W();
                                    const int64_t oh_len = min<int64_t>(dst_h - oh, TILE_OUT_H());
                                    const int64_t ow_len = min<int64_t>(dst_w - ow, TILE_OUT_W());
                                    float *l_dst = dst_
                                        + b * dst_b_stride
                                        + g * dst_g_stride
                                        + ocb * (dst_h * dst_w)
                                        + (oh * dst_w + ow) * CH_DT_BLK();
                                    const float *l_sum_src = sum_src_
                                        + b * sum_src_b_stride
                                        + g * dst_g_stride
                                        + ocb * (dst_h * dst_w)
                                        + (oh * dst_w + ow) * CH_DT_BLK();
                                    float *l_gemm_out = gemm_out_buf
                                        + ocb * tl2_eff
                                        + (tk - tl2) * CH_DT_BLK();

                                    int64_t gemm_out_ti_stride = tl2_eff * sp.padded_oc;
                                    if (sp.override_only) {
                                        l_gemm_out      = gemm_out_buf + (ocb - ocl2) * tl2_eff + (tk - tl2) * CH_DT_BLK();
                                        gemm_out_ti_stride = tl2_eff * ocl2_eff;
                                    }

                                    if (oh_len == TILE_OUT_H() && ow_len == TILE_OUT_W()) {
                                        if (sp.use_nt_store) {
                                            winograd_b4f3_dst_trans_fp32_avx512<true>(
                                                l_gemm_out, l_sum_src,
                                                cvt_bias_ + g * bias_g_stride + ocb,
                                                gemm_out_ti_stride, dst_w * CH_DT_BLK(),
                                                dst_w * CH_DT_BLK(), cp.fuse_flag,
                                                postprocess_buf, l_dst);
                                        } else {
                                            winograd_b4f3_dst_trans_fp32_avx512<false>(
                                                l_gemm_out, l_sum_src,
                                                cvt_bias_ + g * bias_g_stride + ocb,
                                                gemm_out_ti_stride, dst_w * CH_DT_BLK(),
                                                dst_w * CH_DT_BLK(), cp.fuse_flag,
                                                postprocess_buf, l_dst);
                                        }
                                    } else {
                                        float *dst_buf = postprocess_buf + sp.thread_matmul_out_len;
                                        winograd_b4f3_dst_trans_fp32_avx512<false>(
                                            l_gemm_out, l_sum_src,
                                            cvt_bias_ + g * bias_g_stride + ocb,
                                            gemm_out_ti_stride, dst_w * CH_DT_BLK(),
                                            TILE_OUT_W() * CH_DT_BLK(), conv_fuse_flag::none,
                                            postprocess_buf, dst_buf);
                                        if (sp.use_nt_store) {
                                            winograd_b4f3_store_dst_fp32_avx512<true>(
                                                dst_buf, l_sum_src,
                                                oh_len, ow_len,
                                                dst_w * CH_DT_BLK(),
                                                cp.fuse_flag, l_dst);
                                        } else {
                                            winograd_b4f3_store_dst_fp32_avx512<false>(
                                                dst_buf, l_sum_src,
                                                oh_len, ow_len,
                                                dst_w * CH_DT_BLK(),
                                                cp.fuse_flag, l_dst);
                                        }
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
        }
    } // OMP_PARALLEL
    }
    if (sp.use_nt_store) {
        PRAGMA_OMP_PARALLEL()
        {
            _mm_sfence();
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n16cx_winograd_b4f3_fp32_avx512_manager::gen_cvt_weights(
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

    const float mat_G[TILE_IN_H()][KERNEL_H()] = {
        {1.f/4,   0.f,      0.f   },
        {-1.f/6,  -1.f/6,   -1.f/6},
        {-1.f/6,  1.f/6,    -1.f/6},
        {1.f/24,  1.f/12,   1.f/6 },
        {1.f/24,  -1.f/12,  1.f/6 },
        {0.f,     0.f,      1.f   },
    };

    // goihw trans goithtw -> gIthtwOi16o
    for (int64_t g = 0; g < param_.group; ++g) {
        for (int64_t icl2 = 0; icl2 < padded_ic; icl2 += ic_l2_blk) {
            for (int64_t ocb = 0; ocb < padded_oc; ocb += CH_DT_BLK()) {
                const int64_t icl2_eff = min<int64_t>(ic_per_gp - icl2, ic_l2_blk);
                const int64_t ocb_eff = min<int64_t>(oc_per_gp - ocb, CH_DT_BLK());
                float mat_T[TILE_IN_H()][KERNEL_W()];
                for (int64_t ic = icl2; ic < icl2 + icl2_eff; ++ic) {
                    const float *l_flt = filter
                                    + g * oc_per_gp * ic_per_gp * KERNEL_H() * KERNEL_W()
                                    + ocb * ic_per_gp * KERNEL_H() * KERNEL_W()
                                    + ic * KERNEL_H() * KERNEL_W();
                    float *l_cvt_flt = cvt_filter_
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
                                    sum += mat_G[i][k] * l_flt[oc * ic_per_gp * KERNEL_H() * KERNEL_W() + k * KERNEL_W() + j];
                                }
                                mat_T[i][j] = sum;
                            }
                        }
                        // (G * filter) * GT
                        for (int64_t i = 0; i < TILE_IN_H(); ++i) {
                            for (int64_t j = 0; j < TILE_IN_W(); ++j) {
                                float sum = 0.0f;
                                for (int64_t k = 0; k < KERNEL_W(); ++k) {
                                    sum += mat_T[i][k] * mat_G[j][k];
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

bool conv2d_n16cx_winograd_b4f3_fp32_avx512_manager::is_supported()
{
    if (param_.is_pointwise()) {
        return false;
    }
    if (param_.channels / param_.group <= 1.801f * CH_DT_BLK()) {
        return false;
    }
    bool aligned_channels   = param_.channels / param_.group % CH_DT_BLK() == 0;
    bool aligned_num_output = param_.num_output / param_.group % CH_DT_BLK() == 0;
    bool is_required_case   = param_.kernel_h == KERNEL_H() &&
                            param_.kernel_w == KERNEL_W() &&
                            param_.stride_h == STRIDE_H() &&
                            param_.stride_w == STRIDE_W() &&
                            param_.dilation_h == 1 &&
                            param_.dilation_w == 1;


    return (is_required_case) && (param_.group == 1 || (aligned_channels && aligned_num_output));
}

conv2d_fp32_executor *conv2d_n16cx_winograd_b4f3_fp32_avx512_manager::gen_executor()
{
    return new conv2d_n16cx_winograd_b4f3_fp32_avx512_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::x86
