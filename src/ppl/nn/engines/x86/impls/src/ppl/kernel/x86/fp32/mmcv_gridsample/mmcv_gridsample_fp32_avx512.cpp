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

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/mmcv_gridsample/mmcv_gridsample_common.h"
#include "ppl/kernel/x86/common/avx512_tools.h"

#include <immintrin.h>
#include <vector>

namespace ppl { namespace kernel { namespace x86 {

template <bool align_corners, grid_sampler_padding padding_mode>
ppl::common::RetCode mmcv_gridsample_linear_ndarray_fp32_avx512_kernel(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *grid_shape,
    const float *src,
    const float *grid,
    float *dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = grid_shape->GetDim(1);
    const int64_t dst_w    = grid_shape->GetDim(2);

    const int64_t simd_w = 16;
    const __m512 v_0     = _mm512_set1_ps(0);
    const __m512 v_1     = _mm512_set1_ps(1);
    const __m512 v_src_h = _mm512_set1_ps(src_h);
    const __m512 v_src_w = _mm512_set1_ps(src_w);

    std::vector<float> ixs_buffer(dst_h * dst_w * PPL_OMP_MAX_THREADS());
    std::vector<float> iys_buffer(dst_h * dst_w * PPL_OMP_MAX_THREADS());

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t n = 0; n < batch; n++) {
        float *ixs = (float *)ixs_buffer.data() + dst_h * dst_w * PPL_OMP_THREAD_ID();
        float *iys = (float *)iys_buffer.data() + dst_h * dst_w * PPL_OMP_THREAD_ID();

        const float* l_grid = grid + n * dst_h * dst_w * 2;
        for (int64_t h = 0; h < dst_h; h++) {
            for (int64_t w = 0; w < dst_w; w++) {
                float x = l_grid[(h * dst_w + w) * 2 + 0];
                float y = l_grid[(h * dst_w + w) * 2 + 1];

                ixs[h * dst_w + w] = grid_sampler_compute_source_index<float, align_corners, padding_mode>(x, src_w); // TODO: use avx512 to speed up
                iys[h * dst_w + w] = grid_sampler_compute_source_index<float, align_corners, padding_mode>(y, src_h);
            }
        }

        for (int64_t c = 0; c < channels; c++) {
            const float *l_src = src + (n * channels + c) * src_h * src_w;
            float *l_dst       = dst + (n * channels + c) * dst_h * dst_w;
            for (int64_t h = 0; h < dst_h; h++) {
                int64_t w = 0;
                for (; w + simd_w <= dst_w; w += simd_w) {
                    __m512 v_ix = _mm512_loadu_ps(ixs + h * dst_w + w);
                    __m512 v_iy = _mm512_loadu_ps(iys + h * dst_w + w);

                    __m512i v_ix_rnd = _mm512_cvt_roundps_epi32(v_ix, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); // round down, same as floor
                    __m512i v_iy_rnd = _mm512_cvt_roundps_epi32(v_iy, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

                    __m512 v_ix0 = _mm512_cvt_roundepi32_ps(v_ix_rnd, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
                    __m512 v_iy0 = _mm512_cvt_roundepi32_ps(v_iy_rnd, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
                    __m512 v_ix1 = _mm512_add_ps(v_ix0, v_1);
                    __m512 v_iy1 = _mm512_add_ps(v_iy0, v_1);

                    __m512 v_x1_lambda = v_ix - v_ix0;
                    __m512 v_y1_lambda = v_iy - v_iy0;
                    __m512 v_x0_lambda = v_ix1 - v_ix;
                    __m512 v_y0_lambda = v_iy1 - v_iy;

                    __mmask16 m_x0_in = _mm512_kand(_mm512_cmp_ps_mask(v_ix0, v_0, _MM_CMPINT_GE), _mm512_cmp_ps_mask(v_ix0, v_src_w, _MM_CMPINT_LT));
                    __mmask16 m_y0_in = _mm512_kand(_mm512_cmp_ps_mask(v_iy0, v_0, _MM_CMPINT_GE), _mm512_cmp_ps_mask(v_iy0, v_src_h, _MM_CMPINT_LT));
                    __mmask16 m_x1_in = _mm512_kand(_mm512_cmp_ps_mask(v_ix1, v_0, _MM_CMPINT_GE), _mm512_cmp_ps_mask(v_ix1, v_src_w, _MM_CMPINT_LT));
                    __mmask16 m_y1_in = _mm512_kand(_mm512_cmp_ps_mask(v_iy1, v_0, _MM_CMPINT_GE), _mm512_cmp_ps_mask(v_iy1, v_src_h, _MM_CMPINT_LT));

                    __m512 v_src_y0x0 = _mm512_mask_i32gather_ps(v_0, _mm512_kand(m_y0_in, m_x0_in), _mm512_cvt_roundps_epi32(v_iy0 * v_src_w + v_ix0, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC), l_src, 4);
                    __m512 v_src_y0x1 = _mm512_mask_i32gather_ps(v_0, _mm512_kand(m_y0_in, m_x1_in), _mm512_cvt_roundps_epi32(v_iy0 * v_src_w + v_ix1, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC), l_src, 4);
                    __m512 v_src_y1x0 = _mm512_mask_i32gather_ps(v_0, _mm512_kand(m_y1_in, m_x0_in), _mm512_cvt_roundps_epi32(v_iy1 * v_src_w + v_ix0, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC), l_src, 4);
                    __m512 v_src_y1x1 = _mm512_mask_i32gather_ps(v_0, _mm512_kand(m_y1_in, m_x1_in), _mm512_cvt_roundps_epi32(v_iy1 * v_src_w + v_ix1, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC), l_src, 4);

                    __m512 v_dst = v_0;
                    v_dst        = _mm512_mask_add_ps(v_dst, _mm512_kand(m_y0_in, m_x0_in), v_dst, v_src_y0x0 * v_y0_lambda * v_x0_lambda);
                    v_dst        = _mm512_mask_add_ps(v_dst, _mm512_kand(m_y0_in, m_x1_in), v_dst, v_src_y0x1 * v_y0_lambda * v_x1_lambda);
                    v_dst        = _mm512_mask_add_ps(v_dst, _mm512_kand(m_y1_in, m_x0_in), v_dst, v_src_y1x0 * v_y1_lambda * v_x0_lambda);
                    v_dst        = _mm512_mask_add_ps(v_dst, _mm512_kand(m_y1_in, m_x1_in), v_dst, v_src_y1x1 * v_y1_lambda * v_x1_lambda);

                    _mm512_storeu_ps(l_dst + h * dst_w + w, v_dst);
                }
                for (; w < dst_w; w++) {
                    float ix = ixs[h * dst_w + w];
                    float iy = iys[h * dst_w + w];

                    int64_t x0 = std::floor(ix);
                    int64_t y0 = std::floor(iy);
                    int64_t x1 = x0 + 1;
                    int64_t y1 = y0 + 1;

                    float x1_lambda = ix - x0;
                    float y1_lambda = iy - y0;
                    float x0_lambda = x1 - ix;
                    float y0_lambda = y1 - iy;

                    l_dst[h * dst_w + w] = (within_bounds_2d(y0, x0, src_h, src_w) ? l_src[y0 * src_w + x0] * y0_lambda * x0_lambda : 0) +
                                           (within_bounds_2d(y0, x1, src_h, src_w) ? l_src[y0 * src_w + x1] * y0_lambda * x1_lambda : 0) +
                                           (within_bounds_2d(y1, x0, src_h, src_w) ? l_src[y1 * src_w + x0] * y1_lambda * x0_lambda : 0) +
                                           (within_bounds_2d(y1, x1, src_h, src_w) ? l_src[y1 * src_w + x1] * y1_lambda * x1_lambda : 0);
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode mmcv_gridsample_linear_ndarray_fp32_avx512(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *grid_shape,
    const float *src,
    const float *grid,
    const bool align_corners,
    const int64_t padding_mode,
    float *dst)
{
    if (align_corners) {
        if (padding_mode == ZEROS) {
            return mmcv_gridsample_linear_ndarray_fp32_avx512_kernel<true, ZEROS>(src_shape, grid_shape, src, grid, dst);
        } else if (padding_mode == BORDER) {
            return mmcv_gridsample_linear_ndarray_fp32_avx512_kernel<true, BORDER>(src_shape, grid_shape, src, grid, dst);
        } else if (padding_mode == REFLECTION) {
            return mmcv_gridsample_linear_ndarray_fp32_avx512_kernel<true, REFLECTION>(src_shape, grid_shape, src, grid, dst);
        }
    } else {
        if (padding_mode == ZEROS) {
            return mmcv_gridsample_linear_ndarray_fp32_avx512_kernel<false, ZEROS>(src_shape, grid_shape, src, grid, dst);
        } else if (padding_mode == BORDER) {
            return mmcv_gridsample_linear_ndarray_fp32_avx512_kernel<false, BORDER>(src_shape, grid_shape, src, grid, dst);
        } else if (padding_mode == REFLECTION) {
            return mmcv_gridsample_linear_ndarray_fp32_avx512_kernel<false, REFLECTION>(src_shape, grid_shape, src, grid, dst);
        }
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}; // namespace ppl::kernel::x86
