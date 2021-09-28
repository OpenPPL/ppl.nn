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
#include "ppl/kernel/x86/common/mmcv_roialign/mmcv_roialign_common.h"
#include "ppl/kernel/x86/common/avx512_tools.h"

#include <immintrin.h>
#include <float.h>
#include <string>
#include <math.h>

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode mmcv_roialign_n16cx_fp32_avx512(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *output_shape,
    const float *input,
    const float *rois,
    const int64_t aligned,
    const int64_t sampling_ratio,
    const float spatial_scale,
    const int32_t pool_mode, // 0: max, 1: avg
    float *output)
{
    const int64_t n_rois        = rois_shape->GetDim(0);
    const int64_t channels      = input_shape->GetDim(1);
    const int64_t c_blk         = 16;
    const int64_t pad_c         = round_up(channels, c_blk);
    const int64_t height        = input_shape->GetDim(2);
    const int64_t width         = input_shape->GetDim(3);
    const int64_t pooled_height = output_shape->GetDim(2);
    const int64_t pooled_width  = output_shape->GetDim(3);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t n = 0; n < n_rois; n++) {
        int64_t index_n = n * pad_c * pooled_width * pooled_height;

        const float *offset_rois = rois + n * 5;
        int64_t roi_batch_ind    = offset_rois[0];

        // Do not use rounding; this implementation detail is critical
        float offset      = aligned ? 0.5f : 0.0f;
        float roi_start_w = offset_rois[1] * spatial_scale - offset;
        float roi_start_h = offset_rois[2] * spatial_scale - offset;
        float roi_end_w   = offset_rois[3] * spatial_scale - offset;
        float roi_end_h   = offset_rois[4] * spatial_scale - offset;

        float roi_width  = roi_end_w - roi_start_w;
        float roi_height = roi_end_h - roi_start_h;
        if (aligned) {
            // if (roi_width >= 0 && roi_height >= 0) {
            //     return ppl::common::RC_INVALID_VALUE;
            // }
        } else { // for backward-compatibility only
            roi_width  = std::max(roi_width, 1.0f);
            roi_height = std::max(roi_height, 1.0f);
        }
        float bin_size_h = roi_height / static_cast<float>(pooled_height);
        float bin_size_w = roi_width / static_cast<float>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int64_t roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
        int64_t roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

        // When the grid is empty, output zeros == 0/1, instead of NaN.
        const float count   = std::max<int64_t>(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4
        const float r_count = 1.0f / count;

        // we want to precalculate indices and weights shared by all channels,
        // this is the key point of optimization
        std::vector<precalc_info> pre_calc(roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
        pre_calc_for_bilinear_interpolate(height, width, pooled_height, pooled_width, roi_bin_grid_h, roi_bin_grid_w, roi_start_h, roi_start_w, bin_size_h, bin_size_w, roi_bin_grid_h, roi_bin_grid_w, pre_calc);

        const __m512 v_zero    = _mm512_set1_ps(0);
        const __m512 v_flt_min = _mm512_set1_ps(-FLT_MAX);
        const __m512 v_rcount  = _mm512_set1_ps(r_count);

        for (int64_t c = 0; c < pad_c; c += c_blk) {
            int64_t index_n_c         = index_n + c * pooled_width * pooled_height;
            const float *offset_input = input + (roi_batch_ind * pad_c + c) * height * width;
            int64_t pre_calc_index    = 0;

            for (int64_t ph = 0; ph < pooled_height; ph++) {
                for (int64_t pw = 0; pw < pooled_width; pw++) {
                    int64_t index = index_n_c + (ph * pooled_width + pw) * c_blk;

                    if (pool_mode == 1) { // avg pooling
                        __m512 v_dst = v_zero;
                        for (int64_t i = 0; i < roi_bin_grid_h * roi_bin_grid_w; i++) {
                            precalc_info pc   = pre_calc[pre_calc_index];
                            const __m512 v_w1 = _mm512_set1_ps(pc.w1);
                            const __m512 v_w2 = _mm512_set1_ps(pc.w2);
                            const __m512 v_w3 = _mm512_set1_ps(pc.w3);
                            const __m512 v_w4 = _mm512_set1_ps(pc.w4);

                            const __m512 v_src1 = _mm512_loadu_ps(offset_input + pc.pos1 * c_blk);
                            const __m512 v_src2 = _mm512_loadu_ps(offset_input + pc.pos2 * c_blk);
                            const __m512 v_src3 = _mm512_loadu_ps(offset_input + pc.pos3 * c_blk);
                            const __m512 v_src4 = _mm512_loadu_ps(offset_input + pc.pos4 * c_blk);

                            v_dst = v_dst + v_w1 * v_src1 + v_w2 * v_src2 + v_w3 * v_src3 + v_w4 * v_src4;
                            pre_calc_index++;
                        }
                        _mm512_storeu_ps(output + index, v_dst * v_rcount);
                    } else if (pool_mode == 0) { // max pooling
                        __m512 v_dst = v_flt_min;
                        for (int64_t i = 0; i < roi_bin_grid_h * roi_bin_grid_w; i++) {
                            precalc_info pc   = pre_calc[pre_calc_index];
                            const __m512 v_w1 = _mm512_set1_ps(pc.w1);
                            const __m512 v_w2 = _mm512_set1_ps(pc.w2);
                            const __m512 v_w3 = _mm512_set1_ps(pc.w3);
                            const __m512 v_w4 = _mm512_set1_ps(pc.w4);

                            const __m512 v_src1 = _mm512_loadu_ps(offset_input + pc.pos1 * c_blk);
                            const __m512 v_src2 = _mm512_loadu_ps(offset_input + pc.pos2 * c_blk);
                            const __m512 v_src3 = _mm512_loadu_ps(offset_input + pc.pos3 * c_blk);
                            const __m512 v_src4 = _mm512_loadu_ps(offset_input + pc.pos4 * c_blk);

                            v_dst = _mm512_max_ps(v_dst, v_w1 * v_src1 + v_w2 * v_src2 + v_w3 * v_src3 + v_w4 * v_src4);
                            pre_calc_index++;
                        }
                        _mm512_storeu_ps(output + index, v_dst);
                    }
                } // for pw
            } // for ph
        } // for c
    } // for n

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
