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

#include "ppl/kernel/riscv/common/internal_include.h"
#include "ppl/kernel/riscv/common/mmcv_roialign/mmcv_roialign_common.h"

#include <float.h>
#include <string.h>
#include <math.h>

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode mmcv_roialign_ndarray_fp32(
    const ppl::common::TensorShape *input_shape,
    const ppl::common::TensorShape *rois_shape,
    const ppl::common::TensorShape *output_shape,
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
    const int64_t height        = input_shape->GetDim(2);
    const int64_t width         = input_shape->GetDim(3);
    const int64_t pooled_height = output_shape->GetDim(2);
    const int64_t pooled_width  = output_shape->GetDim(3);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t n = 0; n < n_rois; n++) {
        int64_t index_n = n * channels * pooled_width * pooled_height;

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
        const float count = std::max<int64_t>(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

        // we want to precalculate indices and weights shared by all channels,
        // this is the key point of optimization
        std::vector<precalc_info> pre_calc(roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
        pre_calc_for_bilinear_interpolate(height, width, pooled_height, pooled_width, roi_bin_grid_h, roi_bin_grid_w, roi_start_h, roi_start_w, bin_size_h, bin_size_w, roi_bin_grid_h, roi_bin_grid_w, pre_calc);

        for (int64_t c = 0; c < channels; c++) {
            int64_t index_n_c         = index_n + c * pooled_width * pooled_height;
            const float *offset_input = input + (roi_batch_ind * channels + c) * height * width;
            int64_t pre_calc_index    = 0;

            for (int64_t ph = 0; ph < pooled_height; ph++) {
                for (int64_t pw = 0; pw < pooled_width; pw++) {
                    int64_t index = index_n_c + ph * pooled_width + pw;

                    if (pool_mode == 1) { // avg pooling
                        float output_val = 0.;
                        for (int64_t i = 0; i < roi_bin_grid_h * roi_bin_grid_w; i++) {
                            precalc_info pc = pre_calc[pre_calc_index];
                            float val       = pc.w1 * offset_input[pc.pos1] + pc.w2 * offset_input[pc.pos2] +
                                        pc.w3 * offset_input[pc.pos3] + pc.w4 * offset_input[pc.pos4];
                            output_val += val;
                            pre_calc_index += 1;
                        }
                        output[index] = output_val / count;
                    } else if (pool_mode == 0) { // max pooling
                        float maxval = -FLT_MAX;
                        for (int64_t i = 0; i < roi_bin_grid_h * roi_bin_grid_w; i++) {
                            precalc_info pc = pre_calc[pre_calc_index];
                            float val       = pc.w1 * offset_input[pc.pos1] + pc.w2 * offset_input[pc.pos2] +
                                        pc.w3 * offset_input[pc.pos3] + pc.w4 * offset_input[pc.pos4];
                            maxval = max(val, maxval);
                            pre_calc_index += 1;
                        }
                        output[index] = maxval;
                    }
                } // for pw
            } // for ph
        } // for c
    } // for n

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv