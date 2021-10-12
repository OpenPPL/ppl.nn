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
#include <math.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/roialign/roialign_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode roialign_ndarray_fp32(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *batch_indices_shape,
    const float *src,
    const float *rois,
    const int64_t *batch_indices,
    const int32_t mode,
    const int32_t output_height,
    const int32_t output_width,
    const int32_t sampling_ratio,
    const float spatial_scale,
    float *dst)
{
    const int64_t num_rois      = rois_shape->GetDim(0);
    const int64_t channels      = input_shape->GetDim(1);
    const int64_t pooled_height = output_height;
    const int64_t pooled_width  = output_width;
    const int64_t height        = input_shape->GetDim(2);
    const int64_t width         = input_shape->GetDim(3);

    const int64_t rois_offset = 4;

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < num_rois; i++) {
        int64_t index_n             = i * channels * pooled_width * pooled_height;
        const float *p_rois         = rois + i * rois_offset;
        const int64_t roi_batch_ind = batch_indices[i];

        float roi_start_w = p_rois[0] * spatial_scale;
        float roi_start_h = p_rois[1] * spatial_scale;
        float roi_end_w   = p_rois[2] * spatial_scale;
        float roi_end_h   = p_rois[3] * spatial_scale;

        float roi_width  = max(roi_end_w - roi_start_w, 1.0f);
        float roi_height = max(roi_end_h - roi_start_h, 1.0f);
        float bin_size_h = roi_height / pooled_height;
        float bin_size_w = roi_width / pooled_width;

        int64_t roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(::ceil(roi_height / pooled_height)); // e.g., = 2
        int64_t roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(::ceil(roi_width / pooled_width));

        const int64_t count = roi_bin_grid_h * roi_bin_grid_w;

        std::vector<pre_calc_info> pre_calc;
        pre_calc.resize(roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
        pre_calc_for_bilinear_interpolate(
            height,
            width,
            pooled_height,
            pooled_width,
            roi_bin_grid_h,
            roi_bin_grid_w,
            roi_start_h,
            roi_start_w,
            bin_size_h,
            bin_size_w,
            roi_bin_grid_h,
            roi_bin_grid_w,
            pre_calc);

        for (int64_t c = 0; c < channels; c++) {
            int64_t index_n_c      = index_n + c * pooled_width * pooled_height;
            const float *p_src     = src + (roi_batch_ind * channels + c) * height * width;
            int64_t pre_calc_index = 0;

            for (int64_t ph = 0; ph < pooled_height; ph++) {
                for (int64_t pw = 0; pw < pooled_width; pw++) {
                    int64_t index    = index_n_c + ph * pooled_width + pw;
                    float output_val = 0;
                    if (mode == 0) { // avg pooling
                        for (int64_t i = 0; i < roi_bin_grid_h * roi_bin_grid_w; i++) {
                            pre_calc_info pc = pre_calc[pre_calc_index];
                            output_val += pc.w1 * p_src[pc.pos1] + pc.w2 * p_src[pc.pos2] +
                                          pc.w3 * p_src[pc.pos3] + pc.w4 * p_src[pc.pos4];
                            pre_calc_index++;
                        }
                        output_val /= count;
                    } else if (mode == 1) { // max pooling
                        bool max_flag = false;
                        for (int64_t i = 0; i < roi_bin_grid_h * roi_bin_grid_w; i++) {
                            pre_calc_info pc = pre_calc[pre_calc_index];

                            float val = max(max(max(pc.w1 * p_src[pc.pos1], pc.w2 * p_src[pc.pos2]), pc.w3 * p_src[pc.pos3]), pc.w4 * p_src[pc.pos4]);

                            if (!max_flag) {
                                output_val = val;
                                max_flag   = true;
                            } else {
                                output_val = max(output_val, val);
                            }

                            pre_calc_index++;
                        }
                    }
                    dst[index] = output_val;
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
