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

#ifndef __ST_PPL_KERNEL_X86_COMMON_ROIALIGN_ROIALIGN_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_ROIALIGN_ROIALIGN_COMMON_H_

#include <vector>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

struct pre_calc_info {
    int64_t pos1;
    int64_t pos2;
    int64_t pos3;
    int64_t pos4;
    float w1;
    float w2;
    float w3;
    float w4;
};

static inline void pre_calc_for_bilinear_interpolate(
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t iy_upper,
    const int64_t ix_upper,
    float roi_start_h,
    float roi_start_w,
    float bin_size_h,
    float bin_size_w,
    int64_t roi_bin_grid_h,
    int64_t roi_bin_grid_w,
    std::vector<pre_calc_info>& pre_calc)
{
    int64_t pre_calc_index = 0;
    for (int64_t ph = 0; ph < pooled_height; ph++) {
        for (int64_t pw = 0; pw < pooled_width; pw++) {
            for (int64_t iy = 0; iy < iy_upper; iy++) {
                const float yy = roi_start_h + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h;
                for (int64_t ix = 0; ix < ix_upper; ix++) {
                    const float xx = roi_start_w + pw * bin_size_w + (ix + 0.5f) * bin_size_w / roi_bin_grid_w;

                    float x = xx;
                    float y = yy;
                    // deal with: inverse elements are out of feature map boundary
                    if (y < -1.0 || y > height || x < -1.0 || x > width) {
                        // empty
                        pre_calc_info pc;
                        pc.pos1 = 0;
                        pc.pos2 = 0;
                        pc.pos3 = 0;
                        pc.pos4 = 0;
                        pc.w1   = 0;
                        pc.w2   = 0;
                        pc.w3   = 0;
                        pc.w4   = 0;

                        pre_calc[pre_calc_index] = pc;
                        pre_calc_index += 1;
                        continue;
                    }

                    if (y <= 0) {
                        y = 0;
                    }
                    if (x <= 0) {
                        x = 0;
                    }

                    int64_t y_low = static_cast<int64_t>(y);
                    int64_t x_low = static_cast<int64_t>(x);
                    int64_t y_high;
                    int64_t x_high;

                    if (y_low >= height - 1) {
                        y_high = y_low = height - 1;
                        y              = (float)y_low;
                    } else {
                        y_high = y_low + 1;
                    }

                    if (x_low >= width - 1) {
                        x_high = x_low = width - 1;
                        x              = (float)x_low;
                    } else {
                        x_high = x_low + 1;
                    }

                    float ly = y - y_low;
                    float lx = x - x_low;
                    float hy = 1.0f - ly;
                    float hx = 1.0f - lx;
                    float w1 = hy * hx;
                    float w2 = hy * lx;
                    float w3 = ly * hx;
                    float w4 = ly * lx;

                    // save weights and indeces
                    pre_calc_info pc;
                    pc.pos1 = y_low * width + x_low;
                    pc.pos2 = y_low * width + x_high;
                    pc.pos3 = y_high * width + x_low;
                    pc.pos4 = y_high * width + x_high;
                    pc.w1   = w1;
                    pc.w2   = w2;
                    pc.w3   = w3;
                    pc.w4   = w4;

                    pre_calc[pre_calc_index] = pc;

                    pre_calc_index += 1;
                }
            }
        }
    }
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_ROIALIGN_ROIALIGN_COMMON_H_
