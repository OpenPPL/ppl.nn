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
#include "ppl/kernel/x86/common/avx_tools.h"

namespace ppl { namespace kernel { namespace x86 {

void col2im2d_ndarray_fp32_avx(
    const float *col,
    const float *bias,
    const int64_t col_h,
    const int64_t col_w,
    const int64_t num_output,
    const int64_t img_h,
    const int64_t img_w,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t hole_h,
    const int64_t hole_w,
    float *image)
{
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t c_img = 0; c_img < num_output; ++c_img) {
        const int32_t bias_val = bias ? *(int32_t*)(&bias[c_img]) : 0;
        memset32_avx(image + c_img * img_h * img_w, bias_val, img_h * img_w);
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                int64_t c_col    = c_img * kernel_h * kernel_w + kh * kernel_w + kw;
                int64_t w_offset = kw * hole_w;
                int64_t h_offset = kh * hole_h;
                int64_t w_start  = max<int64_t>(div_up(pad_w - w_offset, stride_w), 0);
                int64_t w_end    = min<int64_t>((img_w + pad_w - kernel_w) / stride_w + 1, img_w);
                int64_t iw_start = w_start * stride_w - (pad_w - w_offset);
                for (int64_t h = 0; h < col_h; ++h) {
                    int64_t ih = h * stride_h - pad_h + h_offset;
                    if (ih >= 0 && ih < img_h) {
                        int64_t i_base = (c_img * img_h + ih) * img_w;
                        int64_t o_base = (c_col * col_h + h) * col_w;
                        for (int64_t w = w_start, iw = iw_start; w < w_end; w += 1, iw += stride_w) {
                            image[i_base + iw] += col[o_base + w];
                        }
                    }
                }
            }
        }
    }
}

}}}; // namespace ppl::kernel::x86
