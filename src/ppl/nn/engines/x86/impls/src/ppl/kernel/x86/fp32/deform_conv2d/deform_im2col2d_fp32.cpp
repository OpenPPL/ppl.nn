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

#include <math.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gemm.h"

namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
eT bilinear_interpolate_2d(
    const eT* src,
    const int64_t src_h,
    const int64_t src_w,
    const eT h,
    const eT w)
{
    if (h <= -1 || src_h <= h || w <= -1 || src_w <= w) {
        return 0;
    }

    int64_t h_low = ::floor(h);
    int64_t w_low = ::floor(w);
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    eT lh = h - h_low;
    eT lw = w - w_low;
    eT hh = 1 - lh;
    eT hw = 1 - lw;

    eT v1 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = src[h_low * src_w + w_low];
    eT v2 = 0;
    if (h_low >= 0 && w_high <= src_w - 1)
        v2 = src[h_low * src_w + w_high];
    eT v3 = 0;
    if (h_high <= src_h - 1 && w_low >= 0)
        v3 = src[h_high * src_w + w_low];
    eT v4 = 0;
    if (h_high <= src_h - 1 && w_high <= src_w - 1)
        v4 = src[h_high * src_w + w_high];

    eT w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    eT val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

// output: (channels * kernel_h * kernel_w, dst_h * dst_w)
template <typename eT>
void deform_im2col2d(
    const eT* input,
    const eT* offset,
    const eT* mask,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t channels,
    const int64_t offset_groups,
    const int64_t dst_h,
    const int64_t dst_w,
    const bool use_mask,
    eT* columns)
{
    const int64_t workload = channels * dst_h * dst_w;
PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t index = 0; index < workload; ++index) {
        const int64_t ow = index % dst_w;
        const int64_t oh = (index / dst_w) % dst_h;
        const int64_t ic = index / (dst_w * dst_h);
        const int64_t oc = ic * kernel_h * kernel_w;

        int64_t c_per_offset_grp = channels / offset_groups;
        const int64_t grp_idx = ic / c_per_offset_grp;

        auto columns_ptr = columns + (oc * (dst_h * dst_w) + oh * dst_w + ow);
        auto input_ptr = input + ic * (src_h * src_w);
        auto offset_ptr = offset + grp_idx * 2 * kernel_h * kernel_w * dst_h * dst_w;
        auto mask_ptr = mask;
        if (use_mask) {
            mask_ptr += grp_idx * kernel_h * kernel_w * dst_h * dst_w;
        }

        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                const int64_t mask_idx = kh * kernel_w + kw;
                const int64_t offset_idx = 2 * mask_idx;

                eT mask_value = 1;
                if (use_mask) {
                mask_value =
                    mask_ptr[mask_idx * (dst_h * dst_w) + oh * dst_w + ow];
                }

                const eT offset_h = offset_ptr[offset_idx * (dst_h * dst_w) + oh * dst_w + ow];
                const eT offset_w = offset_ptr[(offset_idx + 1) * (dst_h * dst_w) + oh * dst_w + ow];
                const eT ih = (oh * stride_h - pad_h) + kh * dilation_h + offset_h;
                const eT iw = (ow * stride_w - pad_w) + kw * dilation_w + offset_w;
                *columns_ptr = mask_value * bilinear_interpolate_2d(input_ptr, src_h, src_w, ih, iw);
                columns_ptr += dst_h * dst_w;
            }
        }
    }
}

void deform_im2col2d_fp32(
    const float* input,
    const float* offset,
    const float* mask,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t channels,
    const int64_t offset_groups,
    const int64_t dst_h,
    const int64_t dst_w,
    const bool use_mask,
    float* columns)
{
    deform_im2col2d<float>(
        input, offset, mask,
        src_h, src_w,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        channels, offset_groups,
        dst_h, dst_w,
        use_mask, columns);
}

}}}; // namespace ppl::kernel::x86
