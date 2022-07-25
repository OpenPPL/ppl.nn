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
#include <nmmintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

void im2col2d_ndarray_fp32_sse(
    const float *img,
    const int64_t channels,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t hole_h,
    const int64_t hole_w,
    float *col)
{
    const int64_t simd_w = 4;
    __m128 mmzero        = _mm_setzero_ps();
#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
#endif
    for (int64_t ic = 0; ic < channels; ++ic) {
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                const int64_t expanded_id = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                const int64_t ih          = kh * hole_h;
                const int64_t iw          = kw * hole_w;

                const int64_t oh_beg = max<int64_t>((int64_t)ceilf((pad_h - ih) / (float)(stride_h)), 0);
                const int64_t oh_end = max<int64_t>(oh_beg, min<int64_t>((int64_t)ceilf((src_h + pad_h - ih) / (float)(stride_h)), dst_h));
                const int64_t ow_beg = max<int64_t>((int64_t)ceilf((pad_w - iw) / (float)(stride_w)), 0);
                const int64_t ow_end = max<int64_t>(ow_beg, min<int64_t>((int64_t)ceilf((src_w + pad_w - iw) / (float)(stride_w)), dst_w));

                const float *in_d = img + ic * src_h * src_w;
                float *out_d      = col + expanded_id * dst_w * dst_h;

                for (int64_t oh = 0; oh < oh_beg; ++oh) {
                    int64_t ow = 0;
                    for (; ow <= dst_w - simd_w; ow += simd_w) {
                        _mm_storeu_ps(out_d + oh * dst_w + ow, mmzero);
                    }
                    for (; ow < dst_w; ++ow) {
                        out_d[oh * dst_w + ow] = 0.0f;
                    }
                }

                int64_t ih_paded = oh_beg * stride_h - pad_h + ih;
                int64_t iw_paded = ow_beg * stride_w - pad_w + iw;
                for (int64_t oh = oh_beg; oh < oh_end; ++oh, ih_paded += stride_h) {
                    int64_t ow = 0;
                    for (; ow <= ow_beg - simd_w; ow += simd_w) {
                        _mm_storeu_ps(out_d + oh * dst_w + ow, mmzero);
                    }
                    for (; ow < ow_beg; ++ow) {
                        out_d[oh * dst_w + ow] = 0.0f;
                    }

                    int64_t out_id = oh * dst_w + ow;
                    int64_t in_id  = ih_paded * src_w + iw_paded;
                    for (; ow < ow_end; ++ow, ++out_id, in_id += stride_w) {
                        out_d[out_id] = in_d[in_id];
                    }

                    for (; ow <= dst_w - simd_w; ow += simd_w) {
                        _mm_storeu_ps(out_d + oh * dst_w + ow, mmzero);
                    }
                    for (; ow < dst_w; ++ow) {
                        out_d[oh * dst_w + ow] = 0.0f;
                    }
                }

                for (int64_t oh = oh_end; oh < dst_h; ++oh) {
                    int64_t ow = 0;
                    for (; ow <= dst_w - simd_w; ow += simd_w) {
                        _mm_storeu_ps(out_d + oh * dst_w + ow, mmzero);
                    }
                    for (; ow < dst_w; ++ow) {
                        out_d[oh * dst_w + ow] = 0.0f;
                    }
                }
            }
        }
    }
}

}}}; // namespace ppl::kernel::x86
