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
#include <vector>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reisze2d_ndarray_pytorch_linear_floor_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst)
{
    const int64_t num_imgs = src_shape->GetDim(0) * src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);
    const float hscale     = 1.0f / scale_h;
    const float wscale     = 1.0f / scale_w;

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t ni = 0; ni < num_imgs; ++ni) {
        const float *l_src = src + ni * src_h * src_w;
        float *l_dst       = dst + ni * dst_h * dst_w;
        for (int64_t oh = 0; oh < dst_h; ++oh) {
            float ih = (oh + 0.5f) * hscale - 0.5f;
            int64_t h0, h1;
            float h0_lambda, h1_lambda;
            if (ih < 0) {
                h0        = 0;
                h1        = 0;
                h0_lambda = 1;
                h1_lambda = 0;
            } else {
                h0        = (int64_t)ih;
                h1        = h0 + (h0 < src_h - 1);
                h1_lambda = ih - h0;
                h0_lambda = 1.0f - h1_lambda;
            }

            for (int64_t ow = 0; ow < dst_w; ++ow) {
                float iw = (ow + 0.5f) * wscale - 0.5f;
                int64_t w0, w1;
                float w0_lambda, w1_lambda;
                if (iw < 0) {
                    w0        = 0;
                    w1        = 0;
                    w0_lambda = 1;
                    w1_lambda = 0;
                } else {
                    w0        = (int64_t)iw;
                    w1        = w0 + (w0 < src_w - 1);
                    w1_lambda = iw - w0;
                    w0_lambda = 1.0f - w1_lambda;
                }

                l_dst[oh * dst_w + ow] = l_src[h0 * src_w + w0] * h0_lambda * w0_lambda +
                                         l_src[h0 * src_w + w1] * h0_lambda * w1_lambda +
                                         l_src[h1 * src_w + w0] * h1_lambda * w0_lambda +
                                         l_src[h1 * src_w + w1] * h1_lambda * w1_lambda;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode reisze2d_ndarray_asymmetric_nearest_floor_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst)
{
    const int64_t num_imgs = src_shape->GetDim(0) * src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);
    const float hscale     = 1.0f / scale_h;
    const float wscale     = 1.0f / scale_w;

    std::vector<int64_t> iw_list(dst_w);
    for (int64_t i = 0; i < dst_w; i++) {
        iw_list[i] = static_cast<int64_t>(i * wscale);
    }

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t ni = 0; ni < num_imgs; ++ni) {
        const float *l_src = src + ni * src_h * src_w;
        float *l_dst       = dst + ni * dst_h * dst_w;
        for (int64_t oh = 0; oh < dst_h; ++oh) {
            int64_t ih         = static_cast<int64_t>(oh * hscale);
            float *t_dst       = l_dst + oh * dst_w;
            const float *t_src = l_src + ih * src_w;
            int64_t ow         = 0;
            for (; ow < dst_w - 8; ow += 8) {
                t_dst[ow + 0] = t_src[iw_list[ow + 0]];
                t_dst[ow + 1] = t_src[iw_list[ow + 1]];
                t_dst[ow + 2] = t_src[iw_list[ow + 2]];
                t_dst[ow + 3] = t_src[iw_list[ow + 3]];
                t_dst[ow + 4] = t_src[iw_list[ow + 4]];
                t_dst[ow + 5] = t_src[iw_list[ow + 5]];
                t_dst[ow + 6] = t_src[iw_list[ow + 6]];
                t_dst[ow + 7] = t_src[iw_list[ow + 7]];
            }
            for (; ow < dst_w; ow++) {
                t_dst[ow] = t_src[iw_list[ow]];
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

inline void calc_resize_cubic_coeff(
        const float r,
        const float A,
        float *coeff)
{
    coeff[0] = ((A * (r + 1) - 5 * A) * (r + 1) + 8 * A) * (r + 1) - 4 * A;
    coeff[1] = ((A + 2) * r - (A + 3)) * r * r + 1;
    coeff[2] = ((A + 2) * (1 - r) - (A + 3)) * (1 - r) * (1 - r) + 1;
    coeff[3] = 1.0f - coeff[0] - coeff[1] - coeff[2];
}

template <typename T>
inline T get_value_bounded(
        const T *src,
        const int64_t h,
        const int64_t w,
        const int64_t src_h,
        const int64_t src_w)
{
    int64_t h_ = max<int64_t>(min(h, src_h - 1), 0);
    int64_t w_ = max<int64_t>(min(w, src_w - 1), 0);
    return src[h_ * src_w + w_];
}

#define SRC(h, w) (get_value_bounded(l_src, (h), (w), src_h, src_w))

ppl::common::RetCode reisze2d_ndarray_pytorch_cubic_floor_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    const float cubic_coeff_a,
    float *dst)
{
    const int64_t num_imgs = src_shape->GetDim(0) * src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);
    const float hscale     = 1.0f / scale_h;
    const float wscale     = 1.0f / scale_w;

    std::vector<int64_t> sy_tab(dst_h);
    std::vector<float> coeff_y_tab(dst_h * 4);
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t oh = 0; oh < dst_h; ++oh) {
        float ih   = (oh + 0.5f) * hscale - 0.5f;
        int64_t sy = std::floor(ih);
        float fy   = ih - sy;
        sy_tab[oh] = sy;

        float *coeff_y = coeff_y_tab.data() + 4 * oh;
        calc_resize_cubic_coeff(fy, cubic_coeff_a, coeff_y);
    }

    std::vector<int64_t> sx_tab(dst_w);
    std::vector<float> coeff_x_tab(dst_w * 4);
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t ow = 0; ow < dst_w; ++ow) {
        float iw   = (ow + 0.5f) * wscale - 0.5f;
        int64_t sx = std::floor(iw);
        float fx   = iw - sx;
        sx_tab[ow] = sx;

        float *coeff_x = coeff_x_tab.data() + 4 * ow;
        calc_resize_cubic_coeff(fx, cubic_coeff_a, coeff_x);
    }

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
    PRAGMA_OMP_PARALLEL_FOR()
#endif
    for (int64_t ni = 0; ni < num_imgs; ++ni) {
        for (int64_t oh = 0; oh < dst_h; ++oh) {
            const float *l_src = src + ni * src_h * src_w; // place these to inner loop for parallel_for_collapse
            float *l_dst       = dst + ni * dst_h * dst_w;

            const int64_t sy     = sy_tab[oh];
            const float *coeff_y = coeff_y_tab.data() + 4 * oh;
            for (int64_t ow = 0; ow < dst_w; ++ow) {
                const int64_t sx     = sx_tab[ow];
                const float *coeff_x = coeff_x_tab.data() + 4 * ow;

                l_dst[oh * dst_w + ow] =
                    SRC(sy - 1, sx - 1) * coeff_x[0] * coeff_y[0] + SRC(sy, sx - 1) * coeff_x[0] * coeff_y[1] +
                    SRC(sy + 1, sx - 1) * coeff_x[0] * coeff_y[2] + SRC(sy + 2, sx - 1) * coeff_x[0] * coeff_y[3] +
                    SRC(sy - 1, sx) * coeff_x[1] * coeff_y[0] + SRC(sy, sx) * coeff_x[1] * coeff_y[1] +
                    SRC(sy + 1, sx) * coeff_x[1] * coeff_y[2] + SRC(sy + 2, sx) * coeff_x[1] * coeff_y[3] +
                    SRC(sy - 1, sx + 1) * coeff_x[2] * coeff_y[0] + SRC(sy, sx + 1) * coeff_x[2] * coeff_y[1] +
                    SRC(sy + 1, sx + 1) * coeff_x[2] * coeff_y[2] + SRC(sy + 2, sx + 1) * coeff_x[2] * coeff_y[3] +
                    SRC(sy - 1, sx + 2) * coeff_x[3] * coeff_y[0] + SRC(sy, sx + 2) * coeff_x[3] * coeff_y[1] +
                    SRC(sy + 1, sx + 2) * coeff_x[3] * coeff_y[2] + SRC(sy + 2, sx + 2) * coeff_x[3] * coeff_y[3];
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

#undef SRC

}}} // namespace ppl::kernel::x86
