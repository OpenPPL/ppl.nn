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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_RESIZE2D_NEON_RESIZE2D_NBCX_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_RESIZE2D_NEON_RESIZE2D_NBCX_COMMON_H_

#include <cmath>
#include <vector>
#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, int32_t c_blk>
static ppl::common::RetCode resize2d_nbcx_pytorch_linear_floor_common(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    const float scale_h,
    const float scale_w,
    eT *dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);
    const int64_t padded_c = round_up(channels, c_blk);
    const float hscale     = 1.0f / scale_h;
    const float wscale     = 1.0f / scale_w;
    const int64_t src_hw   = int64_t(src_h) * src_w;
    const int64_t dst_hw   = int64_t(dst_h) * dst_w;

    std::vector<int32_t> w0_tab(dst_w);
    std::vector<int32_t> w1_tab(dst_w);
    std::vector<float> w0_lambda_tab(dst_w);
    std::vector<float> w1_lambda_tab(dst_w);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t ow = 0; ow < dst_w; ow++) {
        float iw = dst_w > 1 ? (ow + 0.5f) * wscale - 0.5f : 0;
        if (iw < 0) {
            w0_tab[ow]        = 0;
            w1_tab[ow]        = 0;
            w0_lambda_tab[ow] = 1;
            w1_lambda_tab[ow] = 0;
        } else {
            w0_tab[ow]        = (int64_t)iw;
            w1_tab[ow]        = w0_tab[ow] + (w0_tab[ow] < src_w - 1);
            w1_lambda_tab[ow] = iw - w0_tab[ow];
            w0_lambda_tab[ow] = 1.0f - w1_lambda_tab[ow];
        }
    }

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t bc = 0; bc < batch * padded_c; bc += c_blk) {
        for (int64_t oh = 0; oh < dst_h; ++oh) {
            const eT *l_src = src + bc * src_hw; // place these to inner loop for parallel_for_collapse
            eT *l_dst       = dst + bc * dst_hw;

            float ih = dst_h > 1 ? (oh + 0.5f) * hscale - 0.5f : 0;
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
                int64_t w0, w1;
                float w0_lambda, w1_lambda;
                w0        = w0_tab[ow];
                w1        = w1_tab[ow];
                w0_lambda = w0_lambda_tab[ow];
                w1_lambda = w1_lambda_tab[ow];

                const vecType v_w0 = vdup_n<eT, eN>(h0_lambda * w0_lambda);
                const vecType v_w1 = vdup_n<eT, eN>(h0_lambda * w1_lambda);
                const vecType v_w2 = vdup_n<eT, eN>(h1_lambda * w0_lambda);
                const vecType v_w3 = vdup_n<eT, eN>(h1_lambda * w1_lambda);

                vecType v_data_0 = vld<eT, eN>(l_src + (h0 * src_w + w0) * c_blk);
                vecType v_data_1 = vld<eT, eN>(l_src + (h0 * src_w + w1) * c_blk);
                vecType v_data_2 = vld<eT, eN>(l_src + (h1 * src_w + w0) * c_blk);
                vecType v_data_3 = vld<eT, eN>(l_src + (h1 * src_w + w1) * c_blk);

                v_data_0 = vmul<vecType>(v_data_0, v_w0);
                v_data_1 = vmul<vecType>(v_data_1, v_w1);
                v_data_2 = vmul<vecType>(v_data_2, v_w2);
                v_data_3 = vmul<vecType>(v_data_3, v_w3);

                vecType v_dst = vadd<vecType>(vadd<vecType>(v_data_0, v_data_1), vadd<vecType>(v_data_2, v_data_3));
                vst<eT, eN>(l_dst + (oh * dst_w + ow) * c_blk, v_dst);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, int32_t c_blk>
static ppl::common::RetCode resize2d_nbcx_asymmetric_nearest_floor_common(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    const float scale_h,
    const float scale_w,
    eT *dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);
    const int64_t padded_c = round_up(channels, c_blk);
    const float hscale     = 1.0f / scale_h;
    const float wscale     = 1.0f / scale_w;

    std::vector<int32_t> iw_list(dst_w);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < dst_w; i++) {
        iw_list[i] = static_cast<int64_t>(i * wscale);
    }

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t ni = 0; ni < batch * padded_c; ni += c_blk) {
        for (int64_t oh = 0; oh < dst_h; ++oh) {
            const eT *l_src = src + ni * src_h * src_w; // place these to inner loop for parallel_for_collapse
            eT *l_dst       = dst + ni * dst_h * dst_w;

            int64_t ih      = static_cast<int64_t>(oh * hscale);
            eT *t_dst       = l_dst + oh * dst_w * c_blk;
            const eT *t_src = l_src + ih * src_w * c_blk;

            int64_t ow = 0;
            for (; ow + 2 <= dst_w; ow += 2) {
                vecType v_src_0 = vld<eT, eN>(t_src + iw_list[ow + 0] * c_blk);
                vecType v_src_1 = vld<eT, eN>(t_src + iw_list[ow + 1] * c_blk);
                vst<eT, eN>(t_dst + (ow + 0) * c_blk, v_src_0);
                vst<eT, eN>(t_dst + (ow + 1) * c_blk, v_src_1);
            }
            for (; ow < dst_w; ++ow) {
                vecType v_src = vld<eT, eN>(t_src + iw_list[ow + 0] * c_blk);
                vst<eT, eN>(t_dst + (ow + 0) * c_blk, v_src);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, int32_t c_blk>
inline typename DT<eT, c_blk>::vecDT get_value_bounded_nbcx(
    const eT *src,
    const int64_t h,
    const int64_t w,
    const int64_t src_h,
    const int64_t src_w)
{
    int64_t h_ = max<int64_t>(min(h, src_h - 1), 0);
    int64_t w_ = max<int64_t>(min(w, src_w - 1), 0);
    return vld<eT, c_blk>(src + h_ * src_w * c_blk + w_ * c_blk);
}

#define SRC(h, w) (get_value_bounded_nbcx<eT, c_blk>(l_src, (h), (w), src_h, src_w))

template <typename eT, int32_t c_blk>
static ppl::common::RetCode resize2d_nbcx_pytorch_cubic_floor_common(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    const float scale_h,
    const float scale_w,
    const float cubic_coeff_a,
    eT *dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);
    const int64_t padded_c = round_up(channels, c_blk);
    const float hscale     = 1.0f / scale_h;
    const float wscale     = 1.0f / scale_w;

    std::vector<int32_t> sy_tab(dst_h);
    std::vector<float> coeff_y_tab(dst_h * 4);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t oh = 0; oh < dst_h; ++oh) {
        float ih   = dst_h > 1 ? (oh + 0.5f) * hscale - 0.5f : 0;
        int64_t sy = std::floor(ih);
        float fy   = ih - sy;
        sy_tab[oh] = sy;

        float *coeff_y = coeff_y_tab.data() + 4 * oh;
        calc_resize_cubic_coeff(fy, cubic_coeff_a, coeff_y);
    }

    std::vector<int32_t> sx_tab(dst_w);
    std::vector<float> coeff_x_tab(dst_w * 4);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t ow = 0; ow < dst_w; ++ow) {
        float iw   = dst_w > 1 ? (ow + 0.5f) * wscale - 0.5f : 0;
        int64_t sx = std::floor(iw);
        float fx   = iw - sx;
        sx_tab[ow] = sx;

        float *coeff_x = coeff_x_tab.data() + 4 * ow;
        calc_resize_cubic_coeff(fx, cubic_coeff_a, coeff_x);
    }

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < padded_c; c += c_blk) {
            for (int64_t oh = 0; oh < dst_h; ++oh) {
                const eT *l_src = src + (n * padded_c + c) * src_h * src_w; // place these to inner loop for parallel_for_collapse
                eT *l_dst       = dst + (n * padded_c + c) * dst_h * dst_w;

                const int64_t sy     = sy_tab[oh];
                const float *coeff_y = coeff_y_tab.data() + 4 * oh;

                const vecType v_coeff_y0 = vdup_n<eT, eN>(coeff_y[0]);
                const vecType v_coeff_y1 = vdup_n<eT, eN>(coeff_y[1]);
                const vecType v_coeff_y2 = vdup_n<eT, eN>(coeff_y[2]);
                const vecType v_coeff_y3 = vdup_n<eT, eN>(coeff_y[3]);
                for (int64_t ow = 0; ow < dst_w; ++ow) {
                    const int64_t sx     = sx_tab[ow];
                    const float *coeff_x = coeff_x_tab.data() + 4 * ow;

                    const vecType v_coeff_x0 = vdup_n<eT, eN>(coeff_x[0]);
                    const vecType v_coeff_x1 = vdup_n<eT, eN>(coeff_x[1]);
                    const vecType v_coeff_x2 = vdup_n<eT, eN>(coeff_x[2]);
                    const vecType v_coeff_x3 = vdup_n<eT, eN>(coeff_x[3]);

                    vecType v_dst =
                        SRC(sy - 1, sx - 1) * v_coeff_x0 * v_coeff_y0 + SRC(sy, sx - 1) * v_coeff_x0 * v_coeff_y1 +
                        SRC(sy + 1, sx - 1) * v_coeff_x0 * v_coeff_y2 + SRC(sy + 2, sx - 1) * v_coeff_x0 * v_coeff_y3 +
                        SRC(sy - 1, sx) * v_coeff_x1 * v_coeff_y0 + SRC(sy, sx) * v_coeff_x1 * v_coeff_y1 +
                        SRC(sy + 1, sx) * v_coeff_x1 * v_coeff_y2 + SRC(sy + 2, sx) * v_coeff_x1 * v_coeff_y3 +
                        SRC(sy - 1, sx + 1) * v_coeff_x2 * v_coeff_y0 + SRC(sy, sx + 1) * v_coeff_x2 * v_coeff_y1 +
                        SRC(sy + 1, sx + 1) * v_coeff_x2 * v_coeff_y2 + SRC(sy + 2, sx + 1) * v_coeff_x2 * v_coeff_y3 +
                        SRC(sy - 1, sx + 2) * v_coeff_x3 * v_coeff_y0 + SRC(sy, sx + 2) * v_coeff_x3 * v_coeff_y1 +
                        SRC(sy + 1, sx + 2) * v_coeff_x3 * v_coeff_y2 + SRC(sy + 2, sx + 2) * v_coeff_x3 * v_coeff_y3;
                    vst<eT, eN>(l_dst + oh * dst_w * c_blk + ow * c_blk, v_dst);
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

#undef SRC

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_RESIZE2D_NEON_RESIZE2D_NBCX_COMMON_H_
