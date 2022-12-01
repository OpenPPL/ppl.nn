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

#include <riscv-vector.h>
#include "ppl/kernel/riscv/common/internal_include.h"

#include "ppl/kernel/riscv/common/mmcv_gridsample/mmcv_gridsample_common.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() 8

template <bool align_corners, grid_sample_padding padding_mode>
ppl::common::RetCode mmcv_gridsample_bilinear_n8cx_fp16_kernel(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* grid_shape,
    const __fp16* src,
    const float* grid,
    __fp16* dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = grid_shape->GetDim(1);
    const int64_t dst_w    = grid_shape->GetDim(2);

    const int64_t padded_channels = round_up(channels, C_BLK());
    const auto vl                 = vsetvli(C_BLK(), RVV_E16, RVV_M1);
    float16xm1_t vfzero           = vfmvvf_float16xm1((__fp16)0.0f, vl);

    for (int64_t n = 0; n < batch; n++) {
        const float* grid_ = grid + n * dst_h * dst_w * 2;
        const __fp16* src_ = src + n * padded_channels * src_h * src_w;
        __fp16* dst_       = dst + n * padded_channels * dst_h * dst_w;
        for (int64_t c = 0; c < padded_channels; c += C_BLK()) {
            const __fp16* src_p = src_ + c * src_h * src_w;
            __fp16* dst_p       = dst_ + c * dst_h * dst_w;
            for (int64_t h = 0; h < dst_h; h++) {
                for (int64_t w = 0; w < dst_w; w++) {
                    float grid_x = grid_[(h * dst_w + w) * 2 + 0];
                    float grid_y = grid_[(h * dst_w + w) * 2 + 1];
                    float ix     = compute_location<float, align_corners, padding_mode>(grid_x, src_w);
                    float iy     = compute_location<float, align_corners, padding_mode>(grid_y, src_h);

                    int64_t x_w = std::floor(ix);
                    int64_t y_n = std::floor(iy);
                    int64_t x_e = x_w + 1;
                    int64_t y_s = y_n + 1;

                    float w_l = ix - x_w;
                    float e_l = 1. - w_l;
                    float n_l = iy - y_n;
                    float s_l = 1. - n_l;

                    float wgt_nw = s_l * e_l;
                    float wgt_ne = s_l * w_l;
                    float wgt_sw = n_l * e_l;
                    float wgt_se = n_l * w_l;

                    float16xm1_t nw_val = within_bounds_2d(y_n, x_w, src_h, src_w) ? vfmulvf_float16xm1(vlev_float16xm1(src_p + (y_n * src_w + x_w) * C_BLK(), vl), (__fp16)wgt_nw, vl) : vfzero;
                    float16xm1_t ne_val = within_bounds_2d(y_n, x_e, src_h, src_w) ? vfmulvf_float16xm1(vlev_float16xm1(src_p + (y_n * src_w + x_e) * C_BLK(), vl), (__fp16)wgt_ne, vl) : vfzero;
                    float16xm1_t sw_val = within_bounds_2d(y_s, x_w, src_h, src_w) ? vfmulvf_float16xm1(vlev_float16xm1(src_p + (y_s * src_w + x_w) * C_BLK(), vl), (__fp16)wgt_sw, vl) : vfzero;
                    float16xm1_t se_val = within_bounds_2d(y_s, x_e, src_h, src_w) ? vfmulvf_float16xm1(vlev_float16xm1(src_p + (y_s * src_w + x_e) * C_BLK(), vl), (__fp16)wgt_se, vl) : vfzero;

                    float16xm1_t out_val = vfaddvv_float16xm1(nw_val, ne_val, vl);
                    out_val              = vfaddvv_float16xm1(out_val, sw_val, vl);
                    out_val              = vfaddvv_float16xm1(out_val, se_val, vl);

                    vsev_float16xm1(dst_p + (h * dst_w + w) * C_BLK(), out_val, vl);
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode mmcv_gridsample_bilinear_n8cx_fp16(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* grid_shape,
    const __fp16* src,
    const float* grid,
    const bool align_corners,
    const int64_t padding_mode,
    __fp16* dst)
{
    if (align_corners) {
        if (ZEROS == padding_mode) {
            return mmcv_gridsample_bilinear_n8cx_fp16_kernel<true, ZEROS>(src_shape, grid_shape, src, grid, dst);
        } else if (BORDER == padding_mode) {
            return mmcv_gridsample_bilinear_n8cx_fp16_kernel<true, BORDER>(src_shape, grid_shape, src, grid, dst);
        } else if (REFLECTION == padding_mode) {
            return mmcv_gridsample_bilinear_n8cx_fp16_kernel<true, REFLECTION>(src_shape, grid_shape, src, grid, dst);
        }
    } else {
        if (ZEROS == padding_mode) {
            return mmcv_gridsample_bilinear_n8cx_fp16_kernel<false, ZEROS>(src_shape, grid_shape, src, grid, dst);
        } else if (BORDER == padding_mode) {
            return mmcv_gridsample_bilinear_n8cx_fp16_kernel<false, BORDER>(src_shape, grid_shape, src, grid, dst);
        } else if (REFLECTION == padding_mode) {
            return mmcv_gridsample_bilinear_n8cx_fp16_kernel<false, REFLECTION>(src_shape, grid_shape, src, grid, dst);
        }
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}; // namespace ppl::kernel::riscv