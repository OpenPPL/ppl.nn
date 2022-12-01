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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_RESIZE2D_RESIZE2D_NBCX_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_RESIZE2D_RESIZE2D_NBCX_COMMON_H_

#include <math.h>
#include <vector>
#include <riscv-vector.h>
#include <type_traits>

#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

template <typename eT, typename T>
inline eT vlev(const T* addr, uint64_t n);
template <>
inline float16xm1_t vlev<float16xm1_t, __fp16>(const __fp16* addr, uint64_t n)
{
    return vlev_float16xm1(addr, n);
}
template <>
inline float32xm1_t vlev<float32xm1_t, float>(const float* addr, uint64_t n)
{
    return vlev_float32xm1(addr, n);
}

template <typename eT, typename T>
inline void vsev(T* addr, eT va, uint64_t n);
template <>
inline void vsev<float16xm1_t, __fp16>(__fp16* addr, float16xm1_t va, uint64_t n)
{
    return vsev_float16xm1(addr, va, n);
}
template <>
inline void vsev<float32xm1_t, float>(float* addr, float32xm1_t va, uint64_t n)
{
    return vsev_float32xm1(addr, va, n);
}

template <typename eT, typename T>
inline eT vfmulvf(eT va, T b, uint64_t n);
template <>
inline float16xm1_t vfmulvf<float16xm1_t, __fp16>(float16xm1_t va, __fp16 b, uint64_t n)
{
    return vfmulvf_float16xm1(va, b, n);
}
template <>
inline float32xm1_t vfmulvf<float32xm1_t, float>(float32xm1_t va, float b, uint64_t n)
{
    return vfmulvf_float32xm1(va, b, n);
}

template <typename eT, typename T>
inline eT vfaddvv(eT va, eT vb, uint64_t n);
template <>
inline float16xm1_t vfaddvv<float16xm1_t, __fp16>(float16xm1_t va, float16xm1_t vb, uint64_t n)
{
    return vfaddvv_float16xm1(va, vb, n);
}
template <>
inline float32xm1_t vfaddvv<float32xm1_t, float>(float32xm1_t va, float32xm1_t vb, uint64_t n)
{
    return vfaddvv_float32xm1(va, vb, n);
}

template <typename eT, typename T, int32_t c_blk>
ppl::common::RetCode resize2d_nbcx_pytorch_linear_floor_common(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const T* src,
    const float scale_h,
    const float scale_w,
    T* dst)
{
    uint64_t vl;
    if (std::is_same<T, float>::value) {
        vl = vsetvli(c_blk, RVV_E32, RVV_M1);
    } else if (std::is_same<T, __fp16>::value) {
        vl = vsetvli(c_blk, RVV_E16, RVV_M1);
    }

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);
    const int64_t padded_c = round_up(channels, c_blk);
    const float hscale     = 1.0f / scale_h;
    const float wscale     = 1.0f / scale_w;

    std::vector<int32_t> w0_tab(dst_w);
    std::vector<int32_t> w1_tab(dst_w);
    std::vector<float> w0_lambda_tab(dst_w);
    std::vector<float> w1_lambda_tab(dst_w);
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

    for (int64_t bc = 0; bc < batch * padded_c; bc += c_blk) {
        const T* src_ = src + bc * src_h * src_w;
        T* dst_       = dst + bc * dst_h * dst_w;
        for (int64_t oh = 0; oh < dst_h; ++oh) {
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

                eT v_data0 = vlev<eT, T>(src_ + (h0 * src_w + w0) * c_blk, vl);
                eT v_data1 = vlev<eT, T>(src_ + (h0 * src_w + w1) * c_blk, vl);
                eT v_data2 = vlev<eT, T>(src_ + (h1 * src_w + w0) * c_blk, vl);
                eT v_data3 = vlev<eT, T>(src_ + (h1 * src_w + w1) * c_blk, vl);

                v_data0 = vfmulvf<eT, T>(v_data0, (T)(h0_lambda * w0_lambda), vl);
                v_data1 = vfmulvf<eT, T>(v_data1, (T)(h0_lambda * w1_lambda), vl);
                v_data2 = vfmulvf<eT, T>(v_data2, (T)(h1_lambda * w0_lambda), vl);
                v_data3 = vfmulvf<eT, T>(v_data3, (T)(h1_lambda * w1_lambda), vl);

                eT v_dst =
                    vfaddvv<eT, T>(vfaddvv<eT, T>(v_data0, v_data1, vl), vfaddvv<eT, T>(v_data2, v_data3, vl), vl);
                vsev<eT, T>(dst_ + (oh * dst_w + ow) * c_blk, v_dst, vl);
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename eT, typename T, int32_t c_blk>
ppl::common::RetCode resize2d_nbcx_asymmetric_nearest_floor_common(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const T* src,
    const float scale_h,
    const float scale_w,
    T* dst)
{
    uint64_t vl;
    if (std::is_same<T, float>::value) {
        vl = vsetvli(c_blk, RVV_E32, RVV_M1);
    } else if (std::is_same<T, __fp16>::value) {
        vl = vsetvli(c_blk, RVV_E16, RVV_M1);
    }

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);
    const int64_t padded_c = round_up(channels, c_blk);
    const float hscale     = 1.0f / scale_h;
    const float wscale     = 1.0f / scale_w;

    std::vector<int64_t> iw_list(dst_w);
    for (int64_t i = 0; i < dst_w; i++) {
        iw_list[i] = static_cast<int64_t>(i * wscale);
    }

    for (int64_t bc = 0; bc < batch * padded_c; bc += c_blk) {
        const T* src_ = src + bc * src_h * src_w;
        T* dst_       = dst + bc * dst_h * dst_w;
        for (int64_t oh = 0; oh < dst_h; ++oh) {
            int64_t ih     = static_cast<int64_t>(oh * hscale);
            const T* src_p = src_ + ih * src_w * c_blk;
            T* dst_p       = dst_ + oh * dst_w * c_blk;
            int64_t ow     = 0;
            for (; ow + 8 < dst_w; ow += 8) {
                vsev<eT, T>(dst_p + (ow + 0) * c_blk, vlev<eT, T>(src_p + iw_list[ow + 0] * c_blk, vl), vl);
                vsev<eT, T>(dst_p + (ow + 1) * c_blk, vlev<eT, T>(src_p + iw_list[ow + 1] * c_blk, vl), vl);
                vsev<eT, T>(dst_p + (ow + 2) * c_blk, vlev<eT, T>(src_p + iw_list[ow + 2] * c_blk, vl), vl);
                vsev<eT, T>(dst_p + (ow + 3) * c_blk, vlev<eT, T>(src_p + iw_list[ow + 3] * c_blk, vl), vl);
                vsev<eT, T>(dst_p + (ow + 4) * c_blk, vlev<eT, T>(src_p + iw_list[ow + 4] * c_blk, vl), vl);
                vsev<eT, T>(dst_p + (ow + 5) * c_blk, vlev<eT, T>(src_p + iw_list[ow + 5] * c_blk, vl), vl);
                vsev<eT, T>(dst_p + (ow + 6) * c_blk, vlev<eT, T>(src_p + iw_list[ow + 6] * c_blk, vl), vl);
                vsev<eT, T>(dst_p + (ow + 7) * c_blk, vlev<eT, T>(src_p + iw_list[ow + 7] * c_blk, vl), vl);
            }
            for (; ow < dst_w; ow++) {
                vsev<eT, T>(dst_p + ow * c_blk, vlev<eT, T>(src_p + iw_list[ow] * c_blk, vl), vl);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_COMMON_RESIZE2D_RESIZE2D_NBCX_COMMON_H_
