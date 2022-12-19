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

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, bool fuse_relu>
static ppl::common::RetCode batchnorm_ndarray_common(
    const ppl::common::TensorShape *src_shape,
    const eT *src,
    const eT *mean,
    const eT *variance,
    const eT *scale,
    const eT *shift,
    const float var_eps,
    eT *dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDimCount() > 1 ? src_shape->GetDim(1) : 1;
    int64_t inner_dims     = 1;
    for (uint32_t i = 2; i < src_shape->GetDimCount(); ++i) {
        inner_dims *= src_shape->GetDim(i);
    }

    const int64_t simd_w      = eN;
    const int64_t unroll_len  = simd_w * 2;
    const int64_t unroll_body = round(inner_dims, unroll_len);

    const vecType v_zero = vdup_n<eT, eN>(0);

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const float mean_var    = mean[c];
            const float shift_var   = shift[c];
            const float var_rcp_var = scale[c] / sqrtf(variance[c] + var_eps);
            const vecType v_mean    = vdup_n<eT, eN>(mean_var);
            const vecType v_shift   = vdup_n<eT, eN>(shift_var);
            const vecType v_var_rcp = vdup_n<eT, eN>(var_rcp_var);

            const eT *p_src = src + n * channels * inner_dims + c * inner_dims;
            eT *p_dst       = dst + n * channels * inner_dims + c * inner_dims;
            for (int64_t i = 0; i < unroll_body; i += unroll_len) {
                vecType v_src_0 = vld<eT, eN>(p_src + i + simd_w * 0);
                vecType v_src_1 = vld<eT, eN>(p_src + i + simd_w * 1);
                vecType v_dst_0 = vadd<vecType>(vmul<vecType>(vsub<vecType>(v_src_0, v_mean), v_var_rcp), v_shift);
                vecType v_dst_1 = vadd<vecType>(vmul<vecType>(vsub<vecType>(v_src_1, v_mean), v_var_rcp), v_shift);
                if (fuse_relu) {
                    v_dst_0 = vmax<vecType>(v_dst_0, v_zero);
                    v_dst_1 = vmax<vecType>(v_dst_1, v_zero);
                }
                vst<eT, eN>(p_dst + i + simd_w * 0, v_dst_0);
                vst<eT, eN>(p_dst + i + simd_w * 1, v_dst_1);
            }
            for (int64_t i = unroll_body; i < inner_dims; i++) {
                eT src_val = p_src[i];
                eT dst_val = (src_val - mean_var) * var_rcp_var + shift_var;
                if (fuse_relu) {
                    dst_val = max(dst_val, (eT)0);
                }
                p_dst[i] = dst_val;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, int32_t c_blk, bool fuse_relu>
static ppl::common::RetCode batchnorm_nbcx_common(
    const ppl::common::TensorShape *src_shape,
    const eT *src,
    const eT *mean,
    const eT *variance,
    const eT *scale,
    const eT *shift,
    const float var_eps,
    eT *dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t height   = src_shape->GetDim(2);
    const int64_t width    = src_shape->GetDim(3);

    const int64_t pad_c      = round_up(channels, c_blk);
    const int64_t inner_dims = height * width;

    const vecType v_zero = vdup_n<eT, eN>(0);
    const vecType v_eps  = vdup_n<eT, eN>((eT)var_eps);

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < pad_c; c += c_blk) {
            const int64_t c_eff = min(channels - c, (int64_t)c_blk);
            vecType v_mean;
            vecType v_shift;
            vecType v_scale;
            vecType v_var;
            if (c_eff == c_blk) {
                v_mean  = vld<eT, eN>(mean + c);
                v_shift = vld<eT, eN>(shift + c);
                v_scale = vld<eT, eN>(scale + c);
                v_var   = vld<eT, eN>(variance + c);
            } else {
                eT temp_mean[c_blk];
                eT temp_shift[c_blk];
                eT temp_scale[c_blk];
                eT temp_var[c_blk];
                for (int64_t i = 0; i < c_eff; i++) {
                    temp_mean[i]  = mean[i + c];
                    temp_shift[i] = shift[i + c];
                    temp_scale[i] = scale[i + c];
                    temp_var[i]   = variance[i + c];
                }
                for (int64_t i = c_eff; i < c_blk; i++) {
                    temp_mean[i]  = 0;
                    temp_shift[i] = 0;
                    temp_scale[i] = 0;
                    temp_var[i]   = 0;
                }
                v_mean  = vld<eT, eN>(temp_mean);
                v_shift = vld<eT, eN>(temp_shift);
                v_scale = vld<eT, eN>(temp_scale);
                v_var   = vld<eT, eN>(temp_var);
            }
            const vecType v_var_rcp = vdiv<vecType>(v_scale, vsqrt<vecType>(vadd<vecType>(v_var, v_eps)));

            const eT *p_src = src + n * pad_c * inner_dims + c * inner_dims;
            eT *p_dst       = dst + n * pad_c * inner_dims + c * inner_dims;

            for (int64_t i = 0; i < inner_dims; i++) {
                vecType v_src = vld<eT, eN>(p_src + i * c_blk);
                vecType v_dst = vadd<vecType>(vmul<vecType>(vsub<vecType>(v_src, v_mean), v_var_rcp), v_shift);
                if (fuse_relu) {
                    v_dst = vmax<vecType>(v_dst, v_zero);
                }
                vst<eT, eN>(p_dst + i * c_blk, v_dst);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, bool fuse_relu>
static ppl::common::RetCode batchnorm_wrapper(
    const ppl::common::TensorShape *src_shape,
    const void *src,
    const void *mean,
    const void *variance,
    const void *scale,
    const void *shift,
    const float var_eps,
    void *dst)
{
    const auto data_format = src_shape->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return batchnorm_ndarray_common<eT, fuse_relu>(src_shape, (const eT *)src, (const eT *)mean, (const eT *)variance, (const eT *)scale, (const eT *)shift, var_eps, (eT *)dst);
    }

    // NBCX
    if (std::is_same<eT, float>::value) {
        if (data_format == ppl::common::DATAFORMAT_N4CX) { // fp32 n4cx
            return batchnorm_nbcx_common<float, 4, fuse_relu>(src_shape, (const float *)src, (const float *)mean, (const float *)variance, (const float *)scale, (const float *)shift, var_eps, (float *)dst);
        }
    }
#ifdef PPLNN_USE_ARMV8_2_FP16
    if (std::is_same<eT, __fp16>::value) {
        if (data_format == ppl::common::DATAFORMAT_N8CX) { // fp16 n8cx
            return batchnorm_nbcx_common<__fp16, 8, fuse_relu>(src_shape, (const __fp16 *)src, (const __fp16 *)mean, (const __fp16 *)variance, (const __fp16 *)scale, (const __fp16 *)shift, var_eps, (__fp16 *)dst);
        }
    }
#endif

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode batchnorm(
    const ppl::common::TensorShape *src_shape,
    const void *src,
    const void *mean,
    const void *variance,
    const void *scale,
    const void *shift,
    const float var_eps,
    const bool fuse_relu,
    void *dst)
{
    const auto data_type = src_shape->GetDataType();
    if (fuse_relu) {
        switch (data_type) {
            case ppl::common::DATATYPE_FLOAT32: return batchnorm_wrapper<float, true>(src_shape, src, mean, variance, scale, shift, var_eps, dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
            case ppl::common::DATATYPE_FLOAT16: return batchnorm_wrapper<__fp16, true>(src_shape, src, mean, variance, scale, shift, var_eps, dst);
#endif
            default: break;
        }
    } else {
        switch (data_type) {
            case ppl::common::DATATYPE_FLOAT32: return batchnorm_wrapper<float, false>(src_shape, src, mean, variance, scale, shift, var_eps, dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
            case ppl::common::DATATYPE_FLOAT16: return batchnorm_wrapper<__fp16, false>(src_shape, src, mean, variance, scale, shift, var_eps, dst);
#endif
            default: break;
        }
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}} // namespace ppl::kernel::arm_server::neon
