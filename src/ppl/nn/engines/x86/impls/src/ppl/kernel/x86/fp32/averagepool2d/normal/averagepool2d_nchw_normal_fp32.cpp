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
#include "ppl/nn/params/onnx/pooling_param.h"

namespace ppl { namespace kernel { namespace x86 {

template <ppl::nn::onnx::PoolingParam::pooling_mode_t pooling_mode, bool ceil_mode>
static ppl::common::RetCode averagepool2d_nchw_normal_fp32_impl(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    float *dst)
{
    const int32_t batch    = src_shape->GetDim(0);
    const int32_t channels = src_shape->GetDim(1);
    const int32_t src_h    = src_shape->GetDim(2);
    const int32_t src_w    = src_shape->GetDim(3);
    const int32_t dst_h    = dst_shape->GetDim(2);
    const int32_t dst_w    = dst_shape->GetDim(3);
#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
#else
    PRAGMA_OMP_PARALLEL_FOR()
#endif
    for (int64_t bc = 0; bc < batch * channels; ++bc) {
        for (int64_t oh = 0; oh < dst_h; ++oh) {
            for (int64_t ow = 0; ow < dst_w; ++ow) {
                const float *p_src = src + bc * src_h * src_w;
                float *p_dst       = dst + bc * dst_h * dst_w;

                const int64_t padded_ihstart = oh * stride_h - pad_h;
                const int64_t padded_iwstart = ow * stride_w - pad_w;
                const int64_t padded_ihend   = ceil_mode ? padded_ihstart + kernel_h : min<int64_t>(padded_ihstart + kernel_h, src_h + pad_h);
                const int64_t padded_iwend   = ceil_mode ? padded_iwstart + kernel_w : min<int64_t>(padded_iwstart + kernel_w, src_w + pad_w);

                const int64_t ihstart = max<int64_t>(padded_ihstart, 0);
                const int64_t iwstart = max<int64_t>(padded_iwstart, 0);
                const int64_t ihend   = min<int64_t>(padded_ihend, src_h);
                const int64_t iwend   = min<int64_t>(padded_iwend, src_w);

                int64_t pool_len = 0;
                if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE) {
                    pool_len = (ihend - ihstart) * (iwend - iwstart);
                } else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
                    pool_len = (padded_ihend - padded_ihstart) * (padded_iwend - padded_iwstart);
                }

                if (pool_len <= 0) {
                    p_dst[oh * dst_w + ow] = 0.0f;
                } else {
                    float sum_val = 0.0f;
                    for (int64_t ih = ihstart; ih < ihend; ++ih) {
                        for (int64_t iw = iwstart; iw < iwend; ++iw) {
                            sum_val += p_src[ih * src_w + iw];
                        }
                    }
                    p_dst[oh * dst_w + ow] = sum_val / pool_len;
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode averagepool2d_nchw_normal_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,
    const int32_t ceil_mode,
    float *dst)
{
    if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE) {
        if (ceil_mode) {
            return averagepool2d_nchw_normal_fp32_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, true>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        } else {
            return averagepool2d_nchw_normal_fp32_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, false>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        }
    } else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
        if (ceil_mode) {
            return averagepool2d_nchw_normal_fp32_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, true>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        } else {
            return averagepool2d_nchw_normal_fp32_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, false>(src_shape, dst_shape, src, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dst);
        }
    }

    return ppl::common::RC_INVALID_VALUE;
}

}}}; // namespace ppl::kernel::x86
