// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for arithmeticitional information
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

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/resize2d/neon//resize2d_ndarray_common.h"
#include "ppl/kernel/arm_server/resize2d/neon//resize2d_nbcx_common.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
static ppl::common::RetCode resize2d_wrapper(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    const float scale_h,
    const float scale_w,
    const int32_t coord_trans_mode,
    const int32_t mode,
    const float cubic_coeff_a,
    eT *dst)
{
    const auto data_format = src_shape->GetDataFormat();

    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        if (coord_trans_mode == 1 &&
            mode == 2) {
            return resize2d_ndarray_pytorch_cubic_floor_common<eT>(src_shape, dst_shape, src, scale_h, scale_w, cubic_coeff_a, dst);
        }
        if (coord_trans_mode == 1 &&
            mode == 1) {
            return resize2d_ndarray_pytorch_linear_floor_common<eT>(src_shape, dst_shape, src, scale_h, scale_w, dst);
        }
        if (coord_trans_mode == 3 &&
            mode == 0) {
            return resize2d_ndarray_asymmetric_nearest_floor_common<eT>(src_shape, dst_shape, src, scale_h, scale_w, dst);
        }
    }

    // NBCX
    if (std::is_same<eT, float>::value) {
        if (data_format == ppl::common::DATAFORMAT_N4CX) {
            if (coord_trans_mode == 1 &&
                mode == 2) {
                return resize2d_nbcx_pytorch_cubic_floor_common<float, 4>(src_shape, dst_shape, (const float *)src, scale_h, scale_w, cubic_coeff_a, (float *)dst);
            }
            if (coord_trans_mode == 1 &&
                mode == 1) {
                return resize2d_nbcx_pytorch_linear_floor_common<float, 4>(src_shape, dst_shape, (const float *)src, scale_h, scale_w, (float *)dst);
            }
            if (coord_trans_mode == 3 &&
                mode == 0) {
                return resize2d_nbcx_asymmetric_nearest_floor_common<float, 4>(src_shape, dst_shape, (const float *)src, scale_h, scale_w, (float *)dst);
            }
        }
    }
#ifdef PPLNN_USE_ARMV8_2_FP16
    if (std::is_same<eT, __fp16>::value) {
        if (data_format == ppl::common::DATAFORMAT_N8CX) {
            if (coord_trans_mode == 1 &&
                mode == 2) {
                return resize2d_nbcx_pytorch_cubic_floor_common<__fp16, 8>(src_shape, dst_shape, (const __fp16 *)src, scale_h, scale_w, cubic_coeff_a, (__fp16 *)dst);
            }
            if (coord_trans_mode == 1 &&
                mode == 1) {
                return resize2d_nbcx_pytorch_linear_floor_common<__fp16, 8>(src_shape, dst_shape, (const __fp16 *)src, scale_h, scale_w, (__fp16 *)dst);
            }
            if (coord_trans_mode == 3 &&
                mode == 0) {
                return resize2d_nbcx_asymmetric_nearest_floor_common<__fp16, 8>(src_shape, dst_shape, (const __fp16 *)src, scale_h, scale_w, (__fp16 *)dst);
            }
        }
    }
#endif

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode resize2d(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const float scale_h,
    const float scale_w,
    const int32_t coord_trans_mode,
    const int32_t mode,
    const float cubic_coeff_a,
    void *dst)
{
    const auto data_type = src_shape->GetDataType();
    switch (data_type) {
        case ppl::common::DATATYPE_FLOAT32: return resize2d_wrapper<float>(src_shape, dst_shape, (const float *)src, scale_h, scale_w, coord_trans_mode, mode, cubic_coeff_a, (float *)dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return resize2d_wrapper<__fp16>(src_shape, dst_shape, (const __fp16 *)src, scale_h, scale_w, coord_trans_mode, mode, cubic_coeff_a, (__fp16 *)dst);
#endif
        default: break;
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}}}; // namespace ppl::kernel::arm_server::neon
