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

#include <limits>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
static ppl::common::RetCode clip_common(
    const ppl::common::TensorShape *src_shape,
    const eT *src,
    const eT *min_ptr,
    const eT *max_ptr,
    const eT min_val,
    const eT max_val,
    eT *dst)
{
    eT clip_min = min_ptr == nullptr ? min_val : min_ptr[0];
    eT clip_max = max_ptr == nullptr ? max_val : max_ptr[0];

    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w      = sizeof(vecType) / sizeof(eT);
    const int64_t length      = src_shape->CalcElementsIncludingPadding();
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    const vecType v_clip_min = vdup_n<eT, eN>(clip_min);
    const vecType v_clip_max = vdup_n<eT, eN>(clip_max);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        vecType v_src_0 = vld<eT, eN>(src + i + simd_w * 0);
        vecType v_src_1 = vld<eT, eN>(src + i + simd_w * 1);
        vecType v_src_2 = vld<eT, eN>(src + i + simd_w * 2);
        vecType v_src_3 = vld<eT, eN>(src + i + simd_w * 3);

        vecType v_dst_0 = vmin<vecType>(vmax<vecType>(v_src_0, v_clip_min), v_clip_max);
        vecType v_dst_1 = vmin<vecType>(vmax<vecType>(v_src_1, v_clip_min), v_clip_max);
        vecType v_dst_2 = vmin<vecType>(vmax<vecType>(v_src_2, v_clip_min), v_clip_max);
        vecType v_dst_3 = vmin<vecType>(vmax<vecType>(v_src_3, v_clip_min), v_clip_max);

        vst<eT, eN>(dst + i + simd_w * 0, v_dst_0);
        vst<eT, eN>(dst + i + simd_w * 1, v_dst_1);
        vst<eT, eN>(dst + i + simd_w * 2, v_dst_2);
        vst<eT, eN>(dst + i + simd_w * 3, v_dst_3);
    }
    for (int64_t i = unroll_body; i < length; i++) {
        dst[i] = min(max(src[i], clip_min), clip_max);
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode clip(
    const ppl::common::TensorShape *src_shape,
    const void *src,
    const void *min_ptr,
    const void *max_ptr,
    const float min_val,
    const float max_val,
    void *dst)
{
    const auto data_type = src_shape->GetDataType();
    switch (data_type) {
        case ppl::common::DATATYPE_FLOAT32: return clip_common<float>(src_shape, (const float *)src, (const float *)min_ptr, (const float *)max_ptr, min_val, max_val, (float *)dst);
        case ppl::common::DATATYPE_INT64: return clip_common<int64_t>(src_shape, (const int64_t *)src, (const int64_t *)min_ptr, (const int64_t *)max_ptr, static_cast<int64_t>(min_val), static_cast<int64_t>(max_val), (int64_t *)dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return clip_common<__fp16>(src_shape, (const __fp16 *)src, (const __fp16 *)min_ptr, (const __fp16 *)max_ptr, static_cast<__fp16>(min_val), static_cast<__fp16>(max_val), (__fp16 *)dst);
#endif
        default: break;
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}}} // namespace ppl::kernel::arm_server::neon
