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
#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
static ppl::common::RetCode abs_common(
    const ppl::common::TensorShape* src_shape,
    const eT* src,
    eT* dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w      = sizeof(vecType) / sizeof(eT);
    const int64_t length      = src_shape->CalcElementsIncludingPadding();
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        const vecType v_src_0 = vld<eT, eN>(src + i + simd_w * 0);
        const vecType v_src_1 = vld<eT, eN>(src + i + simd_w * 1);
        const vecType v_src_2 = vld<eT, eN>(src + i + simd_w * 2);
        const vecType v_src_3 = vld<eT, eN>(src + i + simd_w * 3);

        const vecType v_dst_0 = vabs<vecType>(v_src_0);
        const vecType v_dst_1 = vabs<vecType>(v_src_1);
        const vecType v_dst_2 = vabs<vecType>(v_src_2);
        const vecType v_dst_3 = vabs<vecType>(v_src_3);

        vst<eT, eN>(dst + i + simd_w * 0, v_dst_0);
        vst<eT, eN>(dst + i + simd_w * 1, v_dst_1);
        vst<eT, eN>(dst + i + simd_w * 2, v_dst_2);
        vst<eT, eN>(dst + i + simd_w * 3, v_dst_3);
    }
    for (int64_t i = unroll_body; i < length; i++) {
        dst[i] = src[i] >= 0 ? src[i] : -src[i];
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode abs(
    const ppl::common::TensorShape* src_shape,
    const void* src,
    void* dst)
{
    const auto data_type = src_shape->GetDataType();

    switch (data_type) {
        case ppl::common::DATATYPE_FLOAT32: return abs_common<float>(src_shape, (const float*)src, (float*)dst);
        case ppl::common::DATATYPE_INT64: return abs_common<int64_t>(src_shape, (const int64_t*)src, (int64_t*)dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return abs_common<__fp16>(src_shape, (const __fp16*)src, (__fp16*)dst);
#endif
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}}; // namespace ppl::kernel::arm_server::neon
