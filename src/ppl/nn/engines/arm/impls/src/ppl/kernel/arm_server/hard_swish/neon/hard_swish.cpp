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
static ppl::common::RetCode hard_swish_common(
    const ppl::common::TensorShape* src_shape,
    const eT* src,
    eT* dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w    = sizeof(vecType) / sizeof(eT);
    const int64_t length    = src_shape->CalcElementsIncludingPadding();
    const int64_t simd_body = round(length, simd_w);

    eT r6 = 1.0f / 6.0f;

    vecType v_0  = vdup_n<eT, eN>(0);
    vecType v_3  = vdup_n<eT, eN>(3.0f);
    vecType v_6  = vdup_n<eT, eN>(6.0f);
    vecType v_r6 = vdup_n<eT, eN>(1.0f / 6.0f);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < simd_body; i += simd_w) {
        vecType v_data = vld<eT, eN>(src + i);
        vecType v_temp = v_data + v_3;
        v_temp         = vmax<vecType>(v_temp, v_0);
        v_temp         = vmin<vecType>(v_temp, v_6);
        v_temp         = v_temp * v_r6;
        v_data         = v_data * v_temp;
        vst<eT, eN>(dst + i, v_data);
    }
    for (int64_t i = simd_body; i < length; i++) {
        eT data = src[i];
        eT temp = data + 3.0f;
        temp    = max(temp, (eT)0);
        temp    = min(temp, (eT)6.0f);
        temp *= r6;
        data *= temp;
        dst[i] = data;
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode hard_swish(
    const ppl::common::TensorShape* src_shape,
    const void* src,
    void* dst)
{
    const auto data_type = src_shape->GetDataType();

    switch (data_type) {
        case ppl::common::DATATYPE_FLOAT32: return hard_swish_common<float>(src_shape, (const float*)src, (float*)dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return hard_swish_common<__fp16>(src_shape, (const __fp16*)src, (__fp16*)dst);
#endif
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}}; // namespace ppl::kernel::arm_server::neon
