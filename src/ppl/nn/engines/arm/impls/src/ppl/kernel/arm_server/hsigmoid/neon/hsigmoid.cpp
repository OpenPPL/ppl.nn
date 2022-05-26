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
static ppl::common::RetCode hsigmoid_common(
    const ppl::nn::TensorShape* src_shape,
    const eT* src,
    eT* dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w    = sizeof(vecType) / sizeof(eT);
    const int64_t length    = src_shape->GetElementsIncludingPadding();
    const int64_t simd_body = round(length, simd_w);

    const eT r6 = 1.0f / 6.0f;

    const vecType v_0  = vdup_n<eT, eN>(0);
    const vecType v_1  = vdup_n<eT, eN>(1.0f);
    const vecType v_r6 = vdup_n<eT, eN>(1.0f / 6.0f);
    const vecType v_05 = vdup_n<eT, eN>(0.5f);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < simd_body; i += simd_w) {
        vecType v_data = vld<eT, eN>(src + i);
        v_data         = v_data * v_r6 + v_05;
        v_data         = vmin<vecType>(v_data, v_1);
        v_data         = vmax<vecType>(v_data, v_0);
        vst<eT, eN>(dst + i, v_data);
    }
    for (int64_t i = simd_body; i < length; i++) {
        eT data = src[i];
        data    = data * r6 + 0.5f;
        data    = min(data, (eT)1.0f);
        data    = max(data, (eT)0);
        dst[i]  = data;
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode hsigmoid(
    const ppl::nn::TensorShape* src_shape,
    const void* src,
    void* dst)
{
    const auto data_type = src_shape->GetDataType();

    switch (data_type) {
        case ppl::common::DATATYPE_FLOAT32: return hsigmoid_common<float>(src_shape, (const float*)src, (float*)dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return hsigmoid_common<__fp16>(src_shape, (const __fp16*)src, (__fp16*)dst);
#endif
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}}; // namespace ppl::kernel::arm_server::neon
