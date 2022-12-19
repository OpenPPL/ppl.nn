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

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
static ppl::common::RetCode argmax_ndarray_common(
    const ppl::common::TensorShape *src_shape,
    const eT *src,
    const int64_t axis,
    int64_t *dst)
{
    eT numeric_min_val      = numeric_min<eT>();
    const int64_t real_axis = (axis + src_shape->GetDimCount()) % src_shape->GetDimCount();

    const int64_t argmax_dim = src_shape->GetDim(real_axis);
    int64_t outer_dim        = 1;
    int64_t inner_dim        = 1;
    for (uint32_t i = 0; i < real_axis; i++) {
        outer_dim *= src_shape->GetDim(i);
    }
    for (uint32_t i = real_axis + 1; i < src_shape->GetDimCount(); i++) {
        inner_dim *= src_shape->GetDim(i);
    }

#ifndef PPL_USE_ARM_SERVER_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t i = 0; i < outer_dim; ++i) {
        for (int64_t j = 0; j < inner_dim; ++j) {
            eT max_value = numeric_min_val;
            int64_t idx  = 0;
            for (int64_t k = 0; k < argmax_dim; ++k) {
                if (src[(i * argmax_dim + k) * inner_dim + j] > max_value) {
                    max_value = src[(i * argmax_dim + k) * inner_dim + j];
                    idx       = k;
                }
            }
            dst[i * inner_dim + j] = idx;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode argmax(
    const ppl::common::TensorShape *src_shape,
    const void *src,
    const int64_t axis,
    int64_t *dst)
{
    const auto data_type   = src_shape->GetDataType();
    const auto data_format = src_shape->GetDataFormat();
    if (data_format != ppl::common::DATAFORMAT_NDARRAY) {
        return ppl::common::RC_UNSUPPORTED;
    }

    switch (data_type) {
        case ppl::common::DATATYPE_FLOAT32: return argmax_ndarray_common<float>(src_shape, (const float *)src, axis, dst);
        case ppl::common::DATATYPE_INT64: return argmax_ndarray_common<int64_t>(src_shape, (const int64_t *)src, axis, dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return argmax_ndarray_common<__fp16>(src_shape, (const __fp16 *)src, axis, dst);
#endif
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}}; // namespace ppl::kernel::arm_server::neon
