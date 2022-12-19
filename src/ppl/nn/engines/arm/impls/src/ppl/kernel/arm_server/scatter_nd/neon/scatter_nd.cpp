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

#include "ppl/kernel/arm_server/common/internal_include.h"
#include <string.h>

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
ppl::common::RetCode scatter_nd_ndarray_commmon(
    const eT *src,
    const eT *updates,
    const int64_t *indices,
    const int32_t *strides,
    const int64_t src_length,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim,
    eT *dst)
{
    const int64_t unroll_len  = 64;
    const int64_t unroll_body = round(src_length, unroll_len);
    const int64_t unroll_tail = src_length - unroll_len;

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        memcpy(dst + i, src + i, unroll_len * sizeof(eT));
    }
    if (unroll_tail) {
        memcpy(dst + unroll_body, src + unroll_body, (src_length - unroll_body) * sizeof(eT));
    }

    if (inner_dim > 1) {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t k = 0; k < num_indices; ++k) {
            int64_t offset           = 0;
            const int64_t *l_indices = indices + k * indices_dim;
            const eT *l_updates   = updates + k * inner_dim;
            for (int64_t i = 0; i < indices_dim; ++i) {
                offset += l_indices[i] * strides[i];
            }
            memcpy(dst + offset, l_updates, inner_dim * sizeof(eT));
        }
    } else {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t k = 0; k < num_indices; ++k) {
            int64_t offset           = 0;
            const int64_t *l_indices = indices + k * indices_dim;
            for (int64_t i = 0; i < indices_dim; ++i) {
                offset += l_indices[i] * strides[i];
            }
            dst[offset] = updates[k];
        }
    }

    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode scatter_nd_ndarray(
    const ppl::common::TensorShape *input_shape,
    const void *src,
    const void *updates,
    const int64_t *indices,
    const int32_t *strides,
    const int64_t src_length,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim,
    void *dst)
{
    const auto data_type   = input_shape->GetDataType();
    const auto data_format = input_shape->GetDataFormat();

    if (data_format != ppl::common::DATAFORMAT_NDARRAY) {
        return ppl::common::RC_UNSUPPORTED;
    }

    switch (data_type)
    {
        case ppl::common::DATATYPE_FLOAT32: return scatter_nd_ndarray_commmon<float>((const float*)src, (const float*)updates, indices, 
                                                                                        strides, src_length, inner_dim, num_indices, 
                                                                                        indices_dim, (float*)dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return scatter_nd_ndarray_commmon<__fp16>((const __fp16*)src, (const __fp16*)updates, indices,
                                                                                        strides, src_length, inner_dim, num_indices,
                                                                                        indices_dim, (__fp16*)dst);
#endif
        default:break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}} // namespace ppl::kernel::arm_server::neon
