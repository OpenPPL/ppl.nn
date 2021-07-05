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
#include <string.h>

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode scatter_nd_ndarray_fp32(
    const float *src,
    const float *updates,
    const int64_t *indices,
    const int32_t *strides,
    const int64_t src_length,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim,
    float *dst)
{
    const int64_t unroll_len  = 64;
    const int64_t unroll_body = round(src_length, unroll_len);
    const int64_t unroll_tail = src_length - unroll_len;

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        memcpy(dst + i, src + i, unroll_len * sizeof(float));
    }
    if (unroll_tail) {
        memcpy(dst + unroll_body, src + unroll_body, (src_length - unroll_body) * sizeof(float));
    }

    if (inner_dim > 1) {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t k = 0; k < num_indices; ++k) {
            int64_t offset           = 0;
            const int64_t *l_indices = indices + k * indices_dim;
            const float *l_updates   = updates + k * inner_dim;
            for (int64_t i = 0; i < indices_dim; ++i) {
                offset += l_indices[i] * strides[i];
            }
            memcpy(dst + offset, l_updates, inner_dim * sizeof(float));
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

}}} // namespace ppl::kernel::x86
