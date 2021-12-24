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

ppl::common::RetCode gather_nd_ndarray_fp32(
    const float *src,
    const int64_t *indices,
    const int64_t *strides,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim,
    float *dst)
{
    if (inner_dim > 1) {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t k = 0; k < num_indices; ++k) {
            int64_t offset = 0;
            const int64_t *l_indices = indices + k * indices_dim;
            float *l_dst = dst + k * inner_dim;
            for (int64_t i = 0; i < indices_dim; ++i) {
                offset += l_indices[i] * strides[i];
            }
            memcpy(l_dst, src + offset, inner_dim * sizeof(float));
        }
    } else {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t k = 0; k < num_indices; ++k) {
            int64_t offset = 0;
            const int64_t *l_indices = indices + k * indices_dim;
            for (int64_t i = 0; i < indices_dim; ++i) {
                offset += l_indices[i] * strides[i];
            }
            dst[k] = src[offset];
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86
