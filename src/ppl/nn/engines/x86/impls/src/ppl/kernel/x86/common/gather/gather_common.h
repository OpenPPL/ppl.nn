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

#ifndef __ST_PPL_KERNEL_X86_COMMON_GATHER_GATHER_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_GATHER_GATHER_COMMON_H_

#include "ppl/kernel/x86/common/internal_include.h"
#include <string.h> 

namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
ppl::common::RetCode gather_ndarray_common(
    const eT *src,
    const int64_t *indices,
    const int64_t outer_dim,
    const int64_t gather_dim,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim,
    eT *dst)
{
    if (inner_dim >= 4) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                for (int64_t i = 0; i < indices_dim; ++i) {
                    eT *l_dst = dst + o * num_indices * indices_dim * inner_dim +
                               k * indices_dim * inner_dim + i * inner_dim;
                    int64_t index  = indices[k * indices_dim + i];
                    const eT *l_src = src + o * gather_dim * inner_dim + index * inner_dim;
                    memcpy(l_dst, l_src, inner_dim * sizeof(eT));
                }
            }
        }
    } else if (inner_dim >= 2) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                eT *l_dst =
                    dst + o * num_indices * indices_dim * inner_dim + k * indices_dim * inner_dim;
                const int64_t *l_indices = indices + k * indices_dim;
                const eT *l_src           = src + o * gather_dim * inner_dim;
                if (inner_dim == 2) {
                    for (int64_t i = 0; i < indices_dim; ++i) {
                        l_dst[0] = l_src[l_indices[0] * 2 + 0];
                        l_dst[1] = l_src[l_indices[0] * 2 + 1];
                        l_dst += inner_dim;
                        ++l_indices;
                    }
                } else {
                    for (int64_t i = 0; i < indices_dim; ++i) {
                        l_dst[0] = l_src[l_indices[0] * 3 + 0];
                        l_dst[1] = l_src[l_indices[0] * 3 + 1];
                        l_dst[2] = l_src[l_indices[0] * 3 + 2];
                        l_dst += inner_dim;
                        ++l_indices;
                    }
                }
            }
        }
    } else {
#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                eT *l_dst                 = dst + o * num_indices * indices_dim + k * indices_dim;
                const int64_t *l_indices = indices + k * indices_dim;
                const eT *l_src           = src + o * gather_dim;
                for (int64_t i = 0; i < indices_dim; ++i) {
                    l_dst[0] = l_src[l_indices[0]];
                    ++l_dst;
                    ++l_indices;
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_GATHER_GATHER_COMMON_H_