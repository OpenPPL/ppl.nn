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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_GATHER_GATHER_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_GATHER_GATHER_COMMON_H_

#include "ppl/kernel/riscv/common/internal_include.h"
#include <cstring>

namespace ppl { namespace kernel { namespace riscv {

template <typename T>
ppl::common::RetCode gather_ndarray_common(
    const T* src,
    T* dst,

    const int64_t* indices,
    const int64_t outer_dim,
    const int64_t gather_dim,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim)
{
    if (inner_dim >= 4) {
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                for (int64_t i = 0; i < indices_dim; ++i) {
                    T* dst_l = dst + o * num_indices * indices_dim * inner_dim;
                    dst_l += k * indices_dim * inner_dim + i * inner_dim;
                    int64_t index  = indices[k * indices_dim + i];
                    const T* src_l = src + o * gather_dim * inner_dim + index * inner_dim;
                    memcpy(dst_l, src_l, inner_dim * sizeof(T));
                }
            }
        }
    } else if (inner_dim >= 2) {
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                T* dst_l = dst + o * num_indices * indices_dim * inner_dim;
                dst_l += k * indices_dim * inner_dim;
                const int64_t* indices_l = indices + k * indices_dim;
                const T* src_l           = src + o * gather_dim * inner_dim;
                if (inner_dim == 2) {
                    for (int64_t i = 0; i < indices_dim; ++i) {
                        dst_l[0] = src_l[indices_l[0] * 2 + 0];
                        dst_l[1] = src_l[indices_l[0] * 2 + 1];
                        dst_l += inner_dim;
                        ++indices_l;
                    }
                } else {
                    for (int64_t i = 0; i < indices_dim; ++i) {
                        dst_l[0] = src_l[indices_l[0] * 3 + 0];
                        dst_l[1] = src_l[indices_l[0] * 3 + 1];
                        dst_l[2] = src_l[indices_l[0] * 3 + 2];
                        dst_l += inner_dim;
                        ++indices_l;
                    }
                }
            }
        }
    } else {
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                T* dst_l                 = dst + o * num_indices * indices_dim + k * indices_dim;
                const int64_t* indices_l = indices + k * indices_dim;
                const T* src_l           = src + o * gather_dim;
                for (int64_t i = 0; i < indices_dim; ++i) {
                    dst_l[0] = src_l[indices_l[0]];
                    ++dst_l;
                    ++indices_l;
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; //  namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_COMMON_GATHER_GATHER_COMMON_H_
