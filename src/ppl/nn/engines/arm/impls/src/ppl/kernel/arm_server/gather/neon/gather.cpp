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

#include <vector>
#include <string.h>

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
static ppl::common::RetCode gather_ndarray_common(
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
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                for (int64_t i = 0; i < indices_dim; ++i) {
                    eT *l_dst = dst + o * num_indices * indices_dim * inner_dim +
                                k * indices_dim * inner_dim + i * inner_dim;
                    int64_t index   = indices[k * indices_dim + i];
                    const eT *l_src = src + o * gather_dim * inner_dim + index * inner_dim;
                    memcpy(l_dst, l_src, inner_dim * sizeof(eT));
                }
            }
        }
    } else if (inner_dim >= 2) {
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                eT *l_dst =
                    dst + o * num_indices * indices_dim * inner_dim + k * indices_dim * inner_dim;
                const int64_t *l_indices = indices + k * indices_dim;
                const eT *l_src          = src + o * gather_dim * inner_dim;
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
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                eT *l_dst                = dst + o * num_indices * indices_dim + k * indices_dim;
                const int64_t *l_indices = indices + k * indices_dim;
                const eT *l_src          = src + o * gather_dim;
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

template <typename eT>
static ppl::common::RetCode gather_wrapper(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *indices_shape,
    const void *src,
    const int64_t *indices,
    const int64_t axis,
    void *dst)
{
    // calculate dims
    const uint32_t q    = indices_shape->GetRealDimCount();
    int64_t num_indices = 1;
    int64_t indices_dim = indices_shape->GetDim(q - 1);
    int64_t outer_dim   = 1;
    int64_t inner_dim   = 1;
    int64_t n           = indices_shape->CalcElementsExcludingPadding();

    std::vector<int64_t> real_indices(n);
    if (q != 0) {
        for (uint32_t i = 0; i < q - 1; ++i) {
            num_indices *= indices_shape->GetDim(i);
        }
        for (uint32_t i = 0; i < indices_shape->CalcElementsExcludingPadding(); ++i) {
            real_indices[i] = indices[i] >= 0 ? indices[i] : indices[i] + q;
        }
    }
    if (indices_shape->IsScalar()) {
        real_indices[0] = indices[0] >= 0
                              ? indices[0]
                              : indices[0] + src_shape->GetDim(axis);
    }
    for (int64_t i = 0; i < axis; ++i) {
        outer_dim *= src_shape->GetDim(i);
    }
    int64_t gather_dim = src_shape->GetDim(axis);

    for (uint32_t i = axis + 1; i < src_shape->GetDimCount(); ++i) {
        inner_dim *= src_shape->GetDim(i);
    }

    return gather_ndarray_common<eT>((const eT *)src, indices, outer_dim, gather_dim, inner_dim, num_indices, indices_dim, (eT *)dst);
}

ppl::common::RetCode gather(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *indices_shape,
    const void *src,
    const int64_t *indices,
    const int64_t axis,
    void *dst)
{
    const auto data_type   = src_shape->GetDataType();
    const auto data_format = src_shape->GetDataFormat();
    if (data_format != ppl::common::DATAFORMAT_NDARRAY) {
        return ppl::common::RC_UNSUPPORTED;
    }

    switch (ppl::common::GetSizeOfDataType(data_type)) {
        case 1: return gather_wrapper<uint8_t>(src_shape, indices_shape, src, indices, axis, dst);
        case 2: return gather_wrapper<uint16_t>(src_shape, indices_shape, src, indices, axis, dst);
        case 4: return gather_wrapper<uint32_t>(src_shape, indices_shape, src, indices, axis, dst);
        case 8: return gather_wrapper<uint64_t>(src_shape, indices_shape, src, indices, axis, dst);
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}}} // namespace ppl::kernel::arm_server::neon
