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

#include <string.h> // for memcpy

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode max_unpool_ndarray_fp32(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const float *src,
    const int64_t *indices,
    float *dst)
{
    auto dim_count = src_shape->GetDimCount();
    const int64_t channels_dim = dim_count > 1 ? src_shape->GetDim(1) : 1;
    const int64_t outer_dim = src_shape->GetDim(0) * channels_dim;
    int64_t src_inner_dim = 1;
    int64_t dst_inner_dim = 1;
    for (uint32_t i = 2; i < dim_count; ++i) {
        src_inner_dim *= src_shape->GetDim(i);
        dst_inner_dim *= dst_shape->GetDim(i);
    }

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t n = 0; n < outer_dim; n++) {
        const float *p_src       = src + n * src_inner_dim;
        const int64_t *p_indices = indices + n * src_inner_dim;
        float *p_dst             = dst + n * dst_inner_dim;

        memset(p_dst, 0, dst_inner_dim * sizeof(float));

        const int64_t c = n % channels_dim;
        const int64_t indices_offset = c * dst_inner_dim;
        for (int64_t i = 0; i < src_inner_dim; ++i) {
            p_dst[p_indices[i] - indices_offset] = p_src[i];
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
