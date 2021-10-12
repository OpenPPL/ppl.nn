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

ppl::common::RetCode max_unpool_nchw_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *indices,
    float *dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);

    memset(dst, 0, batch * channels * dst_h * dst_w * sizeof(float));

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t n = 0; n < batch; n++) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t c = 0; c < channels; c++) {
            const float *p_src       = src + (n * channels + c) * src_h * src_w;
            const int64_t *p_indices = indices + (n * channels + c) * src_h * src_w;
            float *p_dst             = dst + (n * channels + c) * dst_h * dst_w;

            const int64_t indices_offset = c * dst_h * dst_w;
            for (int64_t ih = 0; ih < src_h; ih++) {
                for (int64_t iw = 0; iw < src_w; iw++) {
                    const int64_t src_index = ih * src_w + iw;
                    const float data        = p_src[src_index];
                    const int64_t dst_index = p_indices[src_index] - indices_offset;
                    p_dst[dst_index]        = data;
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
