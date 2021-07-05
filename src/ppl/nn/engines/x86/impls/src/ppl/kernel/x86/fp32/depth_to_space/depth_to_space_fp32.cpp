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

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode depth_to_space_ndarray_crd_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t blocksize,
    float *dst)
{
    const int64_t batch = src_shape->GetDim(0);
    const int64_t src_c = src_shape->GetDim(1);
    const int64_t src_h = src_shape->GetDim(2);
    const int64_t src_w = src_shape->GetDim(3);
    const int64_t dst_c = dst_shape->GetDim(1);
    const int64_t dst_h = dst_shape->GetDim(2);
    const int64_t dst_w = dst_shape->GetDim(3);

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
    PRAGMA_OMP_PARALLEL_FOR()
#endif
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t oc = 0; oc < dst_c; ++oc) {
            float *l_dst       = dst + (n * dst_c + oc) * dst_h * dst_w;
            const float *l_src = src + n * src_c * src_h * src_w;
            for (int64_t oh = 0; oh < dst_h; ++oh) {
                const int64_t src_h_idx = oh / blocksize;
                const int64_t sub_h_idx = oh % blocksize;
                for (int64_t ow = 0; ow < dst_w; ++ow) {
                    const int64_t src_w_idx = ow / blocksize;
                    const int64_t sub_w_idx = ow % blocksize;
                    const int64_t src_c_idx = oc * blocksize * blocksize + sub_h_idx * blocksize + sub_w_idx;
                    const int64_t src_idx   = (src_c_idx * src_h + src_h_idx) * src_w + src_w_idx;

                    *l_dst = *(l_src + src_idx);
                    ++l_dst;
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
