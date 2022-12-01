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

#include <cmath>
#include <riscv-vector.h>

#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode softmax_ndarray_fp16(
    const ppl::common::TensorShape* shape,
    const int64_t axis,
    const __fp16* src,
    __fp16* dst)
{
    int64_t outer_dim = 1;
    int64_t inner_dim = 1;
    for (int64_t i = 0; i < axis; i++) {
        outer_dim *= shape->GetDim(i);
    }
    for (int64_t i = axis; i < shape->GetDimCount(); i++) {
        inner_dim *= shape->GetDim(i);
    }

    const auto vl = vsetvli(8, RVV_E16, RVV_M1);

    for (int64_t i = 0; i < outer_dim; i++) {
        const __fp16* src_ = src + i * inner_dim;
        __fp16* dst_       = dst + i * inner_dim;
        // find max
        float16xm1_t vfmax = vfmvvf_float16xm1(-__FLT_MAX__, vl);
        __fp16 fmax        = (__fp16)(-__FLT_MAX__);
        int64_t j          = 0;
        for (; j + 8 < inner_dim; j += 8) {
            vfmax = vfmaxvv_float16xm1(vfmax, vlev_float16xm1(src_ + j, vl), vl);
        }
        for (; j < inner_dim; j++) {
            fmax = std::max(fmax, src_[j]);
        }
        __fp16 max_data[8];
        vsev_float16xm1(max_data, vfmax, vl);
        for (int64_t k = 0; k < 8; k++) {
            fmax = std::max(fmax, max_data[k]);
        }
        // Σ(exp(src - max)) -- precision-tuning
        float sum = 0.0f;
        for (j = 0; j < inner_dim; j++) {
            sum += exp((float)src_[j] - fmax);
        }
        float recp_sum = (double)1.0 / sum;
        // final result
        for (j = 0; j < inner_dim; j++) {
            dst_[j] = (__fp16)(exp((float)src_[j] - fmax) * recp_sum);
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}}; //  namespace ppl::kernel::riscv
