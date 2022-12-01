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

#define C_BLK() ((int64_t)4)

ppl::common::RetCode softmax_ndarray_fp32(
    const ppl::common::TensorShape* shape,
    const int64_t axis,
    const float* src,
    float* dst)
{
    int64_t outer_dim = 1;
    int64_t inner_dim = 1;
    for (int64_t i = 0; i < axis; i++) {
        outer_dim *= shape->GetDim(i);
    }
    for (int64_t i = axis; i < shape->GetDimCount(); i++) {
        inner_dim *= shape->GetDim(i);
    }

    const auto vl = vsetvli(C_BLK(), RVV_E32, RVV_M1);

    for (int64_t i = 0; i < outer_dim; i++) {
        const float* src_  = src + i * inner_dim;
        float* dst_        = dst + i * inner_dim;
        // find max
        float32xm1_t vfmax = vfmvvf_float32xm1(-__FLT_MAX__, vl);
        float fmax         = (float)(-__FLT_MAX__);
        int64_t j          = 0;
        for (; j + C_BLK() < inner_dim; j += C_BLK()) {
            vfmax = vfmaxvv_float32xm1(vfmax, vlev_float32xm1(src_ + j, vl), vl);
        }
        for (; j < inner_dim; j++) {
            fmax = std::max(fmax, src_[j]);
        }
        float max_data[C_BLK()];
        vsev_float32xm1(max_data, vfmax, vl);
        for (int64_t k = 0; k < C_BLK(); k++) {
            fmax = std::max(fmax, max_data[k]);
        }
        vfmax = vfmvvf_float32xm1(fmax, vl);
        // src - max
        for (j = 0; j + C_BLK() < inner_dim; j += C_BLK()) {
            const float* src_p = src_ + j;
            float* dst_p       = dst_ + j;
            vsev_float32xm1(dst_p, vfsubvv_float32xm1(vlev_float32xm1(src_p, vl), vfmax, vl), vl);
        }
        for (; j < inner_dim; j++) {
            dst_[j] = src_[j] - fmax;
        }
        // Σ(exp(src - max))
        float sum = 0.0f;
        for (j = 0; j < inner_dim; j++) {
            sum += exp((float)dst_[j]);
        }
        // final result
        for (j = 0; j < inner_dim; j++) {
            dst_[j] = (float)(exp((float)dst_[j]) * (1.0f / sum));
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}}; //  namespace ppl::kernel::riscv
