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

#include <math.h>
#include <riscv-vector.h>

#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

static inline float32xm1_t _vector128_sigmoid(const float32xm1_t var, const uint64_t vl)
{
    float32xm1_t value = var;
    value              = vfmaxvf_float32xm1(value, -18.0f, vl);
    value              = vfminvf_float32xm1(value, 18.0f, vl);

    float32xm1_t value_squared = vfmulvv_float32xm1(value, value, vl);

    float32xm1_t p;
    p = vfmulvf_float32xm1(value_squared, 4.37031012579801e-11f, vl);
    p = vfaddvf_float32xm1(p, 1.15627324459942e-07f, vl);
    p = vfmulvv_float32xm1(p, value_squared, vl);
    p = vfaddvf_float32xm1(p, 6.08574864600143e-05f, vl);
    p = vfmulvv_float32xm1(p, value_squared, vl);
    p = vfaddvf_float32xm1(p, 8.51377133304701e-03f, vl);
    p = vfmulvv_float32xm1(p, value_squared, vl);
    p = vfaddvf_float32xm1(p, 2.48287947061529e-01f, vl);
    p = vfmulvv_float32xm1(p, value, vl);

    float32xm1_t q;
    q = vfmulvf_float32xm1(value_squared, 6.10247389755681e-13f, vl);
    q = vfaddvf_float32xm1(q, 5.76102136993427e-09f, vl);
    q = vfmulvv_float32xm1(q, value_squared, vl);
    q = vfaddvf_float32xm1(q, 6.29106785017040e-06f, vl);
    q = vfmulvv_float32xm1(q, value_squared, vl);
    q = vfaddvf_float32xm1(q, 1.70198817374094e-03f, vl);
    q = vfmulvv_float32xm1(q, value_squared, vl);
    q = vfaddvf_float32xm1(q, 1.16817656904453e-01f, vl);
    q = vfmulvv_float32xm1(q, value_squared, vl);
    q = vfaddvf_float32xm1(q, 9.93151921023180e-01f, vl);

    float32xm1_t dst = vfaddvf_float32xm1(vfdivvv_float32xm1(p, q, vl), 0.5f, vl);
    return dst;
}

ppl::common::RetCode sigmoid_fp32_vec128(const ppl::common::TensorShape* x_shape, const float* x, float* y)
{
    const auto vl        = vsetvli(4, RVV_E32, RVV_M1);
    const int64_t n_elem = x_shape->CalcElementsIncludingPadding();

    int64_t i = 0;
    for (; i <= n_elem - 16; i += 16) {
        vsev_float32xm1(y + i + 0, _vector128_sigmoid(vlev_float32xm1(x + i + 0, vl), vl), vl);
        vsev_float32xm1(y + i + 4, _vector128_sigmoid(vlev_float32xm1(x + i + 4, vl), vl), vl);
        vsev_float32xm1(y + i + 8, _vector128_sigmoid(vlev_float32xm1(x + i + 8, vl), vl), vl);
        vsev_float32xm1(y + i + 12, _vector128_sigmoid(vlev_float32xm1(x + i + 12, vl), vl), vl);
    }
    for (; i < n_elem - 4; i += 1) {
        vsev_float32xm1(y + i, _vector128_sigmoid(vlev_float32xm1(x + i, vl), vl), vl);
    }
    if (i != n_elem) {
        const auto last_vl = vsetvli(n_elem - i, RVV_E32, RVV_M1);
        vsev_float32xm1(y + i, _vector128_sigmoid(vlev_float32xm1(x + i, vl), vl), last_vl);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv
