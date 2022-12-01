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

#include <riscv-vector.h>
#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() ((int64_t)4)

ppl::common::RetCode relu_fp32(
    const ppl::common::TensorShape* shape,
    const float* src,
    float* dst)
{
    const int64_t total_len  = shape->CalcElementsIncludingPadding();
    const int64_t parall_d   = 32;
    const int64_t unroll_len = parall_d * C_BLK();
    const auto vl            = vsetvli(C_BLK(), RVV_E32, RVV_M1);

    int64_t idx = 0;
    for (; idx + unroll_len < total_len; idx += unroll_len) {
        const float* src_ = src + idx;
        float* dst_       = dst + idx;
        vsev_float32xm1(dst_ + 0 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 0 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 1 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 1 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 2 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 2 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 3 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 3 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 4 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 4 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 5 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 5 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 6 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 6 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 7 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 7 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 8 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 8 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 9 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 9 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 10 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 10 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 11 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 11 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 12 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 12 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 13 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 13 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 14 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 14 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 15 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 15 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 16 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 16 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 17 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 17 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 18 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 18 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 19 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 19 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 20 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 20 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 21 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 21 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 22 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 22 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 23 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 23 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 24 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 24 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 25 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 25 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 26 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 26 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 27 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 27 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 28 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 28 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 29 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 29 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 30 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 30 * C_BLK(), vl), (float)0.0f, vl), vl);
        vsev_float32xm1(dst_ + 31 * C_BLK(), vfmaxvf_float32xm1(vlev_float32xm1(src_ + 31 * C_BLK(), vl), (float)0.0f, vl), vl);
    }
    for (; idx < total_len; idx += C_BLK()) {
        const float* src_ = src + idx;
        float* dst_       = dst + idx;
        vsev_float32xm1(dst_, vfmaxvf_float32xm1(vlev_float32xm1(src_, vl), (float)0.0f, vl), vl);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; //  namespace ppl::kernel::riscv
