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

ppl::common::RetCode clip_fp16(
    const ppl::common::TensorShape* shape,
    const __fp16 clip_max,
    const __fp16 clip_min,
    const __fp16* src,
    __fp16* dst)
{
    const int64_t total_len  = shape->CalcElementsIncludingPadding();
    const int64_t parall_d   = 32;
    const int64_t unroll_len = parall_d * 8;
    const auto vl            = vsetvli(8, RVV_E16, RVV_M1);

    int64_t idx = 0;
    for (; idx + unroll_len < total_len; idx += unroll_len) {
        const __fp16* src_ = src + idx;
        __fp16* dst_       = dst + idx;
        vsev_float16xm1(dst_ + 0 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 0 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 1 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 1 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 2 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 2 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 3 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 3 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 4 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 4 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 5 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 5 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 6 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 6 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 7 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 7 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 8 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 8 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 9 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 9 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 10 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 10 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 11 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 11 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 12 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 12 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 13 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 13 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 14 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 14 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 15 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 15 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 16 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 16 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 17 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 17 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 18 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 18 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 19 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 19 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 20 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 20 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 21 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 21 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 22 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 22 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 23 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 23 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 24 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 24 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 25 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 25 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 26 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 26 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 27 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 27 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 28 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 28 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 29 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 29 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 30 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 30 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
        vsev_float16xm1(dst_ + 31 * 8, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_ + 31 * 8, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
    }
    for (; idx < total_len; idx += 8) {
        const __fp16* src_ = src + idx;
        __fp16* dst_       = dst + idx;
        vsev_float16xm1(dst_, vfmaxvf_float16xm1(vfminvf_float16xm1(vlev_float16xm1(src_, vl), (__fp16)clip_max, vl), (__fp16)clip_min, vl), vl);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; //  namespace ppl::kernel::riscv
