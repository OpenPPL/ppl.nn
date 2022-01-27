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

#ifndef __ST_PPL_KERNEL_RISCV_FP16_REDUCE_REDUCE_KERNEL_FP16_H_
#define __ST_PPL_KERNEL_RISCV_FP16_REDUCE_REDUCE_KERNEL_FP16_H_

#include <cstring>
#include <riscv-vector.h>
#include <float.h>

#include "ppl/kernel/riscv/common/reduce/reduce_common.h"
#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

template <reduce_op_type_t op>
inline __fp16 reduce_init_val_fp16(void)
{
    return 0;
}

template <>
inline __fp16 reduce_init_val_fp16<REDUCE_MAX>(void)
{
    return (__fp16)-FLT_MAX;
}
template <>
inline __fp16 reduce_init_val_fp16<REDUCE_MIN>(void)
{
    return (__fp16)FLT_MAX;
}

//
template <reduce_op_type_t op>
static void reduce_preprocess_fp16(__fp16* dst, int64_t len)
{
    const __fp16 init_val         = reduce_init_val_fp16<op>();
    const float16xm1_t v_init_val = vfmvvf_float16xm1(init_val, vsetvli(8, RVV_E16, RVV_M1));

    const int64_t parall_d   = 16;
    const int64_t unroll_len = parall_d * 8;

    int64_t i = 0;
    for (; i + unroll_len < len; i += unroll_len) {
        vsev_float16xm1(dst + i + 0 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 1 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 2 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 3 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 4 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 5 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 6 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 7 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 8 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 9 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 10 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 11 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 12 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 13 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 14 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
        vsev_float16xm1(dst + i + 15 * 8, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
    }
    for (; i < len; i += 8) {
        vsev_float16xm1(dst + i, v_init_val, vsetvli(8, RVV_E16, RVV_M1));
    }
}

//
template <reduce_op_type_t op>
inline __fp16 reduce_scalar_kernel_fp16(__fp16 a, __fp16 b);

template <>
inline __fp16 reduce_scalar_kernel_fp16<REDUCE_MEAN>(__fp16 a, __fp16 b)
{
    return a + b;
}
template <>
inline __fp16 reduce_scalar_kernel_fp16<REDUCE_MAX>(__fp16 a, __fp16 b)
{
    return a > b ? a : b;
}
template <>
inline __fp16 reduce_scalar_kernel_fp16<REDUCE_MIN>(__fp16 a, __fp16 b)
{
    return a < b ? a : b;
}
template <>
inline __fp16 reduce_scalar_kernel_fp16<REDUCE_SUM>(__fp16 a, __fp16 b)
{
    return a + b;
}
//
template <reduce_op_type_t op>
inline float16xm1_t reduce_vector_kernel_fp16(float16xm1_t a, float16xm1_t b);

template <>
inline float16xm1_t reduce_vector_kernel_fp16<REDUCE_MEAN>(float16xm1_t a, float16xm1_t b)
{
    return vfaddvv_float16xm1(a, b, vsetvli(8, RVV_E16, RVV_M1));
}
template <>
inline float16xm1_t reduce_vector_kernel_fp16<REDUCE_MAX>(float16xm1_t a, float16xm1_t b)
{
    return vfmaxvv_float16xm1(a, b, vsetvli(8, RVV_E16, RVV_M1));
}
template <>
inline float16xm1_t reduce_vector_kernel_fp16<REDUCE_MIN>(float16xm1_t a, float16xm1_t b)
{
    return vfminvv_float16xm1(a, b, vsetvli(8, RVV_E16, RVV_M1));
}
template <>
inline float16xm1_t reduce_vector_kernel_fp16<REDUCE_SUM>(float16xm1_t a, float16xm1_t b)
{
    return vfaddvv_float16xm1(a, b, vsetvli(8, RVV_E16, RVV_M1));
}
//
template <reduce_op_type_t op>
inline __fp16 reduce_vector_all_lanes_kernel_fp16(float16xm1_t v)
{
    __fp16 tmp[8];
    vsev_float16xm1(tmp, v, vsetvli(8, RVV_E16, RVV_M1));
    tmp[0] = reduce_scalar_kernel_fp16<op>(tmp[0], tmp[1]);
    tmp[2] = reduce_scalar_kernel_fp16<op>(tmp[2], tmp[3]);
    tmp[4] = reduce_scalar_kernel_fp16<op>(tmp[4], tmp[5]);
    tmp[6] = reduce_scalar_kernel_fp16<op>(tmp[6], tmp[7]);
    tmp[0] = reduce_scalar_kernel_fp16<op>(tmp[0], tmp[2]);
    tmp[2] = reduce_scalar_kernel_fp16<op>(tmp[4], tmp[6]);
    tmp[0] = reduce_scalar_kernel_fp16<op>(tmp[0], tmp[2]);
    return tmp[0];
}
//
template <reduce_op_type_t op>
static void reduce_postprocess_fp16(__fp16* dst, int64_t len, __fp16 div_val)
{
    if (op == REDUCE_MEAN) {
        const auto vl = vsetvli(8, RVV_E16, RVV_M1);

        const __fp16 rdiv = (__fp16)(1.0f / div_val);

        const int64_t parall_d   = 16;
        const int64_t unroll_len = parall_d * 8;

        int64_t i = 0;
        for (; i + unroll_len < len; i += unroll_len) {
            vsev_float16xm1(dst + i + 0 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 0 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 1 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 1 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 2 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 2 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 3 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 3 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 4 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 4 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 5 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 5 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 6 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 6 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 7 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 7 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 8 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 8 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 9 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 9 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 10 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 10 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 11 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 11 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 12 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 12 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 13 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 13 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 14 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 14 * 8, vl), rdiv, vl), vl);
            vsev_float16xm1(dst + i + 15 * 8, vfmulvf_float16xm1(vlev_float16xm1(dst + i + 15 * 8, vl), rdiv, vl), vl);
        }
        for (; i < len; i += 8) {
            vsev_float16xm1(dst + i, vfmulvf_float16xm1(vlev_float16xm1(dst + i, vl), rdiv, vl), vl);
        }
    }
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP16_REDUCE_REDUCE_KERNEL_FP16_H_
