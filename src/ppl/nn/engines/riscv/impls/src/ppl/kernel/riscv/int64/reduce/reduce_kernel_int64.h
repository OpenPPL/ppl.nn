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

#ifndef __ST_PPL_KERNEL_RISCV_INT64_REDUCE_REDUCE_KERNEL_INT64_H_
#define __ST_PPL_KERNEL_RISCV_INT64_REDUCE_REDUCE_KERNEL_INT64_H_

#include <cstring>
#include <riscv-vector.h>
#include <limits.h>

#include "ppl/kernel/riscv/common/reduce/reduce_common.h"
#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() ((int64_t)2)

template <reduce_op_type_t op>
inline int64_t reduce_init_val_int64(void)
{
    return 0;
}

template <>
inline int64_t reduce_init_val_int64<REDUCE_MAX>(void)
{
    return INT64_MIN;
}
template <>
inline int64_t reduce_init_val_int64<REDUCE_MIN>(void)
{
    return INT64_MAX;
}

//
template <reduce_op_type_t op>
static void reduce_preprocess_scalar_int64(int64_t* dst, int64_t len)
{
    const int64_t init_val = reduce_init_val_int64<op>();
    for (int64_t i = 0; i < len; i += 1) {
        dst[i] = init_val;
    }
}

template <reduce_op_type_t op>
static void reduce_preprocess_int64(int64_t* dst, int64_t len)
{
    const int64_t init_val      = reduce_init_val_int64<op>();
    const int64xm1_t v_init_val = vmvvx_int64xm1(init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));

    const int64_t parall_d   = 16;
    const int64_t unroll_len = parall_d * C_BLK();

    int64_t i = 0;
    for (; i + unroll_len < len; i += unroll_len) {
        vsev_int64xm1(dst + i + 0 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 1 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 2 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 3 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 4 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 5 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 6 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 7 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 8 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 9 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 10 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 11 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 12 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 13 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 14 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
        vsev_int64xm1(dst + i + 15 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
    }
    for (; i < len; i += C_BLK()) {
        vsev_int64xm1(dst + i, v_init_val, vsetvli(C_BLK(), RVV_E64, RVV_M1));
    }
}

//
template <reduce_op_type_t op>
inline int64_t reduce_scalar_kernel_int64(int64_t a, int64_t b);

template <>
inline int64_t reduce_scalar_kernel_int64<REDUCE_MEAN>(int64_t a, int64_t b)
{
    return a + b;
}
template <>
inline int64_t reduce_scalar_kernel_int64<REDUCE_MAX>(int64_t a, int64_t b)
{
    return a > b ? a : b;
}
template <>
inline int64_t reduce_scalar_kernel_int64<REDUCE_MIN>(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
template <>
inline int64_t reduce_scalar_kernel_int64<REDUCE_SUM>(int64_t a, int64_t b)
{
    return a + b;
}
//
template <reduce_op_type_t op>
inline int64xm1_t reduce_vector_kernel_int64(int64xm1_t a, int64xm1_t b);

template <>
inline int64xm1_t reduce_vector_kernel_int64<REDUCE_MEAN>(int64xm1_t a, int64xm1_t b)
{
    return vaddvv_int64xm1(a, b, vsetvli(C_BLK(), RVV_E64, RVV_M1));
}
template <>
inline int64xm1_t reduce_vector_kernel_int64<REDUCE_MAX>(int64xm1_t a, int64xm1_t b)
{
    return vmaxvv_int64xm1(a, b, vsetvli(C_BLK(), RVV_E64, RVV_M1));
}
template <>
inline int64xm1_t reduce_vector_kernel_int64<REDUCE_MIN>(int64xm1_t a, int64xm1_t b)
{
    return vminvv_int64xm1(a, b, vsetvli(C_BLK(), RVV_E64, RVV_M1));
}
template <>
inline int64xm1_t reduce_vector_kernel_int64<REDUCE_SUM>(int64xm1_t a, int64xm1_t b)
{
    return vaddvv_int64xm1(a, b, vsetvli(C_BLK(), RVV_E64, RVV_M1));
}
//
template <reduce_op_type_t op>
inline int64_t reduce_vector_all_lanes_kernel_int64(int64xm1_t v)
{
    int64_t tmp[C_BLK()];
    vsev_int64xm1(tmp, v, vsetvli(C_BLK(), RVV_E64, RVV_M1));
    tmp[0] = reduce_scalar_kernel_int64<op>(tmp[0], tmp[1]);

    return tmp[0];
}
//
template <reduce_op_type_t op>
static void reduce_postprocess_int64(int64_t* dst, int64_t len, int64_t div_val)
{
    if (op == REDUCE_MEAN) {
        const auto vl = vsetvli(C_BLK(), RVV_E64, RVV_M1);

        const int64_t rdiv = (int64_t)(1.0f / div_val);

        const int64_t parall_d   = 16;
        const int64_t unroll_len = parall_d * C_BLK();

        int64_t i = 0;
        for (; i + unroll_len < len; i += unroll_len) {
            vsev_int64xm1(dst + i + 0 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 0 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 1 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 1 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 2 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 2 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 3 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 3 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 4 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 4 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 5 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 5 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 6 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 6 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 7 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 7 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 8 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 8 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 9 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 9 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 10 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 10 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 11 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 11 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 12 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 12 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 13 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 13 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 14 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 14 * C_BLK(), vl), rdiv, vl),
                          vl);
            vsev_int64xm1(dst + i + 15 * C_BLK(),
                          vmulvx_int64xm1(vlev_int64xm1(dst + i + 15 * C_BLK(), vl), rdiv, vl),
                          vl);
        }
        for (; i < len; i += C_BLK()) {
            vsev_int64xm1(dst + i, vmulvx_int64xm1(vlev_int64xm1(dst + i, vl), rdiv, vl), vl);
        }
    }
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_INT64_REDUCE_REDUCE_KERNEL_INT64_H_
