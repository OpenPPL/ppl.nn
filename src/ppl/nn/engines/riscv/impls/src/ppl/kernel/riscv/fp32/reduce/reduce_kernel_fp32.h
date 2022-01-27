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

#ifndef __ST_PPL_KERNEL_RISCV_FP32_REDUCE_REDUCE_KERNEL_FP32_H_
#define __ST_PPL_KERNEL_RISCV_FP32_REDUCE_REDUCE_KERNEL_FP32_H_

#include <cstring>
#include <riscv-vector.h>
#include <float.h>

#include "ppl/kernel/riscv/common/reduce/reduce_common.h"
#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() ((int64_t)4)

template <reduce_op_type_t op>
inline float reduce_init_val_fp32(void)
{
    return 0;
}

template <>
inline float reduce_init_val_fp32<REDUCE_MAX>(void)
{
    return (float)-FLT_MAX;
}
template <>
inline float reduce_init_val_fp32<REDUCE_MIN>(void)
{
    return (float)FLT_MAX;
}

//
template <reduce_op_type_t op>
static void reduce_preprocess_fp32(float* dst, int64_t len)
{
    const float init_val          = reduce_init_val_fp32<op>();
    const float32xm1_t v_init_val = vfmvvf_float32xm1(init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));

    const int64_t parall_d   = 16;
    const int64_t unroll_len = parall_d * C_BLK();

    int64_t i = 0;
    for (; i + unroll_len < len; i += unroll_len) {
        vsev_float32xm1(dst + i + 0 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 1 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 2 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 3 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 4 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 5 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 6 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 7 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 8 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 9 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 10 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 11 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 12 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 13 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 14 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
        vsev_float32xm1(dst + i + 15 * C_BLK(), v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
    }
    for (; i < len; i += C_BLK()) {
        vsev_float32xm1(dst + i, v_init_val, vsetvli(C_BLK(), RVV_E32, RVV_M1));
    }
}

//
template <reduce_op_type_t op>
inline float reduce_scalar_kernel_fp32(float a, float b);

template <>
inline float reduce_scalar_kernel_fp32<REDUCE_MEAN>(float a, float b)
{
    return a + b;
}
template <>
inline float reduce_scalar_kernel_fp32<REDUCE_MAX>(float a, float b)
{
    return a > b ? a : b;
}
template <>
inline float reduce_scalar_kernel_fp32<REDUCE_MIN>(float a, float b)
{
    return a < b ? a : b;
}
template <>
inline float reduce_scalar_kernel_fp32<REDUCE_SUM>(float a, float b)
{
    return a + b;
}
//
template <reduce_op_type_t op>
inline float32xm1_t reduce_vector_kernel_fp32(float32xm1_t a, float32xm1_t b);

template <>
inline float32xm1_t reduce_vector_kernel_fp32<REDUCE_MEAN>(float32xm1_t a, float32xm1_t b)
{
    return vfaddvv_float32xm1(a, b, vsetvli(C_BLK(), RVV_E32, RVV_M1));
}
template <>
inline float32xm1_t reduce_vector_kernel_fp32<REDUCE_MAX>(float32xm1_t a, float32xm1_t b)
{
    return vfmaxvv_float32xm1(a, b, vsetvli(C_BLK(), RVV_E32, RVV_M1));
}
template <>
inline float32xm1_t reduce_vector_kernel_fp32<REDUCE_MIN>(float32xm1_t a, float32xm1_t b)
{
    return vfminvv_float32xm1(a, b, vsetvli(C_BLK(), RVV_E32, RVV_M1));
}
template <>
inline float32xm1_t reduce_vector_kernel_fp32<REDUCE_SUM>(float32xm1_t a, float32xm1_t b)
{
    return vfaddvv_float32xm1(a, b, vsetvli(C_BLK(), RVV_E32, RVV_M1));
}
//
template <reduce_op_type_t op>
inline float reduce_vector_all_lanes_kernel_fp32(float32xm1_t v)
{
    float tmp[C_BLK()];
    vsev_float32xm1(tmp, v, vsetvli(C_BLK(), RVV_E32, RVV_M1));
    tmp[0] = reduce_scalar_kernel_fp32<op>(tmp[0], tmp[1]);
    tmp[2] = reduce_scalar_kernel_fp32<op>(tmp[2], tmp[3]);
    tmp[0] = reduce_scalar_kernel_fp32<op>(tmp[0], tmp[2]);

    return tmp[0];
}
//
template <reduce_op_type_t op>
static void reduce_postprocess_fp32(float* dst, int64_t len, float div_val)
{
    if (op == REDUCE_MEAN) {
        const auto vl = vsetvli(C_BLK(), RVV_E32, RVV_M1);

        const float rdiv = (float)(1.0f / div_val);

        const int64_t parall_d   = 16;
        const int64_t unroll_len = parall_d * C_BLK();

        int64_t i = 0;
        for (; i + unroll_len < len; i += unroll_len) {
            vsev_float32xm1(dst + i + 0 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 0 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 1 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 1 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 2 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 2 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 3 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 3 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 4 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 4 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 5 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 5 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 6 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 6 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 7 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 7 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 8 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 8 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 9 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 9 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 10 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 10 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 11 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 11 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 12 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 12 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 13 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 13 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 14 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 14 * C_BLK(), vl), rdiv, vl), vl);
            vsev_float32xm1(dst + i + 15 * C_BLK(), vfmulvf_float32xm1(vlev_float32xm1(dst + i + 15 * C_BLK(), vl), rdiv, vl), vl);
        }
        for (; i < len; i += C_BLK()) {
            vsev_float32xm1(dst + i, vfmulvf_float32xm1(vlev_float32xm1(dst + i, vl), rdiv, vl), vl);
        }
    }
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP32_REDUCE_REDUCE_KERNEL_FP32_H_
