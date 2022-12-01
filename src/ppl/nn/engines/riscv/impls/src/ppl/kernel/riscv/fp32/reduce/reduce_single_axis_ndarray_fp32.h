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

#ifndef __ST_PPL_KERNEL_RISCV_FP32_REDUCE_REDUCE_SINGLE_AXIS_NDARRAY_FP32_H_
#define __ST_PPL_KERNEL_RISCV_FP32_REDUCE_REDUCE_SINGLE_AXIS_NDARRAY_FP32_H_

#include "ppl/kernel/riscv/fp32/reduce/reduce_kernel_fp32.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() ((int64_t)4)

#define REDUCE_LOOP_VECTOR(R, N)                                                                                            \
    do {                                                                                                                    \
        if (N > 0) res0 = reduce_vector_kernel_fp32<op>(vlev_float32xm1(base_src + R * inner_dim + 0 * C_BLK(), vl), res0); \
        if (N > 1) res1 = reduce_vector_kernel_fp32<op>(vlev_float32xm1(base_src + R * inner_dim + 1 * C_BLK(), vl), res1); \
        if (N > 2) res2 = reduce_vector_kernel_fp32<op>(vlev_float32xm1(base_src + R * inner_dim + 2 * C_BLK(), vl), res2); \
        if (N > 3) res3 = reduce_vector_kernel_fp32<op>(vlev_float32xm1(base_src + R * inner_dim + 3 * C_BLK(), vl), res3); \
    } while (0);

#define REDUCE_LOOP_SCALAR(R, BN, N)                                                                               \
    do {                                                                                                           \
        if (N > 0) res_x[0] = reduce_scalar_kernel_fp32<op>(base_src[R * inner_dim + BN * C_BLK() + 0], res_x[0]); \
        if (N > 1) res_x[1] = reduce_scalar_kernel_fp32<op>(base_src[R * inner_dim + BN * C_BLK() + 1], res_x[1]); \
        if (N > 2) res_x[2] = reduce_scalar_kernel_fp32<op>(base_src[R * inner_dim + BN * C_BLK() + 2], res_x[2]); \
        if (N > 3) res_x[3] = reduce_scalar_kernel_fp32<op>(base_src[R * inner_dim + BN * C_BLK() + 3], res_x[3]); \
    } while (0);

#define REDUCE_INNER_TAIL(BN)                                          \
    do {                                                               \
        const int64_t inner_tail = inner_eff - BN * C_BLK();           \
        float32xm1_t res0, res1, res2, res3;                           \
        if (BN > 0) res0 = vfmvvf_float32xm1(init_val, vl);            \
        if (BN > 1) res1 = vfmvvf_float32xm1(init_val, vl);            \
        if (BN > 2) res2 = vfmvvf_float32xm1(init_val, vl);            \
        if (BN > 3) res3 = vfmvvf_float32xm1(init_val, vl);            \
        float res_x[inner_tail] = {init_val};                          \
                                                                       \
        for (int64_t r = 0; r < reduce_body; r += unroll_reduce) {     \
            /* v_register part */                                      \
            REDUCE_LOOP_VECTOR(0, BN);                                 \
            REDUCE_LOOP_VECTOR(1, BN);                                 \
            REDUCE_LOOP_VECTOR(2, BN);                                 \
            REDUCE_LOOP_VECTOR(3, BN);                                 \
            REDUCE_LOOP_VECTOR(4, BN);                                 \
            REDUCE_LOOP_VECTOR(5, BN);                                 \
            REDUCE_LOOP_VECTOR(6, BN);                                 \
            REDUCE_LOOP_VECTOR(7, BN);                                 \
            /* x_register part */                                      \
            REDUCE_LOOP_SCALAR(0, BN, inner_tail);                     \
            REDUCE_LOOP_SCALAR(1, BN, inner_tail);                     \
            REDUCE_LOOP_SCALAR(2, BN, inner_tail);                     \
            REDUCE_LOOP_SCALAR(3, BN, inner_tail);                     \
            REDUCE_LOOP_SCALAR(4, BN, inner_tail);                     \
            REDUCE_LOOP_SCALAR(5, BN, inner_tail);                     \
            REDUCE_LOOP_SCALAR(6, BN, inner_tail);                     \
            REDUCE_LOOP_SCALAR(7, BN, inner_tail);                     \
                                                                       \
            base_src += unroll_reduce * inner_dim;                     \
        }                                                              \
        for (int64_t r = reduce_body; r < reduce_dim; ++r) {           \
            /* v_register part */                                      \
            REDUCE_LOOP_VECTOR(0, BN);                                 \
            /* x_register part */                                      \
            REDUCE_LOOP_SCALAR(0, BN, inner_tail);                     \
                                                                       \
            base_src += inner_dim;                                     \
        }                                                              \
        if (BN > 0) vsev_float32xm1(base_dst + 0 * C_BLK(), res0, vl); \
        if (BN > 1) vsev_float32xm1(base_dst + 1 * C_BLK(), res1, vl); \
        if (BN > 2) vsev_float32xm1(base_dst + 2 * C_BLK(), res2, vl); \
        if (BN > 3) vsev_float32xm1(base_dst + 3 * C_BLK(), res3, vl); \
        for (int64_t n = 0; n < inner_tail; ++n) {                     \
            base_dst[BN * C_BLK() + n] = res_x[n];                     \
        }                                                              \
    } while (0);

template <reduce_op_type_t op>
ppl::common::RetCode reduce_single_axis_ndarray_fp32(
    const float* src,
    float* dst,

    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const int32_t* axes, // lists of dimensions which axes needs to be reduced
    const int32_t num_axes // number of axes that needs to be reduced
)
{
    int64_t outer_dim       = 1;
    int64_t reduce_dim      = 1;
    int64_t inner_dim       = 1;
    const int64_t dim_count = src_shape->GetDimCount();
    for (int64_t i = 0; i < dim_count; i++) {
        if (i < axes[0]) {
            outer_dim *= src_shape->GetDim(i);
        } else if (i > axes[num_axes - 1]) {
            inner_dim *= src_shape->GetDim(i);
        } else {
            reduce_dim *= src_shape->GetDim(i);
        }
    }

    const int64_t unroll_vector_inner = 4;
    const int64_t unroll_inner        = unroll_vector_inner * C_BLK(); // parallelism of inner_dim
    const int64_t unroll_reduce       = 32 / unroll_vector_inner; // parallelism of reduce_dim
    const int64_t reduce_body         = round(reduce_dim, unroll_reduce);
    const int64_t reduce_tail         = reduce_dim - reduce_body;

    const int64_t reduce_body_no_inner_dim = round(reduce_dim, C_BLK());
    const int64_t reduce_tail_no_inner_dim = reduce_dim - reduce_body_no_inner_dim;

    float init_val = reduce_init_val_fp32<op>();
    const auto vl  = vsetvli(C_BLK(), RVV_E32, RVV_M1);
    for (int64_t o = 0; o < outer_dim; ++o) {
        for (int64_t i = 0; i < inner_dim; i += unroll_inner) {
            const int64_t inner_eff = min<int64_t>(inner_dim - i, unroll_inner);
            const float* base_src   = src + o * reduce_dim * inner_dim + i;
            float* base_dst         = dst + o * inner_dim + i;
            if (inner_eff == unroll_inner) {
                float32xm1_t res0, res1, res2, res3;
                res0 = vfmvvf_float32xm1(init_val, vl);
                res1 = vfmvvf_float32xm1(init_val, vl);
                res2 = vfmvvf_float32xm1(init_val, vl);
                res3 = vfmvvf_float32xm1(init_val, vl);
                for (int64_t r = 0; r < reduce_body; r += unroll_reduce) {
                    REDUCE_LOOP_VECTOR(0, 4);
                    REDUCE_LOOP_VECTOR(1, 4);
                    REDUCE_LOOP_VECTOR(2, 4);
                    REDUCE_LOOP_VECTOR(3, 4);
                    REDUCE_LOOP_VECTOR(4, 4);
                    REDUCE_LOOP_VECTOR(5, 4);
                    REDUCE_LOOP_VECTOR(6, 4);
                    REDUCE_LOOP_VECTOR(7, 4);

                    base_src += unroll_reduce * inner_dim;
                }
                for (int64_t r = reduce_body; r < reduce_dim; ++r) {
                    REDUCE_LOOP_VECTOR(0, 4);
                    base_src += inner_dim;
                }
                vsev_float32xm1(base_dst + 0 * C_BLK(), res0, vl);
                vsev_float32xm1(base_dst + 1 * C_BLK(), res1, vl);
                vsev_float32xm1(base_dst + 2 * C_BLK(), res2, vl);
                vsev_float32xm1(base_dst + 3 * C_BLK(), res3, vl);
            } else if (inner_dim == 1) { // v_register store data of reduce_dim
                float reduce_val = init_val;
                if (reduce_body_no_inner_dim) {
                    float32xm1_t res0 = vfmvvf_float32xm1(init_val, vl);
                    for (int64_t r = 0; r < reduce_body_no_inner_dim; r += C_BLK()) {
                        res0 = reduce_vector_kernel_fp32<op>(vlev_float32xm1(base_src + r, vl), res0);
                    }
                    reduce_val = reduce_vector_all_lanes_kernel_fp32<op>(res0);
                }
                if (reduce_tail_no_inner_dim) {
                    for (int64_t r = reduce_body_no_inner_dim; r < reduce_dim; ++r) {
                        reduce_val = reduce_scalar_kernel_fp32<op>(reduce_val, base_src[r]);
                    }
                }
                base_dst[0] = reduce_val;
            } else { // inner_effective < unroll_inner
                if (inner_eff > 3 * C_BLK()) {
                    REDUCE_INNER_TAIL(3);
                } else if (inner_eff > 2 * C_BLK()) {
                    REDUCE_INNER_TAIL(2);
                } else if (inner_eff > 1 * C_BLK()) {
                    REDUCE_INNER_TAIL(1);
                } else {
                    REDUCE_INNER_TAIL(0);
                }
            }
        }
    }

    // postprocess
    reduce_postprocess_fp32<op>(dst, dst_shape->CalcElementsIncludingPadding(), reduce_dim);

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv

#endif