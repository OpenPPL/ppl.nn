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

#ifndef __ST_PPL_KERNEL_RISCV_INT64_ARITHMETIC_ARITHMETIC_BROADCAST_NDARRAY_INT64_H_
#define __ST_PPL_KERNEL_RISCV_INT64_ARITHMETIC_ARITHMETIC_BROADCAST_NDARRAY_INT64_H_

#include "ppl/nn/runtime/tensor_impl.h"
#include "arithmetic_kernel_int64.h"
#include <cmath>

namespace ppl { namespace kernel { namespace riscv {

template <arithmetic_op_type_t op, bool fuse_relu>
static inline void arithmetic_broadcast_lastdim_no_broadcast_ndarray_int64(const int64_t* src0, const int64_t* src1,
                                                                           int64_t* dst,

                                                                           const int64_t start, const int64_t end) {
    const int64_t parall_d = 16;
    const int64_t unroll_len = parall_d * 2;
    const auto vl = vsetvli(2, RVV_E64, RVV_M1);

    int64_t i = start;
    for (; i + unroll_len < end; i += unroll_len) {
        const int64_t* src0_ = src0 + i;
        const int64_t* src1_ = src1 + i;
        int64_t* dst_ = dst + i;

        int64xm1_t vdata0, vdata1, vdata2, vdata3;
        int64xm1_t vdata4, vdata5, vdata6, vdata7;
        int64xm1_t vdata8, vdata9, vdata10, vdata11;
        int64xm1_t vdata12, vdata13, vdata14, vdata15;

        vdata0 = arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 0 * 2, vl), vlev_int64xm1(src1_ + 0 * 2, vl));
        vdata1 = arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 1 * 2, vl), vlev_int64xm1(src1_ + 1 * 2, vl));
        vdata2 = arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 2 * 2, vl), vlev_int64xm1(src1_ + 2 * 2, vl));
        vdata3 = arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 3 * 2, vl), vlev_int64xm1(src1_ + 3 * 2, vl));
        vdata4 = arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 4 * 2, vl), vlev_int64xm1(src1_ + 4 * 2, vl));
        vdata5 = arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 5 * 2, vl), vlev_int64xm1(src1_ + 5 * 2, vl));
        vdata6 = arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 6 * 2, vl), vlev_int64xm1(src1_ + 6 * 2, vl));
        vdata7 = arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 7 * 2, vl), vlev_int64xm1(src1_ + 7 * 2, vl));
        vdata8 = arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 8 * 2, vl), vlev_int64xm1(src1_ + 8 * 2, vl));
        vdata9 = arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 9 * 2, vl), vlev_int64xm1(src1_ + 9 * 2, vl));
        vdata10 =
            arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 10 * 2, vl), vlev_int64xm1(src1_ + 10 * 2, vl));
        vdata11 =
            arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 11 * 2, vl), vlev_int64xm1(src1_ + 11 * 2, vl));
        vdata12 =
            arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 12 * 2, vl), vlev_int64xm1(src1_ + 12 * 2, vl));
        vdata13 =
            arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 13 * 2, vl), vlev_int64xm1(src1_ + 13 * 2, vl));
        vdata14 =
            arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 14 * 2, vl), vlev_int64xm1(src1_ + 14 * 2, vl));
        vdata15 =
            arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_ + 15 * 2, vl), vlev_int64xm1(src1_ + 15 * 2, vl));

        if (fuse_relu) {
            vsev_int64xm1(dst_ + 0 * 2, vmaxvx_int64xm1(vdata0, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 1 * 2, vmaxvx_int64xm1(vdata1, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 2 * 2, vmaxvx_int64xm1(vdata2, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 3 * 2, vmaxvx_int64xm1(vdata3, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 4 * 2, vmaxvx_int64xm1(vdata4, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 5 * 2, vmaxvx_int64xm1(vdata5, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 6 * 2, vmaxvx_int64xm1(vdata6, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 7 * 2, vmaxvx_int64xm1(vdata7, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 8 * 2, vmaxvx_int64xm1(vdata8, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 9 * 2, vmaxvx_int64xm1(vdata9, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 10 * 2, vmaxvx_int64xm1(vdata10, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 11 * 2, vmaxvx_int64xm1(vdata11, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 12 * 2, vmaxvx_int64xm1(vdata12, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 13 * 2, vmaxvx_int64xm1(vdata13, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 14 * 2, vmaxvx_int64xm1(vdata14, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 15 * 2, vmaxvx_int64xm1(vdata15, (int64_t)0, vl), vl);
        } else {
            vsev_int64xm1(dst_ + 0 * 2, vdata0, vl);
            vsev_int64xm1(dst_ + 1 * 2, vdata1, vl);
            vsev_int64xm1(dst_ + 2 * 2, vdata2, vl);
            vsev_int64xm1(dst_ + 3 * 2, vdata3, vl);
            vsev_int64xm1(dst_ + 4 * 2, vdata4, vl);
            vsev_int64xm1(dst_ + 5 * 2, vdata5, vl);
            vsev_int64xm1(dst_ + 6 * 2, vdata6, vl);
            vsev_int64xm1(dst_ + 7 * 2, vdata7, vl);
            vsev_int64xm1(dst_ + 8 * 2, vdata8, vl);
            vsev_int64xm1(dst_ + 9 * 2, vdata9, vl);
            vsev_int64xm1(dst_ + 10 * 2, vdata10, vl);
            vsev_int64xm1(dst_ + 11 * 2, vdata11, vl);
            vsev_int64xm1(dst_ + 12 * 2, vdata12, vl);
            vsev_int64xm1(dst_ + 13 * 2, vdata13, vl);
            vsev_int64xm1(dst_ + 14 * 2, vdata14, vl);
            vsev_int64xm1(dst_ + 15 * 2, vdata15, vl);
        }
    }
    for (; i + 2 < end; i += 2) {
        const int64_t* src0_ = src0 + i;
        const int64_t* src1_ = src1 + i;
        int64_t* dst_ = dst + i;

        int64xm1_t vdata;
        vdata = arithmetic_vector_kernel_int64<op>(vlev_int64xm1(src0_, vl), vlev_int64xm1(src1_, vl));
        if (fuse_relu) {
            vsev_int64xm1(dst_, vmaxvx_int64xm1(vdata, (int64_t)0, vl), vl);
        } else {
            vsev_int64xm1(dst_, vdata, vl);
        }
    }
    for (; i < end; i++) {
        dst[i] = arithmetic_scalar_kernel_int64<op>(src0[i], src1[i]);
        if (fuse_relu) {
            dst[i] = std::max(dst[i], (int64_t)0);
        }
    }
}

template <arithmetic_op_type_t op, bool fuse_relu>
static inline void arithmetic_broadcast_lastdim_src0_broadcast_ndarray_int64(const int64_t* src0, const int64_t* src1,
                                                                             int64_t* dst,

                                                                             const int64_t start, const int64_t end) {
    const int64_t parall_d = 16;
    const int64_t unroll_len = parall_d * 2;
    const auto vl = vsetvli(2, RVV_E64, RVV_M1);

    const int64_t broadcast_val = src0[0];
    int64xm1_t v_broadcast_val = vmvvx_int64xm1(broadcast_val, vl);

    int64_t i = start;
    for (; i + unroll_len < end; i += unroll_len) {
        const int64_t* src1_ = src1 + i;
        int64_t* dst_ = dst + i;

        int64xm1_t vdata0, vdata1, vdata2, vdata3;
        int64xm1_t vdata4, vdata5, vdata6, vdata7;
        int64xm1_t vdata8, vdata9, vdata10, vdata11;
        int64xm1_t vdata12, vdata13, vdata14, vdata15;

        vdata0 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 0 * 2, vl));
        vdata1 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 1 * 2, vl));
        vdata2 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 2 * 2, vl));
        vdata3 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 3 * 2, vl));
        vdata4 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 4 * 2, vl));
        vdata5 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 5 * 2, vl));
        vdata6 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 6 * 2, vl));
        vdata7 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 7 * 2, vl));
        vdata8 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 8 * 2, vl));
        vdata9 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 9 * 2, vl));
        vdata10 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 10 * 2, vl));
        vdata11 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 11 * 2, vl));
        vdata12 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 12 * 2, vl));
        vdata13 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 13 * 2, vl));
        vdata14 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 14 * 2, vl));
        vdata15 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_ + 15 * 2, vl));

        if (fuse_relu) {
            vsev_int64xm1(dst_ + 0 * 2, vmaxvx_int64xm1(vdata0, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 1 * 2, vmaxvx_int64xm1(vdata1, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 2 * 2, vmaxvx_int64xm1(vdata2, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 3 * 2, vmaxvx_int64xm1(vdata3, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 4 * 2, vmaxvx_int64xm1(vdata4, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 5 * 2, vmaxvx_int64xm1(vdata5, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 6 * 2, vmaxvx_int64xm1(vdata6, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 7 * 2, vmaxvx_int64xm1(vdata7, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 8 * 2, vmaxvx_int64xm1(vdata8, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 9 * 2, vmaxvx_int64xm1(vdata9, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 10 * 2, vmaxvx_int64xm1(vdata10, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 11 * 2, vmaxvx_int64xm1(vdata11, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 12 * 2, vmaxvx_int64xm1(vdata12, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 13 * 2, vmaxvx_int64xm1(vdata13, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 14 * 2, vmaxvx_int64xm1(vdata14, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 15 * 2, vmaxvx_int64xm1(vdata15, (int64_t)0, vl), vl);
        } else {
            vsev_int64xm1(dst_ + 0 * 2, vdata0, vl);
            vsev_int64xm1(dst_ + 1 * 2, vdata1, vl);
            vsev_int64xm1(dst_ + 2 * 2, vdata2, vl);
            vsev_int64xm1(dst_ + 3 * 2, vdata3, vl);
            vsev_int64xm1(dst_ + 4 * 2, vdata4, vl);
            vsev_int64xm1(dst_ + 5 * 2, vdata5, vl);
            vsev_int64xm1(dst_ + 6 * 2, vdata6, vl);
            vsev_int64xm1(dst_ + 7 * 2, vdata7, vl);
            vsev_int64xm1(dst_ + 8 * 2, vdata8, vl);
            vsev_int64xm1(dst_ + 9 * 2, vdata9, vl);
            vsev_int64xm1(dst_ + 10 * 2, vdata10, vl);
            vsev_int64xm1(dst_ + 11 * 2, vdata11, vl);
            vsev_int64xm1(dst_ + 12 * 2, vdata12, vl);
            vsev_int64xm1(dst_ + 13 * 2, vdata13, vl);
            vsev_int64xm1(dst_ + 14 * 2, vdata14, vl);
            vsev_int64xm1(dst_ + 15 * 2, vdata15, vl);
        }
    }
    for (; i + 2 < end; i += 2) {
        const int64_t* src1_ = src1 + i;
        int64_t* dst_ = dst + i;

        int64xm1_t vdata;
        vdata = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src1_, vl));
        if (fuse_relu) {
            vsev_int64xm1(dst_, vmaxvx_int64xm1(vdata, (int64_t)0, vl), vl);
        } else {
            vsev_int64xm1(dst_, vdata, vl);
        }
    }
    for (; i < end; i++) {
        dst[i] = arithmetic_scalar_kernel_int64<op>(broadcast_val, src1[i]);
        if (fuse_relu) {
            dst[i] = std::max(dst[i], (int64_t)0);
        }
    }
}

template <arithmetic_op_type_t op, bool fuse_relu>
static inline void arithmetic_broadcast_lastdim_src1_broadcast_ndarray_int64(const int64_t* src0, const int64_t* src1,
                                                                             int64_t* dst,

                                                                             const int64_t start, const int64_t end) {
    const int64_t parall_d = 16;
    const int64_t unroll_len = parall_d * 2;
    const auto vl = vsetvli(2, RVV_E64, RVV_M1);

    const int64_t broadcast_val = src1[0];
    int64xm1_t v_broadcast_val = vmvvx_int64xm1(broadcast_val, vl);

    int64_t i = start;
    for (; i + unroll_len < end; i += unroll_len) {
        const int64_t* src0_ = src0 + i;
        int64_t* dst_ = dst + i;

        int64xm1_t vdata0, vdata1, vdata2, vdata3;
        int64xm1_t vdata4, vdata5, vdata6, vdata7;
        int64xm1_t vdata8, vdata9, vdata10, vdata11;
        int64xm1_t vdata12, vdata13, vdata14, vdata15;

        vdata0 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 0 * 2, vl));
        vdata1 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 1 * 2, vl));
        vdata2 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 2 * 2, vl));
        vdata3 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 3 * 2, vl));
        vdata4 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 4 * 2, vl));
        vdata5 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 5 * 2, vl));
        vdata6 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 6 * 2, vl));
        vdata7 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 7 * 2, vl));
        vdata8 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 8 * 2, vl));
        vdata9 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 9 * 2, vl));
        vdata10 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 10 * 2, vl));
        vdata11 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 11 * 2, vl));
        vdata12 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 12 * 2, vl));
        vdata13 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 13 * 2, vl));
        vdata14 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 14 * 2, vl));
        vdata15 = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_ + 15 * 2, vl));

        if (fuse_relu) {
            vsev_int64xm1(dst_ + 0 * 2, vmaxvx_int64xm1(vdata0, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 1 * 2, vmaxvx_int64xm1(vdata1, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 2 * 2, vmaxvx_int64xm1(vdata2, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 3 * 2, vmaxvx_int64xm1(vdata3, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 4 * 2, vmaxvx_int64xm1(vdata4, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 5 * 2, vmaxvx_int64xm1(vdata5, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 6 * 2, vmaxvx_int64xm1(vdata6, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 7 * 2, vmaxvx_int64xm1(vdata7, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 8 * 2, vmaxvx_int64xm1(vdata8, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 9 * 2, vmaxvx_int64xm1(vdata9, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 10 * 2, vmaxvx_int64xm1(vdata10, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 11 * 2, vmaxvx_int64xm1(vdata11, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 12 * 2, vmaxvx_int64xm1(vdata12, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 13 * 2, vmaxvx_int64xm1(vdata13, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 14 * 2, vmaxvx_int64xm1(vdata14, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 15 * 2, vmaxvx_int64xm1(vdata15, (int64_t)0, vl), vl);
        } else {
            vsev_int64xm1(dst_ + 0 * 2, vdata0, vl);
            vsev_int64xm1(dst_ + 1 * 2, vdata1, vl);
            vsev_int64xm1(dst_ + 2 * 2, vdata2, vl);
            vsev_int64xm1(dst_ + 3 * 2, vdata3, vl);
            vsev_int64xm1(dst_ + 4 * 2, vdata4, vl);
            vsev_int64xm1(dst_ + 5 * 2, vdata5, vl);
            vsev_int64xm1(dst_ + 6 * 2, vdata6, vl);
            vsev_int64xm1(dst_ + 7 * 2, vdata7, vl);
            vsev_int64xm1(dst_ + 8 * 2, vdata8, vl);
            vsev_int64xm1(dst_ + 9 * 2, vdata9, vl);
            vsev_int64xm1(dst_ + 10 * 2, vdata10, vl);
            vsev_int64xm1(dst_ + 11 * 2, vdata11, vl);
            vsev_int64xm1(dst_ + 12 * 2, vdata12, vl);
            vsev_int64xm1(dst_ + 13 * 2, vdata13, vl);
            vsev_int64xm1(dst_ + 14 * 2, vdata14, vl);
            vsev_int64xm1(dst_ + 15 * 2, vdata15, vl);
        }
    }
    for (; i + 2 < end; i += 2) {
        const int64_t* src0_ = src0 + i;
        int64_t* dst_ = dst + i;

        int64xm1_t vdata;
        vdata = arithmetic_vector_kernel_int64<op>(v_broadcast_val, vlev_int64xm1(src0_, vl));
        if (fuse_relu) {
            vsev_int64xm1(dst_, vmaxvx_int64xm1(vdata, (int64_t)0, vl), vl);
        } else {
            vsev_int64xm1(dst_, vdata, vl);
        }
    }
    for (; i < end; i++) {
        dst[i] = arithmetic_scalar_kernel_int64<op>(broadcast_val, src0[i]);
        if (fuse_relu) {
            dst[i] = std::max(dst[i], (int64_t)0);
        }
    }
}

template <arithmetic_op_type_t op, bool fuse_relu>
static ppl::common::RetCode arithmetic_broadcast_recursive_ndarray_int64(
    const int64_t* src0, const int64_t* src1, int64_t* dst,

    const int64_t* src0_shape, const int64_t* src1_shape, const int64_t* dst_shape, const int64_t* inc0,
    const int64_t* inc1, const int64_t* inc_out, const int64_t dim_count, const int64_t dim_idx,
    parallel_block* block) {
    bool is_first = is_first_dim(block, dim_idx);
    bool is_last = is_last_dim(block, dim_idx);
    const int64_t start = is_first ? block->start[dim_idx] : 0;
    const int64_t end = is_last ? block->end[dim_idx] : dst_shape[dim_idx] - 1;

    if (dim_idx == dim_count - 1) {
        if (src0_shape[dim_idx] == src1_shape[dim_idx]) {
            arithmetic_broadcast_lastdim_no_broadcast_ndarray_int64<op, fuse_relu>(src0, src1, dst, start, end);
        } else if (src0_shape[dim_idx] == 1) {
            arithmetic_broadcast_lastdim_src0_broadcast_ndarray_int64<op, fuse_relu>(src0, src1, dst, start, end);
        } else if (src1_shape[dim_idx] == 1) {
            arithmetic_broadcast_lastdim_src1_broadcast_ndarray_int64<op, fuse_relu>(src0, src1, dst, start, end);
        }
    } else {
        for (block->idx[dim_idx] = start; block->idx[dim_idx] <= end; block->idx[dim_idx]++) {
            int64_t i = block->idx[dim_idx];
            arithmetic_broadcast_recursive_ndarray_int64<op, fuse_relu>(
                src0 + i * inc0[dim_idx], src1 + i * inc1[dim_idx], dst + i * inc_out[dim_idx],

                src0_shape, src1_shape, dst_shape, inc0, inc1, inc_out, dim_count, dim_idx + 1, block);
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <arithmetic_op_type_t op, bool fuse_relu>
static ppl::common::RetCode arithmetic_broadcast_ndarray_int64(const int64_t* src0, const int64_t* src1, int64_t* dst,

                                                               const ppl::nn::TensorShape* src0_shape,
                                                               const ppl::nn::TensorShape* src1_shape,
                                                               const ppl::nn::TensorShape* dst_shape) {
    // pad 1 to input's high dims
    const int64_t dim_count = dst_shape->GetDimCount();
    if (dim_count > PPL_RISCV_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    int64_t padded_src0_shape[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    int64_t padded_src1_shape[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    pad_shape(src0_shape, dim_count, padded_src0_shape);
    pad_shape(src1_shape, dim_count, padded_src1_shape);

    // compress dims
    int64_t real_dim_count = 0;
    int64_t real_src0_shape[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    int64_t real_src1_shape[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    int64_t real_dst_shape[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};

    // remove 1 on high dims to compress dim count
    for (int64_t i = 0; i < dim_count; i++) {
        if (dst_shape->GetDim(i) <= 1 && i != dim_count - 1) {
            continue;
        }
        real_src0_shape[real_dim_count] = padded_src0_shape[i];
        real_src1_shape[real_dim_count] = padded_src1_shape[i];
        real_dst_shape[real_dim_count] = dst_shape->GetDim(i);
        real_dim_count++;
    }

    // merge low dims
    for (int64_t i = real_dim_count - 1; i >= 1; i--) {
        bool cur_dim_input0_need_broadcast = real_src0_shape[i] != real_src1_shape[i] && real_src0_shape[i] == 1;
        bool cur_dim_input1_need_broadcast = real_src0_shape[i] != real_src1_shape[i] && real_src1_shape[i] == 1;
        bool prev_dim_input0_need_broadcast =
            real_src0_shape[i - 1] != real_src1_shape[i - 1] && real_src0_shape[i - 1] == 1;
        bool prev_dim_input1_need_broadcast =
            real_src0_shape[i - 1] != real_src1_shape[i - 1] && real_src1_shape[i - 1] == 1;

        if (cur_dim_input0_need_broadcast == prev_dim_input0_need_broadcast && // can merge
            cur_dim_input1_need_broadcast == prev_dim_input1_need_broadcast) {
            real_src0_shape[i - 1] *= real_src0_shape[i];
            real_src1_shape[i - 1] *= real_src1_shape[i];
            real_dst_shape[i - 1] *= real_dst_shape[i];
            real_dim_count--;
        } else {
            break;
        }
    }

    int64_t inc0[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    int64_t inc1[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    int64_t inc_out[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};

    int64_t stride0 = 1;
    int64_t stride1 = 1;
    int64_t stride_out = 1;

    // prepare incs
    for (int64_t i = real_dim_count - 1; i >= 0; i--) {
        inc0[i] = real_src0_shape[i] == 1 ? 0 : stride0;
        inc1[i] = real_src1_shape[i] == 1 ? 0 : stride1;
        inc_out[i] = stride_out;

        stride0 *= real_src0_shape[i];
        stride1 *= real_src1_shape[i];
        stride_out *= real_dst_shape[i];
    }

    const int64_t total_len = dst_shape->GetElementsExcludingPadding();
    parallel_block block;
    {
        int64_t start_idx = 0;
        int64_t end_idx = total_len - 1;
        idx2dims(start_idx, real_dst_shape, real_dim_count, block.start);
        idx2dims(end_idx, real_dst_shape, real_dim_count, block.end);
        block.id = 0;
        for (int64_t i = 0; i < real_dim_count; i++) {
            block.idx[i] = block.start[i];
        }
    }

    arithmetic_broadcast_recursive_ndarray_int64<op, fuse_relu>(src0, src1, dst,

                                                                real_src0_shape, real_src1_shape, real_dst_shape, inc0,
                                                                inc1, inc_out, real_dim_count, 0, &block);

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_INT64_ARITHMETIC_ARITHMETIC_BROADCAST_NDARRAY_INT64_H_