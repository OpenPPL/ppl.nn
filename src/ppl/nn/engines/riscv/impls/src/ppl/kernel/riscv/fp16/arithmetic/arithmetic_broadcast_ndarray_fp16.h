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

#ifndef __ST_PPL_KERNEL_RISCV_FP16_ARITHMETIC_ARITHMETIC_BROADCAST_NDARRAY_FP16_H_
#define __ST_PPL_KERNEL_RISCV_FP16_ARITHMETIC_ARITHMETIC_BROADCAST_NDARRAY_FP16_H_

#include "ppl/nn/runtime/tensor_impl.h"
#include "arithmetic_kernel_fp16.h"
#include <cmath>

namespace ppl { namespace kernel { namespace riscv {

template <arithmetic_op_type_t op, bool fuse_relu>
static inline void arithmetic_broadcast_lastdim_no_broadcast_ndarray_fp16(
    const __fp16* src0,
    const __fp16* src1,
    __fp16* dst,

    const int64_t start,
    const int64_t end)
{
    const int64_t parall_d   = 16;
    const int64_t unroll_len = parall_d * 8;
    const auto vl            = vsetvli(8, RVV_E16, RVV_M1);

    int64_t i = start;
    for (; i + unroll_len <= end; i += unroll_len) {
        const __fp16* src0_ = src0 + i;
        const __fp16* src1_ = src1 + i;
        __fp16* dst_        = dst + i;

        float16xm1_t vfdata0, vfdata1, vfdata2, vfdata3;
        float16xm1_t vfdata4, vfdata5, vfdata6, vfdata7;
        float16xm1_t vfdata8, vfdata9, vfdata10, vfdata11;
        float16xm1_t vfdata12, vfdata13, vfdata14, vfdata15;

        vfdata0  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 0 * 8, vl), vlev_float16xm1(src1_ + 0 * 8, vl));
        vfdata1  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 1 * 8, vl), vlev_float16xm1(src1_ + 1 * 8, vl));
        vfdata2  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 2 * 8, vl), vlev_float16xm1(src1_ + 2 * 8, vl));
        vfdata3  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 3 * 8, vl), vlev_float16xm1(src1_ + 3 * 8, vl));
        vfdata4  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 4 * 8, vl), vlev_float16xm1(src1_ + 4 * 8, vl));
        vfdata5  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 5 * 8, vl), vlev_float16xm1(src1_ + 5 * 8, vl));
        vfdata6  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 6 * 8, vl), vlev_float16xm1(src1_ + 6 * 8, vl));
        vfdata7  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 7 * 8, vl), vlev_float16xm1(src1_ + 7 * 8, vl));
        vfdata8  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 8 * 8, vl), vlev_float16xm1(src1_ + 8 * 8, vl));
        vfdata9  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 9 * 8, vl), vlev_float16xm1(src1_ + 9 * 8, vl));
        vfdata10 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 10 * 8, vl), vlev_float16xm1(src1_ + 10 * 8, vl));
        vfdata11 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 11 * 8, vl), vlev_float16xm1(src1_ + 11 * 8, vl));
        vfdata12 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 12 * 8, vl), vlev_float16xm1(src1_ + 12 * 8, vl));
        vfdata13 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 13 * 8, vl), vlev_float16xm1(src1_ + 13 * 8, vl));
        vfdata14 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 14 * 8, vl), vlev_float16xm1(src1_ + 14 * 8, vl));
        vfdata15 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 15 * 8, vl), vlev_float16xm1(src1_ + 15 * 8, vl));

        if (fuse_relu) {
            vsev_float16xm1(dst_ + 0 * 8, vfmaxvf_float16xm1(vfdata0, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 1 * 8, vfmaxvf_float16xm1(vfdata1, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 2 * 8, vfmaxvf_float16xm1(vfdata2, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 3 * 8, vfmaxvf_float16xm1(vfdata3, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 4 * 8, vfmaxvf_float16xm1(vfdata4, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 5 * 8, vfmaxvf_float16xm1(vfdata5, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 6 * 8, vfmaxvf_float16xm1(vfdata6, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 7 * 8, vfmaxvf_float16xm1(vfdata7, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 8 * 8, vfmaxvf_float16xm1(vfdata8, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 9 * 8, vfmaxvf_float16xm1(vfdata9, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 10 * 8, vfmaxvf_float16xm1(vfdata10, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 11 * 8, vfmaxvf_float16xm1(vfdata11, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 12 * 8, vfmaxvf_float16xm1(vfdata12, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 13 * 8, vfmaxvf_float16xm1(vfdata13, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 14 * 8, vfmaxvf_float16xm1(vfdata14, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 15 * 8, vfmaxvf_float16xm1(vfdata15, (__fp16)0.0f, vl), vl);
        } else {
            vsev_float16xm1(dst_ + 0 * 8, vfdata0, vl);
            vsev_float16xm1(dst_ + 1 * 8, vfdata1, vl);
            vsev_float16xm1(dst_ + 2 * 8, vfdata2, vl);
            vsev_float16xm1(dst_ + 3 * 8, vfdata3, vl);
            vsev_float16xm1(dst_ + 4 * 8, vfdata4, vl);
            vsev_float16xm1(dst_ + 5 * 8, vfdata5, vl);
            vsev_float16xm1(dst_ + 6 * 8, vfdata6, vl);
            vsev_float16xm1(dst_ + 7 * 8, vfdata7, vl);
            vsev_float16xm1(dst_ + 8 * 8, vfdata8, vl);
            vsev_float16xm1(dst_ + 9 * 8, vfdata9, vl);
            vsev_float16xm1(dst_ + 10 * 8, vfdata10, vl);
            vsev_float16xm1(dst_ + 11 * 8, vfdata11, vl);
            vsev_float16xm1(dst_ + 12 * 8, vfdata12, vl);
            vsev_float16xm1(dst_ + 13 * 8, vfdata13, vl);
            vsev_float16xm1(dst_ + 14 * 8, vfdata14, vl);
            vsev_float16xm1(dst_ + 15 * 8, vfdata15, vl);
        }
    }
    for (; i + 8 <= end; i += 8) {
        const __fp16* src0_ = src0 + i;
        const __fp16* src1_ = src1 + i;
        __fp16* dst_        = dst + i;

        float16xm1_t vfdata;
        vfdata = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_, vl), vlev_float16xm1(src1_, vl));
        if (fuse_relu) {
            vsev_float16xm1(dst_, vfmaxvf_float16xm1(vfdata, (__fp16)0.0f, vl), vl);
        } else {
            vsev_float16xm1(dst_, vfdata, vl);
        }
    }
    for (; i <= end; i++) {
        dst[i] = arithmetic_scalar_kernel_fp16<op>(src0[i], src1[i]);
        if (fuse_relu) {
            dst[i] = std::max(dst[i], (__fp16)0.0f);
        }
    }
}

template <arithmetic_op_type_t op, bool fuse_relu>
static inline void arithmetic_broadcast_lastdim_src0_broadcast_ndarray_fp16(
    const __fp16* src0,
    const __fp16* src1,
    __fp16* dst,

    const int64_t start,
    const int64_t end)
{
    const int64_t parall_d   = 16;
    const int64_t unroll_len = parall_d * 8;
    const auto vl            = vsetvli(8, RVV_E16, RVV_M1);

    const __fp16 broadcast_val   = src0[0];
    float16xm1_t v_broadcast_val = vfmvvf_float16xm1(broadcast_val, vl);

    int64_t i = start;
    for (; i + unroll_len <= end; i += unroll_len) {
        const __fp16* src1_ = src1 + i;
        __fp16* dst_        = dst + i;

        float16xm1_t vfdata0, vfdata1, vfdata2, vfdata3;
        float16xm1_t vfdata4, vfdata5, vfdata6, vfdata7;
        float16xm1_t vfdata8, vfdata9, vfdata10, vfdata11;
        float16xm1_t vfdata12, vfdata13, vfdata14, vfdata15;

        vfdata0  = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 0 * 8, vl));
        vfdata1  = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 1 * 8, vl));
        vfdata2  = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 2 * 8, vl));
        vfdata3  = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 3 * 8, vl));
        vfdata4  = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 4 * 8, vl));
        vfdata5  = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 5 * 8, vl));
        vfdata6  = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 6 * 8, vl));
        vfdata7  = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 7 * 8, vl));
        vfdata8  = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 8 * 8, vl));
        vfdata9  = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 9 * 8, vl));
        vfdata10 = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 10 * 8, vl));
        vfdata11 = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 11 * 8, vl));
        vfdata12 = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 12 * 8, vl));
        vfdata13 = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 13 * 8, vl));
        vfdata14 = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 14 * 8, vl));
        vfdata15 = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_ + 15 * 8, vl));

        if (fuse_relu) {
            vsev_float16xm1(dst_ + 0 * 8, vfmaxvf_float16xm1(vfdata0, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 1 * 8, vfmaxvf_float16xm1(vfdata1, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 2 * 8, vfmaxvf_float16xm1(vfdata2, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 3 * 8, vfmaxvf_float16xm1(vfdata3, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 4 * 8, vfmaxvf_float16xm1(vfdata4, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 5 * 8, vfmaxvf_float16xm1(vfdata5, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 6 * 8, vfmaxvf_float16xm1(vfdata6, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 7 * 8, vfmaxvf_float16xm1(vfdata7, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 8 * 8, vfmaxvf_float16xm1(vfdata8, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 9 * 8, vfmaxvf_float16xm1(vfdata9, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 10 * 8, vfmaxvf_float16xm1(vfdata10, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 11 * 8, vfmaxvf_float16xm1(vfdata11, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 12 * 8, vfmaxvf_float16xm1(vfdata12, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 13 * 8, vfmaxvf_float16xm1(vfdata13, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 14 * 8, vfmaxvf_float16xm1(vfdata14, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 15 * 8, vfmaxvf_float16xm1(vfdata15, (__fp16)0.0f, vl), vl);
        } else {
            vsev_float16xm1(dst_ + 0 * 8, vfdata0, vl);
            vsev_float16xm1(dst_ + 1 * 8, vfdata1, vl);
            vsev_float16xm1(dst_ + 2 * 8, vfdata2, vl);
            vsev_float16xm1(dst_ + 3 * 8, vfdata3, vl);
            vsev_float16xm1(dst_ + 4 * 8, vfdata4, vl);
            vsev_float16xm1(dst_ + 5 * 8, vfdata5, vl);
            vsev_float16xm1(dst_ + 6 * 8, vfdata6, vl);
            vsev_float16xm1(dst_ + 7 * 8, vfdata7, vl);
            vsev_float16xm1(dst_ + 8 * 8, vfdata8, vl);
            vsev_float16xm1(dst_ + 9 * 8, vfdata9, vl);
            vsev_float16xm1(dst_ + 10 * 8, vfdata10, vl);
            vsev_float16xm1(dst_ + 11 * 8, vfdata11, vl);
            vsev_float16xm1(dst_ + 12 * 8, vfdata12, vl);
            vsev_float16xm1(dst_ + 13 * 8, vfdata13, vl);
            vsev_float16xm1(dst_ + 14 * 8, vfdata14, vl);
            vsev_float16xm1(dst_ + 15 * 8, vfdata15, vl);
        }
    }
    for (; i + 8 <= end; i += 8) {
        const __fp16* src1_ = src1 + i;
        __fp16* dst_        = dst + i;

        float16xm1_t vfdata;
        vfdata = arithmetic_vector_kernel_fp16<op>(v_broadcast_val, vlev_float16xm1(src1_, vl));
        if (fuse_relu) {
            vsev_float16xm1(dst_, vfmaxvf_float16xm1(vfdata, (__fp16)0.0f, vl), vl);
        } else {
            vsev_float16xm1(dst_, vfdata, vl);
        }
    }
    for (; i <= end; i++) {
        dst[i] = arithmetic_scalar_kernel_fp16<op>(broadcast_val, src1[i]);
        if (fuse_relu) {
            dst[i] = std::max(dst[i], (__fp16)0.0f);
        }
    }
}

template <arithmetic_op_type_t op, bool fuse_relu>
static inline void arithmetic_broadcast_lastdim_src1_broadcast_ndarray_fp16(
    const __fp16* src0,
    const __fp16* src1,
    __fp16* dst,

    const int64_t start,
    const int64_t end)
{
    const int64_t parall_d   = 16;
    const int64_t unroll_len = parall_d * 8;
    const auto vl            = vsetvli(8, RVV_E16, RVV_M1);

    const __fp16 broadcast_val   = src1[0];
    float16xm1_t v_broadcast_val = vfmvvf_float16xm1(broadcast_val, vl);

    int64_t i = start;
    for (; i + unroll_len <= end; i += unroll_len) {
        const __fp16* src0_ = src0 + i;
        __fp16* dst_        = dst + i;

        float16xm1_t vfdata0, vfdata1, vfdata2, vfdata3;
        float16xm1_t vfdata4, vfdata5, vfdata6, vfdata7;
        float16xm1_t vfdata8, vfdata9, vfdata10, vfdata11;
        float16xm1_t vfdata12, vfdata13, vfdata14, vfdata15;

        vfdata0  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 0  * 8, vl), v_broadcast_val);
        vfdata1  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 1  * 8, vl), v_broadcast_val);
        vfdata2  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 2  * 8, vl), v_broadcast_val);
        vfdata3  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 3  * 8, vl), v_broadcast_val);
        vfdata4  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 4  * 8, vl), v_broadcast_val);
        vfdata5  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 5  * 8, vl), v_broadcast_val);
        vfdata6  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 6  * 8, vl), v_broadcast_val);
        vfdata7  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 7  * 8, vl), v_broadcast_val);
        vfdata8  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 8  * 8, vl), v_broadcast_val);
        vfdata9  = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 9  * 8, vl), v_broadcast_val);
        vfdata10 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 10 * 8, vl), v_broadcast_val);
        vfdata11 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 11 * 8, vl), v_broadcast_val);
        vfdata12 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 12 * 8, vl), v_broadcast_val);
        vfdata13 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 13 * 8, vl), v_broadcast_val);
        vfdata14 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 14 * 8, vl), v_broadcast_val);
        vfdata15 = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_ + 15 * 8, vl), v_broadcast_val);

        if (fuse_relu) {
            vsev_float16xm1(dst_ + 0 * 8, vfmaxvf_float16xm1(vfdata0, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 1 * 8, vfmaxvf_float16xm1(vfdata1, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 2 * 8, vfmaxvf_float16xm1(vfdata2, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 3 * 8, vfmaxvf_float16xm1(vfdata3, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 4 * 8, vfmaxvf_float16xm1(vfdata4, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 5 * 8, vfmaxvf_float16xm1(vfdata5, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 6 * 8, vfmaxvf_float16xm1(vfdata6, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 7 * 8, vfmaxvf_float16xm1(vfdata7, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 8 * 8, vfmaxvf_float16xm1(vfdata8, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 9 * 8, vfmaxvf_float16xm1(vfdata9, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 10 * 8, vfmaxvf_float16xm1(vfdata10, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 11 * 8, vfmaxvf_float16xm1(vfdata11, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 12 * 8, vfmaxvf_float16xm1(vfdata12, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 13 * 8, vfmaxvf_float16xm1(vfdata13, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 14 * 8, vfmaxvf_float16xm1(vfdata14, (__fp16)0.0f, vl), vl);
            vsev_float16xm1(dst_ + 15 * 8, vfmaxvf_float16xm1(vfdata15, (__fp16)0.0f, vl), vl);
        } else {
            vsev_float16xm1(dst_ + 0 * 8, vfdata0, vl);
            vsev_float16xm1(dst_ + 1 * 8, vfdata1, vl);
            vsev_float16xm1(dst_ + 2 * 8, vfdata2, vl);
            vsev_float16xm1(dst_ + 3 * 8, vfdata3, vl);
            vsev_float16xm1(dst_ + 4 * 8, vfdata4, vl);
            vsev_float16xm1(dst_ + 5 * 8, vfdata5, vl);
            vsev_float16xm1(dst_ + 6 * 8, vfdata6, vl);
            vsev_float16xm1(dst_ + 7 * 8, vfdata7, vl);
            vsev_float16xm1(dst_ + 8 * 8, vfdata8, vl);
            vsev_float16xm1(dst_ + 9 * 8, vfdata9, vl);
            vsev_float16xm1(dst_ + 10 * 8, vfdata10, vl);
            vsev_float16xm1(dst_ + 11 * 8, vfdata11, vl);
            vsev_float16xm1(dst_ + 12 * 8, vfdata12, vl);
            vsev_float16xm1(dst_ + 13 * 8, vfdata13, vl);
            vsev_float16xm1(dst_ + 14 * 8, vfdata14, vl);
            vsev_float16xm1(dst_ + 15 * 8, vfdata15, vl);
        }
    }
    for (; i + 8 <= end; i += 8) {
        const __fp16* src0_ = src0 + i;
        __fp16* dst_        = dst + i;

        float16xm1_t vfdata;
        vfdata = arithmetic_vector_kernel_fp16<op>(vlev_float16xm1(src0_, vl), v_broadcast_val);
        if (fuse_relu) {
            vsev_float16xm1(dst_, vfmaxvf_float16xm1(vfdata, (__fp16)0.0f, vl), vl);
        } else {
            vsev_float16xm1(dst_, vfdata, vl);
        }
    }
    for (; i <= end; i++) {
        dst[i] = arithmetic_scalar_kernel_fp16<op>(src0[i], broadcast_val);
        if (fuse_relu) {
            dst[i] = std::max(dst[i], (__fp16)0.0f);
        }
    }
}

template <arithmetic_op_type_t op, bool fuse_relu>
static ppl::common::RetCode arithmetic_broadcast_recursive_ndarray_fp16(
    const __fp16* src0,
    const __fp16* src1,
    __fp16* dst,

    const int64_t* src0_shape,
    const int64_t* src1_shape,
    const int64_t* dst_shape,
    const int64_t* inc0,
    const int64_t* inc1,
    const int64_t* inc_out,
    const int64_t dim_count,
    const int64_t dim_idx,
    parallel_block* block)
{
    bool is_first       = is_first_dim(block, dim_idx);
    bool is_last        = is_last_dim(block, dim_idx);
    const int64_t start = is_first ? block->start[dim_idx] : 0;
    const int64_t end   = is_last ? block->end[dim_idx] : dst_shape[dim_idx] - 1;

    if (dim_idx == dim_count - 1) {
        if (src0_shape[dim_idx] == src1_shape[dim_idx]) {
            arithmetic_broadcast_lastdim_no_broadcast_ndarray_fp16<op, fuse_relu>(src0, src1, dst, start, end);
        } else if (src0_shape[dim_idx] == 1) {
            arithmetic_broadcast_lastdim_src0_broadcast_ndarray_fp16<op, fuse_relu>(src0, src1, dst, start, end);
        } else if (src1_shape[dim_idx] == 1) {
            arithmetic_broadcast_lastdim_src1_broadcast_ndarray_fp16<op, fuse_relu>(src0, src1, dst, start, end);
        }
    } else {
        for (block->idx[dim_idx] = start; block->idx[dim_idx] <= end; block->idx[dim_idx]++) {
            int64_t i = block->idx[dim_idx];
            arithmetic_broadcast_recursive_ndarray_fp16<op, fuse_relu>(
                src0 + i * inc0[dim_idx],
                src1 + i * inc1[dim_idx],
                dst + i * inc_out[dim_idx],

                src0_shape,
                src1_shape,
                dst_shape,
                inc0,
                inc1,
                inc_out,
                dim_count,
                dim_idx + 1,
                block);
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <arithmetic_op_type_t op, bool fuse_relu>
static ppl::common::RetCode arithmetic_broadcast_ndarray_fp16(
    const __fp16* src0,
    const __fp16* src1,
    __fp16* dst,

    const ppl::nn::TensorShape* src0_shape,
    const ppl::nn::TensorShape* src1_shape,
    const ppl::nn::TensorShape* dst_shape)
{
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
    int64_t real_dim_count                               = 0;
    int64_t real_src0_shape[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    int64_t real_src1_shape[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    int64_t real_dst_shape[PPL_RISCV_TENSOR_MAX_DIMS()]  = {0};

    // remove 1 on high dims to compress dim count
    for (int64_t i = 0; i < dim_count; i++) {
        if (dst_shape->GetDim(i) <= 1 && i != dim_count - 1) {
            continue;
        }
        real_src0_shape[real_dim_count] = padded_src0_shape[i];
        real_src1_shape[real_dim_count] = padded_src1_shape[i];
        real_dst_shape[real_dim_count]  = dst_shape->GetDim(i);
        real_dim_count++;
    }

    // merge low dims
    for (int64_t i = real_dim_count - 1; i >= 1; i--) {
        bool cur_dim_input0_need_broadcast  = real_src0_shape[i] != real_src1_shape[i] && real_src0_shape[i] == 1;
        bool cur_dim_input1_need_broadcast  = real_src0_shape[i] != real_src1_shape[i] && real_src1_shape[i] == 1;
        bool prev_dim_input0_need_broadcast = real_src0_shape[i - 1] != real_src1_shape[i - 1] && real_src0_shape[i - 1] == 1;
        bool prev_dim_input1_need_broadcast = real_src0_shape[i - 1] != real_src1_shape[i - 1] && real_src1_shape[i - 1] == 1;

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

    int64_t inc0[PPL_RISCV_TENSOR_MAX_DIMS()]    = {0};
    int64_t inc1[PPL_RISCV_TENSOR_MAX_DIMS()]    = {0};
    int64_t inc_out[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};

    int64_t stride0    = 1;
    int64_t stride1    = 1;
    int64_t stride_out = 1;

    // prepare incs
    for (int64_t i = real_dim_count - 1; i >= 0; i--) {
        inc0[i]    = real_src0_shape[i] == 1 ? 0 : stride0;
        inc1[i]    = real_src1_shape[i] == 1 ? 0 : stride1;
        inc_out[i] = stride_out;

        stride0 *= real_src0_shape[i];
        stride1 *= real_src1_shape[i];
        stride_out *= real_dst_shape[i];
    }

    const int64_t total_len = dst_shape->CalcElementsExcludingPadding();
    parallel_block block;
    {
        int64_t start_idx = 0;
        int64_t end_idx   = total_len - 1;
        idx2dims(start_idx, real_dst_shape, real_dim_count, block.start);
        idx2dims(end_idx, real_dst_shape, real_dim_count, block.end);
        block.id = 0;
        for (int64_t i = 0; i < real_dim_count; i++) {
            block.idx[i] = block.start[i];
        }
    }

    arithmetic_broadcast_recursive_ndarray_fp16<op, fuse_relu>(
        src0,
        src1,
        dst,

        real_src0_shape,
        real_src1_shape,
        real_dst_shape,
        inc0,
        inc1,
        inc_out,
        real_dim_count,
        0,
        &block);

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP16_ARITHMETIC_ARITHMETIC_BROADCAST_NDARRAY_FP16_H_
