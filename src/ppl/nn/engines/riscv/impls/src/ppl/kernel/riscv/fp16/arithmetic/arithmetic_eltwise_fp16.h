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

#ifndef __ST_PPL_KERNEL_RISCV_FP16_ARITHMETIC_ARITHMETIC_ELTWISE_FP16_H_
#define __ST_PPL_KERNEL_RISCV_FP16_ARITHMETIC_ARITHMETIC_ELTWISE_FP16_H_

#include "ppl/nn/runtime/tensor_impl.h"
#include "arithmetic_kernel_fp16.h"

namespace ppl { namespace kernel { namespace riscv {

template <arithmetic_op_type_t _op, bool fuse_relu>
static ppl::common::RetCode arithmetic_eltwise_fp16(
    const ppl::nn::TensorShape* dst_shape,
    const __fp16* src0,
    const __fp16* src1,
    __fp16* dst)
{
    const int64_t total_len  = dst_shape->CalcElementsIncludingPadding();
    const int64_t parall_d   = 16;
    const int64_t unroll_len = parall_d * 8;
    const auto vl            = vsetvli(8, RVV_E16, RVV_M1);

    int64_t idx = 0;
    for (; idx + unroll_len < total_len; idx += unroll_len) {
        const __fp16* src0_ = src0 + idx;
        const __fp16* src1_ = src1 + idx;
        __fp16* dst_        = dst + idx;

        float16xm1_t vfdata0, vfdata1, vfdata2, vfdata3;
        float16xm1_t vfdata4, vfdata5, vfdata6, vfdata7;
        float16xm1_t vfdata8, vfdata9, vfdata10, vfdata11;
        float16xm1_t vfdata12, vfdata13, vfdata14, vfdata15;

        vfdata0  = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 0 * 8, vl), vlev_float16xm1(src1_ + 0 * 8, vl));
        vfdata1  = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 1 * 8, vl), vlev_float16xm1(src1_ + 1 * 8, vl));
        vfdata2  = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 2 * 8, vl), vlev_float16xm1(src1_ + 2 * 8, vl));
        vfdata3  = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 3 * 8, vl), vlev_float16xm1(src1_ + 3 * 8, vl));
        vfdata4  = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 4 * 8, vl), vlev_float16xm1(src1_ + 4 * 8, vl));
        vfdata5  = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 5 * 8, vl), vlev_float16xm1(src1_ + 5 * 8, vl));
        vfdata6  = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 6 * 8, vl), vlev_float16xm1(src1_ + 6 * 8, vl));
        vfdata7  = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 7 * 8, vl), vlev_float16xm1(src1_ + 7 * 8, vl));
        vfdata8  = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 8 * 8, vl), vlev_float16xm1(src1_ + 8 * 8, vl));
        vfdata9  = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 9 * 8, vl), vlev_float16xm1(src1_ + 9 * 8, vl));
        vfdata10 = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 10 * 8, vl), vlev_float16xm1(src1_ + 10 * 8, vl));
        vfdata11 = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 11 * 8, vl), vlev_float16xm1(src1_ + 11 * 8, vl));
        vfdata12 = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 12 * 8, vl), vlev_float16xm1(src1_ + 12 * 8, vl));
        vfdata13 = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 13 * 8, vl), vlev_float16xm1(src1_ + 13 * 8, vl));
        vfdata14 = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 14 * 8, vl), vlev_float16xm1(src1_ + 14 * 8, vl));
        vfdata15 = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_ + 15 * 8, vl), vlev_float16xm1(src1_ + 15 * 8, vl));

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
    for (; idx < total_len; idx += 8) {
        const __fp16* src0_ = src0 + idx;
        const __fp16* src1_ = src1 + idx;
        __fp16* dst_        = dst + idx;

        float16xm1_t vfdata;
        vfdata = arithmetic_vector_kernel_fp16<_op>(vlev_float16xm1(src0_, vl), vlev_float16xm1(src1_, vl));
        if (fuse_relu) {
            vsev_float16xm1(dst_, vfmaxvf_float16xm1(vfdata, (__fp16)0.0f, vl), vl);
        } else {
            vsev_float16xm1(dst_, vfdata, vl);
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP16_ARITHMETIC_ARITHMETIC_ELTWISE_FP16_H_
