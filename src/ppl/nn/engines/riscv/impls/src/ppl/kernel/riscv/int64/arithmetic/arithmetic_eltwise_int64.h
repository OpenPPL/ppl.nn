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

#ifndef __ST_PPL_KERNEL_RISCV_INT64_ARITHMETIC_ARITHMETIC_ELTWISE_INT64_H_
#define __ST_PPL_KERNEL_RISCV_INT64_ARITHMETIC_ARITHMETIC_ELTWISE_INT64_H_

#include "ppl/nn/runtime/tensor_impl.h"
#include "arithmetic_kernel_int64.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() ((int64_t)2)

template <arithmetic_op_type_t _op, bool fuse_relu>
static ppl::common::RetCode arithmetic_eltwise_int64(const ppl::nn::TensorShape* dst_shape, const int64_t* src0,
                                                     const int64_t* src1, int64_t* dst) {
    const int64_t total_len = dst_shape->GetElementsIncludingPadding();
    const int64_t parall_d = 16;
    const int64_t unroll_len = parall_d * C_BLK();
    const auto vl = vsetvli(C_BLK(), RVV_E64, RVV_M1);

    int64_t idx = 0;
    for (; idx + unroll_len < total_len; idx += unroll_len) {
        const int64_t* src0_ = src0 + idx;
        const int64_t* src1_ = src1 + idx;
        int64_t* dst_ = dst + idx;

        int64xm1_t vdata0, vdata1, vdata2, vdata3;
        int64xm1_t vdata4, vdata5, vdata6, vdata7;
        int64xm1_t vdata8, vdata9, vdata10, vdata11;
        int64xm1_t vdata12, vdata13, vdata14, vdata15;

        vdata0 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 0 * C_BLK(), vl),
                                                     vlev_int64xm1(src1_ + 0 * C_BLK(), vl));
        vdata1 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 1 * C_BLK(), vl),
                                                     vlev_int64xm1(src1_ + 1 * C_BLK(), vl));
        vdata2 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 2 * C_BLK(), vl),
                                                     vlev_int64xm1(src1_ + 2 * C_BLK(), vl));
        vdata3 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 3 * C_BLK(), vl),
                                                     vlev_int64xm1(src1_ + 3 * C_BLK(), vl));
        vdata4 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 4 * C_BLK(), vl),
                                                     vlev_int64xm1(src1_ + 4 * C_BLK(), vl));
        vdata5 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 5 * C_BLK(), vl),
                                                     vlev_int64xm1(src1_ + 5 * C_BLK(), vl));
        vdata6 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 6 * C_BLK(), vl),
                                                     vlev_int64xm1(src1_ + 6 * C_BLK(), vl));
        vdata7 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 7 * C_BLK(), vl),
                                                     vlev_int64xm1(src1_ + 7 * C_BLK(), vl));
        vdata8 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 8 * C_BLK(), vl),
                                                     vlev_int64xm1(src1_ + 8 * C_BLK(), vl));
        vdata9 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 9 * C_BLK(), vl),
                                                     vlev_int64xm1(src1_ + 9 * C_BLK(), vl));
        vdata10 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 10 * C_BLK(), vl),
                                                      vlev_int64xm1(src1_ + 10 * C_BLK(), vl));
        vdata11 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 11 * C_BLK(), vl),
                                                      vlev_int64xm1(src1_ + 11 * C_BLK(), vl));
        vdata12 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 12 * C_BLK(), vl),
                                                      vlev_int64xm1(src1_ + 12 * C_BLK(), vl));
        vdata13 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 13 * C_BLK(), vl),
                                                      vlev_int64xm1(src1_ + 13 * C_BLK(), vl));
        vdata14 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 14 * C_BLK(), vl),
                                                      vlev_int64xm1(src1_ + 14 * C_BLK(), vl));
        vdata15 = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_ + 15 * C_BLK(), vl),
                                                      vlev_int64xm1(src1_ + 15 * C_BLK(), vl));

        if (fuse_relu) {
            vsev_int64xm1(dst_ + 0 * C_BLK(), vmaxvx_int64xm1(vdata0, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 1 * C_BLK(), vmaxvx_int64xm1(vdata1, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 2 * C_BLK(), vmaxvx_int64xm1(vdata2, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 3 * C_BLK(), vmaxvx_int64xm1(vdata3, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 4 * C_BLK(), vmaxvx_int64xm1(vdata4, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 5 * C_BLK(), vmaxvx_int64xm1(vdata5, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 6 * C_BLK(), vmaxvx_int64xm1(vdata6, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 7 * C_BLK(), vmaxvx_int64xm1(vdata7, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 8 * C_BLK(), vmaxvx_int64xm1(vdata8, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 9 * C_BLK(), vmaxvx_int64xm1(vdata9, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 10 * C_BLK(), vmaxvx_int64xm1(vdata10, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 11 * C_BLK(), vmaxvx_int64xm1(vdata11, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 12 * C_BLK(), vmaxvx_int64xm1(vdata12, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 13 * C_BLK(), vmaxvx_int64xm1(vdata13, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 14 * C_BLK(), vmaxvx_int64xm1(vdata14, (int64_t)0, vl), vl);
            vsev_int64xm1(dst_ + 15 * C_BLK(), vmaxvx_int64xm1(vdata15, (int64_t)0, vl), vl);
        } else {
            vsev_int64xm1(dst_ + 0 * C_BLK(), vdata0, vl);
            vsev_int64xm1(dst_ + 1 * C_BLK(), vdata1, vl);
            vsev_int64xm1(dst_ + 2 * C_BLK(), vdata2, vl);
            vsev_int64xm1(dst_ + 3 * C_BLK(), vdata3, vl);
            vsev_int64xm1(dst_ + 4 * C_BLK(), vdata4, vl);
            vsev_int64xm1(dst_ + 5 * C_BLK(), vdata5, vl);
            vsev_int64xm1(dst_ + 6 * C_BLK(), vdata6, vl);
            vsev_int64xm1(dst_ + 7 * C_BLK(), vdata7, vl);
            vsev_int64xm1(dst_ + 8 * C_BLK(), vdata8, vl);
            vsev_int64xm1(dst_ + 9 * C_BLK(), vdata9, vl);
            vsev_int64xm1(dst_ + 10 * C_BLK(), vdata10, vl);
            vsev_int64xm1(dst_ + 11 * C_BLK(), vdata11, vl);
            vsev_int64xm1(dst_ + 12 * C_BLK(), vdata12, vl);
            vsev_int64xm1(dst_ + 13 * C_BLK(), vdata13, vl);
            vsev_int64xm1(dst_ + 14 * C_BLK(), vdata14, vl);
            vsev_int64xm1(dst_ + 15 * C_BLK(), vdata15, vl);
        }
    }
    for (; idx < total_len; idx += C_BLK()) {
        const int64_t* src0_ = src0 + idx;
        const int64_t* src1_ = src1 + idx;
        int64_t* dst_ = dst + idx;

        int64xm1_t vdata;
        vdata = arithmetic_vector_kernel_int64<_op>(vlev_int64xm1(src0_, vl), vlev_int64xm1(src1_, vl));
        if (fuse_relu) {
            vsev_int64xm1(dst_, vmaxvx_int64xm1(vdata, (int64_t)0, vl), vl);
        } else {
            vsev_int64xm1(dst_, vdata, vl);
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_INT64_ARITHMETIC_ARITHMETIC_ELTWISE_INT64_H_