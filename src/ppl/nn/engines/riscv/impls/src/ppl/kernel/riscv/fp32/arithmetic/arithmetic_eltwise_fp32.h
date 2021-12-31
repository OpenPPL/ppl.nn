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

#ifndef __ST_PPL_KERNEL_RISCV_FP32_ARITHMETIC_ARITHMETIC_ELTWISE_FP32_H_
#define __ST_PPL_KERNEL_RISCV_FP32_ARITHMETIC_ARITHMETIC_ELTWISE_FP32_H_

#include "ppl/nn/runtime/tensor_impl.h"
#include "arithmetic_kernel_fp32.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() ((int64_t)4)

template <arithmetic_op_type_t _op, bool fuse_relu>
static ppl::common::RetCode arithmetic_eltwise_fp32(const ppl::nn::TensorShape* dst_shape, const float* src0,
                                                    const float* src1, float* dst) {
    const int64_t total_len = dst_shape->GetElementsIncludingPadding();
    const int64_t parall_d = 16;
    const int64_t unroll_len = parall_d * C_BLK();
    const auto vl = vsetvli(C_BLK(), RVV_E32, RVV_M1);

    int64_t idx = 0;
    for (; idx + unroll_len < total_len; idx += unroll_len) {
        const float* src0_ = src0 + idx;
        const float* src1_ = src1 + idx;
        float* dst_ = dst + idx;

        float32xm1_t vfdata0, vfdata1, vfdata2, vfdata3;
        float32xm1_t vfdata4, vfdata5, vfdata6, vfdata7;
        float32xm1_t vfdata8, vfdata9, vfdata10, vfdata11;
        float32xm1_t vfdata12, vfdata13, vfdata14, vfdata15;

        vfdata0 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 0 * C_BLK(), vl),
                                                     vlev_float32xm1(src1_ + 0 * C_BLK(), vl));
        vfdata1 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 1 * C_BLK(), vl),
                                                     vlev_float32xm1(src1_ + 1 * C_BLK(), vl));
        vfdata2 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 2 * C_BLK(), vl),
                                                     vlev_float32xm1(src1_ + 2 * C_BLK(), vl));
        vfdata3 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 3 * C_BLK(), vl),
                                                     vlev_float32xm1(src1_ + 3 * C_BLK(), vl));
        vfdata4 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 4 * C_BLK(), vl),
                                                     vlev_float32xm1(src1_ + 4 * C_BLK(), vl));
        vfdata5 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 5 * C_BLK(), vl),
                                                     vlev_float32xm1(src1_ + 5 * C_BLK(), vl));
        vfdata6 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 6 * C_BLK(), vl),
                                                     vlev_float32xm1(src1_ + 6 * C_BLK(), vl));
        vfdata7 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 7 * C_BLK(), vl),
                                                     vlev_float32xm1(src1_ + 7 * C_BLK(), vl));
        vfdata8 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 8 * C_BLK(), vl),
                                                     vlev_float32xm1(src1_ + 8 * C_BLK(), vl));
        vfdata9 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 9 * C_BLK(), vl),
                                                     vlev_float32xm1(src1_ + 9 * C_BLK(), vl));
        vfdata10 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 10 * C_BLK(), vl),
                                                      vlev_float32xm1(src1_ + 10 * C_BLK(), vl));
        vfdata11 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 11 * C_BLK(), vl),
                                                      vlev_float32xm1(src1_ + 11 * C_BLK(), vl));
        vfdata12 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 12 * C_BLK(), vl),
                                                      vlev_float32xm1(src1_ + 12 * C_BLK(), vl));
        vfdata13 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 13 * C_BLK(), vl),
                                                      vlev_float32xm1(src1_ + 13 * C_BLK(), vl));
        vfdata14 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 14 * C_BLK(), vl),
                                                      vlev_float32xm1(src1_ + 14 * C_BLK(), vl));
        vfdata15 = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_ + 15 * C_BLK(), vl),
                                                      vlev_float32xm1(src1_ + 15 * C_BLK(), vl));

        if (fuse_relu) {
            vsev_float32xm1(dst_ + 0 * C_BLK(), vfmaxvf_float32xm1(vfdata0, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 1 * C_BLK(), vfmaxvf_float32xm1(vfdata1, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 2 * C_BLK(), vfmaxvf_float32xm1(vfdata2, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 3 * C_BLK(), vfmaxvf_float32xm1(vfdata3, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 4 * C_BLK(), vfmaxvf_float32xm1(vfdata4, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 5 * C_BLK(), vfmaxvf_float32xm1(vfdata5, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 6 * C_BLK(), vfmaxvf_float32xm1(vfdata6, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 7 * C_BLK(), vfmaxvf_float32xm1(vfdata7, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 8 * C_BLK(), vfmaxvf_float32xm1(vfdata8, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 9 * C_BLK(), vfmaxvf_float32xm1(vfdata9, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 10 * C_BLK(), vfmaxvf_float32xm1(vfdata10, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 11 * C_BLK(), vfmaxvf_float32xm1(vfdata11, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 12 * C_BLK(), vfmaxvf_float32xm1(vfdata12, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 13 * C_BLK(), vfmaxvf_float32xm1(vfdata13, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 14 * C_BLK(), vfmaxvf_float32xm1(vfdata14, (float)0.0f, vl), vl);
            vsev_float32xm1(dst_ + 15 * C_BLK(), vfmaxvf_float32xm1(vfdata15, (float)0.0f, vl), vl);
        } else {
            vsev_float32xm1(dst_ + 0 * C_BLK(), vfdata0, vl);
            vsev_float32xm1(dst_ + 1 * C_BLK(), vfdata1, vl);
            vsev_float32xm1(dst_ + 2 * C_BLK(), vfdata2, vl);
            vsev_float32xm1(dst_ + 3 * C_BLK(), vfdata3, vl);
            vsev_float32xm1(dst_ + 4 * C_BLK(), vfdata4, vl);
            vsev_float32xm1(dst_ + 5 * C_BLK(), vfdata5, vl);
            vsev_float32xm1(dst_ + 6 * C_BLK(), vfdata6, vl);
            vsev_float32xm1(dst_ + 7 * C_BLK(), vfdata7, vl);
            vsev_float32xm1(dst_ + 8 * C_BLK(), vfdata8, vl);
            vsev_float32xm1(dst_ + 9 * C_BLK(), vfdata9, vl);
            vsev_float32xm1(dst_ + 10 * C_BLK(), vfdata10, vl);
            vsev_float32xm1(dst_ + 11 * C_BLK(), vfdata11, vl);
            vsev_float32xm1(dst_ + 12 * C_BLK(), vfdata12, vl);
            vsev_float32xm1(dst_ + 13 * C_BLK(), vfdata13, vl);
            vsev_float32xm1(dst_ + 14 * C_BLK(), vfdata14, vl);
            vsev_float32xm1(dst_ + 15 * C_BLK(), vfdata15, vl);
        }
    }
    for (; idx < total_len; idx += C_BLK()) {
        const float* src0_ = src0 + idx;
        const float* src1_ = src1 + idx;
        float* dst_ = dst + idx;

        float32xm1_t vfdata;
        vfdata = arithmetic_vector_kernel_fp32<_op>(vlev_float32xm1(src0_, vl), vlev_float32xm1(src1_, vl));
        if (fuse_relu) {
            vsev_float32xm1(dst_, vfmaxvf_float32xm1(vfdata, (float)0.0f, vl), vl);
        } else {
            vsev_float32xm1(dst_, vfdata, vl);
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP32_ARITHMETIC_ARITHMETIC_ELTWISE_FP32_H_