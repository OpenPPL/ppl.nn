
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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_LEAKY_RELU_LEAKY_RELU_NBCX_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_LEAKY_RELU_LEAKY_RELU_NBCX_COMMON_H_

#include <riscv-vector.h>
#include <type_traits>
#include <math.h>

#include "ppl/kernel/riscv/common/internal_include.h"
#include "ppl/kernel/riscv/common/leaky_relu/leaky_relu_kernel.h"

namespace ppl { namespace kernel { namespace riscv {

template <typename T, int32_t v_len>
ppl::common::RetCode leaky_relu_nbcx_common(
    const ppl::common::TensorShape *src_shape,
    const T *src,
    const float alpha,
    T *dst)
{
    uint64_t vl;
    constexpr int32_t c_blk = v_len / (sizeof(T) * 8);
    vl                      = vsetvli<T, v_len>(c_blk);

    const int64_t n_elem      = src_shape->CalcElementsIncludingPadding();
    const int64_t simd_w      = c_blk;
    const int64_t unroll_n    = simd_w * 4;
    const int64_t unroll_body = round(n_elem, unroll_n);

    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        const register_v<T, v_len> v_src_0 = vlev<T, v_len>(src + i + simd_w * 0, vl);
        const register_v<T, v_len> v_src_1 = vlev<T, v_len>(src + i + simd_w * 1, vl);
        const register_v<T, v_len> v_src_2 = vlev<T, v_len>(src + i + simd_w * 2, vl);
        const register_v<T, v_len> v_src_3 = vlev<T, v_len>(src + i + simd_w * 3, vl);

        const register_v<T, v_len> v_ge_0 = vmaxvf<T, v_len>(v_src_0, (T)0.0f, vl);
        const register_v<T, v_len> v_ge_1 = vmaxvf<T, v_len>(v_src_1, (T)0.0f, vl);
        const register_v<T, v_len> v_ge_2 = vmaxvf<T, v_len>(v_src_2, (T)0.0f, vl);
        const register_v<T, v_len> v_ge_3 = vmaxvf<T, v_len>(v_src_3, (T)0.0f, vl);

        const register_v<T, v_len> v_le_0 = vmulvf<T, v_len>(vminvf<T, v_len>(v_src_0, (T)0.0f, vl), (T)alpha, vl);
        const register_v<T, v_len> v_le_1 = vmulvf<T, v_len>(vminvf<T, v_len>(v_src_1, (T)0.0f, vl), (T)alpha, vl);
        const register_v<T, v_len> v_le_2 = vmulvf<T, v_len>(vminvf<T, v_len>(v_src_2, (T)0.0f, vl), (T)alpha, vl);
        const register_v<T, v_len> v_le_3 = vmulvf<T, v_len>(vminvf<T, v_len>(v_src_3, (T)0.0f, vl), (T)alpha, vl);

        vsev<T, v_len>(dst + i + simd_w * 0, vaddvv<register_v<T, v_len>, v_len>(v_ge_0, v_le_0, vl), vl);
        vsev<T, v_len>(dst + i + simd_w * 1, vaddvv<register_v<T, v_len>, v_len>(v_ge_1, v_le_1, vl), vl);
        vsev<T, v_len>(dst + i + simd_w * 2, vaddvv<register_v<T, v_len>, v_len>(v_ge_2, v_le_2, vl), vl);
        vsev<T, v_len>(dst + i + simd_w * 3, vaddvv<register_v<T, v_len>, v_len>(v_ge_3, v_le_3, vl), vl);
    }
    for (int64_t i = unroll_body; i < n_elem; i++) {
        dst[i] = src[i] >= 0 ? src[i] : src[i] * (T)alpha;
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv

#endif
