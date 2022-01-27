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

#ifndef __ST_PPL_KERNEL_RISCV_FP32_CONV2D_COMMON_CONV2D_GEMM_KERNEL_FP32_H_
#define __ST_PPL_KERNEL_RISCV_FP32_CONV2D_COMMON_CONV2D_GEMM_KERNEL_FP32_H_

#include "ppl/kernel/riscv/fp32/conv2d/common/gemm_kernel/conv2d_n4cx_n4cx_gemm_kernel_fp32_vec128.h"
#include "ppl/kernel/riscv/fp32/conv2d/common/gemm_kernel/conv2d_ndarray_n4cx_gemm_kernel_fp32_vec128.h"

namespace ppl { namespace kernel { namespace riscv {

typedef void (*conv2d_gemm_kernel_func_riscv_fp32_type_t)(
    const float* A,
    const float* B,
    float* C,
    const int64_t m,
    const int64_t n,
    const int64_t k);

template <bool first>
conv2d_gemm_kernel_func_riscv_fp32_type_t conv2d_gemm_select_cto4c_kernel_fp32_vec128(int64_t m, int64_t n)
{
    switch (m % 16) {
        case 4:
            switch (n % 7) {
                case 0:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 4, 7, first>;
                case 1:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 4, 1, first>;
                case 2:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 4, 2, first>;
                case 3:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 4, 3, first>;
                case 4:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 4, 4, first>;
                case 5:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 4, 5, first>;
                case 6:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 4, 6, first>;
            }
        case 8:
            switch (n % 7) {
                case 0:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 8, 7, first>;
                case 1:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 8, 1, first>;
                case 2:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 8, 2, first>;
                case 3:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 8, 3, first>;
                case 4:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 8, 4, first>;
                case 5:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 8, 5, first>;
                case 6:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 8, 6, first>;
            }
        case 12:
            switch (n % 7) {
                case 0:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 12, 7, first>;
                case 1:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 12, 1, first>;
                case 2:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 12, 2, first>;
                case 3:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 12, 3, first>;
                case 4:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 12, 4, first>;
                case 5:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 12, 5, first>;
                case 6:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 12, 6, first>;
            }
        case 0:
            switch (n % 7) {
                case 0:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 16, 7, first>;
                case 1:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 16, 1, first>;
                case 2:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 16, 2, first>;
                case 3:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 16, 3, first>;
                case 4:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 16, 4, first>;
                case 5:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 16, 5, first>;
                case 6:
                    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 16, 6, first>;
            }
    }
    return gemm_ndarray_n4cx_fp32_vec128<16, 7, 16, 7, first>;
}

template <bool first>
conv2d_gemm_kernel_func_riscv_fp32_type_t conv2d_gemm_select_4cto4c_kernel_fp32_vec128(int64_t m, int64_t n)
{
    switch (m % 16) {
        case 4:
            switch (n % 7) {
                case 0:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 4, 7, first>;
                case 1:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 4, 1, first>;
                case 2:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 4, 2, first>;
                case 3:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 4, 3, first>;
                case 4:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 4, 4, first>;
                case 5:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 4, 5, first>;
                case 6:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 4, 6, first>;
            }
        case 8:
            switch (n % 7) {
                case 0:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 8, 7, first>;
                case 1:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 8, 1, first>;
                case 2:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 8, 2, first>;
                case 3:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 8, 3, first>;
                case 4:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 8, 4, first>;
                case 5:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 8, 5, first>;
                case 6:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 8, 6, first>;
            }
        case 12:
            switch (n % 7) {
                case 0:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 12, 7, first>;
                case 1:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 12, 1, first>;
                case 2:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 12, 2, first>;
                case 3:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 12, 3, first>;
                case 4:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 12, 4, first>;
                case 5:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 12, 5, first>;
                case 6:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 12, 6, first>;
            }
        case 0:
            switch (n % 7) {
                case 0:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 16, 7, first>;
                case 1:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 16, 1, first>;
                case 2:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 16, 2, first>;
                case 3:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 16, 3, first>;
                case 4:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 16, 4, first>;
                case 5:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 16, 5, first>;
                case 6:
                    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 16, 6, first>;
            }
    }
    return gemm_n4cx_n4cx_fp32_vec128<16, 7, 16, 7, first>;
}

template <int64_t src_atom_c, bool first>
conv2d_gemm_kernel_func_riscv_fp32_type_t conv2d_gemm_select_xcto4c_kernel_fp32_vec128(int64_t m, int64_t n)
{
    switch (src_atom_c) {
        case 1:
            return conv2d_gemm_select_cto4c_kernel_fp32_vec128<first>(m, n);
        case 4:
            return conv2d_gemm_select_4cto4c_kernel_fp32_vec128<first>(m, n);
        default:
            return conv2d_gemm_select_4cto4c_kernel_fp32_vec128<first>(m, n);
    }
}

}}}; // namespace ppl::kernel::riscv

#endif
