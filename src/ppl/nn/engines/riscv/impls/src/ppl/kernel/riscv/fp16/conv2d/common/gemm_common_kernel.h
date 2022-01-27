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

#ifndef PPL3RISCVKERNEL_SRC_FP16_GEMM_COMMON_RVV_1_0_KERNEL_H_
#define PPL3RISCVKERNEL_SRC_FP16_GEMM_COMMON_RVV_1_0_KERNEL_H_

#include "ppl/kernel/riscv/fp16/conv2d/common/gemm_kernel/conv2d_ndarray_n8cx_gemm_kernel_fp16.h"

namespace ppl { namespace kernel { namespace riscv {

typedef void (*conv_gemm_riscv_kernel_m8nx)(const __fp16* kernel_A, const __fp16* kernel_B, __fp16* kernel_C, int64_t k, int64_t total_n);

typedef void (*conv_gemm_riscv_kernel_func_type_t)(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

#ifdef __cplusplus
extern "C" {
#endif

void gemm_common_m8n16_left15_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left15_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left14_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left14_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left13_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left13_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left12_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left12_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left11_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left11_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left10_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left10_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left9_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left9_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left8_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left8_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left7_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left7_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left6_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left6_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left5_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left5_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left4_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left4_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left3_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left3_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left2_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left2_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left1_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left1_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left0_first_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

void gemm_common_m8n16_left0_rv64_fp16(const __fp16* A, const __fp16* B, __fp16* C, int64_t m, int64_t n, int64_t k);

#ifdef __cplusplus
}
#endif

template <int64_t align_n,
          int64_t align_left_n,
          conv_gemm_riscv_kernel_m8nx core_func,
          conv_gemm_riscv_kernel_m8nx core_left_func>
static void conv_gemm_cto8c_kernel_fp16(
    const __fp16* A,
    const __fp16* B,
    __fp16* C,
    int64_t m,
    int64_t n,
    int64_t k)
{
    int64_t mi, ni;

    int64_t kernel_m_stride = k * 8;

    for (mi = 0; mi < m; mi += 8) {
        auto temp_B = B;
        for (ni = 0; ni <= n - align_n; ni += align_n) {
            core_func(A, temp_B, C, k, n);

            C += align_n * 8;
            temp_B += align_n;
        }

        if (align_left_n != 0) {
            core_left_func(A, temp_B, C, k, n);
            C += align_left_n * 8;
        }
        A += kernel_m_stride;
    }
}

template <bool first>
conv_gemm_riscv_kernel_func_type_t conv_gemm_select_cto8c_kernel_fp16(int64_t n)
{
    // TODO: add "int64_t m" to parameters
    switch (n % 24) {
        case 0:
            return conv_gemm_cto8c_kernel_fp16<24, 0, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<0>>;
        case 1:
            return conv_gemm_cto8c_kernel_fp16<24, 1, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<1>>;
        case 2:
            return conv_gemm_cto8c_kernel_fp16<24, 2, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<2>>;
        case 3:
            return conv_gemm_cto8c_kernel_fp16<24, 3, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<3>>;
        case 4:
            return conv_gemm_cto8c_kernel_fp16<24, 4, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<4>>;
        case 5:
            return conv_gemm_cto8c_kernel_fp16<24, 5, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<5>>;
        case 6:
            return conv_gemm_cto8c_kernel_fp16<24, 6, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<6>>;
        case 7:
            return conv_gemm_cto8c_kernel_fp16<24, 7, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<7>>;
        case 8:
            return conv_gemm_cto8c_kernel_fp16<24, 8, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<8>>;
        case 9:
            return conv_gemm_cto8c_kernel_fp16<24, 9, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<9>>;
        case 10:
            return conv_gemm_cto8c_kernel_fp16<24, 10, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<10>>;
        case 11:
            return conv_gemm_cto8c_kernel_fp16<24, 11, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<11>>;
        case 12:
            return conv_gemm_cto8c_kernel_fp16<24, 12, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<12>>;
        case 13:
            return conv_gemm_cto8c_kernel_fp16<24, 13, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<13>>;
        case 14:
            return conv_gemm_cto8c_kernel_fp16<24, 14, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<14>>;
        case 15:
            return conv_gemm_cto8c_kernel_fp16<24, 15, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<15>>;
        case 16:
            return conv_gemm_cto8c_kernel_fp16<24, 16, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<16>>;
        case 17:
            return conv_gemm_cto8c_kernel_fp16<24, 17, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<17>>;
        case 18:
            return conv_gemm_cto8c_kernel_fp16<24, 18, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<18>>;
        case 19:
            return conv_gemm_cto8c_kernel_fp16<24, 19, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<19>>;
        case 20:
            return conv_gemm_cto8c_kernel_fp16<24, 20, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<20>>;
        case 21:
            return conv_gemm_cto8c_kernel_fp16<24, 21, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<21>>;
        case 22:
            return conv_gemm_cto8c_kernel_fp16<24, 22, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<22>>;
        case 23:
            return conv_gemm_cto8c_kernel_fp16<24, 23, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<23>>;
    }
    return conv_gemm_cto8c_kernel_fp16<24, 0, conv_gemm_cto8c_m8nx_kernel_core_fp16<24>, conv_gemm_cto8c_m8nx_kernel_core_fp16<0>>;
    ;
}

template <bool first>
conv_gemm_riscv_kernel_func_type_t conv_gemm_select_kernel_fp16(int64_t n)
{
    switch (n % 16) {
        case 0:
            return first ? gemm_common_m8n16_left0_first_rv64_fp16 : gemm_common_m8n16_left0_rv64_fp16;
        case 1:
            return first ? gemm_common_m8n16_left1_first_rv64_fp16 : gemm_common_m8n16_left1_rv64_fp16;
        case 2:
            return first ? gemm_common_m8n16_left2_first_rv64_fp16 : gemm_common_m8n16_left2_rv64_fp16;
        case 3:
            return first ? gemm_common_m8n16_left3_first_rv64_fp16 : gemm_common_m8n16_left3_rv64_fp16;
        case 4:
            return first ? gemm_common_m8n16_left4_first_rv64_fp16 : gemm_common_m8n16_left4_rv64_fp16;
        case 5:
            return first ? gemm_common_m8n16_left5_first_rv64_fp16 : gemm_common_m8n16_left5_rv64_fp16;
        case 6:
            return first ? gemm_common_m8n16_left6_first_rv64_fp16 : gemm_common_m8n16_left6_rv64_fp16;
        case 7:
            return first ? gemm_common_m8n16_left7_first_rv64_fp16 : gemm_common_m8n16_left7_rv64_fp16;
        case 8:
            return first ? gemm_common_m8n16_left8_first_rv64_fp16 : gemm_common_m8n16_left8_rv64_fp16;
        case 9:
            return first ? gemm_common_m8n16_left9_first_rv64_fp16 : gemm_common_m8n16_left9_rv64_fp16;
        case 10:
            return first ? gemm_common_m8n16_left10_first_rv64_fp16 : gemm_common_m8n16_left10_rv64_fp16;
        case 11:
            return first ? gemm_common_m8n16_left11_first_rv64_fp16 : gemm_common_m8n16_left11_rv64_fp16;
        case 12:
            return first ? gemm_common_m8n16_left12_first_rv64_fp16 : gemm_common_m8n16_left12_rv64_fp16;
        case 13:
            return first ? gemm_common_m8n16_left13_first_rv64_fp16 : gemm_common_m8n16_left13_rv64_fp16;
        case 14:
            return first ? gemm_common_m8n16_left14_first_rv64_fp16 : gemm_common_m8n16_left14_rv64_fp16;
        case 15:
            return first ? gemm_common_m8n16_left15_first_rv64_fp16 : gemm_common_m8n16_left15_rv64_fp16;
    }
    return first ? gemm_common_m8n16_left0_first_rv64_fp16 : gemm_common_m8n16_left0_rv64_fp16;
}

template <int64_t src_atom_c, bool first>
conv_gemm_riscv_kernel_func_type_t conv_gemm_select_xcto8c_kernel_fp16(int64_t m, int64_t n)
{
    switch (src_atom_c) {
        case 1:
            return conv_gemm_select_cto8c_kernel_fp16<first>(n);
        case 8:
            return conv_gemm_select_kernel_fp16<first>(n);
        default:
            return conv_gemm_select_kernel_fp16<first>(n);
    }
}

}}}; // namespace ppl::kernel::riscv

#endif
