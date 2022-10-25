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

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/gemm/gemm_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template<int64_t need_mask, int64_t u_m, int64_t u_n>
void gemm_m8n48_kernel_fp32_avx512_core(int64_t *param) {
    __asm__ __volatile__ (
        ".equ CACHELINE_BYTES, 64\n"
        ".equ P_BYTES, 8\n"
        ".equ D_BYTES, 4\n"
        ".equ LOG2_D_BYTES, 2\n"

        ".equ A_PTR_IDX,        (0 * P_BYTES)\n"
        ".equ B_PTR_IDX,        (1 * P_BYTES)\n"
        ".equ C_PTR_IDX,        (2 * P_BYTES)\n"
        ".equ BIAS_PTR_IDX,     (3 * P_BYTES)\n"
        ".equ SUM_PTR_IDX,      (4 * P_BYTES)\n"
        ".equ M_IDX,            (5 * P_BYTES)\n"
        ".equ K_IDX,            (6 * P_BYTES)\n"
        ".equ LDC_IDX,          (7 * P_BYTES)\n"
        ".equ LDSUM_IDX,        (8 * P_BYTES)\n"
        ".equ ALPHA_IDX,        (9 * P_BYTES)\n"
        ".equ BETA_IDX,         (10 * P_BYTES)\n"
        ".equ BETA_BIAS_IDX,    (11 * P_BYTES)\n"
        ".equ BETA_SUM_IDX,     (12 * P_BYTES)\n"
        ".equ FLAGS_IDX,        (13 * P_BYTES)\n"
        ".equ PRF_C_LDK_IDX,    (14 * P_BYTES)\n"
        ".equ MASK_IDX,         (15 * P_BYTES)\n"

        ".equ N_REG_ELTS, %c[N_REG_ELTS]\n"
        ".equ NEED_MASK, %c[NEED_MASK]\n"
        ".equ U_M, %c[U_M]\n"
        ".equ U_N, %c[U_N]\n"
        ".equ U_K, 4\n"
        ".equ U_K_LOG2, 2\n"
        ".equ U_NR, ((U_N + N_REG_ELTS - 1) / N_REG_ELTS)\n"
        ".equ KERNEL_FLAG_LOAD_C, %c[KERNEL_FLAG_LOAD_C]\n"
        ".equ KERNEL_FLAG_WITH_SUM, %c[KERNEL_FLAG_WITH_SUM]\n"
        ".equ KERNEL_FLAG_ROW_BIAS, %c[KERNEL_FLAG_ROW_BIAS]\n"
        ".equ KERNEL_FLAG_COL_BIAS, %c[KERNEL_FLAG_COL_BIAS]\n"
        ".equ KERNEL_FLAG_SCA_BIAS, %c[KERNEL_FLAG_SCA_BIAS]\n"
        ".equ KERNEL_FLAG_RELU, %c[KERNEL_FLAG_RELU]\n"
        ".equ KERNEL_FLAG_RELU6, %c[KERNEL_FLAG_RELU6]\n"

        ".equ PREFETCH_B_OFFSET, 1536\n" // 16*4*3*8, prefetch next 2uk
        ".equ PREFETCH_A_OFFSET, 256\n"  // 8*4*8, 2uk

        // init masks
        ".if NEED_MASK\n"
        "mov MASK_IDX(%[param]), %%rcx\n"
        "mov $1, %%rax\n"
        "sal %%cl, %%rax\n"
        "sub $1, %%rax\n"                   // (1 << mask) - 1
        "mov $0xffff, %%rcx\n"
        ".if U_NR == 1\n kmovw %%eax, %%k1\n .else\n kmovw %%ecx, %%k1\n .endif\n"
        ".if U_NR == 2\n kmovw %%eax, %%k2\n .else\n kmovw %%ecx, %%k2\n .endif\n"
        ".if U_NR == 3\n kmovw %%eax, %%k3\n .else\n kmovw %%ecx, %%k3\n .endif\n"
        ".endif\n"

        "mov K_IDX(%[param]), %%rax\n"              // k
        "mov PRF_C_LDK_IDX(%[param]), %%r10\n"      // lead_k
        "sar $U_K_LOG2, %%r10\n"
        "mov B_PTR_IDX(%[param]), %%rbx\n"          // b_ptr
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%zmm27\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%zmm28\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%zmm29\n .endif\n"

        "mov FLAGS_IDX(%[param]),        %%rsi\n"
        "mov A_PTR_IDX(%[param]),        %%r15\n"
        "mov C_PTR_IDX(%[param]),        %%r14\n"
        "mov M_IDX(%[param]),            %%r13\n"
        "mov LDC_IDX(%[param]),          %%r11\n"
        "shl $LOG2_D_BYTES, %%r11\n"
        ".if U_M > 3\n"
        "imul $3, %%r11, %%r8\n"                                // ldc3
        ".endif\n"
        ".if U_M > 4\n"
        "lea (%%r14, %%r11, 4), %%r12\n"                        // c_m4
        ".endif\n" 
        "imul $U_M, %%r11, %%r9\n"                              // u_m * ldc

        ".if U_M > 0\n"
        ".if U_NR > 0\n vpxord %%zmm0, %%zmm0, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm1, %%zmm1, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm2, %%zmm2, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vpxord %%zmm3, %%zmm3, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm4, %%zmm4, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm5, %%zmm5, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vpxord %%zmm6, %%zmm6, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm7, %%zmm7, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm8, %%zmm8, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vpxord %%zmm9, %%zmm9, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm10, %%zmm10, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm11, %%zmm11, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vpxord %%zmm12, %%zmm12, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm13, %%zmm13, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm14, %%zmm14, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vpxord %%zmm15, %%zmm15, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm16, %%zmm16, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm17, %%zmm17, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vpxord %%zmm18, %%zmm18, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm19, %%zmm19, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm20, %%zmm20, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vpxord %%zmm21, %%zmm21, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm22, %%zmm22, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm23, %%zmm23, %%zmm23\n .endif\n"
        ".endif\n"




"1:\n" // label_init_session
        "mov %%rax, %%rdx\n" // k
        "sar $U_K_LOG2, %%rdx\n" // purge the k tail, k -> uk
        "sub %%r10, %%rdx\n"
        "jle 20f\n" // label_uk_prf_c
        PPL_X86_INLINE_ASM_ALIGN()
"4:\n" // label_loop_uk_body
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm24\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm25\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm26\n .endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm23\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm27\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm28\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm29\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 0\n prefetcht0 (0 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n .endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm23\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%zmm24\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%zmm25\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%zmm26\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm23\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%zmm27\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%zmm28\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%zmm29\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        "sub $1, %%rdx\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        "lea (U_K * U_N * D_BYTES)(%%rbx), %%rbx\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n prefetcht0 (1 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n .endif\n"
        "lea (U_K * U_M * D_BYTES)(%%r15), %%r15\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm23\n .endif\n"
        ".endif\n"
        "jg 4b\n" // label_loop_uk_body




"20:\n" // label_uk_prf_c
        "sub $U_M, %%r10\n"
        "add $U_M, %%rdx\n"
        "jle 30f\n" // label_uk_after_prf_c
        "lea (CACHELINE_BYTES - D_BYTES)(%%r14), %%rcx\n"
        PPL_X86_INLINE_ASM_ALIGN()
"10:\n" // label_loop_uk_prf_c
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm24\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm25\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm26\n .endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetchw (0 * N_REG_ELTS * D_BYTES)(%%rcx)\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm23\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm27\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm28\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm29\n .endif\n"
        ".if U_NR > 1\n prefetchw (1 * N_REG_ELTS * D_BYTES)(%%rcx)\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 0\n prefetcht0 (0 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n .endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm23\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%zmm24\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%zmm25\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%zmm26\n .endif\n"
        ".if U_NR > 2\n prefetchw (2 * N_REG_ELTS * D_BYTES)(%%rcx)\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm23\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%zmm27\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%zmm28\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%zmm29\n .endif\n"
        "lea (%%rcx, %%r11), %%rcx\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        "sub $1, %%rdx\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        "lea (U_K * U_N * D_BYTES)(%%rbx), %%rbx\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n prefetcht0 (1 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n .endif\n"
        "lea (U_K * U_M * D_BYTES)(%%r15), %%r15\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm23\n .endif\n"
        ".endif\n"
        "jg 10b\n" // label_loop_uk_prf_c




"30:\n" // label_uk_after_prf_c
        "add %%r10, %%rdx\n"
        "jle 5f\n" // label_k_tail
        PPL_X86_INLINE_ASM_ALIGN()
"40:\n" // label_loop_uk_after_prf_c
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm24\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm25\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm26\n .endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm23\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm27\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm28\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm29\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 0\n prefetcht0 (0 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n .endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm23\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%zmm24\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%zmm25\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%zmm26\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm23\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%zmm27\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%zmm28\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%zmm29\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        "sub $1, %%rdx\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        "lea (U_K * U_N * D_BYTES)(%%rbx), %%rbx\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n prefetcht0 (1 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n .endif\n"
        "lea (U_K * U_M * D_BYTES)(%%r15), %%r15\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm23\n .endif\n"
        ".endif\n"
        "jg 40b\n" // label_loop_uk_after_prf_c




"5:\n" // label_k_tail
        "mov %%rax, %%rdx\n"
        "and $(U_K - 1), %%rdx\n"
        "je 6f\n" // label_end_k
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm24\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm25\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%zmm26\n .endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        "cmp $1, %%rdx\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 0\n prefetcht0 (0 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n prefetcht0 (1 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n .endif\n"
        "lea (U_M * D_BYTES)(%%r15), %%r15\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm23\n .endif\n"
        ".endif\n"
        "je 6f\n" // label_end_k
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm27\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm28\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%zmm29\n .endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        "cmp $2, %%rdx\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm24, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm25, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm26, %%zmm20\n .endif\n"
        ".endif\n"
        "lea (U_M * D_BYTES)(%%r15), %%r15\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm24, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm25, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm26, %%zmm23\n .endif\n"
        ".endif\n"
        "je 6f\n" // label_end_k
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n vbroadcastss (4 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n vbroadcastss (5 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n vbroadcastss (6 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm30\n .endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n vbroadcastss (7 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%zmm31\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm30, %%zmm27, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm30, %%zmm28, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm30, %%zmm29, %%zmm20\n .endif\n"
        ".endif\n"
        "lea (U_M * D_BYTES)(%%r15), %%r15\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps %%zmm31, %%zmm27, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%zmm31, %%zmm28, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%zmm31, %%zmm29, %%zmm23\n .endif\n"
        ".endif\n"
"6:\n" // label_end_k




        "vbroadcastss ALPHA_IDX(%[param]), %%zmm24\n" // alpha
        "vbroadcastss BETA_IDX(%[param]), %%zmm25\n"  // beta
        "mov B_PTR_IDX(%[param]), %%rbx\n"            // b_ptr
        "mov PRF_C_LDK_IDX(%[param]), %%r10\n"        // lead_uk
        "sar $U_K_LOG2, %%r10\n"
        "vbroadcastss BETA_BIAS_IDX(%[param]), %%zmm30\n"
        "vbroadcastss BETA_SUM_IDX(%[param]), %%zmm31\n"
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%zmm27\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%zmm28\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%zmm29\n .endif\n"
        // *= alpha
        ".if U_M > 0\n"
        ".if U_NR > 0\n vmulps %%zmm24, %%zmm0, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vmulps %%zmm24, %%zmm1, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vmulps %%zmm24, %%zmm2, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vmulps %%zmm24, %%zmm3, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vmulps %%zmm24, %%zmm4, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vmulps %%zmm24, %%zmm5, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vmulps %%zmm24, %%zmm6, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vmulps %%zmm24, %%zmm7, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vmulps %%zmm24, %%zmm8, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vmulps %%zmm24, %%zmm9, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vmulps %%zmm24, %%zmm10, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vmulps %%zmm24, %%zmm11, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vmulps %%zmm24, %%zmm12, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vmulps %%zmm24, %%zmm13, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vmulps %%zmm24, %%zmm14, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vmulps %%zmm24, %%zmm15, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vmulps %%zmm24, %%zmm16, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vmulps %%zmm24, %%zmm17, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vmulps %%zmm24, %%zmm18, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vmulps %%zmm24, %%zmm19, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vmulps %%zmm24, %%zmm20, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vmulps %%zmm24, %%zmm21, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vmulps %%zmm24, %%zmm22, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vmulps %%zmm24, %%zmm23, %%zmm23\n .endif\n"
        ".endif\n"

        // put sum in the first place, overlaping cache miss
        // += beta_sum * sum
        "test $KERNEL_FLAG_WITH_SUM, %%rsi\n"
        "jz 14f\n" // label_load_sum_end
        "mov SUM_PTR_IDX(%[param]), %%rcx\n"
        "mov LDSUM_IDX(%[param]), %%rdx\n"
        "shl $LOG2_D_BYTES, %%rdx\n" // ldsum
        ".if NEED_MASK\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm0%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm1%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm2%{%%k3}%{z}\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n" // next row
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm3%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm4%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm5%{%%k3}%{z}\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm6%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm7%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm8%{%%k3}%{z}\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm9%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm10%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm11%{%%k3}%{z}\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm12%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm13%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm14%{%%k3}%{z}\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm15%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm16%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm17%{%%k3}%{z}\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm18%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm19%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm20%{%%k3}%{z}\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm21%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm22%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm23%{%%k3}%{z}\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".else\n" // need_mask
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm2\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n" // next row
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm5\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm8\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm11\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm14\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm17\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm20\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm31, %%zmm23\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".endif\n" // need_mask
        "mov %%rcx, SUM_PTR_IDX(%[param])\n"
"14:\n" // label_load_sum_end

        // += beta*C
        "test $KERNEL_FLAG_LOAD_C, %%rsi\n"
        "jz 8f\n" // label_load_c_end
        ".if NEED_MASK\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14), %%zmm25, %%zmm0%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14), %%zmm25, %%zmm1%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14), %%zmm25, %%zmm2%{%%k3}%{z}\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%zmm25, %%zmm3%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%zmm25, %%zmm4%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%zmm25, %%zmm5%{%%k3}%{z}\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2), %%zmm25, %%zmm6%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2), %%zmm25, %%zmm7%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2), %%zmm25, %%zmm8%{%%k3}%{z}\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r8), %%zmm25, %%zmm9%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r8), %%zmm25, %%zmm10%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r8), %%zmm25, %%zmm11%{%%k3}%{z}\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12), %%zmm25, %%zmm12%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12), %%zmm25, %%zmm13%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12), %%zmm25, %%zmm14%{%%k3}%{z}\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%zmm25, %%zmm15%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%zmm25, %%zmm16%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%zmm25, %%zmm17%{%%k3}%{z}\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2), %%zmm25, %%zmm18%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2), %%zmm25, %%zmm19%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2), %%zmm25, %%zmm20%{%%k3}%{z}\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r8), %%zmm25, %%zmm21%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r8), %%zmm25, %%zmm22%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r8), %%zmm25, %%zmm23%{%%k3}%{z}\n .endif\n"
        ".endif\n"
        ".else\n" // need_mask
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14), %%zmm25, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14), %%zmm25, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14), %%zmm25, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%zmm25, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%zmm25, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%zmm25, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2), %%zmm25, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2), %%zmm25, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2), %%zmm25, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r8), %%zmm25, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r8), %%zmm25, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r8), %%zmm25, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12), %%zmm25, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12), %%zmm25, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12), %%zmm25, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%zmm25, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%zmm25, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%zmm25, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2), %%zmm25, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2), %%zmm25, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2), %%zmm25, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r8), %%zmm25, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r8), %%zmm25, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r8), %%zmm25, %%zmm23\n .endif\n"
        ".endif\n"
        ".endif\n" // need_mask
"8:\n" // label_load_c_end

        // += beta_bias * bias
        "mov BIAS_PTR_IDX(%[param]), %%rcx\n"
        "test $KERNEL_FLAG_ROW_BIAS, %%rsi\n"
        "jz 11f\n" // label_row_bias_end
        ".if NEED_MASK\n"
        ".if U_NR > 0\n vmulps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm30, %%zmm24%{%%k1}%{z}\n .endif\n"
        ".if U_NR > 1\n vmulps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm30, %%zmm25%{%%k2}%{z}\n .endif\n"
        ".if U_NR > 2\n vmulps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm30, %%zmm26%{%%k3}%{z}\n .endif\n"
        ".else\n"
        ".if U_NR > 0\n vmulps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm30, %%zmm24\n .endif\n"
        ".if U_NR > 1\n vmulps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm30, %%zmm25\n .endif\n"
        ".if U_NR > 2\n vmulps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%zmm30, %%zmm26\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vaddps %%zmm24, %%zmm0, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm25, %%zmm1, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm2, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vaddps %%zmm24, %%zmm3, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm25, %%zmm4, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm5, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vaddps %%zmm24, %%zmm6, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm25, %%zmm7, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm8, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vaddps %%zmm24, %%zmm9, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm25, %%zmm10, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm11, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vaddps %%zmm24, %%zmm12, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm25, %%zmm13, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm14, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vaddps %%zmm24, %%zmm15, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm25, %%zmm16, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm17, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vaddps %%zmm24, %%zmm18, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm25, %%zmm19, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm20, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vaddps %%zmm24, %%zmm21, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm25, %%zmm22, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm23, %%zmm23\n .endif\n"
        ".endif\n"
"11:\n" // label_row_bias_end
        "test $KERNEL_FLAG_SCA_BIAS, %%rsi\n"
        "jz 12f\n" // label_sca_bias_end
        "vmulps (%%rcx)%{1to16}, %%zmm30, %%zmm26\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm0, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm1, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm2, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm3, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm4, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm5, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm6, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm7, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm8, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm9, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm10, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm11, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm12, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm13, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm14, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm15, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm16, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm17, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm18, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm19, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm20, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm21, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm22, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm23, %%zmm23\n .endif\n"
        ".endif\n"
"12:\n" // label_sca_bias_end
        "test $KERNEL_FLAG_COL_BIAS, %%rsi\n"
        "jz 13f\n" // label_col_bias_end
        ".if U_M > 0\n"
        "vmulps (0 * D_BYTES)(%%rcx)%{1to16}, %%zmm30, %%zmm26\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm0, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm1, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm2, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        "vmulps (1 * D_BYTES)(%%rcx)%{1to16}, %%zmm30, %%zmm26\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm3, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm4, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm5, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        "vmulps (2 * D_BYTES)(%%rcx)%{1to16}, %%zmm30, %%zmm26\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm6, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm7, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm8, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        "vmulps (3 * D_BYTES)(%%rcx)%{1to16}, %%zmm30, %%zmm26\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm9, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm10, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm11, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n"
        "vmulps (4 * D_BYTES)(%%rcx)%{1to16}, %%zmm30, %%zmm26\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm12, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm13, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm14, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        "vmulps (5 * D_BYTES)(%%rcx)%{1to16}, %%zmm30, %%zmm26\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm15, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm16, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm17, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        "vmulps (6 * D_BYTES)(%%rcx)%{1to16}, %%zmm30, %%zmm26\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm18, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm19, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm20, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        "vmulps (7 * D_BYTES)(%%rcx)%{1to16}, %%zmm30, %%zmm26\n"
        ".if U_NR > 0\n vaddps %%zmm26, %%zmm21, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vaddps %%zmm26, %%zmm22, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vaddps %%zmm26, %%zmm23, %%zmm23\n .endif\n"
        ".endif\n"
        "lea (U_M * D_BYTES)(%%rcx), %%rcx\n"
        "mov %%rcx, BIAS_PTR_IDX(%[param])\n"
"13:\n" // label_col_bias_end

        "test $(KERNEL_FLAG_RELU | KERNEL_FLAG_RELU6), %%rsi\n"
        "jz 9f\n" // label_relu_end
        "vpxord %%zmm25, %%zmm25, %%zmm25\n" // 0.0
        "mov $0x40c00000, %%ecx\n"
        "vmovd %%ecx, %%xmm26\n"
        "vbroadcastss %%xmm26, %%zmm26\n" // 6.0
        "test $KERNEL_FLAG_RELU6, %%rsi\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vmaxps %%zmm25, %%zmm0, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vmaxps %%zmm25, %%zmm1, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vmaxps %%zmm25, %%zmm2, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vmaxps %%zmm25, %%zmm3, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vmaxps %%zmm25, %%zmm4, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vmaxps %%zmm25, %%zmm5, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vmaxps %%zmm25, %%zmm6, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vmaxps %%zmm25, %%zmm7, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vmaxps %%zmm25, %%zmm8, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vmaxps %%zmm25, %%zmm9, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vmaxps %%zmm25, %%zmm10, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vmaxps %%zmm25, %%zmm11, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vmaxps %%zmm25, %%zmm12, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vmaxps %%zmm25, %%zmm13, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vmaxps %%zmm25, %%zmm14, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vmaxps %%zmm25, %%zmm15, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vmaxps %%zmm25, %%zmm16, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vmaxps %%zmm25, %%zmm17, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vmaxps %%zmm25, %%zmm18, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vmaxps %%zmm25, %%zmm19, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vmaxps %%zmm25, %%zmm20, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vmaxps %%zmm25, %%zmm21, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vmaxps %%zmm25, %%zmm22, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vmaxps %%zmm25, %%zmm23, %%zmm23\n .endif\n"
        ".endif\n"

        "jz 9f\n" // label_relu_end
        ".if U_M > 0\n"
        ".if U_NR > 0\n vminps %%zmm26, %%zmm0, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vminps %%zmm26, %%zmm1, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vminps %%zmm26, %%zmm2, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vminps %%zmm26, %%zmm3, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vminps %%zmm26, %%zmm4, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vminps %%zmm26, %%zmm5, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vminps %%zmm26, %%zmm6, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vminps %%zmm26, %%zmm7, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vminps %%zmm26, %%zmm8, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vminps %%zmm26, %%zmm9, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vminps %%zmm26, %%zmm10, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vminps %%zmm26, %%zmm11, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vminps %%zmm26, %%zmm12, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vminps %%zmm26, %%zmm13, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vminps %%zmm26, %%zmm14, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vminps %%zmm26, %%zmm15, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vminps %%zmm26, %%zmm16, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vminps %%zmm26, %%zmm17, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vminps %%zmm26, %%zmm18, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vminps %%zmm26, %%zmm19, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vminps %%zmm26, %%zmm20, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vminps %%zmm26, %%zmm21, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vminps %%zmm26, %%zmm22, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vminps %%zmm26, %%zmm23, %%zmm23\n .endif\n"
        ".endif\n"
"9:\n" // label_relu_end

        "sub $U_M, %%r13\n" // m -= u_m
        ".if NEED_MASK\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vmovups %%zmm0, (0 * N_REG_ELTS * D_BYTES)(%%r14)%{%%k1}\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm1, (1 * N_REG_ELTS * D_BYTES)(%%r14)%{%%k2}\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm2, (2 * N_REG_ELTS * D_BYTES)(%%r14)%{%%k3}\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vmovups %%zmm3, (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)%{%%k1}\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm4, (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)%{%%k2}\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm5, (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)%{%%k3}\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vmovups %%zmm6, (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2)%{%%k1}\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm7, (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2)%{%%k2}\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm8, (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2)%{%%k3}\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vmovups %%zmm9, (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r8)%{%%k1}\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm10, (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r8)%{%%k2}\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm11, (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r8)%{%%k3}\n .endif\n"
        ".endif\n"
        "lea (%%r14, %%r9), %%r14\n" // c_m0 += u_m * ldc
        ".if U_M > 4\n"
        ".if U_NR > 0\n vmovups %%zmm12, (0 * N_REG_ELTS * D_BYTES)(%%r12)%{%%k1}\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm13, (1 * N_REG_ELTS * D_BYTES)(%%r12)%{%%k2}\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm14, (2 * N_REG_ELTS * D_BYTES)(%%r12)%{%%k3}\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vmovups %%zmm15, (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)%{%%k1}\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm16, (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)%{%%k2}\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm17, (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)%{%%k3}\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vmovups %%zmm18, (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2)%{%%k1}\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm19, (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2)%{%%k2}\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm20, (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2)%{%%k3}\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vmovups %%zmm21, (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r8)%{%%k1}\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm22, (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r8)%{%%k2}\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm23, (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r8)%{%%k3}\n .endif\n"
        ".endif\n"
        "lea (%%r12, %%r9), %%r12\n" // c_m4 += u_m * ldc
        ".else\n" // need_mask
        ".if U_M > 0\n"
        ".if U_NR > 0\n vmovups %%zmm0, (0 * N_REG_ELTS * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm1, (1 * N_REG_ELTS * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm2, (2 * N_REG_ELTS * D_BYTES)(%%r14)\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vmovups %%zmm3, (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm4, (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm5, (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vmovups %%zmm6, (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2)\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm7, (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2)\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm8, (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11, 2)\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vmovups %%zmm9, (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r8)\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm10, (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r8)\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm11, (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r8)\n .endif\n"
        ".endif\n"
        "lea (%%r14, %%r9), %%r14\n" // c_m0 += u_m * ldc
        ".if U_M > 4\n"
        ".if U_NR > 0\n vmovups %%zmm12, (0 * N_REG_ELTS * D_BYTES)(%%r12)\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm13, (1 * N_REG_ELTS * D_BYTES)(%%r12)\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm14, (2 * N_REG_ELTS * D_BYTES)(%%r12)\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vmovups %%zmm15, (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm16, (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm17, (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vmovups %%zmm18, (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2)\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm19, (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2)\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm20, (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11, 2)\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vmovups %%zmm21, (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r8)\n .endif\n"
        ".if U_NR > 1\n vmovups %%zmm22, (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r8)\n .endif\n"
        ".if U_NR > 2\n vmovups %%zmm23, (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r8)\n .endif\n"
        ".endif\n"
        "lea (%%r12, %%r9), %%r12\n" // c_m4 += u_m * ldc
        ".endif\n"

        ".if U_M > 0\n"
        ".if U_NR > 0\n vpxord %%zmm0, %%zmm0, %%zmm0\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm1, %%zmm1, %%zmm1\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm2, %%zmm2, %%zmm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vpxord %%zmm3, %%zmm3, %%zmm3\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm4, %%zmm4, %%zmm4\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm5, %%zmm5, %%zmm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vpxord %%zmm6, %%zmm6, %%zmm6\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm7, %%zmm7, %%zmm7\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm8, %%zmm8, %%zmm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vpxord %%zmm9, %%zmm9, %%zmm9\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm10, %%zmm10, %%zmm10\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm11, %%zmm11, %%zmm11\n .endif\n"
        ".endif\n"
        ".if U_M > 4\n"
        ".if U_NR > 0\n vpxord %%zmm12, %%zmm12, %%zmm12\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm13, %%zmm13, %%zmm13\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm14, %%zmm14, %%zmm14\n .endif\n"
        ".endif\n"
        ".if U_M > 5\n"
        ".if U_NR > 0\n vpxord %%zmm15, %%zmm15, %%zmm15\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm16, %%zmm16, %%zmm16\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm17, %%zmm17, %%zmm17\n .endif\n"
        ".endif\n"
        ".if U_M > 6\n"
        ".if U_NR > 0\n vpxord %%zmm18, %%zmm18, %%zmm18\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm19, %%zmm19, %%zmm19\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm20, %%zmm20, %%zmm20\n .endif\n"
        ".endif\n"
        ".if U_M > 7\n"
        ".if U_NR > 0\n vpxord %%zmm21, %%zmm21, %%zmm21\n .endif\n"
        ".if U_NR > 1\n vpxord %%zmm22, %%zmm22, %%zmm22\n .endif\n"
        ".if U_NR > 2\n vpxord %%zmm23, %%zmm23, %%zmm23\n .endif\n"
        ".endif\n"

        "jg 1b\n" // label_init_session
        :
        :
        [param]                         "r" (param),
        [N_REG_ELTS]                    "i" (gemm_kernel_fp32_avx512::config::N_REG_ELTS),
        [NEED_MASK]                     "i" (need_mask),
        [U_M]                           "i" (u_m),
        [U_N]                           "i" (u_n),
        [KERNEL_FLAG_LOAD_C]            "i" (gemm_kernel_fp32_avx512::flag::LOAD_C),
        [KERNEL_FLAG_WITH_SUM]          "i" (gemm_kernel_fp32_avx512::flag::WITH_SUM),
        [KERNEL_FLAG_ROW_BIAS]          "i" (gemm_kernel_fp32_avx512::flag::ROW_BIAS),
        [KERNEL_FLAG_COL_BIAS]          "i" (gemm_kernel_fp32_avx512::flag::COL_BIAS),
        [KERNEL_FLAG_SCA_BIAS]          "i" (gemm_kernel_fp32_avx512::flag::SCA_BIAS),
        [KERNEL_FLAG_RELU]              "i" (gemm_kernel_fp32_avx512::flag::RELU),
        [KERNEL_FLAG_RELU6]             "i" (gemm_kernel_fp32_avx512::flag::RELU6)
        :
        "cc",
        "rax", "rbx", "rcx", "rdx",
        "r8" , "r9" , "r10", "r11",
        "r12", "r13", "r14", "r15",
        "rsi",
        "zmm0" , "zmm1" , "zmm2" , "zmm3" , "zmm4" , "zmm5" , "zmm6" , "zmm7" ,
        "zmm8" , "zmm9" , "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
        "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
        "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
        "memory"
    );
}

#endif

template<int64_t need_mask, int64_t u_m, int64_t u_n>
void gemm_m8n48_kernel_fp32_avx512(int64_t *param)
{
#ifdef PPL_USE_X86_INLINE_ASM

    gemm_m8n48_kernel_fp32_avx512_core<need_mask, u_m, u_n>(param);
    return;

#endif

    // reference intrinsic for windows, performance is not tested
    array_param_helper kp(param);
    const int64_t N_REG_ELTS = gemm_kernel_fp32_avx512::config::N_REG_ELTS;
    const int64_t u_nr = div_up(u_n, N_REG_ELTS);
    const int64_t u_k = 4;
    const int64_t u_k_log2 = 2;

    const int64_t prefetch_b_offset = 1536 / sizeof(float);
    const int64_t prefetch_a_offset = 256 / sizeof(float);
    const int64_t cacheline_elts = PPL_X86_CACHELINE_BYTES() / sizeof(float);

    // generate masks
    const __mmask16 k4 = static_cast<__mmask16>((1 << kp.pick<const int64_t>(gemm_kernel_fp32_avx512::param_def::MASK_IDX)) - 1);
    const __mmask16 k1 = need_mask && u_nr == 1 ? k4 : 0xffff;
    const __mmask16 k2 = need_mask && u_nr == 2 ? k4 : 0xffff;
    const __mmask16 k3 = need_mask && u_nr == 3 ? k4 : 0xffff;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    // load constant values
    auto k = kp.pick<int64_t>(gemm_kernel_fp32_avx512::param_def::K_IDX);
    auto prf_c_lduk = kp.pick<int64_t>(gemm_kernel_fp32_avx512::param_def::PRF_C_LDK_IDX) >> u_k_log2;
    auto b_ptr = kp.pick<const float*>(gemm_kernel_fp32_avx512::param_def::B_PTR_IDX);
    if (u_nr > 0) zmm27 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 0 * u_n);
    if (u_nr > 1) zmm28 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 0 * u_n);
    if (u_nr > 2) zmm29 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 0 * u_n);

    auto flags = kp.pick<const gemm_kernel_fp32_avx512::flag_t>(gemm_kernel_fp32_avx512::param_def::FLAGS_IDX);
    auto a_ptr = kp.pick<const float*>(gemm_kernel_fp32_avx512::param_def::A_PTR_IDX);
    auto ldc = kp.pick<const int64_t>(gemm_kernel_fp32_avx512::param_def::LDC_IDX);
    auto ldc3 = ldc + 2 * ldc;
    auto ldcm = u_m * ldc;
    auto c_m0_ptr = kp.pick<float*>(gemm_kernel_fp32_avx512::param_def::C_PTR_IDX);
    auto c_m4_ptr = c_m0_ptr + 4 * ldc;
    auto m = kp.pick<int64_t>(gemm_kernel_fp32_avx512::param_def::M_IDX);

    if (u_m > 0) {
        if (u_nr > 0) zmm0 = _mm512_setzero_ps();
        if (u_nr > 1) zmm1 = _mm512_setzero_ps();
        if (u_nr > 2) zmm2 = _mm512_setzero_ps();
    }
    if (u_m > 1) {
        if (u_nr > 0) zmm3 = _mm512_setzero_ps();
        if (u_nr > 1) zmm4 = _mm512_setzero_ps();
        if (u_nr > 2) zmm5 = _mm512_setzero_ps();
    }
    if (u_m > 2) {
        if (u_nr > 0) zmm6 = _mm512_setzero_ps();
        if (u_nr > 1) zmm7 = _mm512_setzero_ps();
        if (u_nr > 2) zmm8 = _mm512_setzero_ps();
    }
    if (u_m > 3) {
        if (u_nr > 0) zmm9 = _mm512_setzero_ps();
        if (u_nr > 1) zmm10 = _mm512_setzero_ps();
        if (u_nr > 2) zmm11 = _mm512_setzero_ps();
    }
    if (u_m > 4) {
        if (u_nr > 0) zmm12 = _mm512_setzero_ps();
        if (u_nr > 1) zmm13 = _mm512_setzero_ps();
        if (u_nr > 2) zmm14 = _mm512_setzero_ps();
    }
    if (u_m > 5) {
        if (u_nr > 0) zmm15 = _mm512_setzero_ps();
        if (u_nr > 1) zmm16 = _mm512_setzero_ps();
        if (u_nr > 2) zmm17 = _mm512_setzero_ps();
    }
    if (u_m > 6) {
        if (u_nr > 0) zmm18 = _mm512_setzero_ps();
        if (u_nr > 1) zmm19 = _mm512_setzero_ps();
        if (u_nr > 2) zmm20 = _mm512_setzero_ps();
    }
    if (u_m > 7) {
        if (u_nr > 0) zmm21 = _mm512_setzero_ps();
        if (u_nr > 1) zmm22 = _mm512_setzero_ps();
        if (u_nr > 2) zmm23 = _mm512_setzero_ps();
    }

    do {
        auto kl = (k >> u_k_log2) - prf_c_lduk;
        if (kl > 0) {
            do {
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 0 * u_m]);
                if (u_nr > 0) zmm24 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                if (u_nr > 1) zmm25 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                if (u_nr > 2) zmm26 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 0 * u_m]);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm27, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm28, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm29, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 0 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm27, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm28, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm29, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 0 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm27, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm28, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm29, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 0 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm27, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm28, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm29, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 0 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm27, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm28, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm29, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 0 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm27, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm28, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm29, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 0 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm27, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm28, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm29, zmm20);
                }
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 1 * u_m]);
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm27, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm28, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm29, zmm23);
                }
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 1 * u_m]);
                if (u_nr > 0) zmm27 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                if (u_nr > 1) zmm28 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                if (u_nr > 2) zmm29 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm24, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm25, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm26, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 1 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm24, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm25, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm26, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 1 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm24, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm25, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm26, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 1 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm24, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm25, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm26, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 1 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm24, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm25, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm26, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 1 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm24, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm25, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm26, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 1 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm24, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm25, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm26, zmm20);
                }
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 2 * u_m]);
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm24, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm25, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm26, zmm23);
                }
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 2 * u_m]);
                if (u_nr > 0) zmm24 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 3 * u_n);
                if (u_nr > 1) zmm25 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 3 * u_n);
                if (u_nr > 2) zmm26 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 3 * u_n);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm27, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm28, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm29, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 2 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm27, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm28, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm29, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 2 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm27, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm28, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm29, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 2 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm27, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm28, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm29, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 2 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm27, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm28, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm29, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 2 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm27, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm28, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm29, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 2 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm27, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm28, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm29, zmm20);
                }
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 3 * u_m]);
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm27, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm28, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm29, zmm23);
                }
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 3 * u_m]);

                if (u_nr > 0) zmm27 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 4 * u_n);
                if (u_nr > 1) zmm28 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 4 * u_n);
                if (u_nr > 2) zmm29 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 4 * u_n);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm24, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm25, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm26, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 3 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm24, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm25, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm26, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 3 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 4 * u_n + prefetch_b_offset), _MM_HINT_T0);
                kl -= 1;
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm24, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm25, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm26, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 3 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm24, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm25, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm26, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 3 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 4 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm24, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm25, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm26, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 3 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm24, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm25, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm26, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 3 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 4 * u_n + prefetch_b_offset), _MM_HINT_T0);
                b_ptr += u_k * u_n;
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm24, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm25, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm26, zmm20);
                }
                if (u_m > 4) _mm_prefetch((const char*)(a_ptr + 1 * cacheline_elts + prefetch_a_offset), _MM_HINT_T0);
                a_ptr += u_k * u_m;
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm24, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm25, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm26, zmm23);
                }
            } while (kl > 0);
        }

        kl += u_m;
        prf_c_lduk -= u_m;
        if (kl > 0) {
            auto c_pf_ptr = c_m0_ptr + cacheline_elts - 1;
#ifdef PPL_USE_X86_MSVC
            const auto c_mm_hint = _MM_HINT_T0;
#else
            const auto c_mm_hint = _MM_HINT_ET0;
#endif
            do {
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 0 * u_m]);
                if (u_nr > 0) zmm24 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                if (u_nr > 1) zmm25 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                if (u_nr > 2) zmm26 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 0 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(c_pf_ptr + 0 * N_REG_ELTS), c_mm_hint);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm27, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm28, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm29, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 0 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm27, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm28, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm29, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 0 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm27, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm28, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm29, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 0 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm27, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm28, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm29, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 0 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm27, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm28, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm29, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 0 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm27, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm28, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm29, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 0 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm27, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm28, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm29, zmm20);
                }
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 1 * u_m]);
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm27, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm28, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm29, zmm23);
                }
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 1 * u_m]);
                if (u_nr > 0) zmm27 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                if (u_nr > 1) zmm28 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                if (u_nr > 2) zmm29 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                if (u_nr > 1) _mm_prefetch((const char*)(c_pf_ptr + 1 * N_REG_ELTS), c_mm_hint);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm24, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm25, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm26, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 1 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm24, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm25, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm26, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 1 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm24, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm25, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm26, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 1 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm24, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm25, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm26, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 1 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm24, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm25, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm26, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 1 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm24, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm25, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm26, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 1 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm24, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm25, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm26, zmm20);
                }
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 2 * u_m]);
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm24, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm25, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm26, zmm23);
                }
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 2 * u_m]);
                if (u_nr > 0) zmm24 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 3 * u_n);
                if (u_nr > 1) zmm25 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 3 * u_n);
                if (u_nr > 2) zmm26 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 3 * u_n);
                if (u_nr > 2) _mm_prefetch((const char*)(c_pf_ptr + 2 * N_REG_ELTS), c_mm_hint);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm27, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm28, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm29, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 2 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm27, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm28, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm29, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 2 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm27, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm28, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm29, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 2 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm27, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm28, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm29, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 2 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm27, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm28, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm29, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 2 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm27, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm28, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm29, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 2 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm27, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm28, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm29, zmm20);
                }
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 3 * u_m]);
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm27, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm28, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm29, zmm23);
                }
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 3 * u_m]);
                if (u_nr > 0) zmm27 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 4 * u_n);
                if (u_nr > 1) zmm28 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 4 * u_n);
                if (u_nr > 2) zmm29 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 4 * u_n);
                c_pf_ptr += ldc;
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm24, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm25, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm26, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 3 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm24, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm25, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm26, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 3 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 4 * u_n + prefetch_b_offset), _MM_HINT_T0);
                kl -= 1;
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm24, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm25, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm26, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 3 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm24, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm25, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm26, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 3 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 4 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm24, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm25, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm26, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 3 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm24, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm25, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm26, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 3 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 4 * u_n + prefetch_b_offset), _MM_HINT_T0);
                b_ptr += u_k * u_n;
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm24, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm25, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm26, zmm20);
                }
                if (u_m > 4) _mm_prefetch((const char*)(a_ptr + 1 * cacheline_elts + prefetch_a_offset), _MM_HINT_T0);
                a_ptr += u_k * u_m;
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm24, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm25, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm26, zmm23);
                }
            } while (kl > 0);
        }

        kl += prf_c_lduk;
        if (kl > 0) {
            do {
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 0 * u_m]);
                if (u_nr > 0) zmm24 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                if (u_nr > 1) zmm25 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                if (u_nr > 2) zmm26 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 0 * u_m]);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm27, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm28, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm29, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 0 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm27, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm28, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm29, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 0 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm27, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm28, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm29, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 0 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm27, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm28, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm29, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 0 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm27, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm28, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm29, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 0 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm27, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm28, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm29, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 0 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm27, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm28, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm29, zmm20);
                }
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 1 * u_m]);
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm27, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm28, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm29, zmm23);
                }
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 1 * u_m]);
                if (u_nr > 0) zmm27 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                if (u_nr > 1) zmm28 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                if (u_nr > 2) zmm29 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm24, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm25, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm26, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 1 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm24, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm25, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm26, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 1 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm24, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm25, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm26, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 1 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm24, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm25, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm26, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 1 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm24, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm25, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm26, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 1 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm24, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm25, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm26, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 1 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm24, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm25, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm26, zmm20);
                }
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 2 * u_m]);
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm24, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm25, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm26, zmm23);
                }
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 2 * u_m]);
                if (u_nr > 0) zmm24 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 3 * u_n);
                if (u_nr > 1) zmm25 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 3 * u_n);
                if (u_nr > 2) zmm26 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 3 * u_n);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm27, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm28, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm29, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 2 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm27, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm28, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm29, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 2 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm27, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm28, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm29, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 2 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm27, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm28, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm29, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 2 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm27, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm28, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm29, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 2 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm27, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm28, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm29, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 2 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm27, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm28, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm29, zmm20);
                }
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 3 * u_m]);
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm27, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm28, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm29, zmm23);
                }
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 3 * u_m]);
                if (u_nr > 0) zmm27 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 4 * u_n);
                if (u_nr > 1) zmm28 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 4 * u_n);
                if (u_nr > 2) zmm29 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 4 * u_n);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm24, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm25, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm26, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 3 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm24, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm25, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm26, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 3 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 4 * u_n + prefetch_b_offset), _MM_HINT_T0);
                kl -= 1;
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm24, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm25, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm26, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 3 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm24, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm25, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm26, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 3 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 4 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm24, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm25, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm26, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 3 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm24, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm25, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm26, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 3 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 4 * u_n + prefetch_b_offset), _MM_HINT_T0);
                b_ptr += u_k * u_n;
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm24, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm25, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm26, zmm20);
                }
                if (u_m > 4) _mm_prefetch((const char*)(a_ptr + 1 * cacheline_elts + prefetch_a_offset), _MM_HINT_T0);
                a_ptr += u_k * u_m;
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm24, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm25, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm26, zmm23);
                }
            } while (kl > 0);
        }

        kl = k & 3;
        if (kl > 0) {
            {
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 0 * u_m]);
                if (u_nr > 0) zmm24 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                if (u_nr > 1) zmm25 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                if (u_nr > 2) zmm26 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 0 * u_m]);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm27, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm28, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm29, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 0 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm27, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm28, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm29, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 0 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm27, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm28, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm29, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 0 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm27, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm28, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm29, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 0 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm27, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm28, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm29, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 0 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm27, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm28, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm29, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 0 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm27, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm28, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm29, zmm20);
                }
                if (u_m > 4) _mm_prefetch((const char*)(a_ptr + 1 * cacheline_elts + prefetch_a_offset), _MM_HINT_T0);
                a_ptr += u_m;
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm27, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm28, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm29, zmm23);
                }
            }
        }
        if (kl > 1) {
            {
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 0 * u_m]);
                if (u_nr > 0) zmm27 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                if (u_nr > 1) zmm28 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                if (u_nr > 2) zmm29 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 0 * u_m]);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm24, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm25, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm26, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 0 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm24, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm25, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm26, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 0 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm24, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm25, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm26, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 0 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm24, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm25, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm26, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 0 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm24, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm25, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm26, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 0 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm24, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm25, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm26, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 0 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm24, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm25, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm26, zmm20);
                }
                a_ptr += u_m;
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm24, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm25, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm26, zmm23);
                }
            }
        }
        if (kl > 2) {
            {
                if (u_m > 0) zmm30 = _mm512_set1_ps(a_ptr[0 + 0 * u_m]);
                if (u_m > 1) zmm31 = _mm512_set1_ps(a_ptr[1 + 0 * u_m]);
                if (u_m > 0) {
                    if (u_nr > 0) zmm0 = _mm512_fmadd_ps(zmm30, zmm27, zmm0);
                    if (u_nr > 1) zmm1 = _mm512_fmadd_ps(zmm30, zmm28, zmm1);
                    if (u_nr > 2) zmm2 = _mm512_fmadd_ps(zmm30, zmm29, zmm2);
                }
                if (u_m > 2) zmm30 = _mm512_set1_ps(a_ptr[2 + 0 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) zmm3 = _mm512_fmadd_ps(zmm31, zmm27, zmm3);
                    if (u_nr > 1) zmm4 = _mm512_fmadd_ps(zmm31, zmm28, zmm4);
                    if (u_nr > 2) zmm5 = _mm512_fmadd_ps(zmm31, zmm29, zmm5);
                }
                if (u_m > 3) zmm31 = _mm512_set1_ps(a_ptr[3 + 0 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) zmm6 = _mm512_fmadd_ps(zmm30, zmm27, zmm6);
                    if (u_nr > 1) zmm7 = _mm512_fmadd_ps(zmm30, zmm28, zmm7);
                    if (u_nr > 2) zmm8 = _mm512_fmadd_ps(zmm30, zmm29, zmm8);
                }
                if (u_m > 4) zmm30 = _mm512_set1_ps(a_ptr[4 + 0 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) zmm9 = _mm512_fmadd_ps(zmm31, zmm27, zmm9);
                    if (u_nr > 1) zmm10 = _mm512_fmadd_ps(zmm31, zmm28, zmm10);
                    if (u_nr > 2) zmm11 = _mm512_fmadd_ps(zmm31, zmm29, zmm11);
                }
                if (u_m > 5) zmm31 = _mm512_set1_ps(a_ptr[5 + 0 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 4) {
                    if (u_nr > 0) zmm12 = _mm512_fmadd_ps(zmm30, zmm27, zmm12);
                    if (u_nr > 1) zmm13 = _mm512_fmadd_ps(zmm30, zmm28, zmm13);
                    if (u_nr > 2) zmm14 = _mm512_fmadd_ps(zmm30, zmm29, zmm14);
                }
                if (u_m > 6) zmm30 = _mm512_set1_ps(a_ptr[6 + 0 * u_m]);
                if (u_m > 5) {
                    if (u_nr > 0) zmm15 = _mm512_fmadd_ps(zmm31, zmm27, zmm15);
                    if (u_nr > 1) zmm16 = _mm512_fmadd_ps(zmm31, zmm28, zmm16);
                    if (u_nr > 2) zmm17 = _mm512_fmadd_ps(zmm31, zmm29, zmm17);
                }
                if (u_m > 7) zmm31 = _mm512_set1_ps(a_ptr[7 + 0 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * N_REG_ELTS + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 6) {
                    if (u_nr > 0) zmm18 = _mm512_fmadd_ps(zmm30, zmm27, zmm18);
                    if (u_nr > 1) zmm19 = _mm512_fmadd_ps(zmm30, zmm28, zmm19);
                    if (u_nr > 2) zmm20 = _mm512_fmadd_ps(zmm30, zmm29, zmm20);
                }a_ptr += u_m;
                if (u_m > 7) {
                    if (u_nr > 0) zmm21 = _mm512_fmadd_ps(zmm31, zmm27, zmm21);
                    if (u_nr > 1) zmm22 = _mm512_fmadd_ps(zmm31, zmm28, zmm22);
                    if (u_nr > 2) zmm23 = _mm512_fmadd_ps(zmm31, zmm29, zmm23);
                }
            }
        }

        b_ptr = kp.pick<const float*>(gemm_kernel_fp32_avx512::param_def::B_PTR_IDX);
        prf_c_lduk = kp.pick<int64_t>(gemm_kernel_fp32_avx512::param_def::PRF_C_LDK_IDX) >> u_k_log2;
        zmm24 = _mm512_set1_ps(kp.pick<float>(gemm_kernel_fp32_avx512::param_def::ALPHA_IDX));
        zmm25 = _mm512_set1_ps(kp.pick<float>(gemm_kernel_fp32_avx512::param_def::BETA_IDX));
        zmm30 = _mm512_set1_ps(kp.pick<float>(gemm_kernel_fp32_avx512::param_def::BETA_BIAS_IDX));
        zmm31 = _mm512_set1_ps(kp.pick<float>(gemm_kernel_fp32_avx512::param_def::BETA_SUM_IDX));
        if (u_nr > 0) zmm27 = _mm512_loadu_ps(b_ptr + 0 * N_REG_ELTS + 0 * u_n);
        if (u_nr > 1) zmm28 = _mm512_loadu_ps(b_ptr + 1 * N_REG_ELTS + 0 * u_n);
        if (u_nr > 2) zmm29 = _mm512_loadu_ps(b_ptr + 2 * N_REG_ELTS + 0 * u_n);

        // *= alpha
        if (u_m > 0) {
            if (u_nr > 0) zmm0 = _mm512_mul_ps(zmm24, zmm0);
            if (u_nr > 1) zmm1 = _mm512_mul_ps(zmm24, zmm1);
            if (u_nr > 2) zmm2 = _mm512_mul_ps(zmm24, zmm2);
        }
        if (u_m > 1) {
            if (u_nr > 0) zmm3 = _mm512_mul_ps(zmm24, zmm3);
            if (u_nr > 1) zmm4 = _mm512_mul_ps(zmm24, zmm4);
            if (u_nr > 2) zmm5 = _mm512_mul_ps(zmm24, zmm5);
        }
        if (u_m > 2) {
            if (u_nr > 0) zmm6 = _mm512_mul_ps(zmm24, zmm6);
            if (u_nr > 1) zmm7 = _mm512_mul_ps(zmm24, zmm7);
            if (u_nr > 2) zmm8 = _mm512_mul_ps(zmm24, zmm8);
        }
        if (u_m > 3) {
            if (u_nr > 0) zmm9 = _mm512_mul_ps(zmm24, zmm9);
            if (u_nr > 1) zmm10 = _mm512_mul_ps(zmm24, zmm10);
            if (u_nr > 2) zmm11 = _mm512_mul_ps(zmm24, zmm11);
        }
        if (u_m > 4) {
            if (u_nr > 0) zmm12 = _mm512_mul_ps(zmm24, zmm12);
            if (u_nr > 1) zmm13 = _mm512_mul_ps(zmm24, zmm13);
            if (u_nr > 2) zmm14 = _mm512_mul_ps(zmm24, zmm14);
        }
        if (u_m > 5) {
            if (u_nr > 0) zmm15 = _mm512_mul_ps(zmm24, zmm15);
            if (u_nr > 1) zmm16 = _mm512_mul_ps(zmm24, zmm16);
            if (u_nr > 2) zmm17 = _mm512_mul_ps(zmm24, zmm17);
        }
        if (u_m > 6) {
            if (u_nr > 0) zmm18 = _mm512_mul_ps(zmm24, zmm18);
            if (u_nr > 1) zmm19 = _mm512_mul_ps(zmm24, zmm19);
            if (u_nr > 2) zmm20 = _mm512_mul_ps(zmm24, zmm20);
        }
        if (u_m > 7) {
            if (u_nr > 0) zmm21 = _mm512_mul_ps(zmm24, zmm21);
            if (u_nr > 1) zmm22 = _mm512_mul_ps(zmm24, zmm22);
            if (u_nr > 2) zmm23 = _mm512_mul_ps(zmm24, zmm23);
        }

        if (flags & gemm_kernel_fp32_avx512::flag::WITH_SUM) {
            auto sum_ptr = kp.pick<const float*>(gemm_kernel_fp32_avx512::param_def::SUM_PTR_IDX);
            auto ldsum = kp.pick<const int64_t>(gemm_kernel_fp32_avx512::param_def::LDSUM_IDX);
            if (u_m > 0) {
                if (u_nr > 0) zmm0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, sum_ptr + 0 * N_REG_ELTS), zmm31, zmm0);
                if (u_nr > 1) zmm1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, sum_ptr + 1 * N_REG_ELTS), zmm31, zmm1);
                if (u_nr > 2) zmm2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, sum_ptr + 2 * N_REG_ELTS), zmm31, zmm2);
                sum_ptr += ldsum;
            }
            if (u_m > 1) {
                if (u_nr > 0) zmm3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, sum_ptr + 0 * N_REG_ELTS), zmm31, zmm3);
                if (u_nr > 1) zmm4 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, sum_ptr + 1 * N_REG_ELTS), zmm31, zmm4);
                if (u_nr > 2) zmm5 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, sum_ptr + 2 * N_REG_ELTS), zmm31, zmm5);
                sum_ptr += ldsum;
            }
            if (u_m > 2) {
                if (u_nr > 0) zmm6 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, sum_ptr + 0 * N_REG_ELTS), zmm31, zmm6);
                if (u_nr > 1) zmm7 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, sum_ptr + 1 * N_REG_ELTS), zmm31, zmm7);
                if (u_nr > 2) zmm8 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, sum_ptr + 2 * N_REG_ELTS), zmm31, zmm8);
                sum_ptr += ldsum;
            }
            if (u_m > 3) {
                if (u_nr > 0) zmm9 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, sum_ptr + 0 * N_REG_ELTS), zmm31, zmm9);
                if (u_nr > 1) zmm10 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, sum_ptr + 1 * N_REG_ELTS), zmm31, zmm10);
                if (u_nr > 2) zmm11 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, sum_ptr + 2 * N_REG_ELTS), zmm31, zmm11);
                sum_ptr += ldsum;
            }
            if (u_m > 4) {
                if (u_nr > 0) zmm12 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, sum_ptr + 0 * N_REG_ELTS), zmm31, zmm12);
                if (u_nr > 1) zmm13 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, sum_ptr + 1 * N_REG_ELTS), zmm31, zmm13);
                if (u_nr > 2) zmm14 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, sum_ptr + 2 * N_REG_ELTS), zmm31, zmm14);
                sum_ptr += ldsum;
            }
            if (u_m > 5) {
                if (u_nr > 0) zmm15 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, sum_ptr + 0 * N_REG_ELTS), zmm31, zmm15);
                if (u_nr > 1) zmm16 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, sum_ptr + 1 * N_REG_ELTS), zmm31, zmm16);
                if (u_nr > 2) zmm17 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, sum_ptr + 2 * N_REG_ELTS), zmm31, zmm17);
                sum_ptr += ldsum;
            }
            if (u_m > 6) {
                if (u_nr > 0) zmm18 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, sum_ptr + 0 * N_REG_ELTS), zmm31, zmm18);
                if (u_nr > 1) zmm19 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, sum_ptr + 1 * N_REG_ELTS), zmm31, zmm19);
                if (u_nr > 2) zmm20 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, sum_ptr + 2 * N_REG_ELTS), zmm31, zmm20);
                sum_ptr += ldsum;
            }
            if (u_m > 7) {
                if (u_nr > 0) zmm21 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, sum_ptr + 0 * N_REG_ELTS), zmm31, zmm21);
                if (u_nr > 1) zmm22 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, sum_ptr + 1 * N_REG_ELTS), zmm31, zmm22);
                if (u_nr > 2) zmm23 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, sum_ptr + 2 * N_REG_ELTS), zmm31, zmm23);
                sum_ptr += ldsum;
            }
            kp.pick<const float*>(gemm_kernel_fp32_avx512::param_def::SUM_PTR_IDX) = sum_ptr;
        }

        if (flags & gemm_kernel_fp32_avx512::flag::LOAD_C) {
            if (u_m > 0) {
                if (u_nr > 0) zmm0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, c_m0_ptr + 0 * ldc + 0 * N_REG_ELTS), zmm25, zmm0);
                if (u_nr > 1) zmm1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, c_m0_ptr + 0 * ldc + 1 * N_REG_ELTS), zmm25, zmm1);
                if (u_nr > 2) zmm2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, c_m0_ptr + 0 * ldc + 2 * N_REG_ELTS), zmm25, zmm2);
            }
            if (u_m > 1) {
                if (u_nr > 0) zmm3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, c_m0_ptr + 1 * ldc + 0 * N_REG_ELTS), zmm25, zmm3);
                if (u_nr > 1) zmm4 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, c_m0_ptr + 1 * ldc + 1 * N_REG_ELTS), zmm25, zmm4);
                if (u_nr > 2) zmm5 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, c_m0_ptr + 1 * ldc + 2 * N_REG_ELTS), zmm25, zmm5);
            }
            if (u_m > 2) {
                if (u_nr > 0) zmm6 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, c_m0_ptr + 2 * ldc + 0 * N_REG_ELTS), zmm25, zmm6);
                if (u_nr > 1) zmm7 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, c_m0_ptr + 2 * ldc + 1 * N_REG_ELTS), zmm25, zmm7);
                if (u_nr > 2) zmm8 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, c_m0_ptr + 2 * ldc + 2 * N_REG_ELTS), zmm25, zmm8);
            }
            if (u_m > 3) {
                if (u_nr > 0) zmm9 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, c_m0_ptr + 1 * ldc3 + 0 * N_REG_ELTS), zmm25, zmm9);
                if (u_nr > 1) zmm10 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, c_m0_ptr + 1 * ldc3 + 1 * N_REG_ELTS), zmm25, zmm10);
                if (u_nr > 2) zmm11 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, c_m0_ptr + 1 * ldc3 + 2 * N_REG_ELTS), zmm25, zmm11);
            }
            if (u_m > 4) {
                if (u_nr > 0) zmm12 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, c_m4_ptr + 0 * ldc + 0 * N_REG_ELTS), zmm25, zmm12);
                if (u_nr > 1) zmm13 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, c_m4_ptr + 0 * ldc + 1 * N_REG_ELTS), zmm25, zmm13);
                if (u_nr > 2) zmm14 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, c_m4_ptr + 0 * ldc + 2 * N_REG_ELTS), zmm25, zmm14);
            }
            if (u_m > 5) {
                if (u_nr > 0) zmm15 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, c_m4_ptr + 1 * ldc + 0 * N_REG_ELTS), zmm25, zmm15);
                if (u_nr > 1) zmm16 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, c_m4_ptr + 1 * ldc + 1 * N_REG_ELTS), zmm25, zmm16);
                if (u_nr > 2) zmm17 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, c_m4_ptr + 1 * ldc + 2 * N_REG_ELTS), zmm25, zmm17);
            }
            if (u_m > 6) {
                if (u_nr > 0) zmm18 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, c_m4_ptr + 2 * ldc + 0 * N_REG_ELTS), zmm25, zmm18);
                if (u_nr > 1) zmm19 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, c_m4_ptr + 2 * ldc + 1 * N_REG_ELTS), zmm25, zmm19);
                if (u_nr > 2) zmm20 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, c_m4_ptr + 2 * ldc + 2 * N_REG_ELTS), zmm25, zmm20);
            }
            if (u_m > 7) {
                if (u_nr > 0) zmm21 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k1, c_m4_ptr + 1 * ldc3 + 0 * N_REG_ELTS), zmm25, zmm21);
                if (u_nr > 1) zmm22 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k2, c_m4_ptr + 1 * ldc3 + 1 * N_REG_ELTS), zmm25, zmm22);
                if (u_nr > 2) zmm23 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k3, c_m4_ptr + 1 * ldc3 + 2 * N_REG_ELTS), zmm25, zmm23);
            }
        }

        if (flags & gemm_kernel_fp32_avx512::flag::ROW_BIAS) {
            auto bias_ptr = kp.pick<const float*>(gemm_kernel_fp32_avx512::param_def::BIAS_PTR_IDX);
            if (u_nr > 0) zmm24 = _mm512_mul_ps(_mm512_maskz_load_ps(k1, bias_ptr + 0 * N_REG_ELTS), zmm30);
            if (u_nr > 1) zmm25 = _mm512_mul_ps(_mm512_maskz_load_ps(k2, bias_ptr + 1 * N_REG_ELTS), zmm30);
            if (u_nr > 2) zmm26 = _mm512_mul_ps(_mm512_maskz_load_ps(k3, bias_ptr + 2 * N_REG_ELTS), zmm30);
            if (u_m > 0) {
                if (u_nr > 0) zmm0 = _mm512_add_ps(zmm24, zmm0);
                if (u_nr > 1) zmm1 = _mm512_add_ps(zmm25, zmm1);
                if (u_nr > 2) zmm2 = _mm512_add_ps(zmm26, zmm2);
            }
            if (u_m > 1) {
                if (u_nr > 0) zmm3 = _mm512_add_ps(zmm24, zmm3);
                if (u_nr > 1) zmm4 = _mm512_add_ps(zmm25, zmm4);
                if (u_nr > 2) zmm5 = _mm512_add_ps(zmm26, zmm5);
            }
            if (u_m > 2) {
                if (u_nr > 0) zmm6 = _mm512_add_ps(zmm24, zmm6);
                if (u_nr > 1) zmm7 = _mm512_add_ps(zmm25, zmm7);
                if (u_nr > 2) zmm8 = _mm512_add_ps(zmm26, zmm8);
            }
            if (u_m > 3) {
                if (u_nr > 0) zmm9 = _mm512_add_ps(zmm24, zmm9);
                if (u_nr > 1) zmm10 = _mm512_add_ps(zmm25, zmm10);
                if (u_nr > 2) zmm11 = _mm512_add_ps(zmm26, zmm11);
            }
            if (u_m > 4) {
                if (u_nr > 0) zmm12 = _mm512_add_ps(zmm24, zmm12);
                if (u_nr > 1) zmm13 = _mm512_add_ps(zmm25, zmm13);
                if (u_nr > 2) zmm14 = _mm512_add_ps(zmm26, zmm14);
            }
            if (u_m > 5) {
                if (u_nr > 0) zmm15 = _mm512_add_ps(zmm24, zmm15);
                if (u_nr > 1) zmm16 = _mm512_add_ps(zmm25, zmm16);
                if (u_nr > 2) zmm17 = _mm512_add_ps(zmm26, zmm17);
            }
            if (u_m > 6) {
                if (u_nr > 0) zmm18 = _mm512_add_ps(zmm24, zmm18);
                if (u_nr > 1) zmm19 = _mm512_add_ps(zmm25, zmm19);
                if (u_nr > 2) zmm20 = _mm512_add_ps(zmm26, zmm20);
            }
            if (u_m > 7) {
                if (u_nr > 0) zmm21 = _mm512_add_ps(zmm24, zmm21);
                if (u_nr > 1) zmm22 = _mm512_add_ps(zmm25, zmm22);
                if (u_nr > 2) zmm23 = _mm512_add_ps(zmm26, zmm23);
            }
        }
        if (flags & gemm_kernel_fp32_avx512::flag::SCA_BIAS) {
            auto bias_ptr = kp.pick<const float*>(gemm_kernel_fp32_avx512::param_def::BIAS_PTR_IDX);
            zmm26 = _mm512_mul_ps(_mm512_set1_ps(bias_ptr[0]), zmm30);
            if (u_m > 0) {
                if (u_nr > 0) zmm0 = _mm512_add_ps(zmm26, zmm0);
                if (u_nr > 1) zmm1 = _mm512_add_ps(zmm26, zmm1);
                if (u_nr > 2) zmm2 = _mm512_add_ps(zmm26, zmm2);
            }
            if (u_m > 1) {
                if (u_nr > 0) zmm3 = _mm512_add_ps(zmm26, zmm3);
                if (u_nr > 1) zmm4 = _mm512_add_ps(zmm26, zmm4);
                if (u_nr > 2) zmm5 = _mm512_add_ps(zmm26, zmm5);
            }
            if (u_m > 2) {
                if (u_nr > 0) zmm6 = _mm512_add_ps(zmm26, zmm6);
                if (u_nr > 1) zmm7 = _mm512_add_ps(zmm26, zmm7);
                if (u_nr > 2) zmm8 = _mm512_add_ps(zmm26, zmm8);
            }
            if (u_m > 3) {
                if (u_nr > 0) zmm9 = _mm512_add_ps(zmm26, zmm9);
                if (u_nr > 1) zmm10 = _mm512_add_ps(zmm26, zmm10);
                if (u_nr > 2) zmm11 = _mm512_add_ps(zmm26, zmm11);
            }
            if (u_m > 4) {
                if (u_nr > 0) zmm12 = _mm512_add_ps(zmm26, zmm12);
                if (u_nr > 1) zmm13 = _mm512_add_ps(zmm26, zmm13);
                if (u_nr > 2) zmm14 = _mm512_add_ps(zmm26, zmm14);
            }
            if (u_m > 5) {
                if (u_nr > 0) zmm15 = _mm512_add_ps(zmm26, zmm15);
                if (u_nr > 1) zmm16 = _mm512_add_ps(zmm26, zmm16);
                if (u_nr > 2) zmm17 = _mm512_add_ps(zmm26, zmm17);
            }
            if (u_m > 6) {
                if (u_nr > 0) zmm18 = _mm512_add_ps(zmm26, zmm18);
                if (u_nr > 1) zmm19 = _mm512_add_ps(zmm26, zmm19);
                if (u_nr > 2) zmm20 = _mm512_add_ps(zmm26, zmm20);
            }
            if (u_m > 7) {
                if (u_nr > 0) zmm21 = _mm512_add_ps(zmm26, zmm21);
                if (u_nr > 1) zmm22 = _mm512_add_ps(zmm26, zmm22);
                if (u_nr > 2) zmm23 = _mm512_add_ps(zmm26, zmm23);
            }
        }
        if (flags & gemm_kernel_fp32_avx512::flag::COL_BIAS) {
            auto bias_ptr = kp.pick<const float*>(gemm_kernel_fp32_avx512::param_def::BIAS_PTR_IDX);
            if (u_m > 0) {
                zmm26 = _mm512_mul_ps(_mm512_set1_ps(bias_ptr[0]), zmm30);
                if (u_nr > 0) zmm0 = _mm512_add_ps(zmm26, zmm0);
                if (u_nr > 1) zmm1 = _mm512_add_ps(zmm26, zmm1);
                if (u_nr > 2) zmm2 = _mm512_add_ps(zmm26, zmm2);
            }
            if (u_m > 1) {
                zmm26 = _mm512_mul_ps(_mm512_set1_ps(bias_ptr[1]), zmm30);
                if (u_nr > 0) zmm3 = _mm512_add_ps(zmm26, zmm3);
                if (u_nr > 1) zmm4 = _mm512_add_ps(zmm26, zmm4);
                if (u_nr > 2) zmm5 = _mm512_add_ps(zmm26, zmm5);
            }
            if (u_m > 2) {
                zmm26 = _mm512_mul_ps(_mm512_set1_ps(bias_ptr[2]), zmm30);
                if (u_nr > 0) zmm6 = _mm512_add_ps(zmm26, zmm6);
                if (u_nr > 1) zmm7 = _mm512_add_ps(zmm26, zmm7);
                if (u_nr > 2) zmm8 = _mm512_add_ps(zmm26, zmm8);
            }
            if (u_m > 3) {
                zmm26 = _mm512_mul_ps(_mm512_set1_ps(bias_ptr[3]), zmm30);
                if (u_nr > 0) zmm9 = _mm512_add_ps(zmm26, zmm9);
                if (u_nr > 1) zmm10 = _mm512_add_ps(zmm26, zmm10);
                if (u_nr > 2) zmm11 = _mm512_add_ps(zmm26, zmm11);
            }
            if (u_m > 4) {
                zmm26 = _mm512_mul_ps(_mm512_set1_ps(bias_ptr[4]), zmm30);
                if (u_nr > 0) zmm12 = _mm512_add_ps(zmm26, zmm12);
                if (u_nr > 1) zmm13 = _mm512_add_ps(zmm26, zmm13);
                if (u_nr > 2) zmm14 = _mm512_add_ps(zmm26, zmm14);
            }
            if (u_m > 5) {
                zmm26 = _mm512_mul_ps(_mm512_set1_ps(bias_ptr[5]), zmm30);
                if (u_nr > 0) zmm15 = _mm512_add_ps(zmm26, zmm15);
                if (u_nr > 1) zmm16 = _mm512_add_ps(zmm26, zmm16);
                if (u_nr > 2) zmm17 = _mm512_add_ps(zmm26, zmm17);
            }
            if (u_m > 6) {
                zmm26 = _mm512_mul_ps(_mm512_set1_ps(bias_ptr[6]), zmm30);
                if (u_nr > 0) zmm18 = _mm512_add_ps(zmm26, zmm18);
                if (u_nr > 1) zmm19 = _mm512_add_ps(zmm26, zmm19);
                if (u_nr > 2) zmm20 = _mm512_add_ps(zmm26, zmm20);
            }
            if (u_m > 7) {
                zmm26 = _mm512_mul_ps(_mm512_set1_ps(bias_ptr[7]), zmm30);
                if (u_nr > 0) zmm21 = _mm512_add_ps(zmm26, zmm21);
                if (u_nr > 1) zmm22 = _mm512_add_ps(zmm26, zmm22);
                if (u_nr > 2) zmm23 = _mm512_add_ps(zmm26, zmm23);
            }
            bias_ptr += u_m;
            kp.pick<const float*>(gemm_kernel_fp32_avx512::param_def::BIAS_PTR_IDX) = bias_ptr;
        }

        if (flags & (gemm_kernel_fp32_avx512::flag::RELU | gemm_kernel_fp32_avx512::flag::RELU6)) {
            zmm25 = _mm512_setzero_ps();
            zmm26 = _mm512_set1_ps(6.0f);
            if (u_m > 0) {
                if (u_nr > 0) zmm0 = _mm512_max_ps(zmm25, zmm0);
                if (u_nr > 1) zmm1 = _mm512_max_ps(zmm25, zmm1);
                if (u_nr > 2) zmm2 = _mm512_max_ps(zmm25, zmm2);
            }
            if (u_m > 1) {
                if (u_nr > 0) zmm3 = _mm512_max_ps(zmm25, zmm3);
                if (u_nr > 1) zmm4 = _mm512_max_ps(zmm25, zmm4);
                if (u_nr > 2) zmm5 = _mm512_max_ps(zmm25, zmm5);
            }
            if (u_m > 2) {
                if (u_nr > 0) zmm6 = _mm512_max_ps(zmm25, zmm6);
                if (u_nr > 1) zmm7 = _mm512_max_ps(zmm25, zmm7);
                if (u_nr > 2) zmm8 = _mm512_max_ps(zmm25, zmm8);
            }
            if (u_m > 3) {
                if (u_nr > 0) zmm9 = _mm512_max_ps(zmm25, zmm9);
                if (u_nr > 1) zmm10 = _mm512_max_ps(zmm25, zmm10);
                if (u_nr > 2) zmm11 = _mm512_max_ps(zmm25, zmm11);
            }
            if (u_m > 4) {
                if (u_nr > 0) zmm12 = _mm512_max_ps(zmm25, zmm12);
                if (u_nr > 1) zmm13 = _mm512_max_ps(zmm25, zmm13);
                if (u_nr > 2) zmm14 = _mm512_max_ps(zmm25, zmm14);
            }
            if (u_m > 5) {
                if (u_nr > 0) zmm15 = _mm512_max_ps(zmm25, zmm15);
                if (u_nr > 1) zmm16 = _mm512_max_ps(zmm25, zmm16);
                if (u_nr > 2) zmm17 = _mm512_max_ps(zmm25, zmm17);
            }
            if (u_m > 6) {
                if (u_nr > 0) zmm18 = _mm512_max_ps(zmm25, zmm18);
                if (u_nr > 1) zmm19 = _mm512_max_ps(zmm25, zmm19);
                if (u_nr > 2) zmm20 = _mm512_max_ps(zmm25, zmm20);
            }
            if (u_m > 7) {
                if (u_nr > 0) zmm21 = _mm512_max_ps(zmm25, zmm21);
                if (u_nr > 1) zmm22 = _mm512_max_ps(zmm25, zmm22);
                if (u_nr > 2) zmm23 = _mm512_max_ps(zmm25, zmm23);
            }
        }

        if (flags & gemm_kernel_fp32_avx512::flag::RELU6) {
            if (u_m > 0) {
                if (u_nr > 0) zmm0 = _mm512_min_ps(zmm26, zmm0);
                if (u_nr > 1) zmm1 = _mm512_min_ps(zmm26, zmm1);
                if (u_nr > 2) zmm2 = _mm512_min_ps(zmm26, zmm2);
            }
            if (u_m > 1) {
                if (u_nr > 0) zmm3 = _mm512_min_ps(zmm26, zmm3);
                if (u_nr > 1) zmm4 = _mm512_min_ps(zmm26, zmm4);
                if (u_nr > 2) zmm5 = _mm512_min_ps(zmm26, zmm5);
            }
            if (u_m > 2) {
                if (u_nr > 0) zmm6 = _mm512_min_ps(zmm26, zmm6);
                if (u_nr > 1) zmm7 = _mm512_min_ps(zmm26, zmm7);
                if (u_nr > 2) zmm8 = _mm512_min_ps(zmm26, zmm8);
            }
            if (u_m > 3) {
                if (u_nr > 0) zmm9 = _mm512_min_ps(zmm26, zmm9);
                if (u_nr > 1) zmm10 = _mm512_min_ps(zmm26, zmm10);
                if (u_nr > 2) zmm11 = _mm512_min_ps(zmm26, zmm11);
            }
            if (u_m > 4) {
                if (u_nr > 0) zmm12 = _mm512_min_ps(zmm26, zmm12);
                if (u_nr > 1) zmm13 = _mm512_min_ps(zmm26, zmm13);
                if (u_nr > 2) zmm14 = _mm512_min_ps(zmm26, zmm14);
            }
            if (u_m > 5) {
                if (u_nr > 0) zmm15 = _mm512_min_ps(zmm26, zmm15);
                if (u_nr > 1) zmm16 = _mm512_min_ps(zmm26, zmm16);
                if (u_nr > 2) zmm17 = _mm512_min_ps(zmm26, zmm17);
            }
            if (u_m > 6) {
                if (u_nr > 0) zmm18 = _mm512_min_ps(zmm26, zmm18);
                if (u_nr > 1) zmm19 = _mm512_min_ps(zmm26, zmm19);
                if (u_nr > 2) zmm20 = _mm512_min_ps(zmm26, zmm20);
            }
            if (u_m > 7) {
                if (u_nr > 0) zmm21 = _mm512_min_ps(zmm26, zmm21);
                if (u_nr > 1) zmm22 = _mm512_min_ps(zmm26, zmm22);
                if (u_nr > 2) zmm23 = _mm512_min_ps(zmm26, zmm23);
            }
        }

        m -= u_m;
        if (u_m > 0) {
            if (u_nr > 0) _mm512_mask_storeu_ps(c_m0_ptr + 0 * ldc + 0 * N_REG_ELTS, k1, zmm0);
            if (u_nr > 1) _mm512_mask_storeu_ps(c_m0_ptr + 0 * ldc + 1 * N_REG_ELTS, k2, zmm1);
            if (u_nr > 2) _mm512_mask_storeu_ps(c_m0_ptr + 0 * ldc + 2 * N_REG_ELTS, k3, zmm2);
        }
        if (u_m > 1) {
            if (u_nr > 0) _mm512_mask_storeu_ps(c_m0_ptr + 1 * ldc + 0 * N_REG_ELTS, k1, zmm3);
            if (u_nr > 1) _mm512_mask_storeu_ps(c_m0_ptr + 1 * ldc + 1 * N_REG_ELTS, k2, zmm4);
            if (u_nr > 2) _mm512_mask_storeu_ps(c_m0_ptr + 1 * ldc + 2 * N_REG_ELTS, k3, zmm5);
        }
        if (u_m > 2) {
            if (u_nr > 0) _mm512_mask_storeu_ps(c_m0_ptr + 2 * ldc + 0 * N_REG_ELTS, k1, zmm6);
            if (u_nr > 1) _mm512_mask_storeu_ps(c_m0_ptr + 2 * ldc + 1 * N_REG_ELTS, k2, zmm7);
            if (u_nr > 2) _mm512_mask_storeu_ps(c_m0_ptr + 2 * ldc + 2 * N_REG_ELTS, k3, zmm8);
        }
        if (u_m > 3) {
            if (u_nr > 0) _mm512_mask_storeu_ps(c_m0_ptr + 1 * ldc3 + 0 * N_REG_ELTS, k1, zmm9);
            if (u_nr > 1) _mm512_mask_storeu_ps(c_m0_ptr + 1 * ldc3 + 1 * N_REG_ELTS, k2, zmm10);
            if (u_nr > 2) _mm512_mask_storeu_ps(c_m0_ptr + 1 * ldc3 + 2 * N_REG_ELTS, k3, zmm11);
        }
        c_m0_ptr += ldcm;
        if (u_m > 4) {
            if (u_nr > 0) _mm512_mask_storeu_ps(c_m4_ptr + 0 * ldc + 0 * N_REG_ELTS, k1, zmm12);
            if (u_nr > 1) _mm512_mask_storeu_ps(c_m4_ptr + 0 * ldc + 1 * N_REG_ELTS, k2, zmm13);
            if (u_nr > 2) _mm512_mask_storeu_ps(c_m4_ptr + 0 * ldc + 2 * N_REG_ELTS, k3, zmm14);
        }
        if (u_m > 5) {
            if (u_nr > 0) _mm512_mask_storeu_ps(c_m4_ptr + 1 * ldc + 0 * N_REG_ELTS, k1, zmm15);
            if (u_nr > 1) _mm512_mask_storeu_ps(c_m4_ptr + 1 * ldc + 1 * N_REG_ELTS, k2, zmm16);
            if (u_nr > 2) _mm512_mask_storeu_ps(c_m4_ptr + 1 * ldc + 2 * N_REG_ELTS, k3, zmm17);
        }
        if (u_m > 6) {
            if (u_nr > 0) _mm512_mask_storeu_ps(c_m4_ptr + 2 * ldc + 0 * N_REG_ELTS, k1, zmm18);
            if (u_nr > 1) _mm512_mask_storeu_ps(c_m4_ptr + 2 * ldc + 1 * N_REG_ELTS, k2, zmm19);
            if (u_nr > 2) _mm512_mask_storeu_ps(c_m4_ptr + 2 * ldc + 2 * N_REG_ELTS, k3, zmm20);
        }
        if (u_m > 7) {
            if (u_nr > 0) _mm512_mask_storeu_ps(c_m4_ptr + 1 * ldc3 + 0 * N_REG_ELTS, k1, zmm21);
            if (u_nr > 1) _mm512_mask_storeu_ps(c_m4_ptr + 1 * ldc3 + 1 * N_REG_ELTS, k2, zmm22);
            if (u_nr > 2) _mm512_mask_storeu_ps(c_m4_ptr + 1 * ldc3 + 2 * N_REG_ELTS, k3, zmm23);
        }
        c_m4_ptr += ldcm;
        
        if (u_m > 0) {
            if (u_nr > 0) zmm0 = _mm512_setzero_ps();
            if (u_nr > 1) zmm1 = _mm512_setzero_ps();
            if (u_nr > 2) zmm2 = _mm512_setzero_ps();
        }
        if (u_m > 1) {
            if (u_nr > 0) zmm3 = _mm512_setzero_ps();
            if (u_nr > 1) zmm4 = _mm512_setzero_ps();
            if (u_nr > 2) zmm5 = _mm512_setzero_ps();
        }
        if (u_m > 2) {
            if (u_nr > 0) zmm6 = _mm512_setzero_ps();
            if (u_nr > 1) zmm7 = _mm512_setzero_ps();
            if (u_nr > 2) zmm8 = _mm512_setzero_ps();
        }
        if (u_m > 3) {
            if (u_nr > 0) zmm9 = _mm512_setzero_ps();
            if (u_nr > 1) zmm10 = _mm512_setzero_ps();
            if (u_nr > 2) zmm11 = _mm512_setzero_ps();
        }
        if (u_m > 4) {
            if (u_nr > 0) zmm12 = _mm512_setzero_ps();
            if (u_nr > 1) zmm13 = _mm512_setzero_ps();
            if (u_nr > 2) zmm14 = _mm512_setzero_ps();
        }
        if (u_m > 5) {
            if (u_nr > 0) zmm15 = _mm512_setzero_ps();
            if (u_nr > 1) zmm16 = _mm512_setzero_ps();
            if (u_nr > 2) zmm17 = _mm512_setzero_ps();
        }
        if (u_m > 6) {
            if (u_nr > 0) zmm18 = _mm512_setzero_ps();
            if (u_nr > 1) zmm19 = _mm512_setzero_ps();
            if (u_nr > 2) zmm20 = _mm512_setzero_ps();
        }
        if (u_m > 7) {
            if (u_nr > 0) zmm21 = _mm512_setzero_ps();
            if (u_nr > 1) zmm22 = _mm512_setzero_ps();
            if (u_nr > 2) zmm23 = _mm512_setzero_ps();
        }
    } while (m > 0);
}

#define GEMM_KERNEL_FP32_AVX512_TABLE_BLK(NEED_MASK) \
{\
    {\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 1, 1 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 2, 1 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 3, 1 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 4, 1 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 5, 1 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 6, 1 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 7, 1 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 8, 1 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
    },\
    {\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 1, 2 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 2, 2 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 3, 2 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 4, 2 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 5, 2 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 6, 2 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 7, 2 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 8, 2 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
    },\
    {\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 1, 3 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 2, 3 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 3, 3 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 4, 3 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 5, 3 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 6, 3 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 7, 3 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
        gemm_m8n48_kernel_fp32_avx512<NEED_MASK, 8, 3 * gemm_kernel_fp32_avx512::config::N_REG_ELTS>,\
    },\
}\

const gemm_kernel_fp32_avx512::func_t
    gemm_kernel_fp32_avx512::table_[config::NEED_MASK_OPT][config::MAX_N_REGS][config::MAX_M_REGS] =
{
    GEMM_KERNEL_FP32_AVX512_TABLE_BLK(0),
    GEMM_KERNEL_FP32_AVX512_TABLE_BLK(1),
};

}}}; // namespace ppl::kernel::x86
