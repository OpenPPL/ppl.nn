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

#include "ppl/kernel/x86/fp32/gemm/gemm_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template<int64_t need_mask, int64_t u_m, int64_t u_n>
void gemm_m4n24_kernel_fp32_fma_core(int64_t *param) {
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
        ".equ NEXT_B_PTR_IDX,   (15 * P_BYTES)\n"
        ".equ MASK_IDX,         (16 * P_BYTES)\n"

        ".equ N_REG_ELTS, %c[N_REG_ELTS]\n"
        ".equ MAX_N_REGS, %c[MAX_N_REGS]\n"
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

        ".equ PREFETCH_B_OFFSET, 768\n"
        ".equ PREFETCH_A_OFFSET, 384\n"
        ".equ DO_PREFETCH_NEXT_B, (U_NR == MAX_N_REGS && !NEED_MASK)\n"

        "mov K_IDX(%[param]), %%rax\n"              // k
        "mov PRF_C_LDK_IDX(%[param]), %%r10\n"      // lead_k
        "sar $U_K_LOG2, %%r10\n"
        "mov B_PTR_IDX(%[param]), %%rbx\n"          // b_ptr
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".if DO_PREFETCH_NEXT_B\n"
        "mov NEXT_B_PTR_IDX(%[param]), %%r8\n"      // next_b_ptr for prefetching <= do not have register double buffer
        ".endif\n"

        "mov FLAGS_IDX(%[param]),        %%rsi\n"
        "mov A_PTR_IDX(%[param]),        %%r15\n"
        "mov C_PTR_IDX(%[param]),        %%r14\n"
        "mov M_IDX(%[param]),            %%r13\n"
        "mov LDC_IDX(%[param]),          %%r11\n"
        "shl $LOG2_D_BYTES, %%r11\n"
        ".if U_M > 2\n"
        "lea (%%r14, %%r11, 2), %%r12\n"                        // c_m2
        ".endif\n" 
        "imul $U_M, %%r11, %%r9\n"                              // u_m * ldc

        ".if DO_PREFETCH_NEXT_B\n prefetcht2 (0 * CACHELINE_BYTES * D_BYTES)(%%r8)\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vxorps %%ymm0, %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm1, %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm2, %%ymm2, %%ymm2\n .endif\n"
        // prefetch C first 16 col anyway
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES)(%%r14)\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vxorps %%ymm3, %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm4, %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm5, %%ymm5, %%ymm5\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES)(%%r14, %%r11)\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vxorps %%ymm6, %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm7, %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm8, %%ymm8, %%ymm8\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES)(%%r12)\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vxorps %%ymm9, %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm10, %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm11, %%ymm11, %%ymm11\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES)(%%r12, %%r11)\n .endif\n"
        ".endif\n"




"1:\n" // label_init_session
        "mov %%rax, %%rdx\n" // k
        "sar $U_K_LOG2, %%rdx\n" // purge the k tail, k -> uk
        "jle 5f\n"  // label_k_tail
        "sub %%r10, %%rdx\n"
        "jle 30f\n" // label_prf_c
        PPL_X86_INLINE_ASM_ALIGN()
"4:\n" // label_loop_uk_body
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n .endif\n"
        ".if U_NR > 0 && U_M == 1\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 1 && U_M == 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n .endif\n"
        ".if U_NR > 2 && U_M == 1\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES + 0 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n .endif\n"
        ".if U_NR > 0 && U_M == 2\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 1 && U_M == 2\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n .endif\n"
        ".if U_NR > 2 && U_M == 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 0\n prefetcht0 (0 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n .endif\n"
        ".if U_NR > 0 && U_M == 3\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 1 && U_M == 3\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n .endif\n"
        ".if U_NR > 2 && U_M == 3\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n .endif\n"
        ".if U_NR > 0 && U_M == 4\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 1 && U_M == 4\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n .endif\n"
        ".if U_NR > 2 && U_M == 4\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * CACHELINE_BYTES + 0 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n .endif\n"
        ".if U_NR > 0 && U_M == 1\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 1 && U_M == 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n .endif\n"
        ".if U_NR > 2 && U_M == 1\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n .endif\n"
        ".if U_NR > 0 && U_M == 2\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 1 && U_M == 2\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n .endif\n"
        ".if U_NR > 2 && U_M == 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * CACHELINE_BYTES + 0 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n .endif\n"
        ".if U_NR > 0 && U_M == 3\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 1 && U_M == 3\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n .endif\n"
        ".if U_NR > 2 && U_M == 3\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n .endif\n"
        ".if U_NR > 0 && U_M == 4\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 1 && U_M == 4\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n .endif\n"
        ".if U_NR > 2 && U_M == 4\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n .endif\n"
        ".if U_NR > 0 && U_M == 1\n vmovups (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 1 && U_M == 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n .endif\n"
        ".if U_NR > 2 && U_M == 1\n vmovups (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES + 1 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n .endif\n"
        ".if U_NR > 0 && U_M == 2\n vmovups (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 1 && U_M == 2\n vmovups (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n .endif\n"
        ".if U_NR > 2 && U_M == 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n .endif\n"
        ".if U_NR > 0 && U_M == 3\n vmovups (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 1 && U_M == 3\n vmovups (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n .endif\n"
        ".if U_NR > 2 && U_M == 3\n vmovups (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n .endif\n"
        ".if U_NR > 0 && U_M == 4\n vmovups (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 1 && U_M == 4\n vmovups (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n .endif\n"
        ".if U_NR > 2 && U_M == 4\n vmovups (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * CACHELINE_BYTES + 1 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        "sub $1, %%rdx\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n .endif\n"
        ".if U_NR > 0 && U_M == 1\n vmovups (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 1 && U_M == 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n .endif\n"
        ".if U_NR > 2 && U_M == 1\n vmovups (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n .endif\n"
        ".if U_NR > 0 && U_M == 2\n vmovups (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 1 && U_M == 2\n vmovups (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n .endif\n"
        ".if U_NR > 2 && U_M == 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * CACHELINE_BYTES + 1 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n .endif\n"
        ".if U_NR > 0 && U_M == 3\n vmovups (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 1 && U_M == 3\n vmovups (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n .endif\n"
        ".if U_NR > 2 && U_M == 3\n vmovups (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        "lea (U_K * U_M * D_BYTES)(%%r15), %%r15\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n .endif\n"
        ".if U_NR > 0 && U_M == 4\n vmovups (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 1 && U_M == 4\n vmovups (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n .endif\n"
        ".if U_NR > 2 && U_M == 4\n vmovups (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        "lea (U_K * U_N * D_BYTES)(%%rbx), %%rbx\n"
        "jg 4b\n" // label_loop_uk_body



"30:\n" // label_prf_c
        ".if DO_PREFETCH_NEXT_B\n prefetcht2 (1 * CACHELINE_BYTES * D_BYTES)(%%r8)\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES)(%%r14)\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (1 * CACHELINE_BYTES)(%%r14)\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES)(%%r14, %%r11)\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (1 * CACHELINE_BYTES)(%%r14, %%r11)\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES)(%%r12)\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (1 * CACHELINE_BYTES)(%%r12)\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES)(%%r12, %%r11)\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (1 * CACHELINE_BYTES)(%%r12, %%r11)\n .endif\n"
        ".endif\n"
        "add %%r10, %%rdx\n"
        PPL_X86_INLINE_ASM_ALIGN()
"40:\n" // label_loop_uk_after_prf_c
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n .endif\n"
        ".if U_NR > 0 && U_M == 1\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 1 && U_M == 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n .endif\n"
        ".if U_NR > 2 && U_M == 1\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES + 0 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n .endif\n"
        ".if U_NR > 0 && U_M == 2\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 1 && U_M == 2\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n .endif\n"
        ".if U_NR > 2 && U_M == 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 0\n prefetcht0 (0 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n .endif\n"
        ".if U_NR > 0 && U_M == 3\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 1 && U_M == 3\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n .endif\n"
        ".if U_NR > 2 && U_M == 3\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n .endif\n"
        ".if U_NR > 0 && U_M == 4\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 1 && U_M == 4\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n .endif\n"
        ".if U_NR > 2 && U_M == 4\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * CACHELINE_BYTES + 0 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n .endif\n"
        ".if U_NR > 0 && U_M == 1\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 1 && U_M == 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n .endif\n"
        ".if U_NR > 2 && U_M == 1\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n .endif\n"
        ".if U_NR > 0 && U_M == 2\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 1 && U_M == 2\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n .endif\n"
        ".if U_NR > 2 && U_M == 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * CACHELINE_BYTES + 0 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n .endif\n"
        ".if U_NR > 0 && U_M == 3\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 1 && U_M == 3\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n .endif\n"
        ".if U_NR > 2 && U_M == 3\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n .endif\n"
        ".if U_NR > 0 && U_M == 4\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 1 && U_M == 4\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n .endif\n"
        ".if U_NR > 2 && U_M == 4\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n .endif\n"
        ".if U_NR > 0 && U_M == 1\n vmovups (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 1 && U_M == 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n .endif\n"
        ".if U_NR > 2 && U_M == 1\n vmovups (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES + 1 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n .endif\n"
        ".if U_NR > 0 && U_M == 2\n vmovups (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 1 && U_M == 2\n vmovups (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n .endif\n"
        ".if U_NR > 2 && U_M == 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n .endif\n"
        ".if U_NR > 0 && U_M == 3\n vmovups (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 1 && U_M == 3\n vmovups (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n .endif\n"
        ".if U_NR > 2 && U_M == 3\n vmovups (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n .endif\n"
        ".if U_NR > 0 && U_M == 4\n vmovups (0 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 1 && U_M == 4\n vmovups (1 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n .endif\n"
        ".if U_NR > 2 && U_M == 4\n vmovups (2 * N_REG_ELTS * D_BYTES + 3 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * CACHELINE_BYTES + 1 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        "sub $1, %%rdx\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n .endif\n"
        ".if U_NR > 0 && U_M == 1\n vmovups (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 1 && U_M == 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n .endif\n"
        ".if U_NR > 2 && U_M == 1\n vmovups (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n .endif\n"
        ".if U_NR > 0 && U_M == 2\n vmovups (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 1 && U_M == 2\n vmovups (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n .endif\n"
        ".if U_NR > 2 && U_M == 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * CACHELINE_BYTES + 1 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n .endif\n"
        ".if U_NR > 0 && U_M == 3\n vmovups (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 1 && U_M == 3\n vmovups (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n .endif\n"
        ".if U_NR > 2 && U_M == 3\n vmovups (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 3 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        "lea (U_K * U_M * D_BYTES)(%%r15), %%r15\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n .endif\n"
        ".if U_NR > 0 && U_M == 4\n vmovups (0 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 1 && U_M == 4\n vmovups (1 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n .endif\n"
        ".if U_NR > 2 && U_M == 4\n vmovups (2 * N_REG_ELTS * D_BYTES + 4 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        "lea (U_K * U_N * D_BYTES)(%%rbx), %%rbx\n"
        "jg 40b\n" // label_loop_uk_after_prf_c




"5:\n" // label_k_tail
        "mov %%rax, %%rdx\n"
        "and $(U_K - 1), %%rdx\n"
        "je 6f\n" // label_end_k
        "imul $(U_M * D_BYTES), %%rdx, %%rcx\n" // u_m * k_tail
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        "cmp $1, %%rdx\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n .endif\n"
        ".if U_NR > 0 && U_M == 1\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 1 && U_M == 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n .endif\n"
        ".if U_NR > 2 && U_M == 1\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES + 0 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n .endif\n"
        ".if U_NR > 0 && U_M == 2\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 1 && U_M == 2\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n .endif\n"
        ".if U_NR > 2 && U_M == 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 0\n prefetcht0 (0 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n .endif\n"
        ".if U_NR > 0 && U_M == 3\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 1 && U_M == 3\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n .endif\n"
        ".if U_NR > 2 && U_M == 3\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 0 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n .endif\n"
        ".if U_NR > 0 && U_M == 4\n vmovups (0 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 1 && U_M == 4\n vmovups (1 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n .endif\n"
        ".if U_NR > 2 && U_M == 4\n vmovups (2 * N_REG_ELTS * D_BYTES + 1 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        "je 50f\n" // label_end_k_tail
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * CACHELINE_BYTES + 0 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n .endif\n"
        ".if U_NR > 0 && U_M == 1\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 1 && U_M == 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n .endif\n"
        ".if U_NR > 2 && U_M == 1\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        "cmp $2, %%rdx\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n .endif\n"
        ".if U_NR > 0 && U_M == 2\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 1 && U_M == 2\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n .endif\n"
        ".if U_NR > 2 && U_M == 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 2\n prefetcht0 (2 * CACHELINE_BYTES + 0 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n .endif\n"
        ".if U_NR > 0 && U_M == 3\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 1 && U_M == 3\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n .endif\n"
        ".if U_NR > 2 && U_M == 3\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 1 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n .endif\n"
        ".if U_NR > 0 && U_M == 4\n vmovups (0 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 1 && U_M == 4\n vmovups (1 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n .endif\n"
        ".if U_NR > 2 && U_M == 4\n vmovups (2 * N_REG_ELTS * D_BYTES + 2 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        ".endif\n"
        "je 50f\n" // label_end_k_tail
        ".if U_M > 0\n vbroadcastss (0 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n vbroadcastss (1 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_NR > 0\n prefetcht0 (0 * CACHELINE_BYTES + 1 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n vbroadcastss (2 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n vbroadcastss (3 * D_BYTES + 2 * U_M * D_BYTES)(%%r15), %%ymm15\n .endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm15, %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm15, %%ymm14, %%ymm11\n .endif\n"
        ".endif\n"
        ".if U_NR > 1\n prefetcht0 (1 * CACHELINE_BYTES + 1 * U_NR * CACHELINE_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
"50:\n" // label_end_k_tail
        "lea (%%r15, %%rcx), %%r15\n"
"6:\n" // label_end_k



        ".if DO_PREFETCH_NEXT_B\n prefetcht2 (2 * CACHELINE_BYTES * D_BYTES)(%%r8)\n .endif\n"
        "vbroadcastss ALPHA_IDX(%[param]), %%ymm12\n" // alpha
        "vbroadcastss BETA_IDX(%[param]), %%ymm13\n"  // beta
        ".if NEED_MASK\n vmovups MASK_IDX(%[param]), %%ymm15\n .endif\n"
        "mov B_PTR_IDX(%[param]), %%rbx\n"            // b_ptr

        // *= alpha
        ".if U_M > 0\n"
        ".if U_NR > 0\n vmulps %%ymm12, %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vmulps %%ymm12, %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vmulps %%ymm12, %%ymm2, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vmulps %%ymm12, %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vmulps %%ymm12, %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vmulps %%ymm12, %%ymm5, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vmulps %%ymm12, %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vmulps %%ymm12, %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vmulps %%ymm12, %%ymm8, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vmulps %%ymm12, %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vmulps %%ymm12, %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vmulps %%ymm12, %%ymm11, %%ymm11\n .endif\n"
        ".endif\n"

        // put sum in the first place, overlaping cache miss
        // += beta_sum * sum
        "test $KERNEL_FLAG_WITH_SUM, %%rsi\n"
        "jz 14f\n" // label_load_sum_end
        "vbroadcastss BETA_SUM_IDX(%[param]), %%ymm14\n"
        "mov SUM_PTR_IDX(%[param]), %%rcx\n"
        "mov LDSUM_IDX(%[param]), %%rdx\n"
        "shl $LOG2_D_BYTES, %%rdx\n" // ldsum
        ".if NEED_MASK\n"
        ".if U_M > 0\n"
        ".if U_NR == 1\n vmaskmovps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n"
        ".elseif U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm0\n .endif\n"
        ".if U_NR == 2\n vmaskmovps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"
        ".elseif U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm1\n .endif\n"
        ".if U_NR == 3\n vmaskmovps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n"
        ".elseif U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm2\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n" // next row
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR == 1\n vmaskmovps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm3\n"
        ".elseif U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm3\n .endif\n"
        ".if U_NR == 2\n vmaskmovps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
        ".elseif U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm4\n .endif\n"
        ".if U_NR == 3\n vmaskmovps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm5\n"
        ".elseif U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm5\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR == 1\n vmaskmovps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n"
        ".elseif U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm6\n .endif\n"
        ".if U_NR == 2\n vmaskmovps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
        ".elseif U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm7\n .endif\n"
        ".if U_NR == 3\n vmaskmovps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n"
        ".elseif U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm8\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR == 1\n vmaskmovps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm9\n"
        ".elseif U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm9\n .endif\n"
        ".if U_NR == 2\n vmaskmovps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n"
        ".elseif U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm10\n .endif\n"
        ".if U_NR == 3\n vmaskmovps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm14, %%ymm11\n"
        ".elseif U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm11\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".else\n" // need_mask
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm2\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n" // next row
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm5\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm8\n .endif\n"
        "lea (%%rcx, %%rdx), %%rcx\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm14, %%ymm11\n .endif\n"
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
        ".if U_NR == 1\n vmaskmovps (0 * N_REG_ELTS * D_BYTES)(%%r14), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
        ".elseif U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14), %%ymm13, %%ymm0\n .endif\n"
        ".if U_NR == 2\n vmaskmovps (1 * N_REG_ELTS * D_BYTES)(%%r14), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
        ".elseif U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14), %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR == 3\n vmaskmovps (2 * N_REG_ELTS * D_BYTES)(%%r14), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm2\n"
        ".elseif U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14), %%ymm13, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR == 1\n vmaskmovps (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
        ".elseif U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%ymm13, %%ymm3\n .endif\n"
        ".if U_NR == 2\n vmaskmovps (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm4\n"
        ".elseif U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR == 3\n vmaskmovps (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm5\n"
        ".elseif U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%ymm13, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR == 1\n vmaskmovps (0 * N_REG_ELTS * D_BYTES)(%%r12), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm6\n"
        ".elseif U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12), %%ymm13, %%ymm6\n .endif\n"
        ".if U_NR == 2\n vmaskmovps (1 * N_REG_ELTS * D_BYTES)(%%r12), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm7\n"
        ".elseif U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12), %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR == 3\n vmaskmovps (2 * N_REG_ELTS * D_BYTES)(%%r12), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm8\n"
        ".elseif U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12), %%ymm13, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR == 1\n vmaskmovps (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm9\n"
        ".elseif U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%ymm13, %%ymm9\n .endif\n"
        ".if U_NR == 2\n vmaskmovps (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm10\n"
        ".elseif U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR == 3\n vmaskmovps (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%ymm15, %%ymm12\n vfmadd231ps %%ymm12, %%ymm13, %%ymm11\n"
        ".elseif U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%ymm13, %%ymm11\n .endif\n"
        ".endif\n"
        ".else\n" // need_mask
        ".if U_M > 0\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14), %%ymm13, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14), %%ymm13, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14), %%ymm13, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%ymm13, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%ymm13, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11), %%ymm13, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12), %%ymm13, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12), %%ymm13, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12), %%ymm13, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vfmadd231ps (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%ymm13, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%ymm13, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11), %%ymm13, %%ymm11\n .endif\n"
        ".endif\n"
        ".endif\n" // need_mask
"8:\n" // label_load_c_end

        "vbroadcastss BETA_BIAS_IDX(%[param]), %%ymm12\n"
        "mov BIAS_PTR_IDX(%[param]), %%rcx\n"
        // += beta_bias * bias
        "test $KERNEL_FLAG_ROW_BIAS, %%rsi\n"
        "jz 11f\n" // label_row_bias_end
        ".if U_NR > 0\n"
        ".if U_NR == 1 && NEED_MASK\n vmaskmovps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm13\n vmulps %%ymm13, %%ymm12, %%ymm14\n"
        ".else\n vmulps (0 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm12, %%ymm14\n .endif\n"
        ".if U_M > 0\n vaddps %%ymm14, %%ymm0, %%ymm0\n .endif\n"
        ".if U_M > 1\n vaddps %%ymm14, %%ymm3, %%ymm3\n .endif\n"
        ".if U_M > 2\n vaddps %%ymm14, %%ymm6, %%ymm6\n .endif\n"
        ".if U_M > 3\n vaddps %%ymm14, %%ymm9, %%ymm9\n .endif\n"
        ".endif\n"
        ".if U_NR > 1\n"
        ".if U_NR == 2 && NEED_MASK\n vmaskmovps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm13\n vmulps %%ymm13, %%ymm12, %%ymm14\n"
        ".else\n vmulps (1 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm12, %%ymm14\n .endif\n"
        ".if U_M > 0\n vaddps %%ymm14, %%ymm1, %%ymm1\n .endif\n"
        ".if U_M > 1\n vaddps %%ymm14, %%ymm4, %%ymm4\n .endif\n"
        ".if U_M > 2\n vaddps %%ymm14, %%ymm7, %%ymm7\n .endif\n"
        ".if U_M > 3\n vaddps %%ymm14, %%ymm10, %%ymm10\n .endif\n"
        ".endif\n"
        ".if U_NR > 2\n"
        ".if U_NR == 3 && NEED_MASK\n vmaskmovps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm15, %%ymm13\n vmulps %%ymm13, %%ymm12, %%ymm14\n"
        ".else\n vmulps (2 * N_REG_ELTS * D_BYTES)(%%rcx), %%ymm12, %%ymm14\n .endif\n"
        ".if U_M > 0\n vaddps %%ymm14, %%ymm2, %%ymm2\n .endif\n"
        ".if U_M > 1\n vaddps %%ymm14, %%ymm5, %%ymm5\n .endif\n"
        ".if U_M > 2\n vaddps %%ymm14, %%ymm8, %%ymm8\n .endif\n"
        ".if U_M > 3\n vaddps %%ymm14, %%ymm11, %%ymm11\n .endif\n"
        ".endif\n"
"11:\n" // label_row_bias_end
        "test $KERNEL_FLAG_SCA_BIAS, %%rsi\n"
        "jz 12f\n" // label_sca_bias_end
        "vbroadcastss (%%rcx), %%ymm13\n"
        "vmulps %%ymm13, %%ymm12, %%ymm14\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vaddps %%ymm14, %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vaddps %%ymm14, %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vaddps %%ymm14, %%ymm2, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vaddps %%ymm14, %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vaddps %%ymm14, %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vaddps %%ymm14, %%ymm5, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vaddps %%ymm14, %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vaddps %%ymm14, %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vaddps %%ymm14, %%ymm8, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vaddps %%ymm14, %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vaddps %%ymm14, %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vaddps %%ymm14, %%ymm11, %%ymm11\n .endif\n"
        ".endif\n"
"12:\n" // label_sca_bias_end
        "test $KERNEL_FLAG_COL_BIAS, %%rsi\n"
        "jz 13f\n" // label_col_bias_end
        ".if U_M > 0\n"
        "vbroadcastss (0 * D_BYTES)(%%rcx), %%ymm13\n"
        "vmulps %%ymm13, %%ymm12, %%ymm14\n"
        ".if U_NR > 0\n vaddps %%ymm14, %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vaddps %%ymm14, %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vaddps %%ymm14, %%ymm2, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        "vbroadcastss (1 * D_BYTES)(%%rcx), %%ymm13\n"
        "vmulps %%ymm13, %%ymm12, %%ymm14\n"
        ".if U_NR > 0\n vaddps %%ymm14, %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vaddps %%ymm14, %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vaddps %%ymm14, %%ymm5, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        "vbroadcastss (2 * D_BYTES)(%%rcx), %%ymm13\n"
        "vmulps %%ymm13, %%ymm12, %%ymm14\n"
        ".if U_NR > 0\n vaddps %%ymm14, %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vaddps %%ymm14, %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vaddps %%ymm14, %%ymm8, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        "vbroadcastss (3 * D_BYTES)(%%rcx), %%ymm13\n"
        "vmulps %%ymm13, %%ymm12, %%ymm14\n"
        ".if U_NR > 0\n vaddps %%ymm14, %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vaddps %%ymm14, %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vaddps %%ymm14, %%ymm11, %%ymm11\n .endif\n"
        ".endif\n"
        "lea (U_M * D_BYTES)(%%rcx), %%rcx\n"
        "mov %%rcx, BIAS_PTR_IDX(%[param])\n"
"13:\n" // label_col_bias_end

        "test $(KERNEL_FLAG_RELU | KERNEL_FLAG_RELU6), %%rsi\n"
        "jz 9f\n" // label_relu_end
        "vxorps %%ymm13, %%ymm13, %%ymm13\n" // 0.0
        "mov $0x40c00000, %%ecx\n"
        "vmovd %%ecx, %%xmm14\n"
        "vbroadcastss %%xmm14, %%ymm14\n" // 6.0
        "test $KERNEL_FLAG_RELU6, %%rsi\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vmaxps %%ymm13, %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vmaxps %%ymm13, %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vmaxps %%ymm13, %%ymm2, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vmaxps %%ymm13, %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vmaxps %%ymm13, %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vmaxps %%ymm13, %%ymm5, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vmaxps %%ymm13, %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vmaxps %%ymm13, %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vmaxps %%ymm13, %%ymm8, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vmaxps %%ymm13, %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vmaxps %%ymm13, %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vmaxps %%ymm13, %%ymm11, %%ymm11\n .endif\n"
        ".endif\n"

        "jz 9f\n" // label_relu_end
        ".if U_M > 0\n"
        ".if U_NR > 0\n vminps %%ymm14, %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vminps %%ymm14, %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vminps %%ymm14, %%ymm2, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vminps %%ymm14, %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vminps %%ymm14, %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vminps %%ymm14, %%ymm5, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vminps %%ymm14, %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vminps %%ymm14, %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vminps %%ymm14, %%ymm8, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vminps %%ymm14, %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vminps %%ymm14, %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vminps %%ymm14, %%ymm11, %%ymm11\n .endif\n"
        ".endif\n"
"9:\n" // label_relu_end

        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vmovups (1 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vmovups (2 * N_REG_ELTS * D_BYTES + 0 * U_N * D_BYTES)(%%rbx), %%ymm14\n .endif\n"
        "sub $U_M, %%r13\n" // m -= u_m
        ".if DO_PREFETCH_NEXT_B\n lea (3 * CACHELINE_BYTES * D_BYTES)(%%r8), %%r8\n .endif\n"

        ".if NEED_MASK\n"
        ".if U_M > 0\n"
        ".if U_NR == 1\n vmaskmovps %%ymm0, %%ymm15, (0 * N_REG_ELTS * D_BYTES)(%%r14)\n"
        ".elseif U_NR > 0\n vmovups %%ymm0, (0 * N_REG_ELTS * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NR == 2\n vmaskmovps %%ymm1, %%ymm15, (1 * N_REG_ELTS * D_BYTES)(%%r14)\n"
        ".elseif U_NR > 1\n vmovups %%ymm1, (1 * N_REG_ELTS * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NR == 3\n vmaskmovps %%ymm2, %%ymm15, (2 * N_REG_ELTS * D_BYTES)(%%r14)\n"
        ".elseif U_NR > 2\n vmovups %%ymm2, (2 * N_REG_ELTS * D_BYTES)(%%r14)\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR == 1\n vmaskmovps %%ymm3, %%ymm15, (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n"
        ".elseif U_NR > 0\n vmovups %%ymm3, (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n .endif\n"
        ".if U_NR == 2\n vmaskmovps %%ymm4, %%ymm15, (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n"
        ".elseif U_NR > 1\n vmovups %%ymm4, (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n .endif\n"
        ".if U_NR == 3\n vmaskmovps %%ymm5, %%ymm15, (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n"
        ".elseif U_NR > 2\n vmovups %%ymm5, (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n .endif\n"
        ".endif\n"
        "lea (%%r14, %%r9), %%r14\n" // c_m0 += u_m * ldc
        ".if U_M > 2\n"
        ".if U_NR == 1\n vmaskmovps %%ymm6, %%ymm15, (0 * N_REG_ELTS * D_BYTES)(%%r12)\n"
        ".elseif U_NR > 0\n vmovups %%ymm6, (0 * N_REG_ELTS * D_BYTES)(%%r12)\n .endif\n"
        ".if U_NR == 2\n vmaskmovps %%ymm7, %%ymm15, (1 * N_REG_ELTS * D_BYTES)(%%r12)\n"
        ".elseif U_NR > 1\n vmovups %%ymm7, (1 * N_REG_ELTS * D_BYTES)(%%r12)\n .endif\n"
        ".if U_NR == 3\n vmaskmovps %%ymm8, %%ymm15, (2 * N_REG_ELTS * D_BYTES)(%%r12)\n"
        ".elseif U_NR > 2\n vmovups %%ymm8, (2 * N_REG_ELTS * D_BYTES)(%%r12)\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR == 1\n vmaskmovps %%ymm9, %%ymm15, (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n"
        ".elseif U_NR > 0\n vmovups %%ymm9, (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n .endif\n"
        ".if U_NR == 2\n vmaskmovps %%ymm10, %%ymm15, (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n"
        ".elseif U_NR > 1\n vmovups %%ymm10, (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n .endif\n"
        ".if U_NR == 3\n vmaskmovps %%ymm11, %%ymm15, (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n"
        ".elseif U_NR > 2\n vmovups %%ymm11, (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n .endif\n"
        ".endif\n"
        "lea (%%r12, %%r9), %%r12\n" // c_m2 += u_m * ldc
        ".else\n" // need_mask
        ".if U_M > 0\n"
        ".if U_NR > 0\n vmovups %%ymm0, (0 * N_REG_ELTS * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NR > 1\n vmovups %%ymm1, (1 * N_REG_ELTS * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NR > 2\n vmovups %%ymm2, (2 * N_REG_ELTS * D_BYTES)(%%r14)\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vmovups %%ymm3, (0 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n .endif\n"
        ".if U_NR > 1\n vmovups %%ymm4, (1 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n .endif\n"
        ".if U_NR > 2\n vmovups %%ymm5, (2 * N_REG_ELTS * D_BYTES)(%%r14, %%r11)\n .endif\n"
        ".endif\n"
        "lea (%%r14, %%r9), %%r14\n" // c_m0 += u_m * ldc
        ".if U_M > 2\n"
        ".if U_NR > 0\n vmovups %%ymm6, (0 * N_REG_ELTS * D_BYTES)(%%r12)\n .endif\n"
        ".if U_NR > 1\n vmovups %%ymm7, (1 * N_REG_ELTS * D_BYTES)(%%r12)\n .endif\n"
        ".if U_NR > 2\n vmovups %%ymm8, (2 * N_REG_ELTS * D_BYTES)(%%r12)\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vmovups %%ymm9, (0 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n .endif\n"
        ".if U_NR > 1\n vmovups %%ymm10, (1 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n .endif\n"
        ".if U_NR > 2\n vmovups %%ymm11, (2 * N_REG_ELTS * D_BYTES)(%%r12, %%r11)\n .endif\n"
        ".endif\n"
        "lea (%%r12, %%r9), %%r12\n" // c_m2 += u_m * ldc
        ".endif\n" // need_mask

        ".if DO_PREFETCH_NEXT_B\n prefetcht2 (0 * CACHELINE_BYTES * D_BYTES)(%%r8)\n .endif\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vxorps %%ymm0, %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm1, %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm2, %%ymm2, %%ymm2\n .endif\n"
        "prefetcht0 (0 * CACHELINE_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vxorps %%ymm3, %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm4, %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm5, %%ymm5, %%ymm5\n .endif\n"
        "prefetcht0 (0 * CACHELINE_BYTES)(%%r14, %%r11)\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vxorps %%ymm6, %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm7, %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm8, %%ymm8, %%ymm8\n .endif\n"
        "prefetcht0 (0 * CACHELINE_BYTES)(%%r12)\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vxorps %%ymm9, %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm10, %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm11, %%ymm11, %%ymm11\n .endif\n"
        "prefetcht0 (0 * CACHELINE_BYTES)(%%r12, %%r11)\n"
        ".endif\n"

        "jg 1b\n" // label_init_session
        :
        :
        [param]                         "r" (param),
        [N_REG_ELTS]                    "i" (gemm_kernel_fp32_fma::config::N_REG_ELTS),
        [MAX_N_REGS]                    "i" (gemm_kernel_fp32_fma::config::MAX_N_REGS),
        [NEED_MASK]                     "i" (need_mask),
        [U_M]                           "i" (u_m),
        [U_N]                           "i" (u_n),
        [KERNEL_FLAG_LOAD_C]            "i" (gemm_kernel_fp32_fma::flag::LOAD_C),
        [KERNEL_FLAG_WITH_SUM]          "i" (gemm_kernel_fp32_fma::flag::WITH_SUM),
        [KERNEL_FLAG_ROW_BIAS]          "i" (gemm_kernel_fp32_fma::flag::ROW_BIAS),
        [KERNEL_FLAG_COL_BIAS]          "i" (gemm_kernel_fp32_fma::flag::COL_BIAS),
        [KERNEL_FLAG_SCA_BIAS]          "i" (gemm_kernel_fp32_fma::flag::SCA_BIAS),
        [KERNEL_FLAG_RELU]              "i" (gemm_kernel_fp32_fma::flag::RELU),
        [KERNEL_FLAG_RELU6]             "i" (gemm_kernel_fp32_fma::flag::RELU6)
        :
        "cc",
        "rax", "rbx", "rcx", "rdx",
        "r8" , "r9" , "r10", "r11",
        "r12", "r13", "r14", "r15",
        "rsi",
        "ymm0" , "ymm1" , "ymm2" , "ymm3" , "ymm4" , "ymm5" , "ymm6" , "ymm7" ,
        "ymm8" , "ymm9" , "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
        "memory"
    );
}

#endif

template<int64_t need_mask, int64_t u_m, int64_t u_n>
void gemm_m4n24_kernel_fp32_fma(int64_t *param)
{
#ifdef PPL_USE_X86_INLINE_ASM

    gemm_m4n24_kernel_fp32_fma_core<need_mask, u_m, u_n>(param);
    return;

#endif

    // reference intrinsic for windows, performance is not tested
    array_param_helper kp(param);
    const int64_t N_REG_ELTS = gemm_kernel_fp32_fma::config::N_REG_ELTS;
    const int64_t MAX_N_REGS = gemm_kernel_fp32_fma::config::MAX_N_REGS;
    const int64_t u_nr = div_up(u_n, N_REG_ELTS);
    const int64_t u_k = 4;
    const int64_t u_k_log2 = 2;

    const int64_t prefetch_b_offset = 768 / sizeof(float);
    const int64_t prefetch_a_offset = 384 / sizeof(float);
    const int64_t cacheline_elts = PPL_X86_CACHELINE_BYTES() / sizeof(float);
    const bool do_prefetch_next_b = u_nr == MAX_N_REGS && !need_mask;

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    // load constant values
    auto k = kp.pick<int64_t>(gemm_kernel_fp32_fma::param_def::K_IDX);
    auto prf_c_lduk = kp.pick<int64_t>(gemm_kernel_fp32_fma::param_def::PRF_C_LDK_IDX) >> u_k_log2;
    auto b_ptr = kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::B_PTR_IDX);
    if (u_nr > 0) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 0 * u_n);
    if (u_nr > 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 0 * u_n);
    if (u_nr > 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 0 * u_n);
    auto next_b_ptr = do_prefetch_next_b ? kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::NEXT_B_PTR_IDX) : nullptr;

    auto flags = kp.pick<const gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::FLAGS_IDX);
    auto a_ptr = kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX);
    auto ldc = kp.pick<const int64_t>(gemm_kernel_fp32_fma::param_def::LDC_IDX);
    auto ldcm = u_m * ldc;
    auto c_m0_ptr = kp.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX);
    auto c_m2_ptr = c_m0_ptr + 2 * ldc;
    auto m = kp.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX);

    if (do_prefetch_next_b) _mm_prefetch((const char*)(next_b_ptr + 0 * cacheline_elts), _MM_HINT_T2);
    if (u_m > 0) {
        if (u_nr > 0) ymm0 = _mm256_setzero_ps();
        if (u_nr > 1) ymm1 = _mm256_setzero_ps();
        if (u_nr > 2) ymm2 = _mm256_setzero_ps();
        // prefetch C first 16 col anyway
        if (u_nr > 0) _mm_prefetch((const char*)(c_m0_ptr + 0 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
    }
    if (u_m > 1) {
        if (u_nr > 0) ymm3 = _mm256_setzero_ps();
        if (u_nr > 1) ymm4 = _mm256_setzero_ps();
        if (u_nr > 2) ymm5 = _mm256_setzero_ps();
        if (u_nr > 0) _mm_prefetch((const char*)(c_m0_ptr + 1 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
    }
    if (u_m > 2) {
        if (u_nr > 0) ymm6 = _mm256_setzero_ps();
        if (u_nr > 1) ymm7 = _mm256_setzero_ps();
        if (u_nr > 2) ymm8 = _mm256_setzero_ps();
        if (u_nr > 0) _mm_prefetch((const char*)(c_m2_ptr + 0 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
    }
    if (u_m > 3) {
        if (u_nr > 0) ymm9 = _mm256_setzero_ps();
        if (u_nr > 1) ymm10 = _mm256_setzero_ps();
        if (u_nr > 2) ymm11 = _mm256_setzero_ps();
        if (u_nr > 0) _mm_prefetch((const char*)(c_m2_ptr + 1 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
    }

    do {
        auto kl = (k >> u_k_log2) - prf_c_lduk;
        if (kl > 0) {
            do {
                if (u_m > 0) ymm15 = _mm256_set1_ps(a_ptr[0 + 0 * u_m]);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm15, ymm12, ymm0);
                    if (u_nr > 0 && u_m == 1) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm13, ymm1);
                    if (u_nr > 1 && u_m == 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm14, ymm2);
                    if (u_nr > 2 && u_m == 1) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
                if (u_m > 1) ymm15 = _mm256_set1_ps(a_ptr[1 + 0 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 0 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm15, ymm12, ymm3);
                    if (u_nr > 0 && u_m == 2) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm15, ymm13, ymm4);
                    if (u_nr > 1 && u_m == 2) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm14, ymm5);
                    if (u_nr > 2 && u_m == 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
                if (u_m > 2) ymm15 = _mm256_set1_ps(a_ptr[2 + 0 * u_m]);
                if (u_m > 0) _mm_prefetch((const char*)(a_ptr + 0 * cacheline_elts + prefetch_a_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm15, ymm12, ymm6);
                    if (u_nr > 0 && u_m == 3) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);
                    if (u_nr > 1 && u_m == 3) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm14, ymm8);
                    if (u_nr > 2 && u_m == 3) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
                if (u_m > 3) ymm15 = _mm256_set1_ps(a_ptr[3 + 0 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);
                    if (u_nr > 0 && u_m == 4) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm15, ymm13, ymm10);
                    if (u_nr > 1 && u_m == 4) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm14, ymm11);
                    if (u_nr > 2 && u_m == 4) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
                if (u_m > 0) ymm15 = _mm256_set1_ps(a_ptr[0 + 1 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 0 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm15, ymm12, ymm0);
                    if (u_nr > 0 && u_m == 1) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm13, ymm1);
                    if (u_nr > 1 && u_m == 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm14, ymm2);
                    if (u_nr > 2 && u_m == 1) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
                if (u_m > 1) ymm15 = _mm256_set1_ps(a_ptr[1 + 1 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm15, ymm12, ymm3);
                    if (u_nr > 0 && u_m == 2) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm15, ymm13, ymm4);
                    if (u_nr > 1 && u_m == 2) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm14, ymm5);
                    if (u_nr > 2 && u_m == 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
                if (u_m > 2) ymm15 = _mm256_set1_ps(a_ptr[2 + 1 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * cacheline_elts + 0 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm15, ymm12, ymm6);
                    if (u_nr > 0 && u_m == 3) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);
                    if (u_nr > 1 && u_m == 3) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm14, ymm8);
                    if (u_nr > 2 && u_m == 3) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
                if (u_m > 3) ymm15 = _mm256_set1_ps(a_ptr[3 + 1 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);
                    if (u_nr > 0 && u_m == 4) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm15, ymm13, ymm10);
                    if (u_nr > 1 && u_m == 4) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm14, ymm11);
                    if (u_nr > 2 && u_m == 4) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
                if (u_m > 0) ymm15 = _mm256_set1_ps(a_ptr[0 + 2 * u_m]);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm15, ymm12, ymm0);
                    if (u_nr > 0 && u_m == 1) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm13, ymm1);
                    if (u_nr > 1 && u_m == 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm14, ymm2);
                    if (u_nr > 2 && u_m == 1) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 3 * u_n);
                }
                if (u_m > 1) ymm15 = _mm256_set1_ps(a_ptr[1 + 2 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 1 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm15, ymm12, ymm3);
                    if (u_nr > 0 && u_m == 2) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm15, ymm13, ymm4);
                    if (u_nr > 1 && u_m == 2) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm14, ymm5);
                    if (u_nr > 2 && u_m == 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 3 * u_n);
                }
                if (u_m > 2) ymm15 = _mm256_set1_ps(a_ptr[2 + 2 * u_m]);
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm15, ymm12, ymm6);
                    if (u_nr > 0 && u_m == 3) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);
                    if (u_nr > 1 && u_m == 3) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm14, ymm8);
                    if (u_nr > 2 && u_m == 3) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 3 * u_n);
                }
                if (u_m > 3) ymm15 = _mm256_set1_ps(a_ptr[3 + 2 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);
                    if (u_nr > 0 && u_m == 4) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm15, ymm13, ymm10);
                    if (u_nr > 1 && u_m == 4) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm14, ymm11);
                    if (u_nr > 2 && u_m == 4) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 3 * u_n);
                }
                kl -= 1;
                if (u_m > 0) ymm15 = _mm256_set1_ps(a_ptr[0 + 3 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 1 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm15, ymm12, ymm0);
                    if (u_nr > 0 && u_m == 1) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm13, ymm1);
                    if (u_nr > 1 && u_m == 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm14, ymm2);
                    if (u_nr > 2 && u_m == 1) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 4 * u_n);
                }
                if (u_m > 1) ymm15 = _mm256_set1_ps(a_ptr[1 + 3 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm15, ymm12, ymm3);
                    if (u_nr > 0 && u_m == 2) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm15, ymm13, ymm4);
                    if (u_nr > 1 && u_m == 2) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm14, ymm5);
                    if (u_nr > 2 && u_m == 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 4 * u_n);
                }
                if (u_m > 2) ymm15 = _mm256_set1_ps(a_ptr[2 + 3 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * cacheline_elts + 1 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm15, ymm12, ymm6);
                    if (u_nr > 0 && u_m == 3) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);
                    if (u_nr > 1 && u_m == 3) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm14, ymm8);
                    if (u_nr > 2 && u_m == 3) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 4 * u_n);
                }
                if (u_m > 3) ymm15 = _mm256_set1_ps(a_ptr[3 + 3 * u_m]);
                a_ptr += u_k * u_m;
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);
                    if (u_nr > 0 && u_m == 4) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm15, ymm13, ymm10);
                    if (u_nr > 1 && u_m == 4) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm14, ymm11);
                    if (u_nr > 2 && u_m == 4) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 4 * u_n);
                }
                b_ptr += u_k * u_n;
            } while (kl > 0);
        }
        kl += prf_c_lduk;
        if (kl > 0) {
            // prefetch c
            if (do_prefetch_next_b) _mm_prefetch((const char*)(next_b_ptr + 1 * cacheline_elts), _MM_HINT_T2);
            if (u_m > 0) {
                if (u_nr > 0) _mm_prefetch((const char*)(c_m0_ptr + 0 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
                if (u_nr > 2) _mm_prefetch((const char*)(c_m0_ptr + 0 * ldc + 1 * cacheline_elts), _MM_HINT_T0);
            }
            if (u_m > 1) {
                if (u_nr > 0) _mm_prefetch((const char*)(c_m0_ptr + 1 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
                if (u_nr > 2) _mm_prefetch((const char*)(c_m0_ptr + 1 * ldc + 1 * cacheline_elts), _MM_HINT_T0);
            }
            if (u_m > 2) {
                if (u_nr > 0) _mm_prefetch((const char*)(c_m2_ptr + 0 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
                if (u_nr > 2) _mm_prefetch((const char*)(c_m2_ptr + 0 * ldc + 1 * cacheline_elts), _MM_HINT_T0);
            }
            if (u_m > 3) {
                if (u_nr > 0) _mm_prefetch((const char*)(c_m2_ptr + 1 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
                if (u_nr > 2) _mm_prefetch((const char*)(c_m2_ptr + 1 * ldc + 1 * cacheline_elts), _MM_HINT_T0);
            }
            do {
                if (u_m > 0) ymm15 = _mm256_set1_ps(a_ptr[0 + 0 * u_m]);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm15, ymm12, ymm0);
                    if (u_nr > 0 && u_m == 1) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm13, ymm1);
                    if (u_nr > 1 && u_m == 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm14, ymm2);
                    if (u_nr > 2 && u_m == 1) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
                if (u_m > 1) ymm15 = _mm256_set1_ps(a_ptr[1 + 0 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 0 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm15, ymm12, ymm3);
                    if (u_nr > 0 && u_m == 2) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm15, ymm13, ymm4);
                    if (u_nr > 1 && u_m == 2) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm14, ymm5);
                    if (u_nr > 2 && u_m == 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
                if (u_m > 2) ymm15 = _mm256_set1_ps(a_ptr[2 + 0 * u_m]);
                if (u_m > 0) _mm_prefetch((const char*)(a_ptr + 0 * cacheline_elts + prefetch_a_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm15, ymm12, ymm6);
                    if (u_nr > 0 && u_m == 3) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);
                    if (u_nr > 1 && u_m == 3) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm14, ymm8);
                    if (u_nr > 2 && u_m == 3) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
                if (u_m > 3) ymm15 = _mm256_set1_ps(a_ptr[3 + 0 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);
                    if (u_nr > 0 && u_m == 4) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm15, ymm13, ymm10);
                    if (u_nr > 1 && u_m == 4) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm14, ymm11);
                    if (u_nr > 2 && u_m == 4) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
                if (u_m > 0) ymm15 = _mm256_set1_ps(a_ptr[0 + 1 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 0 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm15, ymm12, ymm0);
                    if (u_nr > 0 && u_m == 1) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm13, ymm1);
                    if (u_nr > 1 && u_m == 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm14, ymm2);
                    if (u_nr > 2 && u_m == 1) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
                if (u_m > 1) ymm15 = _mm256_set1_ps(a_ptr[1 + 1 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm15, ymm12, ymm3);
                    if (u_nr > 0 && u_m == 2) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm15, ymm13, ymm4);
                    if (u_nr > 1 && u_m == 2) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm14, ymm5);
                    if (u_nr > 2 && u_m == 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
                if (u_m > 2) ymm15 = _mm256_set1_ps(a_ptr[2 + 1 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * cacheline_elts + 0 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm15, ymm12, ymm6);
                    if (u_nr > 0 && u_m == 3) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);
                    if (u_nr > 1 && u_m == 3) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm14, ymm8);
                    if (u_nr > 2 && u_m == 3) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
                if (u_m > 3) ymm15 = _mm256_set1_ps(a_ptr[3 + 1 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);
                    if (u_nr > 0 && u_m == 4) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm15, ymm13, ymm10);
                    if (u_nr > 1 && u_m == 4) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm14, ymm11);
                    if (u_nr > 2 && u_m == 4) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
                if (u_m > 0) ymm15 = _mm256_set1_ps(a_ptr[0 + 2 * u_m]);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm15, ymm12, ymm0);
                    if (u_nr > 0 && u_m == 1) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm13, ymm1);
                    if (u_nr > 1 && u_m == 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm14, ymm2);
                    if (u_nr > 2 && u_m == 1) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 3 * u_n);
                }
                if (u_m > 1) ymm15 = _mm256_set1_ps(a_ptr[1 + 2 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 1 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm15, ymm12, ymm3);
                    if (u_nr > 0 && u_m == 2) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm15, ymm13, ymm4);
                    if (u_nr > 1 && u_m == 2) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm14, ymm5);
                    if (u_nr > 2 && u_m == 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 3 * u_n);
                }
                if (u_m > 2) ymm15 = _mm256_set1_ps(a_ptr[2 + 2 * u_m]);
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm15, ymm12, ymm6);
                    if (u_nr > 0 && u_m == 3) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);
                    if (u_nr > 1 && u_m == 3) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm14, ymm8);
                    if (u_nr > 2 && u_m == 3) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 3 * u_n);
                }
                if (u_m > 3) ymm15 = _mm256_set1_ps(a_ptr[3 + 2 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);
                    if (u_nr > 0 && u_m == 4) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm15, ymm13, ymm10);
                    if (u_nr > 1 && u_m == 4) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 3 * u_n);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm14, ymm11);
                    if (u_nr > 2 && u_m == 4) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 3 * u_n);
                }
                kl -= 1;
                if (u_m > 0) ymm15 = _mm256_set1_ps(a_ptr[0 + 3 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 1 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm15, ymm12, ymm0);
                    if (u_nr > 0 && u_m == 1) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm13, ymm1);
                    if (u_nr > 1 && u_m == 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm14, ymm2);
                    if (u_nr > 2 && u_m == 1) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 4 * u_n);
                }
                if (u_m > 1) ymm15 = _mm256_set1_ps(a_ptr[1 + 3 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm15, ymm12, ymm3);
                    if (u_nr > 0 && u_m == 2) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm15, ymm13, ymm4);
                    if (u_nr > 1 && u_m == 2) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm14, ymm5);
                    if (u_nr > 2 && u_m == 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 4 * u_n);
                }
                if (u_m > 2) ymm15 = _mm256_set1_ps(a_ptr[2 + 3 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * cacheline_elts + 1 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm15, ymm12, ymm6);
                    if (u_nr > 0 && u_m == 3) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);
                    if (u_nr > 1 && u_m == 3) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm14, ymm8);
                    if (u_nr > 2 && u_m == 3) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 4 * u_n);
                }
                if (u_m > 3) ymm15 = _mm256_set1_ps(a_ptr[3 + 3 * u_m]);
                a_ptr += u_k * u_m;
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);
                    if (u_nr > 0 && u_m == 4) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm15, ymm13, ymm10);
                    if (u_nr > 1 && u_m == 4) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 4 * u_n);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm14, ymm11);
                    if (u_nr > 2 && u_m == 4) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 4 * u_n);
                }
                b_ptr += u_k * u_n;
            } while (kl > 0);
        }

        kl = k & 3;
        if (kl > 0) {
            {
                if (u_m > 0) ymm15 = _mm256_set1_ps(a_ptr[0 + 0 * u_m]);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm15, ymm12, ymm0);
                    if (u_nr > 0 && u_m == 1) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm13, ymm1);
                    if (u_nr > 1 && u_m == 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm14, ymm2);
                    if (u_nr > 2 && u_m == 1) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
                if (u_m > 1) ymm15 = _mm256_set1_ps(a_ptr[1 + 0 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 0 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm15, ymm12, ymm3);
                    if (u_nr > 0 && u_m == 2) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm15, ymm13, ymm4);
                    if (u_nr > 1 && u_m == 2) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm14, ymm5);
                    if (u_nr > 2 && u_m == 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
                if (u_m > 2) ymm15 = _mm256_set1_ps(a_ptr[2 + 0 * u_m]);
                if (u_m > 0) _mm_prefetch((const char*)(a_ptr + 0 * cacheline_elts + prefetch_a_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm15, ymm12, ymm6);
                    if (u_nr > 0 && u_m == 3) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);
                    if (u_nr > 1 && u_m == 3) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm14, ymm8);
                    if (u_nr > 2 && u_m == 3) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
                if (u_m > 3) ymm15 = _mm256_set1_ps(a_ptr[3 + 0 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);
                    if (u_nr > 0 && u_m == 4) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm15, ymm13, ymm10);
                    if (u_nr > 1 && u_m == 4) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 1 * u_n);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm14, ymm11);
                    if (u_nr > 2 && u_m == 4) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 1 * u_n);
                }
            }
        }
        if (kl > 1) {
            {
                if (u_m > 0) ymm15 = _mm256_set1_ps(a_ptr[0 + 1 * u_m]);
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 0 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm15, ymm12, ymm0);
                    if (u_nr > 0 && u_m == 1) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm13, ymm1);
                    if (u_nr > 1 && u_m == 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm14, ymm2);
                    if (u_nr > 2 && u_m == 1) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
                if (u_m > 1) ymm15 = _mm256_set1_ps(a_ptr[1 + 1 * u_m]);
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm15, ymm12, ymm3);
                    if (u_nr > 0 && u_m == 2) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm15, ymm13, ymm4);
                    if (u_nr > 1 && u_m == 2) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm14, ymm5);
                    if (u_nr > 2 && u_m == 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
                if (u_m > 2) ymm15 = _mm256_set1_ps(a_ptr[2 + 1 * u_m]);
                if (u_nr > 2) _mm_prefetch((const char*)(b_ptr + 2 * cacheline_elts + 0 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm15, ymm12, ymm6);
                    if (u_nr > 0 && u_m == 3) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);
                    if (u_nr > 1 && u_m == 3) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm14, ymm8);
                    if (u_nr > 2 && u_m == 3) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
                if (u_m > 3) ymm15 = _mm256_set1_ps(a_ptr[3 + 1 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);
                    if (u_nr > 0 && u_m == 4) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm15, ymm13, ymm10);
                    if (u_nr > 1 && u_m == 4) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 2 * u_n);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm14, ymm11);
                    if (u_nr > 2 && u_m == 4) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 2 * u_n);
                }
            }
        }
        if (kl > 2) {
            {
                if (u_m > 0) ymm15 = _mm256_set1_ps(a_ptr[0 + 2 * u_m]);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm15, ymm12, ymm0);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm13, ymm1);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm14, ymm2);
                }
                if (u_m > 1) ymm15 = _mm256_set1_ps(a_ptr[1 + 2 * u_m]);
                if (u_nr > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 1 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm15, ymm12, ymm3);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm15, ymm13, ymm4);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm14, ymm5);
                }
                if (u_m > 2) ymm15 = _mm256_set1_ps(a_ptr[2 + 2 * u_m]);
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm15, ymm12, ymm6);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm14, ymm8);
                }
                if (u_m > 3) ymm15 = _mm256_set1_ps(a_ptr[3 + 2 * u_m]);
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm15, ymm13, ymm10);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm14, ymm11);
                }
                if (u_nr > 1) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 1 * u_nr * cacheline_elts + prefetch_b_offset), _MM_HINT_T0);
            }
        }
        a_ptr += kl * u_m;

        if (do_prefetch_next_b) _mm_prefetch((const char*)(next_b_ptr + 2 * cacheline_elts), _MM_HINT_T2);
        ymm12 = _mm256_set1_ps(kp.pick<float>(gemm_kernel_fp32_fma::param_def::ALPHA_IDX));
        ymm13 = _mm256_set1_ps(kp.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_IDX));
        if (need_mask) ymm15 = _mm256_loadu_ps(&kp.pick<float>(gemm_kernel_fp32_fma::param_def::MASK_IDX));
        const __m256i &imm15 = _mm256_castps_si256(ymm15);
        b_ptr = kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::B_PTR_IDX);
        prf_c_lduk = kp.pick<int64_t>(gemm_kernel_fp32_fma::param_def::PRF_C_LDK_IDX) >> u_k_log2;

        // *= alpha
        if (u_m > 0) {
            if (u_nr > 0) ymm0 = _mm256_mul_ps(ymm12, ymm0);
            if (u_nr > 1) ymm1 = _mm256_mul_ps(ymm12, ymm1);
            if (u_nr > 2) ymm2 = _mm256_mul_ps(ymm12, ymm2);
        }
        if (u_m > 1) {
            if (u_nr > 0) ymm3 = _mm256_mul_ps(ymm12, ymm3);
            if (u_nr > 1) ymm4 = _mm256_mul_ps(ymm12, ymm4);
            if (u_nr > 2) ymm5 = _mm256_mul_ps(ymm12, ymm5);
        }
        if (u_m > 2) {
            if (u_nr > 0) ymm6 = _mm256_mul_ps(ymm12, ymm6);
            if (u_nr > 1) ymm7 = _mm256_mul_ps(ymm12, ymm7);
            if (u_nr > 2) ymm8 = _mm256_mul_ps(ymm12, ymm8);
        }
        if (u_m > 3) {
            if (u_nr > 0) ymm9 = _mm256_mul_ps(ymm12, ymm9);
            if (u_nr > 1) ymm10 = _mm256_mul_ps(ymm12, ymm10);
            if (u_nr > 2) ymm11 = _mm256_mul_ps(ymm12, ymm11);
        }

        if (flags & gemm_kernel_fp32_fma::flag::WITH_SUM) {
            ymm14 = _mm256_set1_ps(kp.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_SUM_IDX));
            auto sum_ptr = kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::SUM_PTR_IDX);
            auto ldsum = kp.pick<const int64_t>(gemm_kernel_fp32_fma::param_def::LDSUM_IDX);
            if (need_mask) {
                if (u_m > 0) {
                    if (u_nr == 1) ymm0 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 0 * N_REG_ELTS, imm15), ymm14, ymm0);
                    else if (u_nr > 0) ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 0 * N_REG_ELTS), ymm14, ymm0);
                    if (u_nr == 2) ymm1 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 1 * N_REG_ELTS, imm15), ymm14, ymm1);
                    else if (u_nr > 1) ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 1 * N_REG_ELTS), ymm14, ymm1);
                    if (u_nr == 3) ymm2 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 2 * N_REG_ELTS, imm15), ymm14, ymm2);
                    else if (u_nr > 2) ymm2 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 2 * N_REG_ELTS), ymm14, ymm2);
                    sum_ptr += ldsum;
                }
                if (u_m > 1) {
                    if (u_nr == 1) ymm3 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 0 * N_REG_ELTS, imm15), ymm14, ymm3);
                    else if (u_nr > 0) ymm3 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 0 * N_REG_ELTS), ymm14, ymm3);
                    if (u_nr == 2) ymm4 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 1 * N_REG_ELTS, imm15), ymm14, ymm4);
                    else if (u_nr > 1) ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 1 * N_REG_ELTS), ymm14, ymm4);
                    if (u_nr == 3) ymm5 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 2 * N_REG_ELTS, imm15), ymm14, ymm5);
                    else if (u_nr > 2) ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 2 * N_REG_ELTS), ymm14, ymm5);
                    sum_ptr += ldsum;
                }
                if (u_m > 2) {
                    if (u_nr == 1) ymm6 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 0 * N_REG_ELTS, imm15), ymm14, ymm6);
                    else if (u_nr > 0) ymm6 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 0 * N_REG_ELTS), ymm14, ymm6);
                    if (u_nr == 2) ymm7 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 1 * N_REG_ELTS, imm15), ymm14, ymm7);
                    else if (u_nr > 1) ymm7 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 1 * N_REG_ELTS), ymm14, ymm7);
                    if (u_nr == 3) ymm8 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 2 * N_REG_ELTS, imm15), ymm14, ymm8);
                    else if (u_nr > 2) ymm8 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 2 * N_REG_ELTS), ymm14, ymm8);
                    sum_ptr += ldsum;
                }
                if (u_m > 3) {
                    if (u_nr == 1) ymm9 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 0 * N_REG_ELTS, imm15), ymm14, ymm9);
                    else if (u_nr > 0) ymm9 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 0 * N_REG_ELTS), ymm14, ymm9);
                    if (u_nr == 2) ymm10 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 1 * N_REG_ELTS, imm15), ymm14, ymm10);
                    else if (u_nr > 1) ymm10 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 1 * N_REG_ELTS), ymm14, ymm10);
                    if (u_nr == 3) ymm11 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(sum_ptr + 2 * N_REG_ELTS, imm15), ymm14, ymm11);
                    else if (u_nr > 2) ymm11 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 2 * N_REG_ELTS), ymm14, ymm11);
                    sum_ptr += ldsum;
                }
            } else {
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 0 * N_REG_ELTS), ymm14, ymm0);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 1 * N_REG_ELTS), ymm14, ymm1);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 2 * N_REG_ELTS), ymm14, ymm2);
                    sum_ptr += ldsum;
                }
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 0 * N_REG_ELTS), ymm14, ymm3);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 1 * N_REG_ELTS), ymm14, ymm4);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 2 * N_REG_ELTS), ymm14, ymm5);
                    sum_ptr += ldsum;
                }
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 0 * N_REG_ELTS), ymm14, ymm6);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 1 * N_REG_ELTS), ymm14, ymm7);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 2 * N_REG_ELTS), ymm14, ymm8);
                    sum_ptr += ldsum;
                }
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 0 * N_REG_ELTS), ymm14, ymm9);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 1 * N_REG_ELTS), ymm14, ymm10);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(_mm256_loadu_ps(sum_ptr + 2 * N_REG_ELTS), ymm14, ymm11);
                    sum_ptr += ldsum;
                }
            }
            kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::SUM_PTR_IDX) = sum_ptr;
        }

        if (flags & gemm_kernel_fp32_fma::flag::LOAD_C) {
            if (need_mask) {
                if (u_m > 0) {
                    if (u_nr == 1) ymm0 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m0_ptr + 0 * ldc + 0 * N_REG_ELTS, imm15), ymm13, ymm0);
                    else if (u_nr > 0) ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 0 * ldc + 0 * N_REG_ELTS), ymm13, ymm0);
                    if (u_nr == 2) ymm1 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m0_ptr + 0 * ldc + 1 * N_REG_ELTS, imm15), ymm13, ymm1);
                    else if (u_nr > 1) ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 0 * ldc + 1 * N_REG_ELTS), ymm13, ymm1);
                    if (u_nr == 3) ymm2 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m0_ptr + 0 * ldc + 2 * N_REG_ELTS, imm15), ymm13, ymm2);
                    else if (u_nr > 2) ymm2 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 0 * ldc + 2 * N_REG_ELTS), ymm13, ymm2);
                }
                if (u_m > 1) {
                    if (u_nr == 1) ymm3 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m0_ptr + 1 * ldc + 0 * N_REG_ELTS, imm15), ymm13, ymm3);
                    else if (u_nr > 0) ymm3 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 1 * ldc + 0 * N_REG_ELTS), ymm13, ymm3);
                    if (u_nr == 2) ymm4 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m0_ptr + 1 * ldc + 1 * N_REG_ELTS, imm15), ymm13, ymm4);
                    else if (u_nr > 1) ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 1 * ldc + 1 * N_REG_ELTS), ymm13, ymm4);
                    if (u_nr == 3) ymm5 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m0_ptr + 1 * ldc + 2 * N_REG_ELTS, imm15), ymm13, ymm5);
                    else if (u_nr > 2) ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 1 * ldc + 2 * N_REG_ELTS), ymm13, ymm5);
                }
                if (u_m > 2) {
                    if (u_nr == 1) ymm6 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m2_ptr + 0 * ldc + 0 * N_REG_ELTS, imm15), ymm13, ymm6);
                    else if (u_nr > 0) ymm6 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 0 * ldc + 0 * N_REG_ELTS), ymm13, ymm6);
                    if (u_nr == 2) ymm7 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m2_ptr + 0 * ldc + 1 * N_REG_ELTS, imm15), ymm13, ymm7);
                    else if (u_nr > 1) ymm7 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 0 * ldc + 1 * N_REG_ELTS), ymm13, ymm7);
                    if (u_nr == 3) ymm8 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m2_ptr + 0 * ldc + 2 * N_REG_ELTS, imm15), ymm13, ymm8);
                    else if (u_nr > 2) ymm8 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 0 * ldc + 2 * N_REG_ELTS), ymm13, ymm8);
                }
                if (u_m > 3) {
                    if (u_nr == 1) ymm9 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m2_ptr + 1 * ldc + 0 * N_REG_ELTS, imm15), ymm13, ymm9);
                    else if (u_nr > 0) ymm9 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 1 * ldc + 0 * N_REG_ELTS), ymm13, ymm9);
                    if (u_nr == 2) ymm10 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m2_ptr + 1 * ldc + 1 * N_REG_ELTS, imm15), ymm13, ymm10);
                    else if (u_nr > 1) ymm10 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 1 * ldc + 1 * N_REG_ELTS), ymm13, ymm10);
                    if (u_nr == 3) ymm11 = _mm256_fmadd_ps(ymm12 = _mm256_maskload_ps(c_m2_ptr + 1 * ldc + 2 * N_REG_ELTS, imm15), ymm13, ymm11);
                    else if (u_nr > 2) ymm11 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 1 * ldc + 2 * N_REG_ELTS), ymm13, ymm11);
                }
            } else {
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 0 * ldc + 0 * N_REG_ELTS), ymm13, ymm0);
                    if (u_nr > 1) ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 0 * ldc + 1 * N_REG_ELTS), ymm13, ymm1);
                    if (u_nr > 2) ymm2 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 0 * ldc + 2 * N_REG_ELTS), ymm13, ymm2);
                }
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 1 * ldc + 0 * N_REG_ELTS), ymm13, ymm3);
                    if (u_nr > 1) ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 1 * ldc + 1 * N_REG_ELTS), ymm13, ymm4);
                    if (u_nr > 2) ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m0_ptr + 1 * ldc + 2 * N_REG_ELTS), ymm13, ymm5);
                }
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 0 * ldc + 0 * N_REG_ELTS), ymm13, ymm6);
                    if (u_nr > 1) ymm7 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 0 * ldc + 1 * N_REG_ELTS), ymm13, ymm7);
                    if (u_nr > 2) ymm8 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 0 * ldc + 2 * N_REG_ELTS), ymm13, ymm8);
                }
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 1 * ldc + 0 * N_REG_ELTS), ymm13, ymm9);
                    if (u_nr > 1) ymm10 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 1 * ldc + 1 * N_REG_ELTS), ymm13, ymm10);
                    if (u_nr > 2) ymm11 = _mm256_fmadd_ps(_mm256_loadu_ps(c_m2_ptr + 1 * ldc + 2 * N_REG_ELTS), ymm13, ymm11);
                }
            }
        }

        ymm12 = _mm256_set1_ps(kp.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_BIAS_IDX));
        auto bias_ptr = kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::BIAS_PTR_IDX);
        if (flags & gemm_kernel_fp32_fma::flag::ROW_BIAS) {
            if (u_nr > 0) {
                if (u_nr == 1 && need_mask) ymm14 = _mm256_mul_ps(ymm13 = _mm256_maskload_ps(bias_ptr + 0 * N_REG_ELTS, imm15), ymm12);
                else ymm14 = _mm256_mul_ps(_mm256_loadu_ps(bias_ptr + 0 * N_REG_ELTS), ymm12);
                if (u_m > 0) ymm0 = _mm256_add_ps(ymm14, ymm0);
                if (u_m > 1) ymm3 = _mm256_add_ps(ymm14, ymm3);
                if (u_m > 2) ymm6 = _mm256_add_ps(ymm14, ymm6);
                if (u_m > 3) ymm9 = _mm256_add_ps(ymm14, ymm9);
            }
            if (u_nr > 1) {
                if (u_nr == 2 && need_mask) ymm14 = _mm256_mul_ps(ymm13 = _mm256_maskload_ps(bias_ptr + 1 * N_REG_ELTS, imm15), ymm12);
                else ymm14 = _mm256_mul_ps(_mm256_loadu_ps(bias_ptr + 1 * N_REG_ELTS), ymm12);
                if (u_m > 0) ymm1 = _mm256_add_ps(ymm14, ymm1);
                if (u_m > 1) ymm4 = _mm256_add_ps(ymm14, ymm4);
                if (u_m > 2) ymm7 = _mm256_add_ps(ymm14, ymm7);
                if (u_m > 3) ymm10 = _mm256_add_ps(ymm14, ymm10);
            }
            if (u_nr > 2) {
                if (u_nr == 3 && need_mask) ymm14 = _mm256_mul_ps(ymm13 = _mm256_maskload_ps(bias_ptr + 2 * N_REG_ELTS, imm15), ymm12);
                else ymm14 = _mm256_mul_ps(_mm256_loadu_ps(bias_ptr + 2 * N_REG_ELTS), ymm12);
                if (u_m > 0) ymm2 = _mm256_add_ps(ymm14, ymm2);
                if (u_m > 1) ymm5 = _mm256_add_ps(ymm14, ymm5);
                if (u_m > 2) ymm8 = _mm256_add_ps(ymm14, ymm8);
                if (u_m > 3) ymm11 = _mm256_add_ps(ymm14, ymm11);
            }
        }
        if (flags & gemm_kernel_fp32_fma::flag::SCA_BIAS) {
            ymm13 = _mm256_set1_ps(bias_ptr[0]);
            ymm14 = _mm256_mul_ps(ymm13, ymm12);
            if (u_m > 0) {
                if (u_nr > 0) ymm0 = _mm256_add_ps(ymm14, ymm0);
                if (u_nr > 1) ymm1 = _mm256_add_ps(ymm14, ymm1);
                if (u_nr > 2) ymm2 = _mm256_add_ps(ymm14, ymm2);
            }
            if (u_m > 1) {
                if (u_nr > 0) ymm3 = _mm256_add_ps(ymm14, ymm3);
                if (u_nr > 1) ymm4 = _mm256_add_ps(ymm14, ymm4);
                if (u_nr > 2) ymm5 = _mm256_add_ps(ymm14, ymm5);
            }
            if (u_m > 2) {
                if (u_nr > 0) ymm6 = _mm256_add_ps(ymm14, ymm6);
                if (u_nr > 1) ymm7 = _mm256_add_ps(ymm14, ymm7);
                if (u_nr > 2) ymm8 = _mm256_add_ps(ymm14, ymm8);
            }
            if (u_m > 3) {
                if (u_nr > 0) ymm9 = _mm256_add_ps(ymm14, ymm9);
                if (u_nr > 1) ymm10 = _mm256_add_ps(ymm14, ymm10);
                if (u_nr > 2) ymm11 = _mm256_add_ps(ymm14, ymm11);
            }
        }
        if (flags & gemm_kernel_fp32_fma::flag::COL_BIAS) {
            if (u_m > 0) {
                ymm13 = _mm256_set1_ps(bias_ptr[0]);
                ymm14 = _mm256_mul_ps(ymm13, ymm12);
                if (u_nr > 0) ymm0 = _mm256_add_ps(ymm14, ymm0);
                if (u_nr > 1) ymm1 = _mm256_add_ps(ymm14, ymm1);
                if (u_nr > 2) ymm2 = _mm256_add_ps(ymm14, ymm2);
            }
            if (u_m > 1) {
                ymm13 = _mm256_set1_ps(bias_ptr[1]);
                ymm14 = _mm256_mul_ps(ymm13, ymm12);
                if (u_nr > 0) ymm3 = _mm256_add_ps(ymm14, ymm3);
                if (u_nr > 1) ymm4 = _mm256_add_ps(ymm14, ymm4);
                if (u_nr > 2) ymm5 = _mm256_add_ps(ymm14, ymm5);
            }
            if (u_m > 2) {
                ymm13 = _mm256_set1_ps(bias_ptr[2]);
                ymm14 = _mm256_mul_ps(ymm13, ymm12);
                if (u_nr > 0) ymm6 = _mm256_add_ps(ymm14, ymm6);
                if (u_nr > 1) ymm7 = _mm256_add_ps(ymm14, ymm7);
                if (u_nr > 2) ymm8 = _mm256_add_ps(ymm14, ymm8);
            }
            if (u_m > 3) {
                ymm13 = _mm256_set1_ps(bias_ptr[3]);
                ymm14 = _mm256_mul_ps(ymm13, ymm12);
                if (u_nr > 0) ymm9 = _mm256_add_ps(ymm14, ymm9);
                if (u_nr > 1) ymm10 = _mm256_add_ps(ymm14, ymm10);
                if (u_nr > 2) ymm11 = _mm256_add_ps(ymm14, ymm11);
            }
            bias_ptr += u_m;
            kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::BIAS_PTR_IDX) = bias_ptr;
        }

        if (flags & (gemm_kernel_fp32_fma::flag::RELU | gemm_kernel_fp32_fma::flag::RELU6)) {
            ymm13 = _mm256_setzero_ps();
            ymm14 = _mm256_set1_ps(6.0f);
            if (u_m > 0) {
                if (u_nr > 0) ymm0 = _mm256_max_ps(ymm13, ymm0);
                if (u_nr > 1) ymm1 = _mm256_max_ps(ymm13, ymm1);
                if (u_nr > 2) ymm2 = _mm256_max_ps(ymm13, ymm2);
            }
            if (u_m > 1) {
                if (u_nr > 0) ymm3 = _mm256_max_ps(ymm13, ymm3);
                if (u_nr > 1) ymm4 = _mm256_max_ps(ymm13, ymm4);
                if (u_nr > 2) ymm5 = _mm256_max_ps(ymm13, ymm5);
            }
            if (u_m > 2) {
                if (u_nr > 0) ymm6 = _mm256_max_ps(ymm13, ymm6);
                if (u_nr > 1) ymm7 = _mm256_max_ps(ymm13, ymm7);
                if (u_nr > 2) ymm8 = _mm256_max_ps(ymm13, ymm8);
            }
            if (u_m > 3) {
                if (u_nr > 0) ymm9 = _mm256_max_ps(ymm13, ymm9);
                if (u_nr > 1) ymm10 = _mm256_max_ps(ymm13, ymm10);
                if (u_nr > 2) ymm11 = _mm256_max_ps(ymm13, ymm11);
            }
        }

        if (flags & gemm_kernel_fp32_fma::flag::RELU6) {
            if (u_m > 0) {
                if (u_nr > 0) ymm0 = _mm256_min_ps(ymm14, ymm0);
                if (u_nr > 1) ymm1 = _mm256_min_ps(ymm14, ymm1);
                if (u_nr > 2) ymm2 = _mm256_min_ps(ymm14, ymm2);
            }
            if (u_m > 1) {
                if (u_nr > 0) ymm3 = _mm256_min_ps(ymm14, ymm3);
                if (u_nr > 1) ymm4 = _mm256_min_ps(ymm14, ymm4);
                if (u_nr > 2) ymm5 = _mm256_min_ps(ymm14, ymm5);
            }
            if (u_m > 2) {
                if (u_nr > 0) ymm6 = _mm256_min_ps(ymm14, ymm6);
                if (u_nr > 1) ymm7 = _mm256_min_ps(ymm14, ymm7);
                if (u_nr > 2) ymm8 = _mm256_min_ps(ymm14, ymm8);
            }
            if (u_m > 3) {
                if (u_nr > 0) ymm9 = _mm256_min_ps(ymm14, ymm9);
                if (u_nr > 1) ymm10 = _mm256_min_ps(ymm14, ymm10);
                if (u_nr > 2) ymm11 = _mm256_min_ps(ymm14, ymm11);
            }
        }

        if (u_nr > 0) ymm12 = _mm256_loadu_ps(b_ptr + 0 * N_REG_ELTS + 0 * u_n);
        if (u_nr > 1) ymm13 = _mm256_loadu_ps(b_ptr + 1 * N_REG_ELTS + 0 * u_n);
        if (u_nr > 2) ymm14 = _mm256_loadu_ps(b_ptr + 2 * N_REG_ELTS + 0 * u_n);
        m -= u_m;
        if (do_prefetch_next_b) next_b_ptr += 3 * cacheline_elts;

        if (need_mask) {
            if (u_m > 0) {
                if (u_nr == 1) _mm256_maskstore_ps(c_m0_ptr + 0 * ldc + 0 * N_REG_ELTS, imm15, ymm0);
                else if (u_nr > 0) _mm256_storeu_ps(c_m0_ptr + 0 * ldc + 0 * N_REG_ELTS, ymm0);
                if (u_nr == 2) _mm256_maskstore_ps(c_m0_ptr + 0 * ldc + 1 * N_REG_ELTS, imm15, ymm1);
                else if (u_nr > 1) _mm256_storeu_ps(c_m0_ptr + 0 * ldc + 1 * N_REG_ELTS, ymm1);
                if (u_nr == 3) _mm256_maskstore_ps(c_m0_ptr + 0 * ldc + 2 * N_REG_ELTS, imm15, ymm2);
                else if (u_nr > 2) _mm256_storeu_ps(c_m0_ptr + 0 * ldc + 2 * N_REG_ELTS, ymm2);
            }
            if (u_m > 1) {
                if (u_nr == 1) _mm256_maskstore_ps(c_m0_ptr + 1 * ldc + 0 * N_REG_ELTS, imm15, ymm3);
                else if (u_nr > 0) _mm256_storeu_ps(c_m0_ptr + 1 * ldc + 0 * N_REG_ELTS, ymm3);
                if (u_nr == 2) _mm256_maskstore_ps(c_m0_ptr + 1 * ldc + 1 * N_REG_ELTS, imm15, ymm4);
                else if (u_nr > 1) _mm256_storeu_ps(c_m0_ptr + 1 * ldc + 1 * N_REG_ELTS, ymm4);
                if (u_nr == 3) _mm256_maskstore_ps(c_m0_ptr + 1 * ldc + 2 * N_REG_ELTS, imm15, ymm5);
                else if (u_nr > 2) _mm256_storeu_ps(c_m0_ptr + 1 * ldc + 2 * N_REG_ELTS, ymm5);
            }
            c_m0_ptr += ldcm;
            if (u_m > 2) {
                if (u_nr == 1) _mm256_maskstore_ps(c_m2_ptr + 0 * ldc + 0 * N_REG_ELTS, imm15, ymm6);
                else if (u_nr > 0) _mm256_storeu_ps(c_m2_ptr + 0 * ldc + 0 * N_REG_ELTS, ymm6);
                if (u_nr == 2) _mm256_maskstore_ps(c_m2_ptr + 0 * ldc + 1 * N_REG_ELTS, imm15, ymm7);
                else if (u_nr > 1) _mm256_storeu_ps(c_m2_ptr + 0 * ldc + 1 * N_REG_ELTS, ymm7);
                if (u_nr == 3) _mm256_maskstore_ps(c_m2_ptr + 0 * ldc + 2 * N_REG_ELTS, imm15, ymm8);
                else if (u_nr > 2) _mm256_storeu_ps(c_m2_ptr + 0 * ldc + 2 * N_REG_ELTS, ymm8);
            }
            if (u_m > 3) {
                if (u_nr == 1) _mm256_maskstore_ps(c_m2_ptr + 1 * ldc + 0 * N_REG_ELTS, imm15, ymm9);
                else if (u_nr > 0) _mm256_storeu_ps(c_m2_ptr + 1 * ldc + 0 * N_REG_ELTS, ymm9);
                if (u_nr == 2) _mm256_maskstore_ps(c_m2_ptr + 1 * ldc + 1 * N_REG_ELTS, imm15, ymm10);
                else if (u_nr > 1) _mm256_storeu_ps(c_m2_ptr + 1 * ldc + 1 * N_REG_ELTS, ymm10);
                if (u_nr == 3) _mm256_maskstore_ps(c_m2_ptr + 1 * ldc + 2 * N_REG_ELTS, imm15, ymm11);
                else if (u_nr > 2) _mm256_storeu_ps(c_m2_ptr + 1 * ldc + 2 * N_REG_ELTS, ymm11);
            }
            c_m2_ptr += ldcm;
        } else {
            if (u_m > 0) {
                if (u_nr > 0) _mm256_storeu_ps(c_m0_ptr + 0 * ldc + 0 * N_REG_ELTS, ymm0);
                if (u_nr > 1) _mm256_storeu_ps(c_m0_ptr + 0 * ldc + 1 * N_REG_ELTS, ymm1);
                if (u_nr > 2) _mm256_storeu_ps(c_m0_ptr + 0 * ldc + 2 * N_REG_ELTS, ymm2);
            }
            if (u_m > 1) {
                if (u_nr > 0) _mm256_storeu_ps(c_m0_ptr + 1 * ldc + 0 * N_REG_ELTS, ymm3);
                if (u_nr > 1) _mm256_storeu_ps(c_m0_ptr + 1 * ldc + 1 * N_REG_ELTS, ymm4);
                if (u_nr > 2) _mm256_storeu_ps(c_m0_ptr + 1 * ldc + 2 * N_REG_ELTS, ymm5);
            }
            c_m0_ptr += ldcm;
            if (u_m > 2) {
                if (u_nr > 0) _mm256_storeu_ps(c_m2_ptr + 0 * ldc + 0 * N_REG_ELTS, ymm6);
                if (u_nr > 1) _mm256_storeu_ps(c_m2_ptr + 0 * ldc + 1 * N_REG_ELTS, ymm7);
                if (u_nr > 2) _mm256_storeu_ps(c_m2_ptr + 0 * ldc + 2 * N_REG_ELTS, ymm8);
            }
            if (u_m > 3) {
                if (u_nr > 0) _mm256_storeu_ps(c_m2_ptr + 1 * ldc + 0 * N_REG_ELTS, ymm9);
                if (u_nr > 1) _mm256_storeu_ps(c_m2_ptr + 1 * ldc + 1 * N_REG_ELTS, ymm10);
                if (u_nr > 2) _mm256_storeu_ps(c_m2_ptr + 1 * ldc + 2 * N_REG_ELTS, ymm11);
            }
            c_m2_ptr += ldcm;
        }
        
        if (do_prefetch_next_b) _mm_prefetch((const char*)(next_b_ptr + 0 * cacheline_elts), _MM_HINT_T2);
        if (u_m > 0) {
            if (u_nr > 0) ymm0 = _mm256_setzero_ps();
            if (u_nr > 1) ymm1 = _mm256_setzero_ps();
            if (u_nr > 2) ymm2 = _mm256_setzero_ps();
            // prefetch C first 16 col anyway
            if (u_nr > 0) _mm_prefetch((const char*)(c_m0_ptr + 0 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
        }
        if (u_m > 1) {
            if (u_nr > 0) ymm3 = _mm256_setzero_ps();
            if (u_nr > 1) ymm4 = _mm256_setzero_ps();
            if (u_nr > 2) ymm5 = _mm256_setzero_ps();
            if (u_nr > 0) _mm_prefetch((const char*)(c_m0_ptr + 1 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
        }
        if (u_m > 2) {
            if (u_nr > 0) ymm6 = _mm256_setzero_ps();
            if (u_nr > 1) ymm7 = _mm256_setzero_ps();
            if (u_nr > 2) ymm8 = _mm256_setzero_ps();
            if (u_nr > 0) _mm_prefetch((const char*)(c_m2_ptr + 0 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
        }
        if (u_m > 3) {
            if (u_nr > 0) ymm9 = _mm256_setzero_ps();
            if (u_nr > 1) ymm10 = _mm256_setzero_ps();
            if (u_nr > 2) ymm11 = _mm256_setzero_ps();
            if (u_nr > 0) _mm_prefetch((const char*)(c_m2_ptr + 1 * ldc + 0 * cacheline_elts), _MM_HINT_T0);
        }
    } while (m > 0);
}

#define GEMM_KERNEL_FP32_FMA_TABLE_BLK(NEED_MASK) \
{\
    {\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 1, 1 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 2, 1 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 3, 1 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 4, 1 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
    },\
    {\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 1, 2 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 2, 2 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 3, 2 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 4, 2 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
    },\
    {\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 1, 3 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 2, 3 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 3, 3 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
        gemm_m4n24_kernel_fp32_fma<NEED_MASK, 4, 3 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,\
    },\
}\

const gemm_kernel_fp32_fma::func_t
    gemm_kernel_fp32_fma::table_[config::NEED_MASK_OPT][config::MAX_N_REGS][config::MAX_M_REGS] =
{
    GEMM_KERNEL_FP32_FMA_TABLE_BLK(0),
    GEMM_KERNEL_FP32_FMA_TABLE_BLK(1),
};

}}}; // namespace ppl::kernel::x86
