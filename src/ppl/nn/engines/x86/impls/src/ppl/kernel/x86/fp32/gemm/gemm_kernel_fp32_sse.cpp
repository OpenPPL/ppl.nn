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

#include <nmmintrin.h>

#include "ppl/kernel/x86/fp32/gemm/gemm_kernel_fp32_sse.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template<int64_t need_mask, int64_t u_n>
void gemm_m1n48_kernel_fp32_sse_core(int64_t *param) {
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
        ".equ MASKED_C_IDX,     (17 * P_BYTES)\n"
        ".equ MASKED_SUM_IDX,   (21 * P_BYTES)\n"
        ".equ MASKED_BIAS_IDX,  (25 * P_BYTES)\n"
        ".equ SIX_IDX,          (29 * P_BYTES)\n"

        ".equ N_REGB_ELTS, %c[N_REGB_ELTS]\n"
        ".equ MAX_N_REGBS, %c[MAX_N_REGBS]\n"
        ".equ N_REG_ELTS, %c[N_REG_ELTS]\n"
        ".equ NEED_MASK, %c[NEED_MASK]\n"
        ".equ U_N, %c[U_N]\n"
        ".equ U_K, 4\n"
        ".equ U_K_LOG2, 2\n"
        ".equ U_NRB, ((U_N + N_REGB_ELTS - 1) / N_REGB_ELTS)\n"
        ".equ MASK_OFF, (U_N - N_REGB_ELTS)\n"
        ".equ KERNEL_FLAG_LOAD_C, %c[KERNEL_FLAG_LOAD_C]\n"
        ".equ KERNEL_FLAG_WITH_SUM, %c[KERNEL_FLAG_WITH_SUM]\n"
        ".equ KERNEL_FLAG_ROW_BIAS, %c[KERNEL_FLAG_ROW_BIAS]\n"
        ".equ KERNEL_FLAG_COL_BIAS, %c[KERNEL_FLAG_COL_BIAS]\n"
        ".equ KERNEL_FLAG_SCA_BIAS, %c[KERNEL_FLAG_SCA_BIAS]\n"
        ".equ KERNEL_FLAG_RELU, %c[KERNEL_FLAG_RELU]\n"
        ".equ KERNEL_FLAG_RELU6, %c[KERNEL_FLAG_RELU6]\n"

        ".equ PREFETCH_B_OFFSET, 768\n"
        ".equ PREFETCH_A_OFFSET, 192\n"
        ".equ DO_PREFETCH_NEXT_B, (U_NRB == MAX_N_REGBS / 2 && !NEED_MASK)\n"

        "mov K_IDX(%[param]), %%rax\n"              // k
        "mov PRF_C_LDK_IDX(%[param]), %%r10\n"      // lead_k
        "sar $U_K_LOG2, %%r10\n"
        "mov B_PTR_IDX(%[param]), %%rbx\n"          // b_ptr
        "movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        "mov A_PTR_IDX(%[param]), %%r15\n"          // a_ptr
        "movss (0 * D_BYTES)(%%r15), %%xmm15\n shufps $0, %%xmm15, %%xmm15\n"
        ".if DO_PREFETCH_NEXT_B\n"
        "mov NEXT_B_PTR_IDX(%[param]), %%r8\n"      // next_b_ptr for prefetching <= do not have register double buffer
        ".endif\n"

        "mov FLAGS_IDX(%[param]),        %%rsi\n"
        "mov C_PTR_IDX(%[param]),        %%r14\n"
        "mov M_IDX(%[param]),            %%r13\n"
        "mov LDC_IDX(%[param]),          %%r11\n"
        "shl $LOG2_D_BYTES, %%r11\n"
        ".if NEED_MASK\n"
        "mov MASK_IDX(%[param]), %%r9\n"    // mask
        "shl $LOG2_D_BYTES, %%r9\n" // mask * sizeof(float)
        "test $KERNEL_FLAG_ROW_BIAS, %%rsi\n"
        "jz 21f\n" // masked_bias_end
        "mov BIAS_PTR_IDX(%[param]), %%rcx\n"
        "mov %%r9, %%r12\n"
"31:\n" // masked_bias_loop
        "sub $D_BYTES, %%r12\n"
        "mov (MASK_OFF * D_BYTES)(%%rcx, %%r12), %%edx\n"
        "mov %%edx, MASKED_BIAS_IDX(%[param], %%r12)\n"
        "jnz 31b\n"
"21:\n" // masked_bias_end
        ".endif\n"

        ".if DO_PREFETCH_NEXT_B\n prefetcht0 (0 * CACHELINE_BYTES * D_BYTES)(%%r8)\n .endif\n"
        ".if U_NRB > 0\n"
        "xorps %%xmm0, %%xmm0\n"
        "xorps %%xmm1, %%xmm1\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "xorps %%xmm2, %%xmm2\n"
        "xorps %%xmm3, %%xmm3\n"
        ".endif\n"
        ".if U_NRB > 2\n"
        "xorps %%xmm4, %%xmm4\n"
        "xorps %%xmm5, %%xmm5\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "xorps %%xmm6, %%xmm6\n"
        "xorps %%xmm7, %%xmm7\n"
        ".endif\n"
        ".if U_NRB > 4\n"
        "xorps %%xmm8, %%xmm8\n"
        "xorps %%xmm9, %%xmm9\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "xorps %%xmm10, %%xmm10\n"
        "xorps %%xmm11, %%xmm11\n"
        ".endif\n"




"1:\n" // label_init_session
        "mov %%rax, %%rdx\n" // k
        "sar $U_K_LOG2, %%rdx\n" // purge the k tail, k -> uk
        "jle 5f\n"  // label_k_tail
        "sub %%r10, %%rdx\n"
        "jle 30f\n" // label_prf_c
        PPL_X86_INLINE_ASM_ALIGN()
"4:\n" // label_loop_uk_body
        "movss (1 * D_BYTES)(%%r15), %%xmm14\n shufps $0, %%xmm14, %%xmm14\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES + 0 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 0\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm0\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm1\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm3\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES + 0 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm4\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm5\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm6\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm7\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES + 0 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm8\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm9\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm10\n"
        "movups ((6 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm11\n"
        "movups ((6 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        "movss (2 * D_BYTES)(%%r15), %%xmm15\n shufps $0, %%xmm15, %%xmm15\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 0\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm0\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm1\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm3\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm4\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm5\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm6\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm7\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm8\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm9\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm10\n"
        "movups ((6 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm11\n"
        "movups ((6 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        "movss (3 * D_BYTES)(%%r15), %%xmm14\n shufps $0, %%xmm14, %%xmm14\n"
        "sub $1, %%rdx\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 0\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm0\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm1\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm3\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm4\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm5\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm6\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm7\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm8\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm9\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm10\n"
        "movups ((6 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm11\n"
        "movups ((6 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        "movss (4 * D_BYTES)(%%r15), %%xmm15\n shufps $0, %%xmm15, %%xmm15\n"
        "prefetcht0 (0 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n"
        "lea (U_K * D_BYTES)(%%r15), %%r15\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 0\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm0\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm1\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm3\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm4\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm5\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm6\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm7\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm8\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm9\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm10\n"
        "movups ((6 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm11\n"
        "movups ((6 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        "lea (U_K * U_N * D_BYTES)(%%rbx), %%rbx\n"
        "jg 4b\n" // label_loop_uk_body



"30:\n" // label_prf_c
        ".if DO_PREFETCH_NEXT_B\n prefetcht0 (1 * CACHELINE_BYTES * D_BYTES)(%%r8)\n .endif\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES)(%%r14)\n .endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES)(%%r14)\n .endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES)(%%r14)\n .endif\n"

        "add %%r10, %%rdx\n"
        PPL_X86_INLINE_ASM_ALIGN()
"40:\n" // label_loop_uk_after_prf_c
        "movss (1 * D_BYTES)(%%r15), %%xmm14\n shufps $0, %%xmm14, %%xmm14\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES + 0 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 0\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm0\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm1\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm3\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES + 0 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm4\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm5\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm6\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm7\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES + 0 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm8\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm9\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm10\n"
        "movups ((6 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm11\n"
        "movups ((6 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        "movss (2 * D_BYTES)(%%r15), %%xmm15\n shufps $0, %%xmm15, %%xmm15\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 0\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm0\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm1\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm3\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm4\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm5\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm6\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm7\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm8\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm9\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm10\n"
        "movups ((6 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm11\n"
        "movups ((6 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        "movss (3 * D_BYTES)(%%r15), %%xmm14\n shufps $0, %%xmm14, %%xmm14\n"
        "sub $1, %%rdx\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 0\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm0\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm1\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm3\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm4\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm5\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm6\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm7\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm8\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm9\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm10\n"
        "movups ((6 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm11\n"
        "movups ((6 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        "movss (4 * D_BYTES)(%%r15), %%xmm15\n shufps $0, %%xmm15, %%xmm15\n"
        "prefetcht0 (0 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n"
        "lea (U_K * D_BYTES)(%%r15), %%r15\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 0\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm0\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm1\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm3\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm4\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm5\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm6\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm7\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES + 3 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm8\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm9\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm10\n"
        "movups ((6 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm11\n"
        "movups ((6 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        "lea (U_K * U_N * D_BYTES)(%%rbx), %%rbx\n"
        "jg 40b\n" // label_loop_uk_after_prf_c




"5:\n" // label_k_tail
        "mov %%rax, %%rdx\n"
        "and $(U_K - 1), %%rdx\n"
        "je 6f\n" // label_end_k
        "shl $LOG2_D_BYTES, %%rdx\n" // k_tail <<= 2 -> sizeof(float) * k_tail
        "movss (1 * D_BYTES)(%%r15), %%xmm14\n shufps $0, %%xmm14, %%xmm14\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES + 0 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        "cmp $(1 * D_BYTES), %%rdx\n"
        ".if U_NRB > 0\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm0\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm1\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm3\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES + 0 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm4\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm5\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm6\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm7\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES + 0 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm8\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm9\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm10\n"
        "movups ((6 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm11\n"
        "movups ((6 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        "je 50f\n"
        "movss (2 * D_BYTES)(%%r15), %%xmm15\n shufps $0, %%xmm15, %%xmm15\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        "cmp $(2 * D_BYTES), %%rdx\n"
        ".if U_NRB > 0\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm0\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm1\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm3\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm4\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm5\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm6\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm7\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES + 1 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm8\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm9\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm14, %%xmm12\n"
        "addps %%xmm12, %%xmm10\n"
        "movups ((6 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm14, %%xmm13\n"
        "addps %%xmm13, %%xmm11\n"
        "movups ((6 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        "je 50f\n"
        ".if U_NRB > 0\n prefetcht0 (0 * CACHELINE_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 0\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm0\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm1\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm3\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 2\n prefetcht0 (1 * CACHELINE_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm4\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm5\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm6\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm7\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 4\n prefetcht0 (2 * CACHELINE_BYTES + 2 * U_N * D_BYTES + PREFETCH_B_OFFSET)(%%rbx)\n .endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm8\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm9\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm15, %%xmm12\n"
        "addps %%xmm12, %%xmm10\n"
        "movups ((6 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "mulps %%xmm15, %%xmm13\n"
        "addps %%xmm13, %%xmm11\n"
        "movups ((6 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        ".endif\n"
"50:\n" // label_end_k_tail
        "prefetcht0 (0 * CACHELINE_BYTES + PREFETCH_A_OFFSET)(%%r15)\n"
        "lea (%%r15, %%rdx), %%r15\n" // a_ptr += k_tail
"6:\n" // label_end_k



        ".if DO_PREFETCH_NEXT_B\n prefetcht0 (2 * CACHELINE_BYTES * D_BYTES)(%%r8)\n .endif\n"
        "movss ALPHA_IDX(%[param]), %%xmm12\n" // alpha
        "shufps $0, %%xmm12, %%xmm12\n"
        "movss BETA_IDX(%[param]), %%xmm13\n"  // beta
        "shufps $0, %%xmm13, %%xmm13\n"
        "mov B_PTR_IDX(%[param]), %%rbx\n"            // b_ptr

        // *= alpha
        ".if U_NRB > 0\n"
        "mulps %%xmm12, %%xmm0\n"
        "mulps %%xmm12, %%xmm1\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "mulps %%xmm12, %%xmm2\n"
        "mulps %%xmm12, %%xmm3\n"
        ".endif\n"
        ".if U_NRB > 2\n"
        "mulps %%xmm12, %%xmm4\n"
        "mulps %%xmm12, %%xmm5\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "mulps %%xmm12, %%xmm6\n"
        "mulps %%xmm12, %%xmm7\n"
        ".endif\n"
        ".if U_NRB > 4\n"
        "mulps %%xmm12, %%xmm8\n"
        "mulps %%xmm12, %%xmm9\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "mulps %%xmm12, %%xmm10\n"
        "mulps %%xmm12, %%xmm11\n"
        ".endif\n"
        "movss BETA_BIAS_IDX(%[param]), %%xmm12\n"
        "shufps $0, %%xmm12, %%xmm12\n"

        // put sum in the first place, overlaping cache miss
        // += beta_sum * sum
        "test $KERNEL_FLAG_WITH_SUM, %%rsi\n"
        "jz 14f\n" // label_load_sum_end
        "movss BETA_SUM_IDX(%[param]), %%xmm14\n"
        "shufps $0, %%xmm14, %%xmm14\n"
        "mov SUM_PTR_IDX(%[param]), %%rcx\n"
        ".if NEED_MASK\n"
        "mov %%r9, %%r12\n"
"32:\n" // masked_sum_loop
        "sub $D_BYTES, %%r12\n"
        "mov (MASK_OFF * D_BYTES)(%%rcx, %%r12), %%edx\n"
        "mov %%edx, MASKED_SUM_IDX(%[param], %%r12)\n"
        "jnz 32b\n" // masked_sum_loop
        ".endif\n"
        "mov LDSUM_IDX(%[param]), %%rdx\n"
        ".if U_NRB > 0\n"
        ".if U_NRB == 1 && NEED_MASK\n movups (MASKED_SUM_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm0\n"
        ".if U_NRB == 1 && NEED_MASK\n movups (MASKED_SUM_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm1\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        ".if U_NRB == 2 && NEED_MASK\n movups (MASKED_SUM_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm2\n"
        ".if U_NRB == 2 && NEED_MASK\n movups (MASKED_SUM_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm3\n"
        ".endif\n"
        ".if U_NRB > 2\n"
        ".if U_NRB == 3 && NEED_MASK\n movups (MASKED_SUM_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm4\n"
        ".if U_NRB == 3 && NEED_MASK\n movups (MASKED_SUM_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm5\n"
        ".endif\n"
        "lea (%%rcx, %%rdx, D_BYTES), %%rdx\n"
        ".if U_NRB > 3\n"
        ".if U_NRB == 4 && NEED_MASK\n movups (MASKED_SUM_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm6\n"
        ".if U_NRB == 4 && NEED_MASK\n movups (MASKED_SUM_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm7\n"
        ".endif\n"
        ".if U_NRB > 4\n"
        ".if U_NRB == 5 && NEED_MASK\n movups (MASKED_SUM_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm8\n"
        ".if U_NRB == 5 && NEED_MASK\n movups (MASKED_SUM_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm9\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        ".if U_NRB == 6 && NEED_MASK\n movups (MASKED_SUM_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm10\n"
        ".if U_NRB == 6 && NEED_MASK\n movups (MASKED_SUM_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm14, %%xmm15\n"
        "addps %%xmm15, %%xmm11\n"
        ".endif\n"
        "mov %%rdx, SUM_PTR_IDX(%[param])\n"
"14:\n" // label_load_sum_end

        // += beta * C
        "test $KERNEL_FLAG_LOAD_C, %%rsi\n"
        "jz 8f\n" // label_load_c_end
        ".if NEED_MASK\n"
        "mov %%r9, %%r12\n"
"33:\n" // masked_c_load_loop
        "sub $D_BYTES, %%r12\n"
        "mov (MASK_OFF * D_BYTES)(%%r14, %%r12), %%edx\n"
        "mov %%edx, MASKED_C_IDX(%[param], %%r12)\n"
        "jnz 33b\n" // masked_c_load_loop
        ".endif\n"
        ".if U_NRB > 0\n"
        ".if U_NRB == 1 && NEED_MASK\n movups (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm0\n"
        ".if U_NRB == 1 && NEED_MASK\n movups (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm1\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        ".if U_NRB == 2 && NEED_MASK\n movups (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm2\n"
        ".if U_NRB == 2 && NEED_MASK\n movups (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm3\n"
        ".endif\n"
        ".if U_NRB > 2\n"
        ".if U_NRB == 3 && NEED_MASK\n movups (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm4\n"
        ".if U_NRB == 3 && NEED_MASK\n movups (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm5\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        ".if U_NRB == 4 && NEED_MASK\n movups (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm6\n"
        ".if U_NRB == 4 && NEED_MASK\n movups (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm7\n"
        ".endif\n"
        ".if U_NRB > 4\n"
        ".if U_NRB == 5 && NEED_MASK\n movups (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm8\n"
        ".if U_NRB == 5 && NEED_MASK\n movups (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm9\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        ".if U_NRB == 6 && NEED_MASK\n movups (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm10\n"
        ".if U_NRB == 6 && NEED_MASK\n movups (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14), %%xmm15\n .endif\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm15, %%xmm11\n"
        ".endif\n"
"8:\n" // label_load_c_end

        "mov BIAS_PTR_IDX(%[param]), %%rcx\n"
        // += beta_bias * bias
        "test $KERNEL_FLAG_ROW_BIAS, %%rsi\n"
        "jz 11f\n" // label_row_bias_end
        ".if U_NRB > 0\n"
        ".if U_NRB == 1 && NEED_MASK\n movups (MASKED_BIAS_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm0\n"
        ".if U_NRB == 1 && NEED_MASK\n movups (MASKED_BIAS_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm1\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        ".if U_NRB == 2 && NEED_MASK\n movups (MASKED_BIAS_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm2\n"
        ".if U_NRB == 2 && NEED_MASK\n movups (MASKED_BIAS_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm3\n"
        ".endif\n"
        ".if U_NRB > 2\n"
        ".if U_NRB == 3 && NEED_MASK\n movups (MASKED_BIAS_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm4\n"
        ".if U_NRB == 3 && NEED_MASK\n movups (MASKED_BIAS_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm5\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        ".if U_NRB == 4 && NEED_MASK\n movups (MASKED_BIAS_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm6\n"
        ".if U_NRB == 4 && NEED_MASK\n movups (MASKED_BIAS_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm7\n"
        ".endif\n"
        ".if U_NRB > 4\n"
        ".if U_NRB == 5 && NEED_MASK\n movups (MASKED_BIAS_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm8\n"
        ".if U_NRB == 5 && NEED_MASK\n movups (MASKED_BIAS_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm9\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        ".if U_NRB == 6 && NEED_MASK\n movups (MASKED_BIAS_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm10\n"
        ".if U_NRB == 6 && NEED_MASK\n movups (MASKED_BIAS_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param]), %%xmm15\n"
        ".else\n movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n .endif\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm15, %%xmm11\n"
        ".endif\n"
"11:\n" // label_row_bias_end
        "test $(KERNEL_FLAG_SCA_BIAS | KERNEL_FLAG_COL_BIAS), %%rsi\n"
        "jz 12f\n" // label_sca_bias_end
        "movss (%%rcx), %%xmm13\n"
        "shufps $0, %%xmm13, %%xmm13\n"
        "mulps %%xmm12, %%xmm13\n"
        ".if U_NRB > 0\n"
        "addps %%xmm13, %%xmm0\n"
        "addps %%xmm13, %%xmm1\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "addps %%xmm13, %%xmm2\n"
        "addps %%xmm13, %%xmm3\n"
        ".endif\n"
        ".if U_NRB > 2\n"
        "addps %%xmm13, %%xmm4\n"
        "addps %%xmm13, %%xmm5\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "addps %%xmm13, %%xmm6\n"
        "addps %%xmm13, %%xmm7\n"
        ".endif\n"
        ".if U_NRB > 4\n"
        "addps %%xmm13, %%xmm8\n"
        "addps %%xmm13, %%xmm9\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "addps %%xmm13, %%xmm10\n"
        "addps %%xmm13, %%xmm11\n"
        ".endif\n"
"12:\n" // label_sca_bias_end
        "test $KERNEL_FLAG_COL_BIAS, %%rsi\n"
        "jz 13f\n" // label_col_bias_end
        "lea (1 * D_BYTES)(%%rcx), %%rcx\n"
        "mov %%rcx, BIAS_PTR_IDX(%[param])\n"
"13:\n" // label_col_bias_end

        "test $(KERNEL_FLAG_RELU | KERNEL_FLAG_RELU6), %%rsi\n"
        "jz 9f\n" // label_relu_end
        "xorps %%xmm13, %%xmm13\n" // 0.0
        "movups SIX_IDX(%[param]), %%xmm14\n" // 6.0
        "test $KERNEL_FLAG_RELU6, %%rsi\n"
        ".if U_NRB > 0\n"
        "maxps %%xmm13, %%xmm0\n"
        "maxps %%xmm13, %%xmm1\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "maxps %%xmm13, %%xmm2\n"
        "maxps %%xmm13, %%xmm3\n"
        ".endif\n"
        ".if U_NRB > 2\n"
        "maxps %%xmm13, %%xmm4\n"
        "maxps %%xmm13, %%xmm5\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "maxps %%xmm13, %%xmm6\n"
        "maxps %%xmm13, %%xmm7\n"
        ".endif\n"
        ".if U_NRB > 4\n"
        "maxps %%xmm13, %%xmm8\n"
        "maxps %%xmm13, %%xmm9\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "maxps %%xmm13, %%xmm10\n"
        "maxps %%xmm13, %%xmm11\n"
        ".endif\n"

        "jz 9f\n" // label_relu_end
        ".if U_NRB > 0\n"
        "minps %%xmm14, %%xmm0\n"
        "minps %%xmm14, %%xmm1\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "minps %%xmm14, %%xmm2\n"
        "minps %%xmm14, %%xmm3\n"
        ".endif\n"
        ".if U_NRB > 2\n"
        "minps %%xmm14, %%xmm4\n"
        "minps %%xmm14, %%xmm5\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "minps %%xmm14, %%xmm6\n"
        "minps %%xmm14, %%xmm7\n"
        ".endif\n"
        ".if U_NRB > 4\n"
        "minps %%xmm14, %%xmm8\n"
        "minps %%xmm14, %%xmm9\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "minps %%xmm14, %%xmm10\n"
        "minps %%xmm14, %%xmm11\n"
        ".endif\n"
"9:\n" // label_relu_end

        "movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm12\n"
        "movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * U_N) * D_BYTES)(%%rbx), %%xmm13\n"
        "movss (0 * D_BYTES)(%%r15), %%xmm15\n shufps $0, %%xmm15, %%xmm15\n"
        ".if DO_PREFETCH_NEXT_B\n lea (3 * CACHELINE_BYTES * D_BYTES)(%%r8), %%r8\n .endif\n"

        ".if U_NRB > 0\n"
        ".if U_NRB == 1 && NEED_MASK\n movups %%xmm0, (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm0, ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NRB == 1 && NEED_MASK\n movups %%xmm1, (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm1, ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        ".if U_NRB == 2 && NEED_MASK\n movups %%xmm2, (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm2, ((1 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NRB == 2 && NEED_MASK\n movups %%xmm3, (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm3, ((1 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".endif\n"
        ".if U_NRB > 2\n"
        ".if U_NRB == 3 && NEED_MASK\n movups %%xmm4, (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm4, ((2 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NRB == 3 && NEED_MASK\n movups %%xmm5, (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm5, ((2 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        ".if U_NRB == 4 && NEED_MASK\n movups %%xmm6, (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm6, ((3 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NRB == 4 && NEED_MASK\n movups %%xmm7, (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm7, ((3 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".endif\n"
        ".if U_NRB > 4\n"
        ".if U_NRB == 5 && NEED_MASK\n movups %%xmm8, (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm8, ((4 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NRB == 5 && NEED_MASK\n movups %%xmm9, (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm9, ((4 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        ".if U_NRB == 6 && NEED_MASK\n movups %%xmm10, (MASKED_C_IDX + 0 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm10, ((5 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".if U_NRB == 6 && NEED_MASK\n movups %%xmm11, (MASKED_C_IDX + 1 * N_REG_ELTS * D_BYTES)(%[param])\n"
        ".else\n movups %%xmm11, ((5 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n .endif\n"
        ".endif\n"
        ".if NEED_MASK\n"
        "mov %%r9, %%r12\n"
"34:\n" // masked_c_store_loop
        "sub $D_BYTES, %%r12\n"
        "mov MASKED_C_IDX(%[param], %%r12), %%edx\n"
        "mov %%edx, (MASK_OFF * D_BYTES)(%%r14, %%r12)\n"
        "jnz 34b\n" // masked_c_store_loop
        ".endif\n"
        "lea (%%r14, %%r11), %%r14\n"

        "sub $1, %%r13\n" // m -= 1
        ".if DO_PREFETCH_NEXT_B\n prefetcht0 (0 * CACHELINE_BYTES * D_BYTES)(%%r8)\n .endif\n"
        ".if U_NRB > 0\n"
        "xorps %%xmm0, %%xmm0\n"
        "xorps %%xmm1, %%xmm1\n"
        ".endif\n"
        ".if U_NRB > 1\n"
        "xorps %%xmm2, %%xmm2\n"
        "xorps %%xmm3, %%xmm3\n"
        ".endif\n"
        ".if U_NRB > 2\n"
        "xorps %%xmm4, %%xmm4\n"
        "xorps %%xmm5, %%xmm5\n"
        ".endif\n"
        ".if U_NRB > 3\n"
        "xorps %%xmm6, %%xmm6\n"
        "xorps %%xmm7, %%xmm7\n"
        ".endif\n"
        ".if U_NRB > 4\n"
        "xorps %%xmm8, %%xmm8\n"
        "xorps %%xmm9, %%xmm9\n"
        ".endif\n"
        ".if U_NRB > 5\n"
        "xorps %%xmm10, %%xmm10\n"
        "xorps %%xmm11, %%xmm11\n"
        ".endif\n"

        "jg 1b\n" // label_init_session
        :
        :
        [param]                         "r" (param),
        [N_REG_ELTS]                    "i" (gemm_kernel_fp32_sse::config::N_REG_ELTS),
        [N_REGB_ELTS]                   "i" (gemm_kernel_fp32_sse::config::N_REGB_ELTS),
        [MAX_N_REGBS]                   "i" (gemm_kernel_fp32_sse::config::MAX_N_REGBS),
        [NEED_MASK]                     "i" (need_mask),
        [U_N]                           "i" (u_n),
        [KERNEL_FLAG_LOAD_C]            "i" (gemm_kernel_fp32_sse::flag::LOAD_C),
        [KERNEL_FLAG_WITH_SUM]          "i" (gemm_kernel_fp32_sse::flag::WITH_SUM),
        [KERNEL_FLAG_ROW_BIAS]          "i" (gemm_kernel_fp32_sse::flag::ROW_BIAS),
        [KERNEL_FLAG_COL_BIAS]          "i" (gemm_kernel_fp32_sse::flag::COL_BIAS),
        [KERNEL_FLAG_SCA_BIAS]          "i" (gemm_kernel_fp32_sse::flag::SCA_BIAS),
        [KERNEL_FLAG_RELU]              "i" (gemm_kernel_fp32_sse::flag::RELU),
        [KERNEL_FLAG_RELU6]             "i" (gemm_kernel_fp32_sse::flag::RELU6)
        :
        "cc",
        "rax", "rbx", "rcx", "rdx",
        "r8" , "r9" , "r10", "r11",
        "r12", "r13", "r14", "r15",
        "rsi",
        "xmm0" , "xmm1" , "xmm2" , "xmm3" , "xmm4" , "xmm5" , "xmm6" , "xmm7" ,
        "xmm8" , "xmm9" , "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
        "memory"
    );
}

#endif

template<int64_t need_mask, int64_t u_n>
void gemm_m1n48_kernel_fp32_sse(int64_t *param)
{
#ifdef PPL_USE_X86_INLINE_ASM

    gemm_m1n48_kernel_fp32_sse_core<need_mask, u_n>(param);
    return;

#endif

    // reference intrinsic for windows, performance is not tested
    array_param_helper kp(param);
    const int64_t N_REGB_ELTS = gemm_kernel_fp32_sse::config::N_REGB_ELTS;
    const int64_t MAX_N_REGBS = gemm_kernel_fp32_sse::config::MAX_N_REGBS;
    const int64_t N_REG_ELTS = gemm_kernel_fp32_sse::config::N_REG_ELTS;
    const int64_t u_nrb = div_up(u_n, N_REGB_ELTS);
    const int64_t u_k = 4;
    const int64_t u_k_log2 = 2;
    const int64_t mask_off = u_n - N_REGB_ELTS;

    const int64_t prefetch_b_offset = 384 / sizeof(float);
    const int64_t prefetch_a_offset = 192 / sizeof(float);
    const int64_t cacheline_elts = PPL_X86_CACHELINE_BYTES() / sizeof(float); // J5005 64B
    const bool do_prefetch_next_b = u_nrb >= MAX_N_REGBS / 2 && !need_mask;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;

    // load constant values
    auto k = kp.pick<int64_t>(gemm_kernel_fp32_sse::param_def::K_IDX);
    auto prf_c_lduk = kp.pick<int64_t>(gemm_kernel_fp32_sse::param_def::PRF_C_LDK_IDX) >> u_k_log2;
    auto b_ptr = kp.pick<const float*>(gemm_kernel_fp32_sse::param_def::B_PTR_IDX);
    xmm12 = _mm_loadu_ps(b_ptr + 0 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
    xmm13 = _mm_loadu_ps(b_ptr + 0 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
    auto a_ptr = kp.pick<const float*>(gemm_kernel_fp32_sse::param_def::A_PTR_IDX);
    xmm15 = _mm_set1_ps(a_ptr[0]);
    auto next_b_ptr = do_prefetch_next_b ? kp.pick<const float*>(gemm_kernel_fp32_sse::param_def::NEXT_B_PTR_IDX) : nullptr;

    auto flags = kp.pick<const gemm_kernel_fp32_sse::flag_t>(gemm_kernel_fp32_sse::param_def::FLAGS_IDX);
    auto ldc = kp.pick<const int64_t>(gemm_kernel_fp32_sse::param_def::LDC_IDX);
    auto c_ptr = kp.pick<float*>(gemm_kernel_fp32_sse::param_def::C_PTR_IDX);
    auto m = kp.pick<int64_t>(gemm_kernel_fp32_sse::param_def::M_IDX);
    auto mask = need_mask ? kp.pick<int64_t>(gemm_kernel_fp32_sse::param_def::MASK_IDX) : 0;

    if (need_mask && (flags & gemm_kernel_fp32_sse::flag::ROW_BIAS)) {
        auto masked_bias = &kp.pick<float>(gemm_kernel_fp32_sse::param_def::MASKED_BIAS_IDX);
        auto offset_bias = kp.pick<const float*>(gemm_kernel_fp32_sse::param_def::BIAS_PTR_IDX) + mask_off;
        auto i = mask;
        do {
            i -= 1;
            masked_bias[i] = offset_bias[i];
        } while (i);
    }

    if (do_prefetch_next_b) _mm_prefetch((const char*)(next_b_ptr + 0 * cacheline_elts), _MM_HINT_T0);
    if (u_nrb > 0) {
        xmm0 = _mm_setzero_ps();
        xmm1 = _mm_setzero_ps();
    }
    if (u_nrb > 1) {
        xmm2 = _mm_setzero_ps();
        xmm3 = _mm_setzero_ps();
    }
    if (u_nrb > 2) {
        xmm4 = _mm_setzero_ps();
        xmm5 = _mm_setzero_ps();
    }
    if (u_nrb > 3) {
        xmm6 = _mm_setzero_ps();
        xmm7 = _mm_setzero_ps();
    }
    if (u_nrb > 4) {
        xmm8 = _mm_setzero_ps();
        xmm9 = _mm_setzero_ps();
    }
    if (u_nrb > 5) {
        xmm10 = _mm_setzero_ps();
        xmm11 = _mm_setzero_ps();
    }

    do {
        auto kl = (k >> u_k_log2) - prf_c_lduk;
        if (kl > 0) {
            do {
                xmm14 = _mm_set1_ps(a_ptr[1]);
                if (u_nrb > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 0 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 0) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm0 = _mm_add_ps(xmm12, xmm0);
                    xmm12 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm1 = _mm_add_ps(xmm13, xmm1);
                    xmm13 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 1) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm2 = _mm_add_ps(xmm12, xmm2);
                    xmm12 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm3 = _mm_add_ps(xmm13, xmm3);
                    xmm13 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 2) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 0 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 2) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm4 = _mm_add_ps(xmm12, xmm4);
                    xmm12 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm5 = _mm_add_ps(xmm13, xmm5);
                    xmm13 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 3) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm6 = _mm_add_ps(xmm12, xmm6);
                    xmm12 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm7 = _mm_add_ps(xmm13, xmm7);
                    xmm13 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 4) _mm_prefetch((const char*)(b_ptr + 3 * cacheline_elts + 0 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 4) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm8 = _mm_add_ps(xmm12, xmm8);
                    xmm12 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm9 = _mm_add_ps(xmm13, xmm9);
                    xmm13 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 5) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm10 = _mm_add_ps(xmm12, xmm10);
                    xmm12 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm11 = _mm_add_ps(xmm13, xmm11);
                    xmm13 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                xmm15 = _mm_set1_ps(a_ptr[2]);
                if (u_nrb > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 0) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm0 = _mm_add_ps(xmm12, xmm0);
                    xmm12 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm1 = _mm_add_ps(xmm13, xmm1);
                    xmm13 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 1) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm2 = _mm_add_ps(xmm12, xmm2);
                    xmm12 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm3 = _mm_add_ps(xmm13, xmm3);
                    xmm13 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 2) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 2) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm4 = _mm_add_ps(xmm12, xmm4);
                    xmm12 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm5 = _mm_add_ps(xmm13, xmm5);
                    xmm13 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 3) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm6 = _mm_add_ps(xmm12, xmm6);
                    xmm12 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm7 = _mm_add_ps(xmm13, xmm7);
                    xmm13 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 4) _mm_prefetch((const char*)(b_ptr + 3 * cacheline_elts + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 4) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm8 = _mm_add_ps(xmm12, xmm8);
                    xmm12 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm9 = _mm_add_ps(xmm13, xmm9);
                    xmm13 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 5) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm10 = _mm_add_ps(xmm12, xmm10);
                    xmm12 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm11 = _mm_add_ps(xmm13, xmm11);
                    xmm13 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                xmm14 = _mm_set1_ps(a_ptr[3]);
                kl -= 1;
                if (u_nrb > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 0) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm0 = _mm_add_ps(xmm12, xmm0);
                    xmm12 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm1 = _mm_add_ps(xmm13, xmm1);
                    xmm13 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 1) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm2 = _mm_add_ps(xmm12, xmm2);
                    xmm12 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm3 = _mm_add_ps(xmm13, xmm3);
                    xmm13 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 2) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 2) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm4 = _mm_add_ps(xmm12, xmm4);
                    xmm12 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm5 = _mm_add_ps(xmm13, xmm5);
                    xmm13 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 3) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm6 = _mm_add_ps(xmm12, xmm6);
                    xmm12 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm7 = _mm_add_ps(xmm13, xmm7);
                    xmm13 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 4) _mm_prefetch((const char*)(b_ptr + 3 * cacheline_elts + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 4) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm8 = _mm_add_ps(xmm12, xmm8);
                    xmm12 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm9 = _mm_add_ps(xmm13, xmm9);
                    xmm13 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 5) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm10 = _mm_add_ps(xmm12, xmm10);
                    xmm12 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm11 = _mm_add_ps(xmm13, xmm11);
                    xmm13 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                xmm15 = _mm_set1_ps(a_ptr[4]);
                _mm_prefetch((const char*)(a_ptr + 0 * cacheline_elts + prefetch_a_offset), _MM_HINT_T0);
                a_ptr += u_k;
                if (u_nrb > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 0) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm0 = _mm_add_ps(xmm12, xmm0);
                    xmm12 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm1 = _mm_add_ps(xmm13, xmm1);
                    xmm13 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                if (u_nrb > 1) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm2 = _mm_add_ps(xmm12, xmm2);
                    xmm12 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm3 = _mm_add_ps(xmm13, xmm3);
                    xmm13 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                if (u_nrb > 2) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 2) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm4 = _mm_add_ps(xmm12, xmm4);
                    xmm12 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm5 = _mm_add_ps(xmm13, xmm5);
                    xmm13 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                if (u_nrb > 3) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm6 = _mm_add_ps(xmm12, xmm6);
                    xmm12 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm7 = _mm_add_ps(xmm13, xmm7);
                    xmm13 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                if (u_nrb > 4) _mm_prefetch((const char*)(b_ptr + 3 * cacheline_elts + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 4) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm8 = _mm_add_ps(xmm12, xmm8);
                    xmm12 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm9 = _mm_add_ps(xmm13, xmm9);
                    xmm13 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                if (u_nrb > 5) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm10 = _mm_add_ps(xmm12, xmm10);
                    xmm12 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm11 = _mm_add_ps(xmm13, xmm11);
                    xmm13 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                b_ptr += u_k * u_n;
            } while (kl > 0);
        }
        kl += prf_c_lduk;
        if (kl > 0) {
            // prefetch c
            if (do_prefetch_next_b) _mm_prefetch((const char*)(next_b_ptr + 1 * cacheline_elts), _MM_HINT_T0);
            if (u_nrb > 0) _mm_prefetch((const char*)(c_ptr + 0 * cacheline_elts), _MM_HINT_T0);
            if (u_nrb > 2) _mm_prefetch((const char*)(c_ptr + 1 * cacheline_elts), _MM_HINT_T0);
            if (u_nrb > 4) _mm_prefetch((const char*)(c_ptr + 2 * cacheline_elts), _MM_HINT_T0);
            do {
                xmm14 = _mm_set1_ps(a_ptr[1]);
                if (u_nrb > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 0 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 0) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm0 = _mm_add_ps(xmm12, xmm0);
                    xmm12 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm1 = _mm_add_ps(xmm13, xmm1);
                    xmm13 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 1) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm2 = _mm_add_ps(xmm12, xmm2);
                    xmm12 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm3 = _mm_add_ps(xmm13, xmm3);
                    xmm13 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 2) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 0 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 2) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm4 = _mm_add_ps(xmm12, xmm4);
                    xmm12 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm5 = _mm_add_ps(xmm13, xmm5);
                    xmm13 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 3) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm6 = _mm_add_ps(xmm12, xmm6);
                    xmm12 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm7 = _mm_add_ps(xmm13, xmm7);
                    xmm13 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 4) _mm_prefetch((const char*)(b_ptr + 3 * cacheline_elts + 0 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 4) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm8 = _mm_add_ps(xmm12, xmm8);
                    xmm12 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm9 = _mm_add_ps(xmm13, xmm9);
                    xmm13 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 5) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm10 = _mm_add_ps(xmm12, xmm10);
                    xmm12 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm11 = _mm_add_ps(xmm13, xmm11);
                    xmm13 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                xmm15 = _mm_set1_ps(a_ptr[2]);
                if (u_nrb > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 0) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm0 = _mm_add_ps(xmm12, xmm0);
                    xmm12 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm1 = _mm_add_ps(xmm13, xmm1);
                    xmm13 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 1) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm2 = _mm_add_ps(xmm12, xmm2);
                    xmm12 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm3 = _mm_add_ps(xmm13, xmm3);
                    xmm13 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 2) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 2) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm4 = _mm_add_ps(xmm12, xmm4);
                    xmm12 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm5 = _mm_add_ps(xmm13, xmm5);
                    xmm13 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 3) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm6 = _mm_add_ps(xmm12, xmm6);
                    xmm12 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm7 = _mm_add_ps(xmm13, xmm7);
                    xmm13 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 4) _mm_prefetch((const char*)(b_ptr + 3 * cacheline_elts + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 4) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm8 = _mm_add_ps(xmm12, xmm8);
                    xmm12 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm9 = _mm_add_ps(xmm13, xmm9);
                    xmm13 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 5) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm10 = _mm_add_ps(xmm12, xmm10);
                    xmm12 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm11 = _mm_add_ps(xmm13, xmm11);
                    xmm13 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                kl -= 1;
                xmm14 = _mm_set1_ps(a_ptr[3]);
                if (u_nrb > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 0) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm0 = _mm_add_ps(xmm12, xmm0);
                    xmm12 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm1 = _mm_add_ps(xmm13, xmm1);
                    xmm13 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 1) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm2 = _mm_add_ps(xmm12, xmm2);
                    xmm12 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm3 = _mm_add_ps(xmm13, xmm3);
                    xmm13 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 2) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 2) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm4 = _mm_add_ps(xmm12, xmm4);
                    xmm12 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm5 = _mm_add_ps(xmm13, xmm5);
                    xmm13 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 3) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm6 = _mm_add_ps(xmm12, xmm6);
                    xmm12 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm7 = _mm_add_ps(xmm13, xmm7);
                    xmm13 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 4) _mm_prefetch((const char*)(b_ptr + 3 * cacheline_elts + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 4) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm8 = _mm_add_ps(xmm12, xmm8);
                    xmm12 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm9 = _mm_add_ps(xmm13, xmm9);
                    xmm13 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 5) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm10 = _mm_add_ps(xmm12, xmm10);
                    xmm12 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm11 = _mm_add_ps(xmm13, xmm11);
                    xmm13 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                xmm15 = _mm_set1_ps(a_ptr[4]);
                _mm_prefetch((const char*)(a_ptr + 0 * cacheline_elts + prefetch_a_offset), _MM_HINT_T0);
                a_ptr += u_k;
                if (u_nrb > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 0) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm0 = _mm_add_ps(xmm12, xmm0);
                    xmm12 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm1 = _mm_add_ps(xmm13, xmm1);
                    xmm13 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                if (u_nrb > 1) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm2 = _mm_add_ps(xmm12, xmm2);
                    xmm12 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm3 = _mm_add_ps(xmm13, xmm3);
                    xmm13 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                if (u_nrb > 2) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 2) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm4 = _mm_add_ps(xmm12, xmm4);
                    xmm12 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm5 = _mm_add_ps(xmm13, xmm5);
                    xmm13 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                if (u_nrb > 3) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm6 = _mm_add_ps(xmm12, xmm6);
                    xmm12 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm7 = _mm_add_ps(xmm13, xmm7);
                    xmm13 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                if (u_nrb > 4) _mm_prefetch((const char*)(b_ptr + 3 * cacheline_elts + 3 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 4) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm8 = _mm_add_ps(xmm12, xmm8);
                    xmm12 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm9 = _mm_add_ps(xmm13, xmm9);
                    xmm13 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                if (u_nrb > 5) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm10 = _mm_add_ps(xmm12, xmm10);
                    xmm12 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 0 * N_REG_ELTS + 3 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm11 = _mm_add_ps(xmm13, xmm11);
                    xmm13 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 1 * N_REG_ELTS + 3 * u_n);
                }
                b_ptr += u_k * u_n;
            } while (kl > 0);
        }

        kl = k & 3;
        if (kl > 0) {
            {
                xmm14 = _mm_set1_ps(a_ptr[1]);
                if (u_nrb > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 0 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 0) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm0 = _mm_add_ps(xmm12, xmm0);
                    xmm12 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm1 = _mm_add_ps(xmm13, xmm1);
                    xmm13 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 1) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm2 = _mm_add_ps(xmm12, xmm2);
                    xmm12 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm3 = _mm_add_ps(xmm13, xmm3);
                    xmm13 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 2) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 0 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 2) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm4 = _mm_add_ps(xmm12, xmm4);
                    xmm12 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm5 = _mm_add_ps(xmm13, xmm5);
                    xmm13 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 3) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm6 = _mm_add_ps(xmm12, xmm6);
                    xmm12 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm7 = _mm_add_ps(xmm13, xmm7);
                    xmm13 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 4) _mm_prefetch((const char*)(b_ptr + 3 * cacheline_elts + 0 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 4) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm8 = _mm_add_ps(xmm12, xmm8);
                    xmm12 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm9 = _mm_add_ps(xmm13, xmm9);
                    xmm13 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
                if (u_nrb > 5) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm10 = _mm_add_ps(xmm12, xmm10);
                    xmm12 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm11 = _mm_add_ps(xmm13, xmm11);
                    xmm13 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
                }
            }
        }
        if (kl > 1) {
            {
                xmm15 = _mm_set1_ps(a_ptr[2]);
                if (u_nrb > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 0) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm0 = _mm_add_ps(xmm12, xmm0);
                    xmm12 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm1 = _mm_add_ps(xmm13, xmm1);
                    xmm13 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 1) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm2 = _mm_add_ps(xmm12, xmm2);
                    xmm12 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm3 = _mm_add_ps(xmm13, xmm3);
                    xmm13 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 2) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 2) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm4 = _mm_add_ps(xmm12, xmm4);
                    xmm12 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm5 = _mm_add_ps(xmm13, xmm5);
                    xmm13 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 3) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm6 = _mm_add_ps(xmm12, xmm6);
                    xmm12 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm7 = _mm_add_ps(xmm13, xmm7);
                    xmm13 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 4) _mm_prefetch((const char*)(b_ptr + 3 * cacheline_elts + 1 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 4) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm8 = _mm_add_ps(xmm12, xmm8);
                    xmm12 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm9 = _mm_add_ps(xmm13, xmm9);
                    xmm13 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
                if (u_nrb > 5) {
                    xmm12 = _mm_mul_ps(xmm14, xmm12);
                    xmm10 = _mm_add_ps(xmm12, xmm10);
                    xmm12 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 0 * N_REG_ELTS + 1 * u_n);
                    xmm13 = _mm_mul_ps(xmm14, xmm13);
                    xmm11 = _mm_add_ps(xmm13, xmm11);
                    xmm13 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 1 * N_REG_ELTS + 1 * u_n);
                }
            }
        }
        if (kl > 2) {
            {
                if (u_nrb > 0) _mm_prefetch((const char*)(b_ptr + 0 * cacheline_elts + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 0) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm0 = _mm_add_ps(xmm12, xmm0);
                    xmm12 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm1 = _mm_add_ps(xmm13, xmm1);
                    xmm13 = _mm_loadu_ps(b_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 1) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm2 = _mm_add_ps(xmm12, xmm2);
                    xmm12 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm3 = _mm_add_ps(xmm13, xmm3);
                    xmm13 = _mm_loadu_ps(b_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 2) _mm_prefetch((const char*)(b_ptr + 1 * cacheline_elts + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 2) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm4 = _mm_add_ps(xmm12, xmm4);
                    xmm12 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm5 = _mm_add_ps(xmm13, xmm5);
                    xmm13 = _mm_loadu_ps(b_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 3) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm6 = _mm_add_ps(xmm12, xmm6);
                    xmm12 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm7 = _mm_add_ps(xmm13, xmm7);
                    xmm13 = _mm_loadu_ps(b_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 4) _mm_prefetch((const char*)(b_ptr + 3 * cacheline_elts + 2 * u_n + prefetch_b_offset), _MM_HINT_T0);
                if (u_nrb > 4) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm8 = _mm_add_ps(xmm12, xmm8);
                    xmm12 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm9 = _mm_add_ps(xmm13, xmm9);
                    xmm13 = _mm_loadu_ps(b_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
                if (u_nrb > 5) {
                    xmm12 = _mm_mul_ps(xmm15, xmm12);
                    xmm10 = _mm_add_ps(xmm12, xmm10);
                    xmm12 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 0 * N_REG_ELTS + 2 * u_n);
                    xmm13 = _mm_mul_ps(xmm15, xmm13);
                    xmm11 = _mm_add_ps(xmm13, xmm11);
                    xmm13 = _mm_loadu_ps(b_ptr + 6 * N_REGB_ELTS + 1 * N_REG_ELTS + 2 * u_n);
                }
            }
        }
        _mm_prefetch((const char*)(a_ptr + 0 * cacheline_elts + prefetch_a_offset), _MM_HINT_T0);
        a_ptr += kl;

        if (do_prefetch_next_b) _mm_prefetch((const char*)(next_b_ptr + 2 * cacheline_elts), _MM_HINT_T0);
        xmm12 = _mm_set1_ps(kp.pick<float>(gemm_kernel_fp32_sse::param_def::ALPHA_IDX));
        xmm13 = _mm_set1_ps(kp.pick<float>(gemm_kernel_fp32_sse::param_def::BETA_IDX));
        b_ptr = kp.pick<const float*>(gemm_kernel_fp32_sse::param_def::B_PTR_IDX);
        prf_c_lduk = kp.pick<int64_t>(gemm_kernel_fp32_sse::param_def::PRF_C_LDK_IDX) >> u_k_log2;

        // *= alpha
        if (u_nrb > 0) {
            xmm0 = _mm_mul_ps(xmm12, xmm0);
            xmm1 = _mm_mul_ps(xmm12, xmm1);
        }
        if (u_nrb > 1) {
            xmm2 = _mm_mul_ps(xmm12, xmm2);
            xmm3 = _mm_mul_ps(xmm12, xmm3);
        }
        if (u_nrb > 2) {
            xmm4 = _mm_mul_ps(xmm12, xmm4);
            xmm5 = _mm_mul_ps(xmm12, xmm5);
        }
        if (u_nrb > 3) {
            xmm6 = _mm_mul_ps(xmm12, xmm6);
            xmm7 = _mm_mul_ps(xmm12, xmm7);
        }
        if (u_nrb > 4) {
            xmm8 = _mm_mul_ps(xmm12, xmm8);
            xmm9 = _mm_mul_ps(xmm12, xmm9);
        }
        if (u_nrb > 5) {
            xmm10 = _mm_mul_ps(xmm12, xmm10);
            xmm11 = _mm_mul_ps(xmm12, xmm11);
        }
        xmm12 = _mm_set1_ps(kp.pick<float>(gemm_kernel_fp32_sse::param_def::BETA_BIAS_IDX));

        if (flags & gemm_kernel_fp32_sse::flag::WITH_SUM) {
            xmm14 = _mm_set1_ps(kp.pick<float>(gemm_kernel_fp32_sse::param_def::BETA_SUM_IDX));
            auto sum_ptr = kp.pick<const float*>(gemm_kernel_fp32_sse::param_def::SUM_PTR_IDX);
            float *masked_sum;
            if (need_mask) {
                auto offset_sum = sum_ptr + mask_off;
                masked_sum = &kp.pick<float>(gemm_kernel_fp32_sse::param_def::MASKED_SUM_IDX);
                auto i = mask;
                do {
                    i -= 1;
                    masked_sum[i] = offset_sum[i];
                } while (i);
            }
            if (u_nrb > 0) {
                if (u_nrb == 1 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 0 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 0 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm14);
                xmm0 = _mm_add_ps(xmm15, xmm0);
                if (u_nrb == 1 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 1 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 0 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm14);
                xmm1 = _mm_add_ps(xmm15, xmm1);
            }
            if (u_nrb > 1) {
                if (u_nrb == 2 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 0 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm14);
                xmm2 = _mm_add_ps(xmm15, xmm2);
                if (u_nrb == 2 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 1 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm14);
                xmm3 = _mm_add_ps(xmm15, xmm3);
            }
            if (u_nrb > 2) {
                if (u_nrb == 3 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 0 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm14);
                xmm4 = _mm_add_ps(xmm15, xmm4);
                if (u_nrb == 3 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 1 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm14);
                xmm5 = _mm_add_ps(xmm15, xmm5);
            }
            if (u_nrb > 3) {
                if (u_nrb == 4 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 0 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm14);
                xmm6 = _mm_add_ps(xmm15, xmm6);
                if (u_nrb == 4 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 1 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm14);
                xmm7 = _mm_add_ps(xmm15, xmm7);
            }
            if (u_nrb > 4) {
                if (u_nrb == 5 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 0 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm14);
                xmm8 = _mm_add_ps(xmm15, xmm8);
                if (u_nrb == 5 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 1 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm14);
                xmm9 = _mm_add_ps(xmm15, xmm9);
            }
            if (u_nrb > 5) {
                if (u_nrb == 6 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 0 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm14);
                xmm10 = _mm_add_ps(xmm15, xmm10);
                if (u_nrb == 6 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_sum + 1 * N_REG_ELTS), xmm14);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(sum_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm14);
                xmm11 = _mm_add_ps(xmm15, xmm11);
            }
            auto ldsum = kp.pick<const int64_t>(gemm_kernel_fp32_sse::param_def::LDSUM_IDX);
            sum_ptr += ldsum;
            kp.pick<const float*>(gemm_kernel_fp32_sse::param_def::SUM_PTR_IDX) = sum_ptr;
        }

        if (flags & gemm_kernel_fp32_sse::flag::LOAD_C) {
            float *masked_c;
            if (need_mask) {
                auto offset_c = c_ptr + mask_off;
                masked_c = &kp.pick<float>(gemm_kernel_fp32_sse::param_def::MASKED_C_IDX);
                auto i = mask;
                do {
                    i -= 1;
                    masked_c[i] = offset_c[i];
                } while (i);
            }
            if (u_nrb > 0) {
                if (u_nrb == 1 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 0 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 0 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm13);
                xmm0 = _mm_add_ps(xmm15, xmm0);
                if (u_nrb == 1 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 1 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 0 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm13);
                xmm1 = _mm_add_ps(xmm15, xmm1);
            }
            if (u_nrb > 1) {
                if (u_nrb == 2 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 0 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm13);
                xmm2 = _mm_add_ps(xmm15, xmm2);
                if (u_nrb == 2 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 1 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm13);
                xmm3 = _mm_add_ps(xmm15, xmm3);
            }
            if (u_nrb > 2) {
                if (u_nrb == 3 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 0 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm13);
                xmm4 = _mm_add_ps(xmm15, xmm4);
                if (u_nrb == 3 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 1 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm13);
                xmm5 = _mm_add_ps(xmm15, xmm5);
            }
            if (u_nrb > 3) {
                if (u_nrb == 4 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 0 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm13);
                xmm6 = _mm_add_ps(xmm15, xmm6);
                if (u_nrb == 4 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 1 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm13);
                xmm7 = _mm_add_ps(xmm15, xmm7);
            }
            if (u_nrb > 4) {
                if (u_nrb == 5 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 0 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm13);
                xmm8 = _mm_add_ps(xmm15, xmm8);
                if (u_nrb == 5 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 1 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm13);
                xmm9 = _mm_add_ps(xmm15, xmm9);
            }
            if (u_nrb > 5) {
                if (u_nrb == 6 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 0 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm13);
                xmm10 = _mm_add_ps(xmm15, xmm10);
                if (u_nrb == 6 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_c + 1 * N_REG_ELTS), xmm13);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(c_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm13);
                xmm11 = _mm_add_ps(xmm15, xmm11);
            }
        }

        auto bias_ptr = kp.pick<const float*>(gemm_kernel_fp32_sse::param_def::BIAS_PTR_IDX);
        float *masked_bias;
        if (need_mask) masked_bias = &kp.pick<float>(gemm_kernel_fp32_sse::param_def::MASKED_BIAS_IDX);
        if (flags & gemm_kernel_fp32_sse::flag::ROW_BIAS) {
            if (u_nrb > 0) {
                if (u_nrb == 1 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 0 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 0 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm12);
                xmm0 = _mm_add_ps(xmm15, xmm0);
                if (u_nrb == 1 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 1 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 0 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm12);
                xmm1 = _mm_add_ps(xmm15, xmm1);
            }
            if (u_nrb > 1) {
                if (u_nrb == 2 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 0 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm12);
                xmm2 = _mm_add_ps(xmm15, xmm2);
                if (u_nrb == 2 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 1 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm12);
                xmm3 = _mm_add_ps(xmm15, xmm3);
            }
            if (u_nrb > 2) {
                if (u_nrb == 3 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 0 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm12);
                xmm4 = _mm_add_ps(xmm15, xmm4);
                if (u_nrb == 3 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 1 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm12);
                xmm5 = _mm_add_ps(xmm15, xmm5);
            }
            if (u_nrb > 3) {
                if (u_nrb == 4 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 0 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm12);
                xmm6 = _mm_add_ps(xmm15, xmm6);
                if (u_nrb == 4 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 1 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm12);
                xmm7 = _mm_add_ps(xmm15, xmm7);
            }
            if (u_nrb > 4) {
                if (u_nrb == 5 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 0 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm12);
                xmm8 = _mm_add_ps(xmm15, xmm8);
                if (u_nrb == 5 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 1 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm12);
                xmm9 = _mm_add_ps(xmm15, xmm9);
            }
            if (u_nrb > 5) {
                if (u_nrb == 6 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 0 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS), xmm12);
                xmm10 = _mm_add_ps(xmm15, xmm10);
                if (u_nrb == 6 && need_mask) xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(masked_bias + 1 * N_REG_ELTS), xmm12);
                else xmm15 = _mm_mul_ps(xmm15 = _mm_loadu_ps(bias_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS), xmm12);
                xmm11 = _mm_add_ps(xmm15, xmm11);
            }
        }
        if (flags & (gemm_kernel_fp32_sse::flag::SCA_BIAS | gemm_kernel_fp32_sse::flag::COL_BIAS)) {
            xmm15 = _mm_set1_ps(bias_ptr[0]);
            xmm15 = _mm_mul_ps(xmm13, xmm12);
            if (u_nrb > 0) {
                xmm0 = _mm_add_ps(xmm15, xmm0);
                xmm1 = _mm_add_ps(xmm15, xmm1);
            }
            if (u_nrb > 1) {
                xmm2 = _mm_add_ps(xmm15, xmm2);
                xmm3 = _mm_add_ps(xmm15, xmm3);
            }
            if (u_nrb > 2) {
                xmm4 = _mm_add_ps(xmm15, xmm4);
                xmm5 = _mm_add_ps(xmm15, xmm5);
            }
            if (u_nrb > 3) {
                xmm6 = _mm_add_ps(xmm15, xmm6);
                xmm7 = _mm_add_ps(xmm15, xmm7);
            }
            if (u_nrb > 4) {
                xmm8 = _mm_add_ps(xmm15, xmm8);
                xmm9 = _mm_add_ps(xmm15, xmm9);
            }
            if (u_nrb > 5) {
                xmm10 = _mm_add_ps(xmm15, xmm10);
                xmm11 = _mm_add_ps(xmm15, xmm11);
            }
        }
        if (flags & gemm_kernel_fp32_sse::flag::COL_BIAS) {
            bias_ptr += 1;
            kp.pick<const float*>(gemm_kernel_fp32_sse::param_def::BIAS_PTR_IDX) = bias_ptr;
        }

        if (flags & (gemm_kernel_fp32_sse::flag::RELU | gemm_kernel_fp32_sse::flag::RELU6)) {
            auto six = &kp.pick<float>(gemm_kernel_fp32_sse::param_def::SIX_IDX);
            xmm13 = _mm_setzero_ps();
            xmm14 = _mm_loadu_ps(six);
            if (u_nrb > 0) {
                xmm0 = _mm_max_ps(xmm13, xmm0);
                xmm1 = _mm_max_ps(xmm13, xmm1);
            }
            if (u_nrb > 1) {
                xmm2 = _mm_max_ps(xmm13, xmm2);
                xmm3 = _mm_max_ps(xmm13, xmm3);
            }
            if (u_nrb > 2) {
                xmm4 = _mm_max_ps(xmm13, xmm4);
                xmm5 = _mm_max_ps(xmm13, xmm5);
            }
            if (u_nrb > 3) {
                xmm6 = _mm_max_ps(xmm13, xmm6);
                xmm7 = _mm_max_ps(xmm13, xmm7);
            }
            if (u_nrb > 4) {
                xmm8 = _mm_max_ps(xmm13, xmm8);
                xmm9 = _mm_max_ps(xmm13, xmm9);
            }
            if (u_nrb > 5) {
                xmm10 = _mm_max_ps(xmm13, xmm10);
                xmm11 = _mm_max_ps(xmm13, xmm11);
            }
        }

        if (flags & gemm_kernel_fp32_sse::flag::RELU6) {
            if (u_nrb > 0) {
                xmm0 = _mm_min_ps(xmm14, xmm0);
                xmm1 = _mm_min_ps(xmm14, xmm1);
            }
            if (u_nrb > 1) {
                xmm2 = _mm_min_ps(xmm14, xmm2);
                xmm3 = _mm_min_ps(xmm14, xmm3);
            }
            if (u_nrb > 2) {
                xmm4 = _mm_min_ps(xmm14, xmm4);
                xmm5 = _mm_min_ps(xmm14, xmm5);
            }
            if (u_nrb > 3) {
                xmm6 = _mm_min_ps(xmm14, xmm6);
                xmm7 = _mm_min_ps(xmm14, xmm7);
            }
            if (u_nrb > 4) {
                xmm8 = _mm_min_ps(xmm14, xmm8);
                xmm9 = _mm_min_ps(xmm14, xmm9);
            }
            if (u_nrb > 5) {
                xmm10 = _mm_min_ps(xmm14, xmm10);
                xmm11 = _mm_min_ps(xmm14, xmm11);
            }
        }

        xmm12 = _mm_loadu_ps(b_ptr + 0 * N_REGB_ELTS + 0 * N_REG_ELTS + 0 * u_n);
        xmm13 = _mm_loadu_ps(b_ptr + 0 * N_REGB_ELTS + 1 * N_REG_ELTS + 0 * u_n);
        xmm15 = _mm_set1_ps(a_ptr[0]);
        if (do_prefetch_next_b) next_b_ptr += 3 * cacheline_elts;

        float *masked_c;
        if (need_mask) masked_c = &kp.pick<float>(gemm_kernel_fp32_sse::param_def::MASKED_C_IDX);
        if (u_nrb > 0) {
            if (u_nrb == 1 && need_mask) _mm_storeu_ps(masked_c + 0 * N_REG_ELTS, xmm0);
            else _mm_storeu_ps(c_ptr + 0 * N_REGB_ELTS + 0 * N_REG_ELTS, xmm0);
            if (u_nrb == 1 && need_mask) _mm_storeu_ps(masked_c + 1 * N_REG_ELTS, xmm1);
            else _mm_storeu_ps(c_ptr + 0 * N_REGB_ELTS + 1 * N_REG_ELTS, xmm1);
        }
        if (u_nrb > 1) {
            if (u_nrb == 2 && need_mask) _mm_storeu_ps(masked_c + 0 * N_REG_ELTS, xmm2);
            else _mm_storeu_ps(c_ptr + 1 * N_REGB_ELTS + 0 * N_REG_ELTS, xmm2);
            if (u_nrb == 2 && need_mask) _mm_storeu_ps(masked_c + 1 * N_REG_ELTS, xmm3);
            else _mm_storeu_ps(c_ptr + 1 * N_REGB_ELTS + 1 * N_REG_ELTS, xmm3);
        }
        if (u_nrb > 2) {
            if (u_nrb == 3 && need_mask) _mm_storeu_ps(masked_c + 0 * N_REG_ELTS, xmm4);
            else _mm_storeu_ps(c_ptr + 2 * N_REGB_ELTS + 0 * N_REG_ELTS, xmm4);
            if (u_nrb == 3 && need_mask) _mm_storeu_ps(masked_c + 1 * N_REG_ELTS, xmm5);
            else _mm_storeu_ps(c_ptr + 2 * N_REGB_ELTS + 1 * N_REG_ELTS, xmm5);
        }
        if (u_nrb > 3) {
            if (u_nrb == 4 && need_mask) _mm_storeu_ps(masked_c + 0 * N_REG_ELTS, xmm6);
            else _mm_storeu_ps(c_ptr + 3 * N_REGB_ELTS + 0 * N_REG_ELTS, xmm6);
            if (u_nrb == 4 && need_mask) _mm_storeu_ps(masked_c + 1 * N_REG_ELTS, xmm7);
            else _mm_storeu_ps(c_ptr + 3 * N_REGB_ELTS + 1 * N_REG_ELTS, xmm7);
        }
        if (u_nrb > 4) {
            if (u_nrb == 5 && need_mask) _mm_storeu_ps(masked_c + 0 * N_REG_ELTS, xmm8);
            else _mm_storeu_ps(c_ptr + 4 * N_REGB_ELTS + 0 * N_REG_ELTS, xmm8);
            if (u_nrb == 5 && need_mask) _mm_storeu_ps(masked_c + 1 * N_REG_ELTS, xmm9);
            else _mm_storeu_ps(c_ptr + 4 * N_REGB_ELTS + 1 * N_REG_ELTS, xmm9);
        }
        if (u_nrb > 5) {
            if (u_nrb == 6 && need_mask) _mm_storeu_ps(masked_c + 0 * N_REG_ELTS, xmm10);
            else _mm_storeu_ps(c_ptr + 5 * N_REGB_ELTS + 0 * N_REG_ELTS, xmm10);
            if (u_nrb == 6 && need_mask) _mm_storeu_ps(masked_c + 1 * N_REG_ELTS, xmm11);
            else _mm_storeu_ps(c_ptr + 5 * N_REGB_ELTS + 1 * N_REG_ELTS, xmm11);
        }
        if (need_mask) {
            auto offset_c = c_ptr + mask_off;
            auto i = mask;
            do {
                i -= 1;
                offset_c[i] = masked_c[i];
            } while (i);
        }
        c_ptr += ldc;

        m -= 1;
        if (do_prefetch_next_b) _mm_prefetch((const char*)(next_b_ptr + 0 * cacheline_elts), _MM_HINT_T0);
        if (u_nrb > 0) {
            xmm0 = _mm_setzero_ps();
            xmm1 = _mm_setzero_ps();
        }
        if (u_nrb > 1) {
            xmm2 = _mm_setzero_ps();
            xmm3 = _mm_setzero_ps();
        }
        if (u_nrb > 2) {
            xmm4 = _mm_setzero_ps();
            xmm5 = _mm_setzero_ps();
        }
        if (u_nrb > 3) {
            xmm6 = _mm_setzero_ps();
            xmm7 = _mm_setzero_ps();
        }
        if (u_nrb > 4) {
            xmm8 = _mm_setzero_ps();
            xmm9 = _mm_setzero_ps();
        }
        if (u_nrb > 5) {
            xmm10 = _mm_setzero_ps();
            xmm11 = _mm_setzero_ps();
        }
    } while (m > 0);
}

#define GEMM_KERNEL_FP32_SSE_TABLE_BLK(NEED_MASK) \
{\
    {gemm_m1n48_kernel_fp32_sse<NEED_MASK, 1 * gemm_kernel_fp32_sse::config::N_REGB_ELTS>,},\
    {gemm_m1n48_kernel_fp32_sse<NEED_MASK, 2 * gemm_kernel_fp32_sse::config::N_REGB_ELTS>,},\
    {gemm_m1n48_kernel_fp32_sse<NEED_MASK, 3 * gemm_kernel_fp32_sse::config::N_REGB_ELTS>,},\
    {gemm_m1n48_kernel_fp32_sse<NEED_MASK, 4 * gemm_kernel_fp32_sse::config::N_REGB_ELTS>,},\
    {gemm_m1n48_kernel_fp32_sse<NEED_MASK, 5 * gemm_kernel_fp32_sse::config::N_REGB_ELTS>,},\
    {gemm_m1n48_kernel_fp32_sse<NEED_MASK, 6 * gemm_kernel_fp32_sse::config::N_REGB_ELTS>,},\
}\

const gemm_kernel_fp32_sse::func_t
    gemm_kernel_fp32_sse::table_[config::NEED_MASK_OPT][config::MAX_N_REGBS][config::MAX_M_REGS] =
{
    GEMM_KERNEL_FP32_SSE_TABLE_BLK(0),
    GEMM_KERNEL_FP32_SSE_TABLE_BLK(1),
};

}}}; // namespace ppl::kernel::x86
