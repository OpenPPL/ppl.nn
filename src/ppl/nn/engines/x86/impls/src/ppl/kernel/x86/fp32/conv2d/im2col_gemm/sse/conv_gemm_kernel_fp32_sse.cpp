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

#include "ppl/kernel/x86/fp32/conv2d/im2col_gemm/sse/conv_gemm_kernel_fp32_sse.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template <int32_t u_n>
void conv_gemm_kernel_fp32_sse_kernel_core(
    int64_t *param)
{
    static float six[4] = {6.0f, 6.0f, 6.0f, 6.0f};
    __asm__ __volatile__ (
        ".equ P_BYTES, 8\n"
        ".equ D_BYTES, 4\n"

        ".equ A_PTR_IDX, (0 * P_BYTES)\n"
        ".equ PACKED_B_PTR_IDX, (1 * P_BYTES)\n"
        ".equ C_PTR_IDX, (2 * P_BYTES)\n"
        ".equ V_PTR_IDX, (3 * P_BYTES)\n"
        ".equ H_PTR_IDX, (4 * P_BYTES)\n"
        ".equ M_IDX, (5 * P_BYTES)\n"
        ".equ K_IDX, (6 * P_BYTES)\n"
        ".equ LDA_IDX, (7 * P_BYTES)\n"
        ".equ LDPACKED_B_IDX, (8 * P_BYTES)\n"
        ".equ LDC_IDX, (9 * P_BYTES)\n"
        ".equ LDH_IDX, (10 * P_BYTES)\n"
        ".equ FLAGS_IDX, (11 * P_BYTES)\n"

        ".equ N_REGB_ELTS, %c[N_REGB_ELTS]\n"
        ".equ N_REG_ELTS, %c[N_REG_ELTS]\n"
        ".equ U_N, %c[U_N]\n"
        ".equ U_NB, ((U_N + N_REGB_ELTS - 1) / N_REGB_ELTS)\n"
        ".equ U_K, %c[U_K]\n"
        ".equ KERNEL_FLAG_LOAD_H, %c[KERNEL_FLAG_LOAD_H]\n"
        ".equ KERNEL_FLAG_ADD_V, %c[KERNEL_FLAG_ADD_V]\n"
        ".equ KERNEL_FLAG_RELU, %c[KERNEL_FLAG_RELU]\n"
        ".equ KERNEL_FLAG_RELU6, %c[KERNEL_FLAG_RELU6]\n"

        "mov LDA_IDX(%[param]), %%r8\n"
        "mov LDPACKED_B_IDX(%[param]), %%r9\n"
        "mov FLAGS_IDX(%[param]), %%r11\n"
        "mov A_PTR_IDX(%[param]), %%r12\n"
        "mov H_PTR_IDX(%[param]), %%r13\n"
        "mov C_PTR_IDX(%[param]), %%r14\n"
        "mov M_IDX(%[param]), %%r15\n"
"1:\n" // label_init_session
        "test $KERNEL_FLAG_LOAD_H, %%r11\n"
        "jz 2f\n" // label_init_zero
        "mov LDH_IDX(%[param]), %%rax\n"
        ".if U_NB > 0\n"
        "movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm0\n"
        "movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm1\n"
        ".endif\n"
        ".if U_NB > 1\n"
        "movups ((1 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm2\n"
        "movups ((1 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm3\n"
        ".endif\n"
        ".if U_NB > 2\n"
        "movups ((2 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm4\n"
        "movups ((2 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm5\n"
        ".endif\n"
        ".if U_NB > 3\n"
        "movups ((3 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm6\n"
        "movups ((3 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm7\n"
        ".endif\n"
        ".if U_NB > 4\n"
        "movups ((4 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm8\n"
        "movups ((4 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm9\n"
        ".endif\n"
        ".if U_NB > 5\n"
        "movups ((5 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm10\n"
        "movups ((5 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r13), %%xmm11\n"
        ".endif\n"
        "lea (%%r13, %%rax, D_BYTES), %%r13\n"
        "jmp 4f\n" // label_compute_session
"2:\n" // label_init_zero
        ".if U_NB > 0\n"
        "xorps %%xmm0, %%xmm0\n"
        "xorps %%xmm1, %%xmm1\n"
        ".endif\n"
        ".if U_NB > 1\n"
        "xorps %%xmm2, %%xmm2\n"
        "xorps %%xmm3, %%xmm3\n"
        ".endif\n"
        ".if U_NB > 2\n"
        "xorps %%xmm4, %%xmm4\n"
        "xorps %%xmm5, %%xmm5\n"
        ".endif\n"
        ".if U_NB > 3\n"
        "xorps %%xmm6, %%xmm6\n"
        "xorps %%xmm7, %%xmm7\n"
        ".endif\n"
        ".if U_NB > 4\n"
        "xorps %%xmm8, %%xmm8\n"
        "xorps %%xmm9, %%xmm9\n"
        ".endif\n"
        ".if U_NB > 5\n"
        "xorps %%xmm10, %%xmm10\n"
        "xorps %%xmm11, %%xmm11\n"
        ".endif\n"

"4:\n" // label_compute_session
        "mov %%r12, %%rax\n"
        "mov PACKED_B_PTR_IDX(%[param]), %%rbx\n"
        "mov K_IDX(%[param]), %%r10\n"
        ".if U_NB > 3\n"
        "lea (%%rbx, %%r9, 2 * D_BYTES), %%rcx\n"
        "lea (%%rcx, %%r9, 1 * D_BYTES), %%rcx\n"
        ".endif\n"
        "cmp $U_K, %%r10\n"
        "jl 6f\n" // label_k_remain
        PPL_X86_INLINE_ASM_ALIGN()
"5:\n" // label_k_body
        ".irp K,0,1,2,3,4,5,6,7\n"
        "movss (\\K * D_BYTES)(%%rax), %%xmm12\n"
        "shufps $0, %%xmm12, %%xmm12\n"
        ".if U_NB > 0\n"
        "movups ((\\K * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rbx), %%xmm14\n"
        "movups ((\\K * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rbx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm0\n"
        "addps %%xmm15, %%xmm1\n"
        ".endif\n"
        ".if U_NB > 1\n"
        "movups ((\\K * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rbx, %%r9, 1 * D_BYTES), %%xmm14\n"
        "movups ((\\K * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rbx, %%r9, 1 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm2\n"
        "addps %%xmm15, %%xmm3\n"
        ".endif\n"
        ".if U_NB > 2\n"
        "movups ((\\K * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rbx, %%r9, 2 * D_BYTES), %%xmm14\n"
        "movups ((\\K * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rbx, %%r9, 2 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm4\n"
        "addps %%xmm15, %%xmm5\n"
        ".endif\n"
        ".if U_NB > 3\n"
        "movups ((\\K * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm14\n"
        "movups ((\\K * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm6\n"
        "addps %%xmm15, %%xmm7\n"
        ".endif\n"
        ".if U_NB > 4\n"
        "movups ((\\K * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx, %%r9, 1 * D_BYTES), %%xmm14\n"
        "movups ((\\K * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx, %%r9, 1 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm8\n"
        "addps %%xmm15, %%xmm9\n"
        ".endif\n"
        ".if U_NB > 5\n"
        "movups ((\\K * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx, %%r9, 2 * D_BYTES), %%xmm14\n"
        "movups ((\\K * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx, %%r9, 2 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm10\n"
        "addps %%xmm15, %%xmm11\n"
        ".endif\n"
        ".endr\n"
        ".if U_NB > 0\n lea (U_K * N_REGB_ELTS * D_BYTES)(%%rbx), %%rbx\n .endif\n"
        ".if U_NB > 3\n lea (U_K * N_REGB_ELTS * D_BYTES)(%%rcx), %%rcx\n .endif\n"
        "lea (U_K * D_BYTES)(%%rax), %%rax\n"
        "sub $U_K, %%r10\n"
        "cmp $U_K, %%r10\n"
        "jge 5b\n" // label_k_body
        "cmp $0, %%r10\n"
        "je 7f\n" // label_finalize_session
"6:\n" // label_k_remain
        "movss (0 * D_BYTES)(%%rax), %%xmm12\n"
        "shufps $0, %%xmm12, %%xmm12\n"
        ".if U_NB > 0\n"
        "movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rbx), %%xmm14\n"
        "movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rbx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm0\n"
        "addps %%xmm15, %%xmm1\n"
        ".endif\n"
        ".if U_NB > 1\n"
        "movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rbx, %%r9, 1 * D_BYTES), %%xmm14\n"
        "movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rbx, %%r9, 1 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm2\n"
        "addps %%xmm15, %%xmm3\n"
        ".endif\n"
        ".if U_NB > 2\n"
        "movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rbx, %%r9, 2 * D_BYTES), %%xmm14\n"
        "movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rbx, %%r9, 2 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm4\n"
        "addps %%xmm15, %%xmm5\n"
        ".endif\n"
        ".if U_NB > 3\n"
        "movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm14\n"
        "movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm6\n"
        "addps %%xmm15, %%xmm7\n"
        ".endif\n"
        ".if U_NB > 4\n"
        "movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx, %%r9, 1 * D_BYTES), %%xmm14\n"
        "movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx, %%r9, 1 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm8\n"
        "addps %%xmm15, %%xmm9\n"
        ".endif\n"
        ".if U_NB > 5\n"
        "movups ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rcx, %%r9, 2 * D_BYTES), %%xmm14\n"
        "movups ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rcx, %%r9, 2 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm10\n"
        "addps %%xmm15, %%xmm11\n"
        ".endif\n"
        ".if U_NB > 0\n lea (N_REGB_ELTS * D_BYTES)(%%rbx), %%rbx\n .endif\n"
        ".if U_NB > 3\n lea (N_REGB_ELTS * D_BYTES)(%%rcx), %%rcx\n .endif\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%r10\n"
        "cmp $0, %%r10\n"
        "jne 6b\n" // label_k_remain

"7:\n" // label_finalize_session
        "test $KERNEL_FLAG_ADD_V, %%r11\n"
        "jz 8f\n" // label_add_v_end
        "mov V_PTR_IDX(%[param]), %%rax\n"
        ".if U_NB > 0\n"
        "addps ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm0\n"
        "addps ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm1\n"
        ".endif\n"
        ".if U_NB > 1\n"
        "addps ((1 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm2\n"
        "addps ((1 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm3\n"
        ".endif\n"
        ".if U_NB > 2\n"
        "addps ((2 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm4\n"
        "addps ((2 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm5\n"
        ".endif\n"
        ".if U_NB > 3\n"
        "addps ((3 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm6\n"
        "addps ((3 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm7\n"
        ".endif\n"
        ".if U_NB > 4\n"
        "addps ((4 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm8\n"
        "addps ((4 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm9\n"
        ".endif\n"
        ".if U_NB > 5\n"
        "addps ((5 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm10\n"
        "addps ((5 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%rax), %%xmm11\n"
        ".endif\n"
"8:\n" // label_add_v_end
        "test $(KERNEL_FLAG_RELU | KERNEL_FLAG_RELU6), %%r11\n"
        "jz 9f\n" // label_relu_end
        "xorps %%xmm12, %%xmm12\n"
        ".if U_NB > 0\n"
        "maxps %%xmm12, %%xmm0\n"
        "maxps %%xmm12, %%xmm1\n"
        ".endif\n"
        ".if U_NB > 1\n"
        "maxps %%xmm12, %%xmm2\n"
        "maxps %%xmm12, %%xmm3\n"
        ".endif\n"
        ".if U_NB > 2\n"
        "maxps %%xmm12, %%xmm4\n"
        "maxps %%xmm12, %%xmm5\n"
        ".endif\n"
        ".if U_NB > 3\n"
        "maxps %%xmm12, %%xmm6\n"
        "maxps %%xmm12, %%xmm7\n"
        ".endif\n"
        ".if U_NB > 4\n"
        "maxps %%xmm12, %%xmm8\n"
        "maxps %%xmm12, %%xmm9\n"
        ".endif\n"
        ".if U_NB > 5\n"
        "maxps %%xmm12, %%xmm10\n"
        "maxps %%xmm12, %%xmm11\n"
        ".endif\n"
        "test $KERNEL_FLAG_RELU6, %%r11\n"
        "jz 9f\n" // label_relu_end
        "movups (%[six]), %%xmm13\n"
        ".if U_NB > 0\n"
        "minps %%xmm13, %%xmm0\n"
        "minps %%xmm13, %%xmm1\n"
        ".endif\n"
        ".if U_NB > 1\n"
        "minps %%xmm13, %%xmm2\n"
        "minps %%xmm13, %%xmm3\n"
        ".endif\n"
        ".if U_NB > 2\n"
        "minps %%xmm13, %%xmm4\n"
        "minps %%xmm13, %%xmm5\n"
        ".endif\n"
        ".if U_NB > 3\n"
        "minps %%xmm13, %%xmm6\n"
        "minps %%xmm13, %%xmm7\n"
        ".endif\n"
        ".if U_NB > 4\n"
        "minps %%xmm13, %%xmm8\n"
        "minps %%xmm13, %%xmm9\n"
        ".endif\n"
        ".if U_NB > 5\n"
        "minps %%xmm13, %%xmm10\n"
        "minps %%xmm13, %%xmm11\n"
        ".endif\n"
"9:\n" // label_relu_end
        "mov LDC_IDX(%[param]), %%rbx\n"
        ".if U_NB > 0\n"
        "movups %%xmm0, ((0 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        "movups %%xmm1, ((0 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_NB > 1\n"
        "movups %%xmm2, ((1 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        "movups %%xmm3, ((1 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_NB > 2\n"
        "movups %%xmm4, ((2 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        "movups %%xmm5, ((2 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_NB > 3\n"
        "movups %%xmm6, ((3 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        "movups %%xmm7, ((3 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_NB > 4\n"
        "movups %%xmm8, ((4 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        "movups %%xmm9, ((4 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_NB > 5\n"
        "movups %%xmm10, ((5 * N_REGB_ELTS + 0 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        "movups %%xmm11, ((5 * N_REGB_ELTS + 1 * N_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        "sub $1, %%r15\n"
        "cmp $0, %%r15\n"
        "lea (%%r12, %%r8, D_BYTES), %%r12\n"
        "lea (%%r14, %%rbx, D_BYTES), %%r14\n"
        "jne 1b\n" // label_init_session
        :
        :
        [param]                       "r" (param),
        [six]                         "r" (six),
        [N_REGB_ELTS]                 "i" (conv_gemm_kernel_fp32_sse::config::n_regb_elts),
        [N_REG_ELTS]                  "i" (conv_gemm_kernel_fp32_sse::config::n_reg_elts),
        [U_N]                         "i" (u_n),
        [U_K]                         "i" (conv_gemm_kernel_fp32_sse::config::unroll_k),
        [KERNEL_FLAG_LOAD_H]          "i" (conv_gemm_kernel_fp32_sse::flag::load_h),
        [KERNEL_FLAG_ADD_V]           "i" (conv_gemm_kernel_fp32_sse::flag::add_v),
        [KERNEL_FLAG_RELU]            "i" (conv_gemm_kernel_fp32_sse::flag::relu),
        [KERNEL_FLAG_RELU6]           "i" (conv_gemm_kernel_fp32_sse::flag::relu6)
        :
        "cc",
        "rax", "rbx", "rcx",
        "r8" , "r9" , "r10", "r11", "r12", "r13", "r14", "r15",
        "xmm0" , "xmm1" , "xmm2" , "xmm3" , "xmm4" , "xmm5" , "xmm6" , "xmm7" ,
        "xmm8" , "xmm9" , "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
        "memory"
    );
}

#endif

template <int32_t u_n>
void conv_gemm_fp32_sse_kernel(
    int64_t *param)
{

#ifdef PPL_USE_X86_INLINE_ASM
    conv_gemm_kernel_fp32_sse_kernel_core<u_n>(param);
    return;
#endif

#define K_COMPUTE_STEP(K) do {\
    xmm12 = _mm_set1_ps(ka[(K)]);\
    if (u_nb > 0) {\
        xmm14 = _mm_loadu_ps(kpacked_b_n24 + 0 * ldpacked_b + (K) * n_regb_elts + 0 * n_reg_elts);\
        xmm15 = _mm_loadu_ps(kpacked_b_n24 + 0 * ldpacked_b + (K) * n_regb_elts + 1 * n_reg_elts);\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm0 = _mm_add_ps(xmm0, xmm14);\
        xmm1 = _mm_add_ps(xmm1, xmm15);\
    }\
    if (u_nb > 1) {\
        xmm14 = _mm_loadu_ps(kpacked_b_n24 + 1 * ldpacked_b + (K) * n_regb_elts + 0 * n_reg_elts);\
        xmm15 = _mm_loadu_ps(kpacked_b_n24 + 1 * ldpacked_b + (K) * n_regb_elts + 1 * n_reg_elts);\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm2 = _mm_add_ps(xmm2, xmm14);\
        xmm3 = _mm_add_ps(xmm3, xmm15);\
    }\
    if (u_nb > 2) {\
        xmm14 = _mm_loadu_ps(kpacked_b_n24 + 2 * ldpacked_b + (K) * n_regb_elts + 0 * n_reg_elts);\
        xmm15 = _mm_loadu_ps(kpacked_b_n24 + 2 * ldpacked_b + (K) * n_regb_elts + 1 * n_reg_elts);\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm4 = _mm_add_ps(xmm4, xmm14);\
        xmm5 = _mm_add_ps(xmm5, xmm15);\
    }\
    if (u_nb > 3) {\
        xmm14 = _mm_loadu_ps(kpacked_b_n48 + 0 * ldpacked_b + (K) * n_regb_elts + 0 * n_reg_elts);\
        xmm15 = _mm_loadu_ps(kpacked_b_n48 + 0 * ldpacked_b + (K) * n_regb_elts + 1 * n_reg_elts);\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm6 = _mm_add_ps(xmm6, xmm14);\
        xmm7 = _mm_add_ps(xmm7, xmm15);\
    }\
    if (u_nb > 4) {\
        xmm14 = _mm_loadu_ps(kpacked_b_n48 + 1 * ldpacked_b + (K) * n_regb_elts + 0 * n_reg_elts);\
        xmm15 = _mm_loadu_ps(kpacked_b_n48 + 1 * ldpacked_b + (K) * n_regb_elts + 1 * n_reg_elts);\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm8 = _mm_add_ps(xmm8, xmm14);\
        xmm9 = _mm_add_ps(xmm9, xmm15);\
    }\
    if (u_nb > 5) {\
        xmm14 = _mm_loadu_ps(kpacked_b_n48 + 2 * ldpacked_b + (K) * n_regb_elts + 0 * n_reg_elts);\
        xmm15 = _mm_loadu_ps(kpacked_b_n48 + 2 * ldpacked_b + (K) * n_regb_elts + 1 * n_reg_elts);\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm10 = _mm_add_ps(xmm10, xmm14);\
        xmm11 = _mm_add_ps(xmm11, xmm15);\
    }\
} while (0)

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;

    array_param_helper kp(param);
    const int64_t n_regb_elts = conv_gemm_kernel_fp32_sse::config::n_regb_elts;
    const int64_t n_reg_elts = conv_gemm_kernel_fp32_sse::config::n_reg_elts;
    const int64_t u_nb       = div_up(u_n, n_regb_elts);
    const int64_t u_k        = 8;
    const int64_t flags      = kp.pick<const int64_t>(conv_gemm_kernel_fp32_sse::param_def::flags_idx);

    const float *a_ptr = kp.pick<const float*>(conv_gemm_kernel_fp32_sse::param_def::a_ptr_idx);
    const float *h_ptr = kp.pick<const float*>(conv_gemm_kernel_fp32_sse::param_def::h_ptr_idx);
    float *c_ptr       = kp.pick<float*>(conv_gemm_kernel_fp32_sse::param_def::c_ptr_idx);
    int64_t m          = kp.pick<int64_t>(conv_gemm_kernel_fp32_sse::param_def::m_idx);
    do {
        if (flags & conv_gemm_kernel_fp32_sse::flag::load_h) {
            if (u_nb > 0) {
                xmm0 = _mm_loadu_ps(h_ptr + 0 * n_regb_elts + 0 * n_reg_elts);
                xmm1 = _mm_loadu_ps(h_ptr + 0 * n_regb_elts + 1 * n_reg_elts);
            }
            if (u_nb > 1) {
                xmm2 = _mm_loadu_ps(h_ptr + 1 * n_regb_elts + 0 * n_reg_elts);
                xmm3 = _mm_loadu_ps(h_ptr + 1 * n_regb_elts + 1 * n_reg_elts);
            }
            if (u_nb > 2) {
                xmm4 = _mm_loadu_ps(h_ptr + 2 * n_regb_elts + 0 * n_reg_elts);
                xmm5 = _mm_loadu_ps(h_ptr + 2 * n_regb_elts + 1 * n_reg_elts);
            }
            if (u_nb > 3) {
                xmm6 = _mm_loadu_ps(h_ptr + 3 * n_regb_elts + 0 * n_reg_elts);
                xmm7 = _mm_loadu_ps(h_ptr + 3 * n_regb_elts + 1 * n_reg_elts);
            }
            if (u_nb > 4) {
                xmm8 = _mm_loadu_ps(h_ptr + 4 * n_regb_elts + 0 * n_reg_elts);
                xmm9 = _mm_loadu_ps(h_ptr + 4 * n_regb_elts + 1 * n_reg_elts);
            }
            if (u_nb > 5) {
                xmm10 = _mm_loadu_ps(h_ptr + 5 * n_regb_elts + 0 * n_reg_elts);
                xmm11 = _mm_loadu_ps(h_ptr + 5 * n_regb_elts + 1 * n_reg_elts);
            }
            h_ptr += kp.pick<const int64_t>(conv_gemm_kernel_fp32_sse::param_def::ldh_idx);
        } else {
            if (u_nb > 0) {
                xmm0 = _mm_setzero_ps();
                xmm1 = _mm_setzero_ps();
            }
            if (u_nb > 1) {
                xmm2 = _mm_setzero_ps();
                xmm3 = _mm_setzero_ps();
            }
            if (u_nb > 2) {
                xmm4 = _mm_setzero_ps();
                xmm5 = _mm_setzero_ps();
            }
            if (u_nb > 3) {
                xmm6 = _mm_setzero_ps();
                xmm7 = _mm_setzero_ps();
            }
            if (u_nb > 4) {
                xmm8 = _mm_setzero_ps();
                xmm9 = _mm_setzero_ps();
            }
            if (u_nb > 5) {
                xmm10 = _mm_setzero_ps();
                xmm11 = _mm_setzero_ps();
            }
        }
        
        const int64_t ldpacked_b = kp.pick<const int64_t>(conv_gemm_kernel_fp32_sse::param_def::ldpacked_b_idx);
        const float *kpacked_b_n24 = kp.pick<const float*>(conv_gemm_kernel_fp32_sse::param_def::packed_b_ptr_idx);
        const float *kpacked_b_n48 = u_nb > 3 ? kpacked_b_n24 + 3 * ldpacked_b : nullptr;

        const float *ka = a_ptr;
        int64_t k       = kp.pick<const int64_t>(conv_gemm_kernel_fp32_sse::param_def::k_idx);
        while (k >= u_k) {
            k -= u_k;
            K_COMPUTE_STEP(0);
            K_COMPUTE_STEP(1);
            K_COMPUTE_STEP(2);
            K_COMPUTE_STEP(3);
            K_COMPUTE_STEP(4);
            K_COMPUTE_STEP(5);
            K_COMPUTE_STEP(6);
            K_COMPUTE_STEP(7);
            if (u_nb > 0) kpacked_b_n24 += u_k * n_regb_elts;
            if (u_nb > 3) kpacked_b_n48 += u_k * n_regb_elts;
            ka += u_k;
        }
        while (k > 0) {
            k -= 1;
            K_COMPUTE_STEP(0);
            if (u_nb > 0) kpacked_b_n24 += n_regb_elts;
            if (u_nb > 3) kpacked_b_n48 += n_regb_elts;
            ka += 1;
        }

        if (flags & conv_gemm_kernel_fp32_sse::flag::add_v) {
            const float *v_ptr = kp.pick<const float*>(conv_gemm_kernel_fp32_sse::param_def::v_ptr_idx);
            if (u_nb > 0) {
                xmm0 = _mm_add_ps(_mm_loadu_ps(v_ptr + 0 * n_regb_elts + 0 * n_reg_elts), xmm0);
                xmm1 = _mm_add_ps(_mm_loadu_ps(v_ptr + 0 * n_regb_elts + 1 * n_reg_elts), xmm1);
            }
            if (u_nb > 1) {
                xmm2 = _mm_add_ps(_mm_loadu_ps(v_ptr + 1 * n_regb_elts + 0 * n_reg_elts), xmm2);
                xmm3 = _mm_add_ps(_mm_loadu_ps(v_ptr + 1 * n_regb_elts + 1 * n_reg_elts), xmm3);
            }
            if (u_nb > 2) {
                xmm4 = _mm_add_ps(_mm_loadu_ps(v_ptr + 2 * n_regb_elts + 0 * n_reg_elts), xmm4);
                xmm5 = _mm_add_ps(_mm_loadu_ps(v_ptr + 2 * n_regb_elts + 1 * n_reg_elts), xmm5);
            }
            if (u_nb > 3) {
                xmm6 = _mm_add_ps(_mm_loadu_ps(v_ptr + 3 * n_regb_elts + 0 * n_reg_elts), xmm6);
                xmm7 = _mm_add_ps(_mm_loadu_ps(v_ptr + 3 * n_regb_elts + 1 * n_reg_elts), xmm7);
            }
            if (u_nb > 4) {
                xmm8 = _mm_add_ps(_mm_loadu_ps(v_ptr + 4 * n_regb_elts + 0 * n_reg_elts), xmm8);
                xmm9 = _mm_add_ps(_mm_loadu_ps(v_ptr + 4 * n_regb_elts + 1 * n_reg_elts), xmm9);
            }
            if (u_nb > 5) {
                xmm10 = _mm_add_ps(_mm_loadu_ps(v_ptr + 5 * n_regb_elts + 0 * n_reg_elts), xmm10);
                xmm11 = _mm_add_ps(_mm_loadu_ps(v_ptr + 5 * n_regb_elts + 1 * n_reg_elts), xmm11);
            }
        }

        if (flags & (conv_gemm_kernel_fp32_sse::flag::relu | conv_gemm_kernel_fp32_sse::flag::relu6)) {
            xmm14 = _mm_setzero_ps();
            if (u_nb > 0) {
                xmm0 = _mm_max_ps(xmm0, xmm14);
                xmm1 = _mm_max_ps(xmm1, xmm14);
            }
            if (u_nb > 1) {
                xmm2 = _mm_max_ps(xmm2, xmm14);
                xmm3 = _mm_max_ps(xmm3, xmm14);
            }
            if (u_nb > 2) {
                xmm4 = _mm_max_ps(xmm4, xmm14);
                xmm5 = _mm_max_ps(xmm5, xmm14);
            }
            if (u_nb > 3) {
                xmm6 = _mm_max_ps(xmm6, xmm14);
                xmm7 = _mm_max_ps(xmm7, xmm14);
            }
            if (u_nb > 4) {
                xmm8 = _mm_max_ps(xmm8, xmm14);
                xmm9 = _mm_max_ps(xmm9, xmm14);
            }
            if (u_nb > 5) {
                xmm10 = _mm_max_ps(xmm10, xmm14);
                xmm11 = _mm_max_ps(xmm11, xmm14);
            }

            if (flags & conv_gemm_kernel_fp32_sse::flag::relu6) {
                xmm13 = _mm_set1_ps(6.0f);
                if (u_nb > 0) {
                    xmm0 = _mm_min_ps(xmm0, xmm13);
                    xmm1 = _mm_min_ps(xmm1, xmm13);
                }
                if (u_nb > 1) {
                    xmm2 = _mm_min_ps(xmm2, xmm13);
                    xmm3 = _mm_min_ps(xmm3, xmm13);
                }
                if (u_nb > 2) {
                    xmm4 = _mm_min_ps(xmm4, xmm13);
                    xmm5 = _mm_min_ps(xmm5, xmm13);
                }
                if (u_nb > 3) {
                    xmm6 = _mm_min_ps(xmm6, xmm13);
                    xmm7 = _mm_min_ps(xmm7, xmm13);
                }
                if (u_nb > 4) {
                    xmm8 = _mm_min_ps(xmm8, xmm13);
                    xmm9 = _mm_min_ps(xmm9, xmm13);
                }
                if (u_nb > 5) {
                    xmm10 = _mm_min_ps(xmm10, xmm13);
                    xmm11 = _mm_min_ps(xmm11, xmm13);
                }
            }
        }

        if (u_nb > 0) {
            _mm_storeu_ps(c_ptr + 0 * n_regb_elts + 0 * n_reg_elts, xmm0);
            _mm_storeu_ps(c_ptr + 0 * n_regb_elts + 1 * n_reg_elts, xmm1);
        }
        if (u_nb > 1) {
            _mm_storeu_ps(c_ptr + 1 * n_regb_elts + 0 * n_reg_elts, xmm2);
            _mm_storeu_ps(c_ptr + 1 * n_regb_elts + 1 * n_reg_elts, xmm3);
        }
        if (u_nb > 2) {
            _mm_storeu_ps(c_ptr + 2 * n_regb_elts + 0 * n_reg_elts, xmm4);
            _mm_storeu_ps(c_ptr + 2 * n_regb_elts + 1 * n_reg_elts, xmm5);
        }
        if (u_nb > 3) {
            _mm_storeu_ps(c_ptr + 3 * n_regb_elts + 0 * n_reg_elts, xmm6);
            _mm_storeu_ps(c_ptr + 3 * n_regb_elts + 1 * n_reg_elts, xmm7);
        }
        if (u_nb > 4) {
            _mm_storeu_ps(c_ptr + 4 * n_regb_elts + 0 * n_reg_elts, xmm8);
            _mm_storeu_ps(c_ptr + 4 * n_regb_elts + 1 * n_reg_elts, xmm9);
        }
        if (u_nb > 5) {
            _mm_storeu_ps(c_ptr + 5 * n_regb_elts + 0 * n_reg_elts, xmm10);
            _mm_storeu_ps(c_ptr + 5 * n_regb_elts + 1 * n_reg_elts, xmm11);
        }
        a_ptr += kp.pick<const int64_t>(conv_gemm_kernel_fp32_sse::param_def::lda_idx);
        c_ptr += kp.pick<const int64_t>(conv_gemm_kernel_fp32_sse::param_def::ldc_idx);
        m -= 1;
    } while (m > 0);
#undef K_COMPUTE_STEP
}

const conv_gemm_kernel_fp32_sse::func_t
    conv_gemm_kernel_fp32_sse::table_[config::max_n_regbs] =
{
    conv_gemm_fp32_sse_kernel<1 * conv_gemm_kernel_fp32_sse::config::n_regb_elts>,\
    conv_gemm_fp32_sse_kernel<2 * conv_gemm_kernel_fp32_sse::config::n_regb_elts>,\
    conv_gemm_fp32_sse_kernel<3 * conv_gemm_kernel_fp32_sse::config::n_regb_elts>,\
    conv_gemm_fp32_sse_kernel<4 * conv_gemm_kernel_fp32_sse::config::n_regb_elts>,\
    conv_gemm_fp32_sse_kernel<5 * conv_gemm_kernel_fp32_sse::config::n_regb_elts>,\
    conv_gemm_fp32_sse_kernel<6 * conv_gemm_kernel_fp32_sse::config::n_regb_elts>,\
};

}}};
