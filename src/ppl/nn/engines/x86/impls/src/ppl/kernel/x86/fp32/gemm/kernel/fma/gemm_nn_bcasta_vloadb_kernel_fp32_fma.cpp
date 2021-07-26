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

#include "ppl/kernel/x86/fp32/gemm/kernel/fma/gemm_nn_bcasta_vloadb_kernel_fp32_fma.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template <int64_t ker_form, bool nt_store, bool prefetch_a, int64_t m_len>
void gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel_core(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
#define SIX_IDX() 6
    PICK_PARAM(float, priv_param, SIX_IDX()) = 6.0f;
#undef SIX_IDX
    __asm__ __volatile__ (
        ".equ P_BYTES, 8\n"
        ".equ D_BYTES, 4\n"

        ".equ N_RF_BLK, 8\n"
        ".equ M6_K_DT_BLK, 16\n"
        ".equ M6_N_DT_BLK, 16\n"

        ".equ A_IDX, (0 * P_BYTES)\n" // matrix A, const float*
        ".equ B_IDX, (1 * P_BYTES)\n" // matrix B, const float*
        ".equ V_IDX, (2 * P_BYTES)\n" // broadcast vector V, const float*
        ".equ H_IDX, (3 * P_BYTES)\n" // history matrix H (usually pass C here), const float*
        ".equ C_IDX, (4 * P_BYTES)\n" // matrix C, float*
        ".equ M_IDX, (5 * P_BYTES)\n" // critical: M % m_len == 0, int64_t
        ".equ SIX_IDX, (6 * P_BYTES)\n"

        ".equ K_IDX, (0 * P_BYTES)\n" // int64_t
        ".equ A_MBLK_STRIDE_IDX, (1 * P_BYTES)\n" // int64_t
        ".equ A_KBLK_STRIDE_IDX, (2 * P_BYTES)\n" // int64_t
        ".equ H_M_STRIDE_IDX, (3 * P_BYTES)\n" // int64_t
        ".equ C_M_STRIDE_IDX, (4 * P_BYTES)\n" // int64_t
        ".equ FLAGS_IDX, (5 * P_BYTES)\n" // uint64_t
        ".equ ALPHA_IDX, (6 * P_BYTES)\n" // float
        ".equ BETA_IDX, (7 * P_BYTES)\n" // float

        ".equ KER_FORM, %c[KER_FORM]\n"
        ".equ NT_STORE, %c[NT_STORE]\n"
        ".equ PREFETCH_A, %c[PREFETCH_A]\n"
        ".equ PREFETCH_B, 1\n"
        ".equ KERNEL_FLAG_LOAD_C, %c[KERNEL_FLAG_LOAD_C]\n"
        ".equ KERNEL_FLAG_ADD_V, %c[KERNEL_FLAG_ADD_V]\n"
        ".equ KERNEL_FLAG_ADD_H, %c[KERNEL_FLAG_ADD_H]\n"
        ".equ KERNEL_FLAG_RELU, %c[KERNEL_FLAG_RELU]\n"
        ".equ KERNEL_FLAG_RELU6, %c[KERNEL_FLAG_RELU6]\n"
        ".equ M_LEN, %c[M_LEN]\n"

        ".equ KER_FORM_NONE, 0\n"
        ".equ KER_FORM_GEMM, 1\n"
        ".equ KER_FORM_CONV, 2\n"

        PPL_X86_INLINE_ASM_ALIGN()
        ".if KER_FORM != KER_FORM_NONE\n"
        "mov H_IDX(%[priv_param]), %%r15\n" // mb_h
        ".endif\n" // .if KER_FORM != KER_FORM_NONE
        "mov A_IDX(%[priv_param]), %%r14\n" // mb_a
        "mov C_IDX(%[priv_param]), %%r13\n" // mb_c
        "mov M_IDX(%[priv_param]), %%r12\n" // m

"1:\n" // label_init_session
        "mov FLAGS_IDX(%[shar_param]), %%r11\n"
        "test $KERNEL_FLAG_LOAD_C, %%r11\n"
        "jz 2f\n" // label_setzero_c
        "mov %%r13, %%r9\n"
        ".if M_LEN > 0\n"
        "vmovups (0 * N_RF_BLK * D_BYTES)(%%r9), %%ymm0\n"
        "vmovups (1 * N_RF_BLK * D_BYTES)(%%r9), %%ymm10\n"
        ".endif\n"
        ".if M_LEN > 1\n"
        "mov C_M_STRIDE_IDX(%[shar_param]), %%r10\n"
        "lea (%%r9, %%r10, D_BYTES), %%r9\n"
        "vmovups (0 * N_RF_BLK * D_BYTES)(%%r9), %%ymm1\n"
        "vmovups (1 * N_RF_BLK * D_BYTES)(%%r9), %%ymm11\n"
        ".endif\n"
        ".if M_LEN > 2\n"
        "lea (%%r9, %%r10, D_BYTES), %%r9\n"
        "vmovups (0 * N_RF_BLK * D_BYTES)(%%r9), %%ymm2\n"
        "vmovups (1 * N_RF_BLK * D_BYTES)(%%r9), %%ymm12\n"
        ".endif\n"
        ".if M_LEN > 3\n"
        "lea (%%r9, %%r10, D_BYTES), %%r9\n"
        "vmovups (0 * N_RF_BLK * D_BYTES)(%%r9), %%ymm3\n"
        "vmovups (1 * N_RF_BLK * D_BYTES)(%%r9), %%ymm13\n"
        ".endif\n"
        ".if M_LEN > 4\n"
        "lea (%%r9, %%r10, D_BYTES), %%r9\n"
        "vmovups (0 * N_RF_BLK * D_BYTES)(%%r9), %%ymm4\n"
        "vmovups (1 * N_RF_BLK * D_BYTES)(%%r9), %%ymm14\n"
        ".endif\n"
        ".if M_LEN > 5\n"
        "lea (%%r9, %%r10, D_BYTES), %%r9\n"
        "vmovups (0 * N_RF_BLK * D_BYTES)(%%r9), %%ymm5\n"
        "vmovups (1 * N_RF_BLK * D_BYTES)(%%r9), %%ymm15\n"
        ".endif\n"
        "jmp 3f\n" // label_compute_session
"2:\n" // label_setzero_c
        ".if M_LEN > 0\n"
        "vxorps %%ymm0, %%ymm0, %%ymm0\n"
        "vxorps %%ymm1, %%ymm1, %%ymm1\n"
        ".endif\n"
        ".if M_LEN > 1\n"
        "vxorps %%ymm2, %%ymm2, %%ymm2\n"
        "vxorps %%ymm3, %%ymm3, %%ymm3\n"
        ".endif\n"
        ".if M_LEN > 2\n"
        "vxorps %%ymm4, %%ymm4, %%ymm4\n"
        "vxorps %%ymm5, %%ymm5, %%ymm5\n"
        ".endif\n"
        ".if M_LEN > 3\n"
        "vxorps %%ymm10, %%ymm10, %%ymm10\n"
        "vxorps %%ymm11, %%ymm11, %%ymm11\n"
        ".endif\n"
        ".if M_LEN > 4\n"
        "vxorps %%ymm12, %%ymm12, %%ymm12\n"
        "vxorps %%ymm13, %%ymm13, %%ymm13\n"
        ".endif\n"
        ".if M_LEN > 5\n"
        "vxorps %%ymm14, %%ymm14, %%ymm14\n"
        "vxorps %%ymm15, %%ymm15, %%ymm15\n"
        ".endif\n"

"3:\n" // label_compute_session
        "mov A_KBLK_STRIDE_IDX(%[shar_param]), %%r11\n"
        "mov K_IDX(%[shar_param]), %%r10\n"
        "mov B_IDX(%[priv_param]), %%rbx\n"
        "mov %%r14, %%rax\n"
        "cmp $M6_K_DT_BLK, %%r10\n"
        "jl 5f\n" // label_k_remain
"4:\n" // label_k_body
        PPL_X86_INLINE_ASM_ALIGN()
        "lea (%%rax, %%r11, D_BYTES), %%rcx\n"
        ".if PREFETCH_A\n"
        ".if M_LEN > 0\n prefetcht0 (0 * M6_K_DT_BLK * D_BYTES)(%%rcx)\n .endif\n"
        ".if M_LEN > 1\n prefetcht0 (1 * M6_K_DT_BLK * D_BYTES)(%%rcx)\n .endif\n"
        ".if M_LEN > 2\n prefetcht0 (2 * M6_K_DT_BLK * D_BYTES)(%%rcx)\n .endif\n"
        ".if M_LEN > 3\n prefetcht0 (3 * M6_K_DT_BLK * D_BYTES)(%%rcx)\n .endif\n"
        ".if M_LEN > 4\n prefetcht0 (4 * M6_K_DT_BLK * D_BYTES)(%%rcx)\n .endif\n"
        ".if M_LEN > 5\n prefetcht0 (5 * M6_K_DT_BLK * D_BYTES)(%%rcx)\n .endif\n"
        ".endif\n" // if PREFETCH_A
        ".if M_LEN > 1\n" // UNROLL OR NOT
        ".irp K,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
        "vmovups ((\\K * M6_N_DT_BLK + 0 * N_RF_BLK) * D_BYTES)(%%rbx), %%ymm6\n"
        "vmovups ((\\K * M6_N_DT_BLK + 1 * N_RF_BLK) * D_BYTES)(%%rbx), %%ymm7\n"
        ".if M_LEN > 0\n vbroadcastss ((\\K + 0 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm8\n .endif\n"
        ".if M_LEN > 1\n vbroadcastss ((\\K + 1 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm9\n .endif\n"
        ".if M_LEN > 0\n"
        "vfmadd231ps %%ymm6, %%ymm8, %%ymm0\n"
        "vfmadd231ps %%ymm7, %%ymm8, %%ymm10\n"
        ".endif\n"
        ".if M_LEN > 2\n vbroadcastss ((\\K + 2 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm8\n .endif\n"
        ".if M_LEN > 1\n"
        "vfmadd231ps %%ymm6, %%ymm9, %%ymm1\n"
        "vfmadd231ps %%ymm7, %%ymm9, %%ymm11\n"
        ".endif\n"
        ".if M_LEN > 3\n vbroadcastss ((\\K + 3 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm9\n .endif\n"
        ".if M_LEN > 2\n"
        "vfmadd231ps %%ymm6, %%ymm8, %%ymm2\n"
        "vfmadd231ps %%ymm7, %%ymm8, %%ymm12\n"
        ".endif\n"
        ".if M_LEN > 4\n vbroadcastss ((\\K + 4 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm8\n .endif\n"
        ".if M_LEN > 3\n"
        "vfmadd231ps %%ymm6, %%ymm9, %%ymm3\n"
        "vfmadd231ps %%ymm7, %%ymm9, %%ymm13\n"
        ".endif\n"
        ".if M_LEN > 5\n vbroadcastss ((\\K + 5 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm9\n .endif\n"
        ".if M_LEN > 4\n"
        "vfmadd231ps %%ymm6, %%ymm8, %%ymm4\n"
        "vfmadd231ps %%ymm7, %%ymm8, %%ymm14\n"
        ".endif\n"
        ".if PREFETCH_B\n"
        "prefetcht0 ((\\K * M6_N_DT_BLK + M6_K_DT_BLK * M6_N_DT_BLK) * D_BYTES)(%%rbx)\n"
        ".endif\n" // .if PREFETCH_B
        ".if M_LEN > 5\n"
        "vfmadd231ps %%ymm6, %%ymm9, %%ymm5\n"
        "vfmadd231ps %%ymm7, %%ymm9, %%ymm15\n"
        ".endif\n"
        ".endr\n" // .irp K
        "lea (M6_K_DT_BLK * M6_N_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        ".else\n" // UNROLL OR NOT
        "mov %%rax, %%rdx\n"
        "mov $M6_K_DT_BLK, %%r8\n"
"10:\n" // label_k_loop
        "vmovups ((0 * M6_N_DT_BLK + 0 * N_RF_BLK) * D_BYTES)(%%rbx), %%ymm6\n"
        "vmovups ((0 * M6_N_DT_BLK + 1 * N_RF_BLK) * D_BYTES)(%%rbx), %%ymm7\n"
        ".if M_LEN > 0\n vbroadcastss ((0 + 0 * M6_K_DT_BLK) * D_BYTES)(%%rdx), %%ymm8\n .endif\n"
        ".if M_LEN > 1\n vbroadcastss ((0 + 1 * M6_K_DT_BLK) * D_BYTES)(%%rdx), %%ymm9\n .endif\n"
        ".if M_LEN > 0\n"
        "vfmadd231ps %%ymm6, %%ymm8, %%ymm0\n"
        "vfmadd231ps %%ymm7, %%ymm8, %%ymm10\n"
        ".endif\n"
        ".if M_LEN > 2\n vbroadcastss ((0 + 2 * M6_K_DT_BLK) * D_BYTES)(%%rdx), %%ymm8\n .endif\n"
        ".if M_LEN > 1\n"
        "vfmadd231ps %%ymm6, %%ymm9, %%ymm1\n"
        "vfmadd231ps %%ymm7, %%ymm9, %%ymm11\n"
        ".endif\n"
        ".if M_LEN > 3\n vbroadcastss ((0 + 3 * M6_K_DT_BLK) * D_BYTES)(%%rdx), %%ymm9\n .endif\n"
        ".if M_LEN > 2\n"
        "vfmadd231ps %%ymm6, %%ymm8, %%ymm2\n"
        "vfmadd231ps %%ymm7, %%ymm8, %%ymm12\n"
        ".endif\n"
        ".if M_LEN > 4\n vbroadcastss ((0 + 4 * M6_K_DT_BLK) * D_BYTES)(%%rdx), %%ymm8\n .endif\n"
        ".if M_LEN > 3\n"
        "vfmadd231ps %%ymm6, %%ymm9, %%ymm3\n"
        "vfmadd231ps %%ymm7, %%ymm9, %%ymm13\n"
        ".endif\n"
        ".if M_LEN > 5\n vbroadcastss ((0 + 5 * M6_K_DT_BLK) * D_BYTES)(%%rdx), %%ymm9\n .endif\n"
        ".if M_LEN > 4\n"
        "vfmadd231ps %%ymm6, %%ymm8, %%ymm4\n"
        "vfmadd231ps %%ymm7, %%ymm8, %%ymm14\n"
        ".endif\n"
        ".if PREFETCH_B\n"
        "prefetcht0 ((0 * M6_N_DT_BLK + M6_K_DT_BLK * M6_N_DT_BLK) * D_BYTES)(%%rbx)\n"
        ".endif\n" // .if PREFETCH_B
        ".if M_LEN > 5\n"
        "vfmadd231ps %%ymm6, %%ymm9, %%ymm5\n"
        "vfmadd231ps %%ymm7, %%ymm9, %%ymm15\n"
        ".endif\n"
        "add $D_BYTES, %%rdx\n"
        "lea (M6_N_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        "sub $1, %%r8\n"
        "cmp $0, %%r8\n"
        "jne 10b\n" // label_k_loop
        ".endif\n" // UNROLL OR NOT
        "mov %%rcx, %%rax\n"
        "sub $M6_K_DT_BLK, %%r10\n"
        "cmp $M6_K_DT_BLK, %%r10\n"
        "jge 4b\n" // label_k_body
        "cmp $0, %%r10\n"
        "je 6f\n" // label_finalize_session
"5:\n" // label_k_remain
        "vmovups ((0 * M6_N_DT_BLK + 0 * N_RF_BLK) * D_BYTES)(%%rbx), %%ymm6\n"
        "vmovups ((0 * M6_N_DT_BLK + 1 * N_RF_BLK) * D_BYTES)(%%rbx), %%ymm7\n"
        ".if M_LEN > 0\n vbroadcastss ((0 + 0 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm8\n .endif\n"
        ".if M_LEN > 1\n vbroadcastss ((0 + 1 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm9\n .endif\n"
        ".if M_LEN > 0\n"
        "vfmadd231ps %%ymm6, %%ymm8, %%ymm0\n"
        "vfmadd231ps %%ymm7, %%ymm8, %%ymm10\n"
        ".endif\n"
        ".if M_LEN > 2\n vbroadcastss ((0 + 2 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm8\n .endif\n"
        ".if M_LEN > 1\n"
        "vfmadd231ps %%ymm6, %%ymm9, %%ymm1\n"
        "vfmadd231ps %%ymm7, %%ymm9, %%ymm11\n"
        ".endif\n"
        ".if M_LEN > 3\n vbroadcastss ((0 + 3 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm9\n .endif\n"
        ".if M_LEN > 2\n"
        "vfmadd231ps %%ymm6, %%ymm8, %%ymm2\n"
        "vfmadd231ps %%ymm7, %%ymm8, %%ymm12\n"
        ".endif\n"
        ".if M_LEN > 4\n vbroadcastss ((0 + 4 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm8\n .endif\n"
        ".if M_LEN > 3\n"
        "vfmadd231ps %%ymm6, %%ymm9, %%ymm3\n"
        "vfmadd231ps %%ymm7, %%ymm9, %%ymm13\n"
        ".endif\n"
        ".if M_LEN > 5\n vbroadcastss ((0 + 5 * M6_K_DT_BLK) * D_BYTES)(%%rax), %%ymm9\n .endif\n"
        ".if M_LEN > 4\n"
        "vfmadd231ps %%ymm6, %%ymm8, %%ymm4\n"
        "vfmadd231ps %%ymm7, %%ymm8, %%ymm14\n"
        ".endif\n"
        ".if M_LEN > 5\n"
        "vfmadd231ps %%ymm6, %%ymm9, %%ymm5\n"
        "vfmadd231ps %%ymm7, %%ymm9, %%ymm15\n"
        ".endif\n"
        "add $D_BYTES, %%rax\n"
        "lea (M6_N_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        "sub $1, %%r10\n"
        "cmp $0, %%r10\n"
        "jne 5b\n" // label_k_remain

"6:\n" // label_finalize_session
        ".if KER_FORM != KER_FORM_NONE\n"
        "mov FLAGS_IDX(%[shar_param]), %%r11\n"
        "test $KERNEL_FLAG_ADD_V, %%r11\n"
        "jz 7f\n" // label_addv_end
        "mov V_IDX(%[priv_param]), %%r10\n"
        "vmovups (0 * N_RF_BLK * D_BYTES)(%%r10), %%ymm8\n"
        "vmovups (1 * N_RF_BLK * D_BYTES)(%%r10), %%ymm9\n"
        ".if M_LEN > 0\n"
        "vaddps %%ymm8, %%ymm0, %%ymm0\n"
        "vaddps %%ymm8, %%ymm1, %%ymm1\n"
        ".endif\n"
        ".if M_LEN > 1\n"
        "vaddps %%ymm8, %%ymm2, %%ymm2\n"
        "vaddps %%ymm8, %%ymm3, %%ymm3\n"
        ".endif\n"
        ".if M_LEN > 2\n"
        "vaddps %%ymm8, %%ymm4, %%ymm4\n"
        "vaddps %%ymm8, %%ymm5, %%ymm5\n"
        ".endif\n"
        ".if M_LEN > 3\n"
        "vaddps %%ymm9, %%ymm10, %%ymm10\n"
        "vaddps %%ymm9, %%ymm11, %%ymm11\n"
        ".endif\n"
        ".if M_LEN > 4\n"
        "vaddps %%ymm9, %%ymm12, %%ymm12\n"
        "vaddps %%ymm9, %%ymm13, %%ymm13\n"
        ".endif\n"
        ".if M_LEN > 5\n"
        "vaddps %%ymm9, %%ymm14, %%ymm14\n"
        "vaddps %%ymm9, %%ymm15, %%ymm15\n"
        ".endif\n"
"7:\n" // label_addv_end
        "test $KERNEL_FLAG_ADD_H, %%r11\n"
        "jz 8f\n" // label_addh_end
        "mov H_M_STRIDE_IDX(%[shar_param]), %%r10\n"
        ".if M_LEN > 0\n"
        "vaddps (0 * N_RF_BLK * D_BYTES)(%%r15), %%ymm0, %%ymm0\n"
        "vaddps (1 * N_RF_BLK * D_BYTES)(%%r15), %%ymm10, %%ymm10\n"
        "lea (%%r15, %%r10, D_BYTES), %%r15\n"
        ".endif\n"
        ".if M_LEN > 1\n"
        "vaddps (0 * N_RF_BLK * D_BYTES)(%%r15), %%ymm1, %%ymm1\n"
        "vaddps (1 * N_RF_BLK * D_BYTES)(%%r15), %%ymm11, %%ymm11\n"
        "lea (%%r15, %%r10, D_BYTES), %%r15\n"
        ".endif\n"
        ".if M_LEN > 2\n"
        "vaddps (0 * N_RF_BLK * D_BYTES)(%%r15), %%ymm2, %%ymm2\n"
        "vaddps (1 * N_RF_BLK * D_BYTES)(%%r15), %%ymm12, %%ymm12\n"
        "lea (%%r15, %%r10, D_BYTES), %%r15\n"
        ".endif\n"
        ".if M_LEN > 3\n"
        "vaddps (0 * N_RF_BLK * D_BYTES)(%%r15), %%ymm3, %%ymm3\n"
        "vaddps (1 * N_RF_BLK * D_BYTES)(%%r15), %%ymm13, %%ymm13\n"
        "lea (%%r15, %%r10, D_BYTES), %%r15\n"
        ".endif\n"
        ".if M_LEN > 4\n"
        "vaddps (0 * N_RF_BLK * D_BYTES)(%%r15), %%ymm4, %%ymm4\n"
        "vaddps (1 * N_RF_BLK * D_BYTES)(%%r15), %%ymm14, %%ymm14\n"
        "lea (%%r15, %%r10, D_BYTES), %%r15\n"
        ".endif\n"
        ".if M_LEN > 5\n"
        "vaddps (0 * N_RF_BLK * D_BYTES)(%%r15), %%ymm5, %%ymm5\n"
        "vaddps (1 * N_RF_BLK * D_BYTES)(%%r15), %%ymm15, %%ymm15\n"
        "lea (%%r15, %%r10, D_BYTES), %%r15\n"
        ".endif\n"
"8:\n" // label_addh_end
        "test $(KERNEL_FLAG_RELU | KERNEL_FLAG_RELU6), %%r11\n"
        "jz 9f\n" // label_relu_end
        "vxorps %%ymm8, %%ymm8, %%ymm8\n"
        ".if M_LEN > 0\n"
        "vmaxps %%ymm8, %%ymm0, %%ymm0\n"
        "vmaxps %%ymm8, %%ymm1, %%ymm1\n"
        ".endif\n"
        ".if M_LEN > 1\n"
        "vmaxps %%ymm8, %%ymm2, %%ymm2\n"
        "vmaxps %%ymm8, %%ymm3, %%ymm3\n"
        ".endif\n"
        ".if M_LEN > 2\n"
        "vmaxps %%ymm8, %%ymm4, %%ymm4\n"
        "vmaxps %%ymm8, %%ymm5, %%ymm5\n"
        ".endif\n"
        ".if M_LEN > 3\n"
        "vmaxps %%ymm8, %%ymm10, %%ymm10\n"
        "vmaxps %%ymm8, %%ymm11, %%ymm11\n"
        ".endif\n"
        ".if M_LEN > 4\n"
        "vmaxps %%ymm8, %%ymm12, %%ymm12\n"
        "vmaxps %%ymm8, %%ymm13, %%ymm13\n"
        ".endif\n"
        ".if M_LEN > 0\n"
        "vmaxps %%ymm8, %%ymm14, %%ymm14\n"
        "vmaxps %%ymm8, %%ymm15, %%ymm15\n"
        ".endif\n"
        "test $KERNEL_FLAG_RELU6, %%r11\n"
        "jz 9f\n" // label_relu_end
        "vbroadcastss SIX_IDX(%[priv_param]), %%ymm9\n"
        ".if M_LEN > 0\n"
        "vminps %%ymm9, %%ymm0, %%ymm0\n"
        "vminps %%ymm9, %%ymm1, %%ymm1\n"
        ".endif\n"
        ".if M_LEN > 1\n"
        "vminps %%ymm9, %%ymm2, %%ymm2\n"
        "vminps %%ymm9, %%ymm3, %%ymm3\n"
        ".endif\n"
        ".if M_LEN > 2\n"
        "vminps %%ymm9, %%ymm4, %%ymm4\n"
        "vminps %%ymm9, %%ymm5, %%ymm5\n"
        ".endif\n"
        ".if M_LEN > 3\n"
        "vminps %%ymm9, %%ymm10, %%ymm10\n"
        "vminps %%ymm9, %%ymm11, %%ymm11\n"
        ".endif\n"
        ".if M_LEN > 4\n"
        "vminps %%ymm9, %%ymm12, %%ymm12\n"
        "vminps %%ymm9, %%ymm13, %%ymm13\n"
        ".endif\n"
        ".if M_LEN > 5\n"
        "vminps %%ymm9, %%ymm14, %%ymm14\n"
        "vminps %%ymm9, %%ymm15, %%ymm15\n"
        ".endif\n"
"9:\n" // label_relu_end
        ".endif\n" // .if KER_FORM != KER_FORM_NONE
        "mov C_M_STRIDE_IDX(%[shar_param]), %%r10\n"
        ".if NT_STORE\n"
        ".if M_LEN > 0\n"
        "vmovntps %%ymm0, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovntps %%ymm10, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".if M_LEN > 1\n"
        "vmovntps %%ymm1, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovntps %%ymm11, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".if M_LEN > 2\n"
        "vmovntps %%ymm2, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovntps %%ymm12, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".if M_LEN > 3\n"
        "vmovntps %%ymm3, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovntps %%ymm13, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".if M_LEN > 4\n"
        "vmovntps %%ymm4, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovntps %%ymm14, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".if M_LEN > 5\n"
        "vmovntps %%ymm5, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovntps %%ymm15, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".else\n" // .if NT_STORE
        ".if M_LEN > 0\n"
        "vmovups %%ymm0, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovups %%ymm10, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".if M_LEN > 1\n"
        "vmovups %%ymm1, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovups %%ymm11, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".if M_LEN > 2\n"
        "vmovups %%ymm2, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovups %%ymm12, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".if M_LEN > 3\n"
        "vmovups %%ymm3, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovups %%ymm13, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".if M_LEN > 4\n"
        "vmovups %%ymm4, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovups %%ymm14, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".if M_LEN > 5\n"
        "vmovups %%ymm5, (0 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "vmovups %%ymm15, (1 * N_RF_BLK * D_BYTES)(%%r13)\n"
        "lea (%%r13, %%r10, D_BYTES), %%r13\n"
        ".endif\n"
        ".endif\n" // .if NT_STORE
        "mov A_MBLK_STRIDE_IDX(%[shar_param]), %%r10\n"
        "lea (%%r14, %%r10, D_BYTES), %%r14\n" // mb_a += a_mblk_stride
        "sub $M_LEN, %%r12\n"
        "cmp $0, %%r12\n"
        "jne 1b\n" // label_init_session
        :
        :
        [priv_param]                    "r" (priv_param),
        [shar_param]                    "r" (shar_param),
        [KER_FORM]                      "i" (ker_form),
        [NT_STORE]                      "i" (nt_store),
        [PREFETCH_A]                    "i" (prefetch_a),
        [M_LEN]                         "i" (m_len),
        [KERNEL_FLAG_LOAD_C]            "i" (KERNEL_FLAG_LOAD_C()),
        [KERNEL_FLAG_ADD_V]             "i" (KERNEL_FLAG_ADD_V()),
        [KERNEL_FLAG_ADD_H]             "i" (KERNEL_FLAG_ADD_H()),
        [KERNEL_FLAG_RELU]              "i" (KERNEL_FLAG_RELU()),
        [KERNEL_FLAG_RELU6]             "i" (KERNEL_FLAG_RELU6())
        :
        "cc",
        "rax", "rbx", "rcx", "rdx",
        "r8" , "r9" , "r10", "r11", "r12", "r13", "r14", "r15",
        "ymm0" , "ymm1" , "ymm2" , "ymm3" , "ymm4" , "ymm5" , "ymm6" , "ymm7" ,
        "ymm8" , "ymm9" , "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
        "memory"
    );
}

#endif

template <int64_t ker_form, bool nt_store, bool prefetch_a, int64_t n_len, int64_t m_len>
void gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel(
    const int64_t *priv_param,
    const int64_t *shar_param)
{

#ifdef PPL_USE_X86_INLINE_ASM
    if (ker_form != KER_FORM_GEMM() && n_len == 2 * N_RF_BLK() && m_len == 6) {
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel_core<ker_form, nt_store, prefetch_a, m_len>(priv_param, shar_param);
        return;
    }
#endif

#define K_COMPUTE_STEP(K) do {\
    if (n_len > 0 * N_RF_BLK()) ymm6 = _mm256_loadu_ps(b + (K) * M6_N_DT_BLK() + 0 * N_RF_BLK());\
    if (n_len > 1 * N_RF_BLK()) ymm7 = _mm256_loadu_ps(b + (K) * M6_N_DT_BLK() + 1 * N_RF_BLK());\
    if (m_len > 0) {\
        ymm8 = _mm256_set1_ps(a[(K) + 0 * M6_K_DT_BLK()]);\
        if (n_len > 0 * N_RF_BLK()) ymm0 = _mm256_fmadd_ps(ymm6, ymm8, ymm0);\
        if (n_len > 1 * N_RF_BLK()) ymm10 = _mm256_fmadd_ps(ymm7, ymm8, ymm10);\
    }\
    if (m_len > 1) {\
        ymm9 = _mm256_set1_ps(a[(K) + 1 * M6_K_DT_BLK()]);\
        if (n_len > 0 * N_RF_BLK()) ymm1 = _mm256_fmadd_ps(ymm6, ymm9, ymm1);\
        if (n_len > 1 * N_RF_BLK()) ymm11 = _mm256_fmadd_ps(ymm7, ymm9, ymm11);\
    }\
    if (m_len > 2) {\
        ymm8 = _mm256_set1_ps(a[(K) + 2 * M6_K_DT_BLK()]);\
        if (n_len > 0 * N_RF_BLK()) ymm2 = _mm256_fmadd_ps(ymm6, ymm8, ymm2);\
        if (n_len > 1 * N_RF_BLK()) ymm12 = _mm256_fmadd_ps(ymm7, ymm8, ymm12);\
    }\
    if (m_len > 3) {\
        ymm9 = _mm256_set1_ps(a[(K) + 3 * M6_K_DT_BLK()]);\
        if (n_len > 0 * N_RF_BLK()) ymm3 = _mm256_fmadd_ps(ymm6, ymm9, ymm3);\
        if (n_len > 1 * N_RF_BLK()) ymm13 = _mm256_fmadd_ps(ymm7, ymm9, ymm13);\
    }\
    if (m_len > 4) {\
        ymm8 = _mm256_set1_ps(a[(K) + 4 * M6_K_DT_BLK()]);\
        if (n_len > 0 * N_RF_BLK()) ymm4 = _mm256_fmadd_ps(ymm6, ymm8, ymm4);\
        if (n_len > 1 * N_RF_BLK()) ymm14 = _mm256_fmadd_ps(ymm7, ymm8, ymm14);\
    }\
    if (m_len > 5) {\
        ymm9 = _mm256_set1_ps(a[(K) + 5 * M6_K_DT_BLK()]);\
        if (n_len > 0 * N_RF_BLK()) ymm5 = _mm256_fmadd_ps(ymm6, ymm9, ymm5);\
        if (n_len > 1 * N_RF_BLK()) ymm15 = _mm256_fmadd_ps(ymm7, ymm9, ymm15);\
    }\
} while (false)

// #define DO_PREFETCH_B
#ifdef DO_PREFETCH_B
#define K_PREFETCH_STEP(K) do {\
    if (m_len > 2) _mm_prefetch((const char*)(b + (K) * M6_N_DT_BLK() + M6_K_DT_BLK() * M6_N_DT_BLK()), _MM_HINT_T0);\
} while (false)
#else
#define K_PREFETCH_STEP(K)
#endif
    const float *mb_h;
    if (ker_form != KER_FORM_NONE()) {
        mb_h = PICK_PARAM(const float*, priv_param, H_IDX());
    }
    const float *mb_a = PICK_PARAM(const float*, priv_param, A_IDX());
    float *mb_c = PICK_PARAM(float*, priv_param, C_IDX());
    int64_t m = PICK_PARAM(int64_t, priv_param, M_IDX());
    do {
        __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
        __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
        { // session - initialize
            const uint64_t flags = PICK_PARAM(const uint64_t, shar_param, FLAGS_IDX());
            if (flags & KERNEL_FLAG_LOAD_C()) {
                const int64_t c_m_stride = PICK_PARAM(const int64_t, shar_param, C_M_STRIDE_IDX());
                const float *m_c = mb_c;
                if (m_len > 0) {
                    if (n_len > 0 * N_RF_BLK()) ymm0 = _mm256_loadu_ps(m_c + 0 * N_RF_BLK());
                    if (n_len > 1 * N_RF_BLK()) ymm10 = _mm256_loadu_ps(m_c + 1 * N_RF_BLK());
                }
                if (m_len > 1) {
                    m_c += c_m_stride;
                    if (n_len > 0 * N_RF_BLK()) ymm1 = _mm256_loadu_ps(m_c + 0 * N_RF_BLK());
                    if (n_len > 1 * N_RF_BLK()) ymm11 = _mm256_loadu_ps(m_c + 1 * N_RF_BLK());
                }
                if (m_len > 2) {
                    m_c += c_m_stride;
                    if (n_len > 0 * N_RF_BLK()) ymm2 = _mm256_loadu_ps(m_c + 0 * N_RF_BLK());
                    if (n_len > 1 * N_RF_BLK()) ymm12 = _mm256_loadu_ps(m_c + 1 * N_RF_BLK());
                }
                if (m_len > 3) {
                    m_c += c_m_stride;
                    if (n_len > 0 * N_RF_BLK()) ymm3 = _mm256_loadu_ps(m_c + 0 * N_RF_BLK());
                    if (n_len > 1 * N_RF_BLK()) ymm13 = _mm256_loadu_ps(m_c + 1 * N_RF_BLK());
                }
                if (m_len > 4) {
                    m_c += c_m_stride;
                    if (n_len > 0 * N_RF_BLK()) ymm4 = _mm256_loadu_ps(m_c + 0 * N_RF_BLK());
                    if (n_len > 1 * N_RF_BLK()) ymm14 = _mm256_loadu_ps(m_c + 1 * N_RF_BLK());
                }
                if (m_len > 5) {
                    m_c += c_m_stride;
                    if (n_len > 0 * N_RF_BLK()) ymm5 = _mm256_loadu_ps(m_c + 0 * N_RF_BLK());
                    if (n_len > 1 * N_RF_BLK()) ymm15 = _mm256_loadu_ps(m_c + 1 * N_RF_BLK());
                }
            } else {
                if (n_len > 0 * N_RF_BLK()) {
                    if (m_len > 0) ymm0 = _mm256_setzero_ps();
                    if (m_len > 1) ymm1 = _mm256_setzero_ps();
                    if (m_len > 2) ymm2 = _mm256_setzero_ps();
                    if (m_len > 3) ymm3 = _mm256_setzero_ps();
                    if (m_len > 4) ymm4 = _mm256_setzero_ps();
                    if (m_len > 5) ymm5 = _mm256_setzero_ps();
                }
                if (n_len > 1 * N_RF_BLK()) {
                    if (m_len > 0) ymm10 = _mm256_setzero_ps();
                    if (m_len > 1) ymm11 = _mm256_setzero_ps();
                    if (m_len > 2) ymm12 = _mm256_setzero_ps();
                    if (m_len > 3) ymm13 = _mm256_setzero_ps();
                    if (m_len > 4) ymm14 = _mm256_setzero_ps();
                    if (m_len > 5) ymm15 = _mm256_setzero_ps();
                }
            }
        }

        { // session - compute
            int64_t k = PICK_PARAM(int64_t, shar_param, K_IDX());
            const float *b = PICK_PARAM(const float*, priv_param, B_IDX());
            const float *a = mb_a;
            const int64_t a_kblk_stride = PICK_PARAM(const int64_t, shar_param, A_KBLK_STRIDE_IDX());
            while (k >= M6_K_DT_BLK()) {
                k -= M6_K_DT_BLK();
                if (prefetch_a) {
                    const float *next_a = a + a_kblk_stride;
                    if (m_len > 0) _mm_prefetch((const char*)(next_a + 0 * M6_K_DT_BLK()), _MM_HINT_T0);
                    if (m_len > 1) _mm_prefetch((const char*)(next_a + 1 * M6_K_DT_BLK()), _MM_HINT_T0);
                    if (m_len > 2) _mm_prefetch((const char*)(next_a + 2 * M6_K_DT_BLK()), _MM_HINT_T0);
                    if (m_len > 3) _mm_prefetch((const char*)(next_a + 3 * M6_K_DT_BLK()), _MM_HINT_T0);
                    if (m_len > 4) _mm_prefetch((const char*)(next_a + 4 * M6_K_DT_BLK()), _MM_HINT_T0);
                    if (m_len > 5) _mm_prefetch((const char*)(next_a + 5 * M6_K_DT_BLK()), _MM_HINT_T0);
                }
                if (n_len == 2 * N_RF_BLK() && m_len > 3) {
                    K_COMPUTE_STEP(0);
                    K_PREFETCH_STEP(0);
                    K_COMPUTE_STEP(1);
                    K_PREFETCH_STEP(1);
                    K_COMPUTE_STEP(2);
                    K_PREFETCH_STEP(2);
                    K_COMPUTE_STEP(3);
                    K_PREFETCH_STEP(3);
                    K_COMPUTE_STEP(4);
                    K_PREFETCH_STEP(4);
                    K_COMPUTE_STEP(5);
                    K_PREFETCH_STEP(5);
                    K_COMPUTE_STEP(6);
                    K_PREFETCH_STEP(6);
                    K_COMPUTE_STEP(7);
                    K_PREFETCH_STEP(7);
                    K_COMPUTE_STEP(8);
                    K_PREFETCH_STEP(8);
                    K_COMPUTE_STEP(9);
                    K_PREFETCH_STEP(9);
                    K_COMPUTE_STEP(10);
                    K_PREFETCH_STEP(10);
                    K_COMPUTE_STEP(11);
                    K_PREFETCH_STEP(11);
                    K_COMPUTE_STEP(12);
                    K_PREFETCH_STEP(12);
                    K_COMPUTE_STEP(13);
                    K_PREFETCH_STEP(13);
                    K_COMPUTE_STEP(14);
                    K_PREFETCH_STEP(14);
                    K_COMPUTE_STEP(15);
                    K_PREFETCH_STEP(15);
                } else {
                    for (int64_t k = 0; k < M6_K_DT_BLK(); ++k) {
                        K_COMPUTE_STEP(k);
                        K_PREFETCH_STEP(k);
                    }
                }
                a += a_kblk_stride;
                b += M6_K_DT_BLK() * M6_N_DT_BLK();
            }
            while (k > 0) {
                --k;
                K_COMPUTE_STEP(0);
                a += 1;
                b += M6_N_DT_BLK();
            }
        }
        
        { // session - finalize
            if (ker_form != KER_FORM_NONE()) {
                const uint64_t flags = PICK_PARAM(const uint64_t, shar_param, FLAGS_IDX());
                if (ker_form == KER_FORM_CONV()) {
                    if (flags & KERNEL_FLAG_ADD_V()) {
                        const float *v = PICK_PARAM(const float*, priv_param, V_IDX());
                        if (n_len > 0 * N_RF_BLK()) {
                            ymm8 = _mm256_loadu_ps(v + 0 * N_RF_BLK());
                            if (m_len > 0) ymm0 = _mm256_add_ps(ymm8, ymm0);
                            if (m_len > 1) ymm1 = _mm256_add_ps(ymm8, ymm1);
                            if (m_len > 2) ymm2 = _mm256_add_ps(ymm8, ymm2);
                            if (m_len > 3) ymm3 = _mm256_add_ps(ymm8, ymm3);
                            if (m_len > 4) ymm4 = _mm256_add_ps(ymm8, ymm4);
                            if (m_len > 5) ymm5 = _mm256_add_ps(ymm8, ymm5);
                        }
                        if (n_len > 1 * N_RF_BLK()) {
                            ymm9 = _mm256_loadu_ps(v + 1 * N_RF_BLK());
                            if (m_len > 0) ymm10 = _mm256_add_ps(ymm9, ymm10);
                            if (m_len > 1) ymm11 = _mm256_add_ps(ymm9, ymm11);
                            if (m_len > 2) ymm12 = _mm256_add_ps(ymm9, ymm12);
                            if (m_len > 3) ymm13 = _mm256_add_ps(ymm9, ymm13);
                            if (m_len > 4) ymm14 = _mm256_add_ps(ymm9, ymm14);
                            if (m_len > 5) ymm15 = _mm256_add_ps(ymm9, ymm15);
                        }
                    }
                    if (flags & KERNEL_FLAG_ADD_H()) {
                        const int64_t h_m_stride = PICK_PARAM(const int64_t, shar_param, H_M_STRIDE_IDX());
                        if (m_len > 0) {
                            if (n_len > 0 * N_RF_BLK()) ymm0 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm0);
                            if (n_len > 1 * N_RF_BLK()) ymm10 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm10);
                            mb_h += h_m_stride;
                        }
                        if (m_len > 1) {
                            if (n_len > 0 * N_RF_BLK()) ymm1 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm1);
                            if (n_len > 1 * N_RF_BLK()) ymm11 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm11);
                            mb_h += h_m_stride;
                        }
                        if (m_len > 2) {
                            if (n_len > 0 * N_RF_BLK()) ymm2 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm2);
                            if (n_len > 1 * N_RF_BLK()) ymm12 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm12);
                            mb_h += h_m_stride;
                        }
                        if (m_len > 3) {
                            if (n_len > 0 * N_RF_BLK()) ymm3 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm3);
                            if (n_len > 1 * N_RF_BLK()) ymm13 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm13);
                            mb_h += h_m_stride;
                        }
                        if (m_len > 4) {
                            if (n_len > 0 * N_RF_BLK()) ymm4 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm4);
                            if (n_len > 1 * N_RF_BLK()) ymm14 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm14);
                            mb_h += h_m_stride;
                        }
                        if (m_len > 5) {
                            if (n_len > 0 * N_RF_BLK()) ymm5 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm5);
                            if (n_len > 1 * N_RF_BLK()) ymm15 = _mm256_add_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm15);
                            mb_h += h_m_stride;
                        }
                    }

                    if (flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
                        ymm8 = _mm256_setzero_ps();
                        if (n_len > 0 * N_RF_BLK()) {
                            if (m_len > 0) ymm0 = _mm256_max_ps(ymm8, ymm0);
                            if (m_len > 1) ymm1 = _mm256_max_ps(ymm8, ymm1);
                            if (m_len > 2) ymm2 = _mm256_max_ps(ymm8, ymm2);
                            if (m_len > 3) ymm3 = _mm256_max_ps(ymm8, ymm3);
                            if (m_len > 4) ymm4 = _mm256_max_ps(ymm8, ymm4);
                            if (m_len > 5) ymm5 = _mm256_max_ps(ymm8, ymm5);
                        }
                        if (n_len > 1 * N_RF_BLK()) {
                            if (m_len > 0) ymm10 = _mm256_max_ps(ymm8, ymm10);
                            if (m_len > 1) ymm11 = _mm256_max_ps(ymm8, ymm11);
                            if (m_len > 2) ymm12 = _mm256_max_ps(ymm8, ymm12);
                            if (m_len > 3) ymm13 = _mm256_max_ps(ymm8, ymm13);
                            if (m_len > 4) ymm14 = _mm256_max_ps(ymm8, ymm14);
                            if (m_len > 5) ymm15 = _mm256_max_ps(ymm8, ymm15);
                        }
                    }
                    if (flags & KERNEL_FLAG_RELU6()) {
                        ymm9 = _mm256_set1_ps(6.0f);
                        if (n_len > 0 * N_RF_BLK()) {
                            if (m_len > 0) ymm0 = _mm256_min_ps(ymm9, ymm0);
                            if (m_len > 1) ymm1 = _mm256_min_ps(ymm9, ymm1);
                            if (m_len > 2) ymm2 = _mm256_min_ps(ymm9, ymm2);
                            if (m_len > 3) ymm3 = _mm256_min_ps(ymm9, ymm3);
                            if (m_len > 4) ymm4 = _mm256_min_ps(ymm9, ymm4);
                            if (m_len > 5) ymm5 = _mm256_min_ps(ymm9, ymm5);
                        }
                        if (n_len > 1 * N_RF_BLK()) {
                            if (m_len > 0) ymm10 = _mm256_min_ps(ymm9, ymm10);
                            if (m_len > 1) ymm11 = _mm256_min_ps(ymm9, ymm11);
                            if (m_len > 2) ymm12 = _mm256_min_ps(ymm9, ymm12);
                            if (m_len > 3) ymm13 = _mm256_min_ps(ymm9, ymm13);
                            if (m_len > 4) ymm14 = _mm256_min_ps(ymm9, ymm14);
                            if (m_len > 5) ymm15 = _mm256_min_ps(ymm9, ymm15);
                        }
                    }
                } else if (ker_form == KER_FORM_GEMM()) {
                    if (flags & KERNEL_FLAG_MUL_C()) {
                        const float alpha = PICK_PARAM(const float, shar_param, ALPHA_IDX());
                        ymm8 = _mm256_set1_ps(alpha);
                        if (n_len > 0 * N_RF_BLK()) {
                            if (m_len > 0) ymm0 = _mm256_mul_ps(ymm8, ymm0);
                            if (m_len > 1) ymm1 = _mm256_mul_ps(ymm8, ymm1);
                            if (m_len > 2) ymm2 = _mm256_mul_ps(ymm8, ymm2);
                            if (m_len > 3) ymm3 = _mm256_mul_ps(ymm8, ymm3);
                            if (m_len > 4) ymm4 = _mm256_mul_ps(ymm8, ymm4);
                            if (m_len > 5) ymm5 = _mm256_mul_ps(ymm8, ymm5);
                        }
                        if (n_len > 1 * N_RF_BLK()) {
                            if (m_len > 0) ymm10 = _mm256_mul_ps(ymm8, ymm10);
                            if (m_len > 1) ymm11 = _mm256_mul_ps(ymm8, ymm11);
                            if (m_len > 2) ymm12 = _mm256_mul_ps(ymm8, ymm12);
                            if (m_len > 3) ymm13 = _mm256_mul_ps(ymm8, ymm13);
                            if (m_len > 4) ymm14 = _mm256_mul_ps(ymm8, ymm14);
                            if (m_len > 5) ymm15 = _mm256_mul_ps(ymm8, ymm15);
                        }
                    }
                    if (flags & KERNEL_FLAG_FMA_V()) {
                        const float *v = PICK_PARAM(const float*, priv_param, V_IDX());
                        const float beta = PICK_PARAM(const float, shar_param, BETA_IDX());
                        ymm9 = _mm256_set1_ps(beta);
                        if (n_len > 0 * N_RF_BLK()) {
                            ymm8 = _mm256_loadu_ps(v + 0 * N_RF_BLK());
                            if (m_len > 0) ymm0 = _mm256_fmadd_ps(ymm8, ymm9, ymm0);
                            if (m_len > 1) ymm1 = _mm256_fmadd_ps(ymm8, ymm9, ymm1);
                            if (m_len > 2) ymm2 = _mm256_fmadd_ps(ymm8, ymm9, ymm2);
                            if (m_len > 3) ymm3 = _mm256_fmadd_ps(ymm8, ymm9, ymm3);
                            if (m_len > 4) ymm4 = _mm256_fmadd_ps(ymm8, ymm9, ymm4);
                            if (m_len > 5) ymm5 = _mm256_fmadd_ps(ymm8, ymm9, ymm5);
                        }
                        if (n_len > 1 * N_RF_BLK()) {
                            ymm7 = _mm256_loadu_ps(v + 1 * N_RF_BLK());
                            if (m_len > 0) ymm10 = _mm256_fmadd_ps(ymm7, ymm9, ymm10);
                            if (m_len > 1) ymm11 = _mm256_fmadd_ps(ymm7, ymm9, ymm11);
                            if (m_len > 2) ymm12 = _mm256_fmadd_ps(ymm7, ymm9, ymm12);
                            if (m_len > 3) ymm13 = _mm256_fmadd_ps(ymm7, ymm9, ymm13);
                            if (m_len > 4) ymm14 = _mm256_fmadd_ps(ymm7, ymm9, ymm14);
                            if (m_len > 5) ymm15 = _mm256_fmadd_ps(ymm7, ymm9, ymm15);
                        }
                    } else if (flags & KERNEL_FLAG_FMA_H()) {
                        const int64_t h_m_stride = PICK_PARAM(const int64_t, shar_param, H_M_STRIDE_IDX());
                        const float beta = PICK_PARAM(const float, shar_param, BETA_IDX());
                        ymm9 = _mm256_set1_ps(beta);
                        if (m_len > 0) {
                            if (n_len > 0 * N_RF_BLK()) ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm9, ymm0);
                            if (n_len > 1 * N_RF_BLK()) ymm10 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm9, ymm10);
                            mb_h += h_m_stride;
                        }
                        if (m_len > 1) {
                            if (n_len > 0 * N_RF_BLK()) ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm9, ymm1);
                            if (n_len > 1 * N_RF_BLK()) ymm11 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm9, ymm11);
                            mb_h += h_m_stride;
                        }
                        if (m_len > 2) {
                            if (n_len > 0 * N_RF_BLK()) ymm2 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm9, ymm2);
                            if (n_len > 1 * N_RF_BLK()) ymm12 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm9, ymm12);
                            mb_h += h_m_stride;
                        }
                        if (m_len > 3) {
                            if (n_len > 0 * N_RF_BLK()) ymm3 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm9, ymm3);
                            if (n_len > 1 * N_RF_BLK()) ymm13 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm9, ymm13);
                            mb_h += h_m_stride;
                        }
                        if (m_len > 4) {
                            if (n_len > 0 * N_RF_BLK()) ymm4 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm9, ymm4);
                            if (n_len > 1 * N_RF_BLK()) ymm14 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm9, ymm14);
                            mb_h += h_m_stride;
                        }
                        if (m_len > 5) {
                            if (n_len > 0 * N_RF_BLK()) ymm5 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 0 * N_RF_BLK()), ymm9, ymm5);
                            if (n_len > 1 * N_RF_BLK()) ymm15 = _mm256_fmadd_ps(_mm256_loadu_ps(mb_h + 1 * N_RF_BLK()), ymm9, ymm15);
                            mb_h += h_m_stride;
                        }
                    }
                }
            }

            if (nt_store) {
                const int64_t c_m_stride = PICK_PARAM(const int64_t, shar_param, C_M_STRIDE_IDX());
                if (m_len > 0) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_stream_ps(mb_c + 0 * N_RF_BLK(), ymm0);
                    if (n_len > 1 * N_RF_BLK()) _mm256_stream_ps(mb_c + 1 * N_RF_BLK(), ymm10);
                    mb_c += c_m_stride;
                }
                if (m_len > 1) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_stream_ps(mb_c + 0 * N_RF_BLK(), ymm1);
                    if (n_len > 1 * N_RF_BLK()) _mm256_stream_ps(mb_c + 1 * N_RF_BLK(), ymm11);
                    mb_c += c_m_stride;
                }
                if (m_len > 2) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_stream_ps(mb_c + 0 * N_RF_BLK(), ymm2);
                    if (n_len > 1 * N_RF_BLK()) _mm256_stream_ps(mb_c + 1 * N_RF_BLK(), ymm12);
                    mb_c += c_m_stride;
                }
                if (m_len > 3) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_stream_ps(mb_c + 0 * N_RF_BLK(), ymm3);
                    if (n_len > 1 * N_RF_BLK()) _mm256_stream_ps(mb_c + 1 * N_RF_BLK(), ymm13);
                    mb_c += c_m_stride;
                }
                if (m_len > 4) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_stream_ps(mb_c + 0 * N_RF_BLK(), ymm4);
                    if (n_len > 1 * N_RF_BLK()) _mm256_stream_ps(mb_c + 1 * N_RF_BLK(), ymm14);
                    mb_c += c_m_stride;
                }
                if (m_len > 5) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_stream_ps(mb_c + 0 * N_RF_BLK(), ymm5);
                    if (n_len > 1 * N_RF_BLK()) _mm256_stream_ps(mb_c + 1 * N_RF_BLK(), ymm15);
                    mb_c += c_m_stride;
                }
            } else {
                const int64_t c_m_stride = PICK_PARAM(const int64_t, shar_param, C_M_STRIDE_IDX());
                if (m_len > 0) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 0 * N_RF_BLK(), ymm0);
                    if (n_len > 1 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 1 * N_RF_BLK(), ymm10);
                    mb_c += c_m_stride;
                }
                if (m_len > 1) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 0 * N_RF_BLK(), ymm1);
                    if (n_len > 1 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 1 * N_RF_BLK(), ymm11);
                    mb_c += c_m_stride;
                }
                if (m_len > 2) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 0 * N_RF_BLK(), ymm2);
                    if (n_len > 1 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 1 * N_RF_BLK(), ymm12);
                    mb_c += c_m_stride;
                }
                if (m_len > 3) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 0 * N_RF_BLK(), ymm3);
                    if (n_len > 1 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 1 * N_RF_BLK(), ymm13);
                    mb_c += c_m_stride;
                }
                if (m_len > 4) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 0 * N_RF_BLK(), ymm4);
                    if (n_len > 1 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 1 * N_RF_BLK(), ymm14);
                    mb_c += c_m_stride;
                }
                if (m_len > 5) {
                    if (n_len > 0 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 0 * N_RF_BLK(), ymm5);
                    if (n_len > 1 * N_RF_BLK()) _mm256_storeu_ps(mb_c + 1 * N_RF_BLK(), ymm15);
                    mb_c += c_m_stride;
                }
            }
        }
        { // next m block
            const int64_t a_mblk_stride = PICK_PARAM(const int64_t, shar_param, A_MBLK_STRIDE_IDX());
            mb_a += a_mblk_stride;
            m -= m_len;
        }
    } while (m > 0);
#undef K_COMPUTE_STEP
#undef K_PREFETCH_STEP

#ifdef DO_PREFETCH_B
#undef DO_PREFETCH_B
#endif
}

#define N16M6_KERNEL_TABLE_BLK(KER_FORM, NT_STORE, PREFTH_A) \
{\
    {\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 1 * N_RF_BLK(), 1>,\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 1 * N_RF_BLK(), 2>,\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 1 * N_RF_BLK(), 3>,\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 1 * N_RF_BLK(), 4>,\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 1 * N_RF_BLK(), 5>,\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 1 * N_RF_BLK(), 6>,\
    },\
    {\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 2 * N_RF_BLK(), 1>,\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 2 * N_RF_BLK(), 2>,\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 2 * N_RF_BLK(), 3>,\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 2 * N_RF_BLK(), 4>,\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 2 * N_RF_BLK(), 5>,\
        gemm_nn_bcasta_vloadb_m6n16k16_fp32_fma_kernel<KER_FORM, NT_STORE, PREFTH_A, 2 * N_RF_BLK(), 6>,\
    },\
}

gemm_nn_bcasta_vloadb_m6n16k16_kernel_fp32_fma_func_t
gemm_nn_bcasta_vloadb_m6n16k16_kernel_fp32_fma_table[KER_FORM_OPT()][NT_STORE_OPT()][PREFTH_A_OPT()][M6_N_RF()][M6_M_RF()] = 
{
    {
        {
            N16M6_KERNEL_TABLE_BLK(KER_FORM_NONE(), false, false),
            N16M6_KERNEL_TABLE_BLK(KER_FORM_NONE(), false, true),
        },
        {
            N16M6_KERNEL_TABLE_BLK(KER_FORM_NONE(), true, false),
            N16M6_KERNEL_TABLE_BLK(KER_FORM_NONE(), true, true),
        },
    },
    {
        {
            N16M6_KERNEL_TABLE_BLK(KER_FORM_GEMM(), false, false),
            N16M6_KERNEL_TABLE_BLK(KER_FORM_GEMM(), false, true),
        },
        {
            N16M6_KERNEL_TABLE_BLK(KER_FORM_GEMM(), true, false),
            N16M6_KERNEL_TABLE_BLK(KER_FORM_GEMM(), true, true),
        },
    },
    {
        {
            N16M6_KERNEL_TABLE_BLK(KER_FORM_CONV(), false, false),
            N16M6_KERNEL_TABLE_BLK(KER_FORM_CONV(), false, true),
        },
        {
            N16M6_KERNEL_TABLE_BLK(KER_FORM_CONV(), true, false),
            N16M6_KERNEL_TABLE_BLK(KER_FORM_CONV(), true, true),
        },
    },
};

}}};
