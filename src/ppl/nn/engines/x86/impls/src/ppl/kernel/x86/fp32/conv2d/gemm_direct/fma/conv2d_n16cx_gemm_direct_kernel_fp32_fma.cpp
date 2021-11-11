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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_GEMM_DIRECT_FMA_CONV2D_N16CX_GEMM_DIRECT_BLK1X6_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_GEMM_DIRECT_FMA_CONV2D_N16CX_GEMM_DIRECT_BLK1X6_KERNEL_FP32_FMA_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/fma/conv2d_n16cx_gemm_direct_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template <bool nt_store, int32_t u_s>
void conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel_core(int64_t *param)
{
    __asm__ __volatile__ (
        ".equ P_BYTES, 8\n"
        ".equ D_BYTES, 4\n"

        ".equ SRC_PTR_IDX,  (0 * P_BYTES)\n"
        ".equ HIS_PTR_IDX,  (1 * P_BYTES)\n"
        ".equ DST_PTR_IDX,  (2 * P_BYTES)\n"
        ".equ FLT_PTR_IDX,  (3 * P_BYTES)\n"
        ".equ BIAS_PTR_IDX, (4 * P_BYTES)\n"
        ".equ SPACE_IDX,    (5 * P_BYTES)\n"
        ".equ CHANNELS_IDX,       (6 * P_BYTES)\n"
        ".equ SRC_ICB_STRIDE_IDX, (7 * P_BYTES)\n"
        ".equ FLAGS_IDX,          (8 * P_BYTES)\n"

        ".equ NT_STORE, %c[NT_STORE]\n"
        ".equ U_S, %c[U_S]\n"
        ".equ IC_DATA_BLK, %c[IC_DATA_BLK]\n"
        ".equ OC_DATA_BLK, %c[OC_DATA_BLK]\n"
        ".equ OC_REG_ELTS, %c[OC_REG_ELTS]\n"
        ".equ KERNEL_FLAG_LD_BIAS, %c[KERNEL_FLAG_LD_BIAS]\n"
        ".equ KERNEL_FLAG_AD_BIAS, %c[KERNEL_FLAG_AD_BIAS]\n"
        ".equ KERNEL_FLAG_RELU, %c[KERNEL_FLAG_RELU]\n"
        ".equ KERNEL_FLAG_RELU6, %c[KERNEL_FLAG_RELU6]\n"

        "mov SRC_ICB_STRIDE_IDX(%[param]), %%r8\n"
        "mov FLAGS_IDX(%[param]), %%r11\n"
        "mov SRC_PTR_IDX(%[param]), %%r12\n"
        "mov HIS_PTR_IDX(%[param]), %%r13\n"
        "mov DST_PTR_IDX(%[param]), %%r14\n"
        "mov SPACE_IDX(%[param]), %%r15\n"
"1:\n" // label_init_session
        "test $KERNEL_FLAG_LD_BIAS, %%r11\n"
        "jz 2f\n" // label_load_h
        "mov BIAS_PTR_IDX(%[param]), %%r10\n"
        ".if U_S > 0\n"
            "vmovups (0 * OC_REG_ELTS * D_BYTES)(%%r10), %%ymm0\n"
            "vmovups (1 * OC_REG_ELTS * D_BYTES)(%%r10), %%ymm1\n"
        ".endif\n"
        ".if U_S > 1\n"
            "vmovaps %%ymm0, %%ymm2\n"
            "vmovaps %%ymm1, %%ymm3\n"
        ".endif\n"
        ".if U_S > 2\n"
            "vmovaps %%ymm0, %%ymm4\n"
            "vmovaps %%ymm1, %%ymm5\n"
        ".endif\n"
        ".if U_S > 3\n"
            "vmovaps %%ymm0, %%ymm6\n"
            "vmovaps %%ymm1, %%ymm7\n"
        ".endif\n"
        ".if U_S > 4\n"
            "vmovaps %%ymm0, %%ymm8\n"
            "vmovaps %%ymm1, %%ymm9\n"
        ".endif\n"
        ".if U_S > 5\n"
            "vmovaps %%ymm0, %%ymm10\n"
            "vmovaps %%ymm1, %%ymm11\n"
        ".endif\n"
        "jmp 3f\n" // label_load_h_end
"2:\n" // label_load_h
        ".if U_S > 0\n"
            "vmovups ((0 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm0\n"
            "vmovups ((0 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm1\n"
        ".endif\n"
        ".if U_S > 1\n"
            "vmovups ((1 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm2\n"
            "vmovups ((1 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm3\n"
        ".endif\n"
        ".if U_S > 2\n"
            "vmovups ((2 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm4\n"
            "vmovups ((2 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm5\n"
        ".endif\n"
        ".if U_S > 3\n"
            "vmovups ((3 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm6\n"
            "vmovups ((3 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm7\n"
        ".endif\n"
        ".if U_S > 4\n"
            "vmovups ((4 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm8\n"
            "vmovups ((4 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm9\n"
        ".endif\n"
        ".if U_S > 5\n"
            "vmovups ((5 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm10\n"
            "vmovups ((5 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r13), %%ymm11\n"
        ".endif\n"
"3:\n" // label_load_h_end
        "test $KERNEL_FLAG_AD_BIAS, %%r11\n"
        "jz 4f\n" // label_compute_session
        "mov BIAS_PTR_IDX(%[param]), %%r10\n"
        "vmovups (0 * OC_REG_ELTS * D_BYTES)(%%r10), %%ymm14\n"
        "vmovups (1 * OC_REG_ELTS * D_BYTES)(%%r10), %%ymm15\n"
        ".if U_S > 0\n"
            "vaddps %%ymm14, %%ymm0, %%ymm0\n"
            "vaddps %%ymm15, %%ymm1, %%ymm1\n"
        ".endif\n"
        ".if U_S > 1\n"
            "vaddps %%ymm14, %%ymm2, %%ymm2\n"
            "vaddps %%ymm15, %%ymm3, %%ymm3\n"
        ".endif\n"
        ".if U_S > 2\n"
            "vaddps %%ymm14, %%ymm4, %%ymm4\n"
            "vaddps %%ymm15, %%ymm5, %%ymm5\n"
        ".endif\n"
        ".if U_S > 3\n"
            "vaddps %%ymm14, %%ymm6, %%ymm6\n"
            "vaddps %%ymm15, %%ymm7, %%ymm7\n"
        ".endif\n"
        ".if U_S > 4\n"
            "vaddps %%ymm14, %%ymm8, %%ymm8\n"
            "vaddps %%ymm15, %%ymm9, %%ymm9\n"
        ".endif\n"
        ".if U_S > 5\n"
            "vaddps %%ymm14, %%ymm10, %%ymm10\n"
            "vaddps %%ymm15, %%ymm11, %%ymm11\n"
        ".endif\n"

"4:\n" // label_compute_session
        "mov %%r12, %%rax\n"
        "mov FLT_PTR_IDX(%[param]), %%rbx\n"
        "mov CHANNELS_IDX(%[param]), %%r10\n"
        "cmp $IC_DATA_BLK, %%r10\n"
        "jl 6f\n" // label_ic_remain
"5:\n" // label_ic_body
        PPL_X86_INLINE_ASM_ALIGN()
        ".irp IC,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
        "vmovups ((\\IC * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%rbx), %%ymm14\n"
        "vmovups ((\\IC * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%rbx), %%ymm15\n"
        ".if U_S > 3\n"
        "prefetcht0 ((\\IC * OC_DATA_BLK + IC_DATA_BLK * OC_DATA_BLK) * D_BYTES)(%%rbx)\n"
        ".endif\n"
        ".if U_S > 0\n"
        "vbroadcastss ((\\IC + 0 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm1\n"
        ".endif\n"
        ".if U_S > 1\n"
        "vbroadcastss ((\\IC + 1 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm2\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm3\n"
        ".endif\n"
        ".if U_S > 2\n"
        "vbroadcastss ((\\IC + 2 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm5\n"
        ".endif\n"
        ".if U_S > 3\n"
        "vbroadcastss ((\\IC + 3 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm6\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n"
        ".endif\n"
        ".if U_S > 4\n"
        "vbroadcastss ((\\IC + 4 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n"
        ".endif\n"
        ".if U_S > 5\n"
        "vbroadcastss ((\\IC + 5 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm10\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm11\n"
        ".endif\n"
        ".endr\n"
        "lea (IC_DATA_BLK * OC_DATA_BLK * D_BYTES)(%%rbx), %%rbx\n"
        "lea (%%rax, %%r8, D_BYTES), %%rax\n"
        "sub $IC_DATA_BLK, %%r10\n"
        "cmp $IC_DATA_BLK, %%r10\n"
        "jge 5b\n" // label_ic_body
        "cmp $0, %%r10\n"
        "je 7f\n" // label_finalize_session
"6:\n" // label_ic_remain
        "vmovups ((0 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%rbx), %%ymm14\n"
        "vmovups ((0 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%rbx), %%ymm15\n"
        ".if U_S > 0\n"
        "vbroadcastss ((0 + 0 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm1\n"
        ".endif\n"
        ".if U_S > 1\n"
        "vbroadcastss ((0 + 1 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm2\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm3\n"
        ".endif\n"
        ".if U_S > 2\n"
        "vbroadcastss ((0 + 2 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm5\n"
        ".endif\n"
        ".if U_S > 3\n"
        "vbroadcastss ((0 + 3 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm6\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n"
        ".endif\n"
        ".if U_S > 4\n"
        "vbroadcastss ((0 + 4 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n"
        ".endif\n"
        ".if U_S > 5\n"
        "vbroadcastss ((0 + 5 * IC_DATA_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm10\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm11\n"
        ".endif\n"
        "lea (OC_DATA_BLK * D_BYTES)(%%rbx), %%rbx\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%r10\n"
        "cmp $0, %%r10\n"
        "jne 6b\n" // label_ic_remain

"7:\n" // label_finalize_session
        "test $(KERNEL_FLAG_RELU | KERNEL_FLAG_RELU6), %%r11\n"
        "jz 8f\n" // label_relu_end
        "vxorps %%ymm14, %%ymm14, %%ymm14\n"
        ".if U_S > 0\n"
            "vmaxps %%ymm14, %%ymm0, %%ymm0\n"
            "vmaxps %%ymm14, %%ymm1, %%ymm1\n"
        ".endif\n"
        ".if U_S > 1\n"
            "vmaxps %%ymm14, %%ymm2, %%ymm2\n"
            "vmaxps %%ymm14, %%ymm3, %%ymm3\n"
        ".endif\n"
        ".if U_S > 2\n"
            "vmaxps %%ymm14, %%ymm4, %%ymm4\n"
            "vmaxps %%ymm14, %%ymm5, %%ymm5\n"
        ".endif\n"
        ".if U_S > 3\n"
            "vmaxps %%ymm14, %%ymm6, %%ymm6\n"
            "vmaxps %%ymm14, %%ymm7, %%ymm7\n"
        ".endif\n"
        ".if U_S > 4\n"
            "vmaxps %%ymm14, %%ymm8, %%ymm8\n"
            "vmaxps %%ymm14, %%ymm9, %%ymm9\n"
        ".endif\n"
        ".if U_S > 5\n"
            "vmaxps %%ymm14, %%ymm10, %%ymm10\n"
            "vmaxps %%ymm14, %%ymm11, %%ymm11\n"
        ".endif\n"
        "test $KERNEL_FLAG_RELU6, %%r11\n"
        "jz 8f\n" // label_relu_end
        "mov $0x40c00000, %%ecx\n"
        "vmovd %%ecx, %%xmm15\n"
        "vbroadcastss %%xmm15, %%ymm15\n"
        ".if U_S > 0\n"
            "vminps %%ymm15, %%ymm0, %%ymm0\n"
            "vminps %%ymm15, %%ymm1, %%ymm1\n"
        ".endif\n"
        ".if U_S > 1\n"
            "vminps %%ymm15, %%ymm2, %%ymm2\n"
            "vminps %%ymm15, %%ymm3, %%ymm3\n"
        ".endif\n"
        ".if U_S > 2\n"
            "vminps %%ymm15, %%ymm4, %%ymm4\n"
            "vminps %%ymm15, %%ymm5, %%ymm5\n"
        ".endif\n"
        ".if U_S > 3\n"
            "vminps %%ymm15, %%ymm6, %%ymm6\n"
            "vminps %%ymm15, %%ymm7, %%ymm7\n"
        ".endif\n"
        ".if U_S > 4\n"
            "vminps %%ymm15, %%ymm8, %%ymm8\n"
            "vminps %%ymm15, %%ymm9, %%ymm9\n"
        ".endif\n"
        ".if U_S > 5\n"
            "vminps %%ymm15, %%ymm10, %%ymm10\n"
            "vminps %%ymm15, %%ymm11, %%ymm11\n"
        ".endif\n"
"8:\n" // label_relu_end

        ".if NT_STORE\n"
        ".if U_S > 0\n"
            "vmovntps %%ymm0, ((0 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm1, ((0 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_S > 1\n"
            "vmovntps %%ymm2, ((1 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm3, ((1 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_S > 2\n"
            "vmovntps %%ymm4, ((2 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm5, ((2 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_S > 3\n"
            "vmovntps %%ymm6, ((3 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm7, ((3 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_S > 4\n"
            "vmovntps %%ymm8, ((4 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm9, ((4 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_S > 5\n"
            "vmovntps %%ymm10, ((5 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm11, ((5 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".else\n"
        ".if U_S > 0\n"
            "vmovups %%ymm0, ((0 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm1, ((0 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_S > 1\n"
            "vmovups %%ymm2, ((1 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm3, ((1 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_S > 2\n"
            "vmovups %%ymm4, ((2 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm5, ((2 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_S > 3\n"
            "vmovups %%ymm6, ((3 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm7, ((3 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_S > 4\n"
            "vmovups %%ymm8, ((4 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm9, ((4 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if U_S > 5\n"
            "vmovups %%ymm10, ((5 * OC_DATA_BLK + 0 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm11, ((5 * OC_DATA_BLK + 1 * OC_REG_ELTS) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".endif\n"
        "sub $U_S, %%r15\n"
        "cmp $0, %%r15\n"
        "lea (U_S * IC_DATA_BLK * D_BYTES)(%%r12), %%r12\n"
        "lea (U_S * OC_DATA_BLK * D_BYTES)(%%r13), %%r13\n"
        "lea (U_S * OC_DATA_BLK * D_BYTES)(%%r14), %%r14\n"
        "jne 1b\n" // label_init_session
        :
        :
        [param]                       "r" (param),
        [NT_STORE]                    "i" (nt_store),
        [U_S]                         "i" (u_s),
        [IC_DATA_BLK]                 "i" (conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::IC_DATA_BLK),
        [OC_DATA_BLK]                 "i" (conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_DATA_BLK),
        [OC_REG_ELTS]                 "i" (conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS),
        [KERNEL_FLAG_LD_BIAS]         "i" (conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::LOAD_BIAS),
        [KERNEL_FLAG_AD_BIAS]         "i" (conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::ADD_BIAS),
        [KERNEL_FLAG_RELU]            "i" (conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::RELU),
        [KERNEL_FLAG_RELU6]           "i" (conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::RELU6)
        :
        "cc",
        "rax", "rbx", "rcx",
        "r8" , "r9" , "r10", "r11", "r12", "r13", "r14", "r15",
        "ymm0" , "ymm1" , "ymm2" , "ymm3" , "ymm4" , "ymm5" , "ymm6" , "ymm7" ,
        "ymm8" , "ymm9" , "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
        "memory"
    );
}

#endif

template <bool nt_store, int32_t u_oc, int32_t u_s>
void conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel(int64_t *param)
{
#ifdef PPL_USE_X86_INLINE_ASM
    if (u_oc == 2 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS) {
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel_core<nt_store, u_s>(param);
        return;
    }
#endif

#define IC_COMPUTE_STEP(IC) do {\
    if (u_ocr > 0) ymm14 = _mm256_loadu_ps(ic_flt + (IC) * OC_DATA_BLK + 0 * OC_REG_ELTS);\
    if (u_ocr > 1) ymm15 = _mm256_loadu_ps(ic_flt + (IC) * OC_DATA_BLK + 1 * OC_REG_ELTS);\
    if (u_s > 0) {\
        ymm12 = _mm256_set1_ps(ic_src[(IC) + 0 * IC_DATA_BLK]);\
        if (u_ocr > 0) ymm0 = _mm256_fmadd_ps(ymm14, ymm12, ymm0);\
        if (u_ocr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm12, ymm1);\
    }\
    if (u_s > 1) {\
        ymm13 = _mm256_set1_ps(ic_src[(IC) + 1 * IC_DATA_BLK]);\
        if (u_ocr > 0) ymm2 = _mm256_fmadd_ps(ymm14, ymm13, ymm2);\
        if (u_ocr > 1) ymm3 = _mm256_fmadd_ps(ymm15, ymm13, ymm3);\
    }\
    if (u_s > 2) {\
        ymm12 = _mm256_set1_ps(ic_src[(IC) + 2 * IC_DATA_BLK]);\
        if (u_ocr > 0) ymm4 = _mm256_fmadd_ps(ymm14, ymm12, ymm4);\
        if (u_ocr > 1) ymm5 = _mm256_fmadd_ps(ymm15, ymm12, ymm5);\
    }\
    if (u_s > 3) {\
        ymm13 = _mm256_set1_ps(ic_src[(IC) + 3 * IC_DATA_BLK]);\
        if (u_ocr > 0) ymm6 = _mm256_fmadd_ps(ymm14, ymm13, ymm6);\
        if (u_ocr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);\
    }\
    if (u_s > 4) {\
        ymm12 = _mm256_set1_ps(ic_src[(IC) + 4 * IC_DATA_BLK]);\
        if (u_ocr > 0) ymm8 = _mm256_fmadd_ps(ymm14, ymm12, ymm8);\
        if (u_ocr > 1) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);\
    }\
    if (u_s > 5) {\
        ymm13 = _mm256_set1_ps(ic_src[(IC) + 5 * IC_DATA_BLK]);\
        if (u_ocr > 0) ymm10 = _mm256_fmadd_ps(ymm14, ymm13, ymm10);\
        if (u_ocr > 1) ymm11 = _mm256_fmadd_ps(ymm15, ymm13, ymm11);\
    }\
} while (0)

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    const int64_t IC_DATA_BLK = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::IC_DATA_BLK;
    const int64_t OC_DATA_BLK = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_DATA_BLK;
    const int64_t OC_REG_ELTS = conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS;
    const int64_t u_ocr = div_up(u_oc, OC_REG_ELTS);

    array_param_helper ker_p(param);

    const int64_t src_icb_stride = ker_p.pick<const int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SRC_ICB_STRIDE_IDX);
    const int64_t flags = ker_p.pick<const int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::FLAGS_IDX);

    const float *src = ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SRC_PTR_IDX);
    const float *his = ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::HIS_PTR_IDX);
    float *dst       = ker_p.pick<float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::DST_PTR_IDX);
    int64_t space    = ker_p.pick<const int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::SPACE_IDX);
    do {
        if (flags & conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::LOAD_BIAS) {
            const float* bias = ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::BIAS_PTR_IDX);
            if (u_s > 0) {
                if (u_ocr > 0) ymm0 = _mm256_loadu_ps(bias + 0 * OC_REG_ELTS);
                if (u_ocr > 1) ymm1 = _mm256_loadu_ps(bias + 1 * OC_REG_ELTS);
            }
            if (u_s > 1) {
                if (u_ocr > 0) ymm2 = ymm0;
                if (u_ocr > 1) ymm3 = ymm1;
            }
            if (u_s > 2) {
                if (u_ocr > 0) ymm4 = ymm0;
                if (u_ocr > 1) ymm5 = ymm1;
            }
            if (u_s > 3) {
                if (u_ocr > 0) ymm6 = ymm0;
                if (u_ocr > 1) ymm7 = ymm1;
            }
            if (u_s > 4) {
                if (u_ocr > 0) ymm8 = ymm0;
                if (u_ocr > 1) ymm9 = ymm1;
            }
            if (u_s > 5) {
                if (u_ocr > 0) ymm10 = ymm0;
                if (u_ocr > 1) ymm11 = ymm1;
            }
        } else {
            if (u_s > 0) {
                if (u_ocr > 0) ymm0 = _mm256_loadu_ps(his + 0 * OC_DATA_BLK + 0 * OC_REG_ELTS);
                if (u_ocr > 1) ymm1 = _mm256_loadu_ps(his + 0 * OC_DATA_BLK + 1 * OC_REG_ELTS);
            }
            if (u_s > 1) {
                if (u_ocr > 0) ymm2 = _mm256_loadu_ps(his + 1 * OC_DATA_BLK + 0 * OC_REG_ELTS);
                if (u_ocr > 1) ymm3 = _mm256_loadu_ps(his + 1 * OC_DATA_BLK + 1 * OC_REG_ELTS);
            }
            if (u_s > 2) {
                if (u_ocr > 0) ymm4 = _mm256_loadu_ps(his + 2 * OC_DATA_BLK + 0 * OC_REG_ELTS);
                if (u_ocr > 1) ymm5 = _mm256_loadu_ps(his + 2 * OC_DATA_BLK + 1 * OC_REG_ELTS);
            }
            if (u_s > 3) {
                if (u_ocr > 0) ymm6 = _mm256_loadu_ps(his + 3 * OC_DATA_BLK + 0 * OC_REG_ELTS);
                if (u_ocr > 1) ymm7 = _mm256_loadu_ps(his + 3 * OC_DATA_BLK + 1 * OC_REG_ELTS);
            }
            if (u_s > 4) {
                if (u_ocr > 0) ymm8 = _mm256_loadu_ps(his + 4 * OC_DATA_BLK + 0 * OC_REG_ELTS);
                if (u_ocr > 1) ymm9 = _mm256_loadu_ps(his + 4 * OC_DATA_BLK + 1 * OC_REG_ELTS);
            }
            if (u_s > 5) {
                if (u_ocr > 0) ymm10 = _mm256_loadu_ps(his + 5 * OC_DATA_BLK + 0 * OC_REG_ELTS);
                if (u_ocr > 1) ymm11 = _mm256_loadu_ps(his + 5 * OC_DATA_BLK + 1 * OC_REG_ELTS);
            }
        }

        if (flags & conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::ADD_BIAS) {
            const float* bias = ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::BIAS_PTR_IDX);
            if (u_ocr > 0) ymm14 = _mm256_loadu_ps(bias + 0 * OC_REG_ELTS);
            if (u_ocr > 1) ymm15 = _mm256_loadu_ps(bias + 1 * OC_REG_ELTS);
            if (u_s > 0) {
                if (u_ocr > 0) ymm0 = _mm256_add_ps(ymm14, ymm0);
                if (u_ocr > 1) ymm1 = _mm256_add_ps(ymm15, ymm1);
            }
            if (u_s > 1) {
                if (u_ocr > 0) ymm2 = _mm256_add_ps(ymm14, ymm2);
                if (u_ocr > 1) ymm3 = _mm256_add_ps(ymm15, ymm3);
            }
            if (u_s > 2) {
                if (u_ocr > 0) ymm4 = _mm256_add_ps(ymm14, ymm4);
                if (u_ocr > 1) ymm5 = _mm256_add_ps(ymm15, ymm5);
            }
            if (u_s > 3) {
                if (u_ocr > 0) ymm6 = _mm256_add_ps(ymm14, ymm6);
                if (u_ocr > 1) ymm7 = _mm256_add_ps(ymm15, ymm7);
            }
            if (u_s > 4) {
                if (u_ocr > 0) ymm8 = _mm256_add_ps(ymm14, ymm8);
                if (u_ocr > 1) ymm9 = _mm256_add_ps(ymm15, ymm9);
            }
            if (u_s > 5) {
                if (u_ocr > 0) ymm10 = _mm256_add_ps(ymm14, ymm10);
                if (u_ocr > 1) ymm11 = _mm256_add_ps(ymm15, ymm11);
            }
        }
        
        const float *ic_src = src;
        const float *ic_flt = ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::FLT_PTR_IDX);
        int64_t channels     = ker_p.pick<const int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_fma::param_def::CHANNELS_IDX);
        while (channels >= IC_DATA_BLK) {
            channels -= IC_DATA_BLK;
            IC_COMPUTE_STEP(0);
            IC_COMPUTE_STEP(1);
            IC_COMPUTE_STEP(2);
            IC_COMPUTE_STEP(3);

            IC_COMPUTE_STEP(4);
            IC_COMPUTE_STEP(5);
            IC_COMPUTE_STEP(6);
            IC_COMPUTE_STEP(7);

            IC_COMPUTE_STEP(8);
            IC_COMPUTE_STEP(9);
            IC_COMPUTE_STEP(10);
            IC_COMPUTE_STEP(11);

            IC_COMPUTE_STEP(12);
            IC_COMPUTE_STEP(13);
            IC_COMPUTE_STEP(14);
            IC_COMPUTE_STEP(15);
            ic_flt += IC_DATA_BLK * OC_DATA_BLK;
            ic_src += src_icb_stride;
        }
        if (channels > 0) {
            for (int64_t ic = 0; ic < channels; ++ic) {
                IC_COMPUTE_STEP(0);
                ic_src += 1;
                ic_flt += OC_DATA_BLK;
            }
        }
        
        if (flags & (conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::RELU | conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::RELU6)) {
            ymm14 = _mm256_setzero_ps();
            if (u_s > 0) {
                if (u_ocr > 0) ymm0 = _mm256_max_ps(ymm14, ymm0);
                if (u_ocr > 1) ymm1 = _mm256_max_ps(ymm14, ymm1);
            }
            if (u_s > 1) {
                if (u_ocr > 0) ymm2 = _mm256_max_ps(ymm14, ymm2);
                if (u_ocr > 1) ymm3 = _mm256_max_ps(ymm14, ymm3);
            }
            if (u_s > 2) {
                if (u_ocr > 0) ymm4 = _mm256_max_ps(ymm14, ymm4);
                if (u_ocr > 1) ymm5 = _mm256_max_ps(ymm14, ymm5);
            }
            if (u_s > 3) {
                if (u_ocr > 0) ymm6 = _mm256_max_ps(ymm14, ymm6);
                if (u_ocr > 1) ymm7 = _mm256_max_ps(ymm14, ymm7);
            }
            if (u_s > 4) {
                if (u_ocr > 0) ymm8 = _mm256_max_ps(ymm14, ymm8);
                if (u_ocr > 1) ymm9 = _mm256_max_ps(ymm14, ymm9);
            }
            if (u_s > 5) {
                if (u_ocr > 0) ymm10 = _mm256_max_ps(ymm14, ymm10);
                if (u_ocr > 1) ymm11 = _mm256_max_ps(ymm14, ymm11);
            }
        }
        if (flags & conv2d_n16cx_gemm_direct_kernel_fp32_fma::flag::RELU6) {
            ymm15 = _mm256_set1_ps(6.0f);
            if (u_s > 0) {
                if (u_ocr > 0) ymm0 = _mm256_min_ps(ymm15, ymm0);
                if (u_ocr > 1) ymm1 = _mm256_min_ps(ymm15, ymm1);
            }
            if (u_s > 1) {
                if (u_ocr > 0) ymm2 = _mm256_min_ps(ymm15, ymm2);
                if (u_ocr > 1) ymm3 = _mm256_min_ps(ymm15, ymm3);
            }
            if (u_s > 2) {
                if (u_ocr > 0) ymm4 = _mm256_min_ps(ymm15, ymm4);
                if (u_ocr > 1) ymm5 = _mm256_min_ps(ymm15, ymm5);
            }
            if (u_s > 3) {
                if (u_ocr > 0) ymm6 = _mm256_min_ps(ymm15, ymm6);
                if (u_ocr > 1) ymm7 = _mm256_min_ps(ymm15, ymm7);
            }
            if (u_s > 4) {
                if (u_ocr > 0) ymm8 = _mm256_min_ps(ymm15, ymm8);
                if (u_ocr > 1) ymm9 = _mm256_min_ps(ymm15, ymm9);
            }
            if (u_s > 5) {
                if (u_ocr > 0) ymm10 = _mm256_min_ps(ymm15, ymm10);
                if (u_ocr > 1) ymm11 = _mm256_min_ps(ymm15, ymm11);
            }
        }

        if (nt_store) {
            if (u_s > 0) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 0 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm0);
                if (u_ocr > 1) _mm256_stream_ps(dst + 0 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm1);
            }
            if (u_s > 1) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 1 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm2);
                if (u_ocr > 1) _mm256_stream_ps(dst + 1 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm3);
            }
            if (u_s > 2) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 2 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm4);
                if (u_ocr > 1) _mm256_stream_ps(dst + 2 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm5);
            }
            if (u_s > 3) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 3 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm6);
                if (u_ocr > 1) _mm256_stream_ps(dst + 3 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm7);
            }
            if (u_s > 4) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 4 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm8);
                if (u_ocr > 1) _mm256_stream_ps(dst + 4 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm9);
            }
            if (u_s > 5) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 5 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm10);
                if (u_ocr > 1) _mm256_stream_ps(dst + 5 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm11);
            }
        } else {
            if (u_s > 0) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 0 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm0);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 0 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm1);
            }
            if (u_s > 1) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 1 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm2);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 1 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm3);
            }
            if (u_s > 2) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 2 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm4);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 2 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm5);
            }
            if (u_s > 3) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 3 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm6);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 3 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm7);
            }
            if (u_s > 4) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 4 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm8);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 4 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm9);
            }
            if (u_s > 5) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 5 * OC_DATA_BLK + 0 * OC_REG_ELTS, ymm10);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 5 * OC_DATA_BLK + 1 * OC_REG_ELTS, ymm11);
            }
        }
        src += u_s * IC_DATA_BLK;
        his += u_s * OC_DATA_BLK;
        dst += u_s * OC_DATA_BLK;
        space -= u_s;
    } while (space > 0);
#undef IC_COMPUTE_STEP
}

#define GEMM_DIRECT_KERNEL_TABLE_BLK(NT_STORE) \
{\
    {\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 1>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 2>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 3>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 4>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 5>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 6>,\
    },\
    {\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 1>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 2>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 3>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 4>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 5>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * conv2d_n16cx_gemm_direct_kernel_fp32_fma::config::OC_REG_ELTS, 6>,\
    },\
}

const conv2d_n16cx_gemm_direct_kernel_fp32_fma::func_t
    conv2d_n16cx_gemm_direct_kernel_fp32_fma::table_[config::NT_STORE_OPT][config::MAX_OC_REGS][config::MAX_S_REGS] =
{
    GEMM_DIRECT_KERNEL_TABLE_BLK(false),
    GEMM_DIRECT_KERNEL_TABLE_BLK(true),
};

}}};

#endif
