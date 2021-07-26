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

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template <bool nt_store, int32_t hw_len>
void conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel_core(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
    __asm__ __volatile__ (
        ".equ P_BYTES, 8\n"
        ".equ D_BYTES, 4\n"

        ".equ CH_DT_BLK, 16\n"
        ".equ CH_RF_BLK, 8\n"

        ".equ SRC_IDX,  (0 * P_BYTES)\n"
        ".equ HIS_IDX,  (1 * P_BYTES)\n"
        ".equ DST_IDX,  (2 * P_BYTES)\n"
        ".equ FLT_IDX,  (3 * P_BYTES)\n"
        ".equ BIAS_IDX, (4 * P_BYTES)\n"
        ".equ HW_IDX,   (5 * P_BYTES)\n"

        ".equ CHANNELS_IDX,       (0 * P_BYTES)\n"
        ".equ SRC_ICB_STRIDE_IDX, (1 * P_BYTES)\n"
        ".equ FLAGS_IDX,          (2 * P_BYTES)\n"
        ".equ SIX_IDX,            (3 * P_BYTES)\n"

        ".equ NT_STORE, %c[NT_STORE]\n"
        ".equ HW_LEN, %c[HW_LEN]\n"
        ".equ KERNEL_FLAG_LD_BIAS, %c[KERNEL_FLAG_LD_BIAS]\n"
        ".equ KERNEL_FLAG_AD_BIAS, %c[KERNEL_FLAG_AD_BIAS]\n"
        ".equ KERNEL_FLAG_RELU, %c[KERNEL_FLAG_RELU]\n"
        ".equ KERNEL_FLAG_RELU6, %c[KERNEL_FLAG_RELU6]\n"

        "mov SRC_ICB_STRIDE_IDX(%[shar_param]), %%r8\n"
        "mov FLAGS_IDX(%[shar_param]), %%r11\n"
        "mov SRC_IDX(%[priv_param]), %%r12\n"
        "mov HIS_IDX(%[priv_param]), %%r13\n"
        "mov DST_IDX(%[priv_param]), %%r14\n"
        "mov HW_IDX(%[priv_param]), %%r15\n"
"1:\n" // label_init_session
        "test $KERNEL_FLAG_LD_BIAS, %%r11\n"
        "jz 2f\n" // label_load_h
        "mov BIAS_IDX(%[priv_param]), %%r10\n"
        ".if HW_LEN > 0\n"
            "vmovups (0 * CH_RF_BLK * D_BYTES)(%%r10), %%ymm0\n"
            "vmovups (1 * CH_RF_BLK * D_BYTES)(%%r10), %%ymm1\n"
        ".endif\n"
        ".if HW_LEN > 1\n"
            "vmovaps %%ymm0, %%ymm2\n"
            "vmovaps %%ymm1, %%ymm3\n"
        ".endif\n"
        ".if HW_LEN > 2\n"
            "vmovaps %%ymm0, %%ymm4\n"
            "vmovaps %%ymm1, %%ymm5\n"
        ".endif\n"
        ".if HW_LEN > 3\n"
            "vmovaps %%ymm0, %%ymm6\n"
            "vmovaps %%ymm1, %%ymm7\n"
        ".endif\n"
        ".if HW_LEN > 4\n"
            "vmovaps %%ymm0, %%ymm8\n"
            "vmovaps %%ymm1, %%ymm9\n"
        ".endif\n"
        ".if HW_LEN > 5\n"
            "vmovaps %%ymm0, %%ymm10\n"
            "vmovaps %%ymm1, %%ymm11\n"
        ".endif\n"
        "jmp 3f\n" // label_load_h_end
"2:\n" // label_load_h
        ".if HW_LEN > 0\n"
            "vmovups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm0\n"
            "vmovups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm1\n"
        ".endif\n"
        ".if HW_LEN > 1\n"
            "vmovups ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm2\n"
            "vmovups ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm3\n"
        ".endif\n"
        ".if HW_LEN > 2\n"
            "vmovups ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm4\n"
            "vmovups ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm5\n"
        ".endif\n"
        ".if HW_LEN > 3\n"
            "vmovups ((3 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm6\n"
            "vmovups ((3 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm7\n"
        ".endif\n"
        ".if HW_LEN > 4\n"
            "vmovups ((4 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm8\n"
            "vmovups ((4 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm9\n"
        ".endif\n"
        ".if HW_LEN > 5\n"
            "vmovups ((5 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm10\n"
            "vmovups ((5 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r13), %%ymm11\n"
        ".endif\n"
"3:\n" // label_load_h_end
        "test $KERNEL_FLAG_AD_BIAS, %%r11\n"
        "jz 4f\n" // label_compute_session
        "mov BIAS_IDX(%[priv_param]), %%r10\n"
        "vmovups (0 * CH_RF_BLK * D_BYTES)(%%r10), %%ymm14\n"
        "vmovups (1 * CH_RF_BLK * D_BYTES)(%%r10), %%ymm15\n"
        ".if HW_LEN > 0\n"
            "vaddps %%ymm14, %%ymm0, %%ymm0\n"
            "vaddps %%ymm15, %%ymm1, %%ymm1\n"
        ".endif\n"
        ".if HW_LEN > 1\n"
            "vaddps %%ymm14, %%ymm2, %%ymm2\n"
            "vaddps %%ymm15, %%ymm3, %%ymm3\n"
        ".endif\n"
        ".if HW_LEN > 2\n"
            "vaddps %%ymm14, %%ymm4, %%ymm4\n"
            "vaddps %%ymm15, %%ymm5, %%ymm5\n"
        ".endif\n"
        ".if HW_LEN > 3\n"
            "vaddps %%ymm14, %%ymm6, %%ymm6\n"
            "vaddps %%ymm15, %%ymm7, %%ymm7\n"
        ".endif\n"
        ".if HW_LEN > 4\n"
            "vaddps %%ymm14, %%ymm8, %%ymm8\n"
            "vaddps %%ymm15, %%ymm9, %%ymm9\n"
        ".endif\n"
        ".if HW_LEN > 5\n"
            "vaddps %%ymm14, %%ymm10, %%ymm10\n"
            "vaddps %%ymm15, %%ymm11, %%ymm11\n"
        ".endif\n"

"4:\n" // label_compute_session
        "mov %%r12, %%rax\n"
        "mov FLT_IDX(%[priv_param]), %%rbx\n"
        "mov CHANNELS_IDX(%[shar_param]), %%r10\n"
        "cmp $CH_DT_BLK, %%r10\n"
        "jl 6f\n" // label_ic_remain
"5:\n" // label_ic_body
        PPL_X86_INLINE_ASM_ALIGN()
        ".irp IC,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
        "vmovups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx), %%ymm14\n"
        "vmovups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx), %%ymm15\n"
        ".if HW_LEN > 0\n"
        "vbroadcastss ((\\IC + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm1\n"
        ".endif\n"
        ".if HW_LEN > 1\n"
        "vbroadcastss ((\\IC + 1 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm2\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm3\n"
        ".endif\n"
        ".if HW_LEN > 2\n"
        "vbroadcastss ((\\IC + 2 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm5\n"
        ".endif\n"
        ".if HW_LEN > 3\n"
        "vbroadcastss ((\\IC + 3 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm6\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n"
        ".endif\n"
        ".if HW_LEN > 4\n"
        "vbroadcastss ((\\IC + 4 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n"
        ".endif\n"
        ".if HW_LEN > 5\n"
        "vbroadcastss ((\\IC + 5 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm10\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm11\n"
        ".endif\n"
        ".endr\n"
        "lea (CH_DT_BLK * CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        "lea (%%rax, %%r8, D_BYTES), %%rax\n"
        "sub $CH_DT_BLK, %%r10\n"
        "cmp $CH_DT_BLK, %%r10\n"
        "jge 5b\n" // label_ic_body
        "cmp $0, %%r10\n"
        "je 7f\n" // label_finalize_session
"6:\n" // label_ic_remain
        "vmovups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx), %%ymm14\n"
        "vmovups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx), %%ymm15\n"
        ".if HW_LEN > 0\n"
        "vbroadcastss ((0 + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm1\n"
        ".endif\n"
        ".if HW_LEN > 1\n"
        "vbroadcastss ((0 + 1 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm2\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm3\n"
        ".endif\n"
        ".if HW_LEN > 2\n"
        "vbroadcastss ((0 + 2 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm5\n"
        ".endif\n"
        ".if HW_LEN > 3\n"
        "vbroadcastss ((0 + 3 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm6\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n"
        ".endif\n"
        ".if HW_LEN > 4\n"
        "vbroadcastss ((0 + 4 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n"
        ".endif\n"
        ".if HW_LEN > 5\n"
        "vbroadcastss ((0 + 5 * CH_DT_BLK) * D_BYTES)(%%rax), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm10\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm11\n"
        ".endif\n"
        "lea (CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%r10\n"
        "cmp $0, %%r10\n"
        "jne 6b\n" // label_ic_remain

"7:\n" // label_finalize_session
        "test $(KERNEL_FLAG_RELU | KERNEL_FLAG_RELU6), %%r11\n"
        "jz 8f\n" // label_relu_end
        "vxorps %%ymm14, %%ymm14, %%ymm14\n"
        ".if HW_LEN > 0\n"
            "vmaxps %%ymm14, %%ymm0, %%ymm0\n"
            "vmaxps %%ymm14, %%ymm1, %%ymm1\n"
        ".endif\n"
        ".if HW_LEN > 1\n"
            "vmaxps %%ymm14, %%ymm2, %%ymm2\n"
            "vmaxps %%ymm14, %%ymm3, %%ymm3\n"
        ".endif\n"
        ".if HW_LEN > 2\n"
            "vmaxps %%ymm14, %%ymm4, %%ymm4\n"
            "vmaxps %%ymm14, %%ymm5, %%ymm5\n"
        ".endif\n"
        ".if HW_LEN > 3\n"
            "vmaxps %%ymm14, %%ymm6, %%ymm6\n"
            "vmaxps %%ymm14, %%ymm7, %%ymm7\n"
        ".endif\n"
        ".if HW_LEN > 4\n"
            "vmaxps %%ymm14, %%ymm8, %%ymm8\n"
            "vmaxps %%ymm14, %%ymm9, %%ymm9\n"
        ".endif\n"
        ".if HW_LEN > 5\n"
            "vmaxps %%ymm14, %%ymm10, %%ymm10\n"
            "vmaxps %%ymm14, %%ymm11, %%ymm11\n"
        ".endif\n"
        "test $KERNEL_FLAG_RELU6, %%r11\n"
        "jz 8f\n" // label_relu_end
        "vbroadcastss SIX_IDX(%[shar_param]), %%ymm15\n"
        ".if HW_LEN > 0\n"
            "vminps %%ymm15, %%ymm0, %%ymm0\n"
            "vminps %%ymm15, %%ymm1, %%ymm1\n"
        ".endif\n"
        ".if HW_LEN > 1\n"
            "vminps %%ymm15, %%ymm2, %%ymm2\n"
            "vminps %%ymm15, %%ymm3, %%ymm3\n"
        ".endif\n"
        ".if HW_LEN > 2\n"
            "vminps %%ymm15, %%ymm4, %%ymm4\n"
            "vminps %%ymm15, %%ymm5, %%ymm5\n"
        ".endif\n"
        ".if HW_LEN > 3\n"
            "vminps %%ymm15, %%ymm6, %%ymm6\n"
            "vminps %%ymm15, %%ymm7, %%ymm7\n"
        ".endif\n"
        ".if HW_LEN > 4\n"
            "vminps %%ymm15, %%ymm8, %%ymm8\n"
            "vminps %%ymm15, %%ymm9, %%ymm9\n"
        ".endif\n"
        ".if HW_LEN > 5\n"
            "vminps %%ymm15, %%ymm10, %%ymm10\n"
            "vminps %%ymm15, %%ymm11, %%ymm11\n"
        ".endif\n"
"8:\n" // label_relu_end

        ".if NT_STORE\n"
        ".if HW_LEN > 0\n"
            "vmovntps %%ymm0, ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm1, ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if HW_LEN > 1\n"
            "vmovntps %%ymm2, ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm3, ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if HW_LEN > 2\n"
            "vmovntps %%ymm4, ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm5, ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if HW_LEN > 3\n"
            "vmovntps %%ymm6, ((3 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm7, ((3 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if HW_LEN > 4\n"
            "vmovntps %%ymm8, ((4 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm9, ((4 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if HW_LEN > 5\n"
            "vmovntps %%ymm10, ((5 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovntps %%ymm11, ((5 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".else\n"
        ".if HW_LEN > 0\n"
            "vmovups %%ymm0, ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm1, ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if HW_LEN > 1\n"
            "vmovups %%ymm2, ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm3, ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if HW_LEN > 2\n"
            "vmovups %%ymm4, ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm5, ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if HW_LEN > 3\n"
            "vmovups %%ymm6, ((3 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm7, ((3 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if HW_LEN > 4\n"
            "vmovups %%ymm8, ((4 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm9, ((4 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if HW_LEN > 5\n"
            "vmovups %%ymm10, ((5 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
            "vmovups %%ymm11, ((5 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".endif\n"
        "sub $HW_LEN, %%r15\n"
        "cmp $0, %%r15\n"
        "lea (HW_LEN * CH_DT_BLK * D_BYTES)(%%r12), %%r12\n"
        "lea (HW_LEN * CH_DT_BLK * D_BYTES)(%%r13), %%r13\n"
        "lea (HW_LEN * CH_DT_BLK * D_BYTES)(%%r14), %%r14\n"
        "jne 1b\n" // label_init_session
        :
        :
        [priv_param]                  "r" (priv_param),
        [shar_param]                  "r" (shar_param),
        [NT_STORE]                    "i" (nt_store),
        [HW_LEN]                      "i" (hw_len),
        [KERNEL_FLAG_LD_BIAS]         "i" (KERNEL_FLAG_LD_BIAS()),
        [KERNEL_FLAG_AD_BIAS]         "i" (KERNEL_FLAG_AD_BIAS()),
        [KERNEL_FLAG_RELU]            "i" (KERNEL_FLAG_RELU()),
        [KERNEL_FLAG_RELU6]           "i" (KERNEL_FLAG_RELU6())
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

template <bool nt_store, int32_t oc_len, int32_t hw_len>
void conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
#ifdef PPL_USE_X86_INLINE_ASM
    if (oc_len == 2 * CH_RF_BLK() && hw_len == 6) {
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel_core<nt_store, hw_len>(priv_param, shar_param);
        return;
    }
#endif

#define IC_COMPUTE_STEP(IC) do {\
    if (oc_len > 0 * CH_RF_BLK()) ymm14 = _mm256_loadu_ps(ic_flt + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
    if (oc_len > 1 * CH_RF_BLK()) ymm15 = _mm256_loadu_ps(ic_flt + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
    if (hw_len > 0) {\
        ymm12 = _mm256_set1_ps(ic_src[(IC) + 0 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_fmadd_ps(ymm14, ymm12, ymm0);\
        if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_fmadd_ps(ymm15, ymm12, ymm1);\
    }\
    if (hw_len > 1) {\
        ymm13 = _mm256_set1_ps(ic_src[(IC) + 1 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_fmadd_ps(ymm14, ymm13, ymm2);\
        if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_fmadd_ps(ymm15, ymm13, ymm3);\
    }\
    if (hw_len > 2) {\
        ymm12 = _mm256_set1_ps(ic_src[(IC) + 2 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_fmadd_ps(ymm14, ymm12, ymm4);\
        if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_fmadd_ps(ymm15, ymm12, ymm5);\
    }\
    if (hw_len > 3) {\
        ymm13 = _mm256_set1_ps(ic_src[(IC) + 3 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_fmadd_ps(ymm14, ymm13, ymm6);\
        if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);\
    }\
    if (hw_len > 4) {\
        ymm12 = _mm256_set1_ps(ic_src[(IC) + 4 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_fmadd_ps(ymm14, ymm12, ymm8);\
        if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);\
    }\
    if (hw_len > 5) {\
        ymm13 = _mm256_set1_ps(ic_src[(IC) + 5 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_fmadd_ps(ymm14, ymm13, ymm10);\
        if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_fmadd_ps(ymm15, ymm13, ymm11);\
    }\
} while (0)

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    const int64_t src_icb_stride = shar_param[SRC_ICB_STRIDE_IDX()];
    const int64_t kernel_flags = shar_param[FLAGS_IDX()];

    const float *src = PICK_PARAM(const float*, priv_param, SRC_IDX());
    const float *his = PICK_PARAM(const float*, priv_param, HIS_IDX());
    float *dst       = PICK_PARAM(float*, priv_param, DST_IDX());
    int64_t hw       = priv_param[HW_IDX()];
    do {
        if (kernel_flags & KERNEL_FLAG_LD_BIAS()) {
            const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
            if (hw_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_loadu_ps(bias + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_loadu_ps(bias + 1 * CH_RF_BLK());
            }
            if (hw_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = ymm0;
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = ymm1;
            }
            if (hw_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = ymm0;
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = ymm1;
            }
            if (hw_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = ymm0;
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = ymm1;
            }
            if (hw_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = ymm0;
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = ymm1;
            }
            if (hw_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = ymm0;
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = ymm1;
            }
        } else {
            if (hw_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_loadu_ps(his + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_loadu_ps(his + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (hw_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_loadu_ps(his + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_loadu_ps(his + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (hw_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_loadu_ps(his + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_loadu_ps(his + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (hw_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_loadu_ps(his + 3 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_loadu_ps(his + 3 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (hw_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_loadu_ps(his + 4 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_loadu_ps(his + 4 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (hw_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_loadu_ps(his + 5 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_loadu_ps(his + 5 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
        }

        if (kernel_flags & KERNEL_FLAG_AD_BIAS()) {
            const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
            if (oc_len > 0 * CH_RF_BLK()) ymm14 = _mm256_loadu_ps(bias + 0 * CH_RF_BLK());
            if (oc_len > 1 * CH_RF_BLK()) ymm15 = _mm256_loadu_ps(bias + 1 * CH_RF_BLK());
            if (hw_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_add_ps(ymm14, ymm0);
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_add_ps(ymm15, ymm1);
            }
            if (hw_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_add_ps(ymm14, ymm2);
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_add_ps(ymm15, ymm3);
            }
            if (hw_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_add_ps(ymm14, ymm4);
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_add_ps(ymm15, ymm5);
            }
            if (hw_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_add_ps(ymm14, ymm6);
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_add_ps(ymm15, ymm7);
            }
            if (hw_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_add_ps(ymm14, ymm8);
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_add_ps(ymm15, ymm9);
            }
            if (hw_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_add_ps(ymm14, ymm10);
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_add_ps(ymm15, ymm11);
            }
        }
        
        const float *icb_src = src;
        const float *icb_flt = PICK_PARAM(const float*, priv_param, FLT_IDX());
        int64_t channels     = shar_param[CHANNELS_IDX()];
        while (channels >= CH_DT_BLK()) {
            channels -= CH_DT_BLK();
            const float *ic_src = icb_src;
            const float *ic_flt = icb_flt;
            for (int64_t ic = 0; ic < CH_DT_BLK(); ++ic) {
                IC_COMPUTE_STEP(0);
                ic_src += 1;
                ic_flt += CH_DT_BLK();
            }
            icb_flt += CH_DT_BLK() * CH_DT_BLK();
            icb_src += src_icb_stride;
        }
        if (channels > 0) {
            const float *ic_src = icb_src;
            const float *ic_flt = icb_flt;
            for (int64_t ic = 0; ic < channels; ++ic) {
                IC_COMPUTE_STEP(0);
                ic_src += 1;
                ic_flt += CH_DT_BLK();
            }
        }
        
        if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
            ymm14 = _mm256_setzero_ps();
            if (hw_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_max_ps(ymm14, ymm0);
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_max_ps(ymm14, ymm1);
            }
            if (hw_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_max_ps(ymm14, ymm2);
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_max_ps(ymm14, ymm3);
            }
            if (hw_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_max_ps(ymm14, ymm4);
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_max_ps(ymm14, ymm5);
            }
            if (hw_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_max_ps(ymm14, ymm6);
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_max_ps(ymm14, ymm7);
            }
            if (hw_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_max_ps(ymm14, ymm8);
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_max_ps(ymm14, ymm9);
            }
            if (hw_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_max_ps(ymm14, ymm10);
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_max_ps(ymm14, ymm11);
            }
        }
        if (kernel_flags & KERNEL_FLAG_RELU6()) {
            ymm15 = _mm256_set1_ps(6.0f);
            if (hw_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_min_ps(ymm15, ymm0);
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_min_ps(ymm15, ymm1);
            }
            if (hw_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_min_ps(ymm15, ymm2);
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_min_ps(ymm15, ymm3);
            }
            if (hw_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_min_ps(ymm15, ymm4);
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_min_ps(ymm15, ymm5);
            }
            if (hw_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_min_ps(ymm15, ymm6);
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_min_ps(ymm15, ymm7);
            }
            if (hw_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_min_ps(ymm15, ymm8);
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_min_ps(ymm15, ymm9);
            }
            if (hw_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_min_ps(ymm15, ymm10);
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_min_ps(ymm15, ymm11);
            }
        }

        if (nt_store) {
            if (hw_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm0);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm1);
            }
            if (hw_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm2);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm3);
            }
            if (hw_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm4);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm5);
            }
            if (hw_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 3 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm6);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 3 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm7);
            }
            if (hw_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 4 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm8);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 4 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm9);
            }
            if (hw_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_stream_ps(dst + 5 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm10);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_stream_ps(dst + 5 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm11);
            }
        } else {
            if (hw_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm0);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm1);
            }
            if (hw_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm2);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm3);
            }
            if (hw_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm4);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm5);
            }
            if (hw_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 3 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm6);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 3 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm7);
            }
            if (hw_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 4 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm8);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 4 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm9);
            }
            if (hw_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(dst + 5 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm10);
                if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(dst + 5 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm11);
            }
        }
        src += hw_len * CH_DT_BLK();
        his += hw_len * CH_DT_BLK();
        dst += hw_len * CH_DT_BLK();
        hw -= hw_len;
    } while (hw > 0);
#undef IC_COMPUTE_STEP
}

#define GEMM_DIRECT_KERNEL_TABLE_BLK(NT_STORE) \
{\
    {\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * CH_RF_BLK(), 1>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * CH_RF_BLK(), 2>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * CH_RF_BLK(), 3>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * CH_RF_BLK(), 4>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * CH_RF_BLK(), 5>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 1 * CH_RF_BLK(), 6>,\
    },\
    {\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * CH_RF_BLK(), 1>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * CH_RF_BLK(), 2>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * CH_RF_BLK(), 3>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * CH_RF_BLK(), 4>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * CH_RF_BLK(), 5>,\
        conv2d_n16cx_gemm_direct_fp32_fma_blk1x6_kernel<NT_STORE, 2 * CH_RF_BLK(), 6>,\
    },\
}

conv2d_n16cx_gemm_direct_kernel_fp32_fma_func_t
conv2d_n16cx_gemm_direct_kernel_fp32_fma_table[NT_STORE_OPT()][MAX_OC_RF()][MAX_HW_RF()] =
{
    GEMM_DIRECT_KERNEL_TABLE_BLK(false),
    GEMM_DIRECT_KERNEL_TABLE_BLK(true),
};

}}};

#endif
