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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_GEMM_DIRECT_AVX512_CONV2D_N16CX_GEMM_DIRECT_BLK1X31_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_GEMM_DIRECT_AVX512_CONV2D_N16CX_GEMM_DIRECT_BLK1X31_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/common/array_param_helper.h"
#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/avx512/conv2d_n16cx_gemm_direct_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template <bool nt_store, int32_t u_s>
void conv2d_n16cx_gemm_direct_fp32_avx512_blk1x31_kernel_core(int64_t *param)
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
        ".equ HIS_OCB_STRIDE_IDX, (8 * P_BYTES)\n"
        ".equ DST_OCB_STRIDE_IDX, (9 * P_BYTES)\n"
        ".equ FLT_OCB_STRIDE_IDX, (10 * P_BYTES)\n"
        ".equ FLAGS_IDX,          (11 * P_BYTES)\n"

        ".equ NT_STORE, %c[NT_STORE]\n"
        ".equ U_S, %c[U_S]\n"
        ".equ IC_DATA_BLK, %c[IC_DATA_BLK]\n"
        ".equ OC_DATA_BLK, %c[OC_DATA_BLK]\n"
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
        ".if U_S > 0\n vmovups (0 * OC_DATA_BLK * D_BYTES)(%%r10), %%zmm0\n .endif\n"
        ".if U_S > 1\n vmovaps %%zmm0, %%zmm1\n .endif\n"
        ".if U_S > 2\n vmovaps %%zmm0, %%zmm2\n .endif\n"
        ".if U_S > 3\n vmovaps %%zmm0, %%zmm3\n .endif\n"
        ".if U_S > 4\n vmovaps %%zmm0, %%zmm4\n .endif\n"
        ".if U_S > 5\n vmovaps %%zmm0, %%zmm5\n .endif\n"
        ".if U_S > 6\n vmovaps %%zmm0, %%zmm6\n .endif\n"
        ".if U_S > 7\n vmovaps %%zmm0, %%zmm7\n .endif\n"
        ".if U_S > 8\n vmovaps %%zmm0, %%zmm8\n .endif\n"
        ".if U_S > 9\n vmovaps %%zmm0, %%zmm9\n .endif\n"
        ".if U_S > 10\n vmovaps %%zmm0, %%zmm10\n .endif\n"
        ".if U_S > 11\n vmovaps %%zmm0, %%zmm11\n .endif\n"
        ".if U_S > 12\n vmovaps %%zmm0, %%zmm12\n .endif\n"
        ".if U_S > 13\n vmovaps %%zmm0, %%zmm13\n .endif\n"
        ".if U_S > 14\n vmovaps %%zmm0, %%zmm14\n .endif\n"
        ".if U_S > 15\n vmovaps %%zmm0, %%zmm15\n .endif\n"
        ".if U_S > 16\n vmovaps %%zmm0, %%zmm16\n .endif\n"
        ".if U_S > 17\n vmovaps %%zmm0, %%zmm17\n .endif\n"
        ".if U_S > 18\n vmovaps %%zmm0, %%zmm18\n .endif\n"
        ".if U_S > 19\n vmovaps %%zmm0, %%zmm19\n .endif\n"
        ".if U_S > 20\n vmovaps %%zmm0, %%zmm20\n .endif\n"
        ".if U_S > 21\n vmovaps %%zmm0, %%zmm21\n .endif\n"
        ".if U_S > 22\n vmovaps %%zmm0, %%zmm22\n .endif\n"
        ".if U_S > 23\n vmovaps %%zmm0, %%zmm23\n .endif\n"
        ".if U_S > 24\n vmovaps %%zmm0, %%zmm24\n .endif\n"
        ".if U_S > 25\n vmovaps %%zmm0, %%zmm25\n .endif\n"
        ".if U_S > 26\n vmovaps %%zmm0, %%zmm26\n .endif\n"
        ".if U_S > 27\n vmovaps %%zmm0, %%zmm27\n .endif\n"
        ".if U_S > 28\n vmovaps %%zmm0, %%zmm28\n .endif\n"
        ".if U_S > 29\n vmovaps %%zmm0, %%zmm29\n .endif\n"
        ".if U_S > 30\n vmovaps %%zmm0, %%zmm30\n .endif\n"
        "jmp 3f\n" // label_load_h_end
"2:\n" // label_load_h
        ".if U_S > 0\n vmovups (0 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm0\n .endif\n"
        ".if U_S > 1\n vmovups (1 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm1\n .endif\n"
        ".if U_S > 2\n vmovups (2 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm2\n .endif\n"
        ".if U_S > 3\n vmovups (3 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm3\n .endif\n"
        ".if U_S > 4\n vmovups (4 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm4\n .endif\n"
        ".if U_S > 5\n vmovups (5 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm5\n .endif\n"
        ".if U_S > 6\n vmovups (6 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm6\n .endif\n"
        ".if U_S > 7\n vmovups (7 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm7\n .endif\n"
        ".if U_S > 8\n vmovups (8 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm8\n .endif\n"
        ".if U_S > 9\n vmovups (9 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm9\n .endif\n"
        ".if U_S > 10\n vmovups (10 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm10\n .endif\n"
        ".if U_S > 11\n vmovups (11 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm11\n .endif\n"
        ".if U_S > 12\n vmovups (12 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm12\n .endif\n"
        ".if U_S > 13\n vmovups (13 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm13\n .endif\n"
        ".if U_S > 14\n vmovups (14 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm14\n .endif\n"
        ".if U_S > 15\n vmovups (15 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm15\n .endif\n"
        ".if U_S > 16\n vmovups (16 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm16\n .endif\n"
        ".if U_S > 17\n vmovups (17 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm17\n .endif\n"
        ".if U_S > 18\n vmovups (18 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm18\n .endif\n"
        ".if U_S > 19\n vmovups (19 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm19\n .endif\n"
        ".if U_S > 20\n vmovups (20 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm20\n .endif\n"
        ".if U_S > 21\n vmovups (21 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm21\n .endif\n"
        ".if U_S > 22\n vmovups (22 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm22\n .endif\n"
        ".if U_S > 23\n vmovups (23 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm23\n .endif\n"
        ".if U_S > 24\n vmovups (24 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm24\n .endif\n"
        ".if U_S > 25\n vmovups (25 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm25\n .endif\n"
        ".if U_S > 26\n vmovups (26 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm26\n .endif\n"
        ".if U_S > 27\n vmovups (27 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm27\n .endif\n"
        ".if U_S > 28\n vmovups (28 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm28\n .endif\n"
        ".if U_S > 29\n vmovups (29 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm29\n .endif\n"
        ".if U_S > 30\n vmovups (30 * OC_DATA_BLK * D_BYTES)(%%r13), %%zmm30\n .endif\n"
"3:\n" // label_load_h_end
        "test $KERNEL_FLAG_AD_BIAS, %%r11\n"
        "jz 4f\n" // label_compute_session
        "mov BIAS_PTR_IDX(%[param]), %%r10\n"
        "vmovups (0 * OC_DATA_BLK * D_BYTES)(%%r10), %%zmm31\n"
        ".if U_S > 0\n vaddps %%zmm31, %%zmm0, %%zmm0\n .endif\n"
        ".if U_S > 1\n vaddps %%zmm31, %%zmm1, %%zmm1\n .endif\n"
        ".if U_S > 2\n vaddps %%zmm31, %%zmm2, %%zmm2\n .endif\n"
        ".if U_S > 3\n vaddps %%zmm31, %%zmm3, %%zmm3\n .endif\n"
        ".if U_S > 4\n vaddps %%zmm31, %%zmm4, %%zmm4\n .endif\n"
        ".if U_S > 5\n vaddps %%zmm31, %%zmm5, %%zmm5\n .endif\n"
        ".if U_S > 6\n vaddps %%zmm31, %%zmm6, %%zmm6\n .endif\n"
        ".if U_S > 7\n vaddps %%zmm31, %%zmm7, %%zmm7\n .endif\n"
        ".if U_S > 8\n vaddps %%zmm31, %%zmm8, %%zmm8\n .endif\n"
        ".if U_S > 9\n vaddps %%zmm31, %%zmm9, %%zmm9\n .endif\n"
        ".if U_S > 10\n vaddps %%zmm31, %%zmm10, %%zmm10\n .endif\n"
        ".if U_S > 11\n vaddps %%zmm31, %%zmm11, %%zmm11\n .endif\n"
        ".if U_S > 12\n vaddps %%zmm31, %%zmm12, %%zmm12\n .endif\n"
        ".if U_S > 13\n vaddps %%zmm31, %%zmm13, %%zmm13\n .endif\n"
        ".if U_S > 14\n vaddps %%zmm31, %%zmm14, %%zmm14\n .endif\n"
        ".if U_S > 15\n vaddps %%zmm31, %%zmm15, %%zmm15\n .endif\n"
        ".if U_S > 16\n vaddps %%zmm31, %%zmm16, %%zmm16\n .endif\n"
        ".if U_S > 17\n vaddps %%zmm31, %%zmm17, %%zmm17\n .endif\n"
        ".if U_S > 18\n vaddps %%zmm31, %%zmm18, %%zmm18\n .endif\n"
        ".if U_S > 19\n vaddps %%zmm31, %%zmm19, %%zmm19\n .endif\n"
        ".if U_S > 20\n vaddps %%zmm31, %%zmm20, %%zmm20\n .endif\n"
        ".if U_S > 21\n vaddps %%zmm31, %%zmm21, %%zmm21\n .endif\n"
        ".if U_S > 22\n vaddps %%zmm31, %%zmm22, %%zmm22\n .endif\n"
        ".if U_S > 23\n vaddps %%zmm31, %%zmm23, %%zmm23\n .endif\n"
        ".if U_S > 24\n vaddps %%zmm31, %%zmm24, %%zmm24\n .endif\n"
        ".if U_S > 25\n vaddps %%zmm31, %%zmm25, %%zmm25\n .endif\n"
        ".if U_S > 26\n vaddps %%zmm31, %%zmm26, %%zmm26\n .endif\n"
        ".if U_S > 27\n vaddps %%zmm31, %%zmm27, %%zmm27\n .endif\n"
        ".if U_S > 28\n vaddps %%zmm31, %%zmm28, %%zmm28\n .endif\n"
        ".if U_S > 29\n vaddps %%zmm31, %%zmm29, %%zmm29\n .endif\n"
        ".if U_S > 30\n vaddps %%zmm31, %%zmm30, %%zmm30\n .endif\n"

"4:\n" // label_compute_session
        "mov %%r12, %%rax\n"
        "mov FLT_PTR_IDX(%[param]), %%rbx\n"
        "mov CHANNELS_IDX(%[param]), %%r10\n"
        "cmp $IC_DATA_BLK, %%r10\n"
        "jl 6f\n" // label_ic_remain
"5:\n" // label_ic_body
        PPL_X86_INLINE_ASM_ALIGN()
        ".if U_S < 9\n"
        ".irp IC,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
        "vmovups (\\IC * OC_DATA_BLK * D_BYTES)(%%rbx), %%zmm31\n"
        ".if U_S > 7\n prefetcht0 ((\\IC * OC_DATA_BLK + IC_DATA_BLK * OC_DATA_BLK) * D_BYTES)(%%rbx)\n .endif\n"
        ".if U_S > 0\n vfmadd231ps ((\\IC + 0 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm0\n .endif\n"
        ".if U_S > 1\n vfmadd231ps ((\\IC + 1 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm1\n .endif\n"
        ".if U_S > 2\n vfmadd231ps ((\\IC + 2 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm2\n .endif\n"
        ".if U_S > 3\n vfmadd231ps ((\\IC + 3 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm3\n .endif\n"
        ".if U_S > 4\n vfmadd231ps ((\\IC + 4 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm4\n .endif\n"
        ".if U_S > 5\n vfmadd231ps ((\\IC + 5 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm5\n .endif\n"
        ".if U_S > 6\n vfmadd231ps ((\\IC + 6 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm6\n .endif\n"
        ".if U_S > 7\n vfmadd231ps ((\\IC + 7 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm7\n .endif\n"
        ".if U_S > 8\n vfmadd231ps ((\\IC + 8 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm8\n .endif\n"
        ".if U_S > 9\n vfmadd231ps ((\\IC + 9 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm9\n .endif\n"
        ".if U_S > 10\n vfmadd231ps ((\\IC + 10 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm10\n .endif\n"
        ".if U_S > 11\n vfmadd231ps ((\\IC + 11 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm11\n .endif\n"
        ".if U_S > 12\n vfmadd231ps ((\\IC + 12 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm12\n .endif\n"
        ".if U_S > 13\n vfmadd231ps ((\\IC + 13 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm13\n .endif\n"
        ".if U_S > 14\n vfmadd231ps ((\\IC + 14 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm14\n .endif\n"
        ".if U_S > 15\n vfmadd231ps ((\\IC + 15 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm15\n .endif\n"
        ".if U_S > 16\n vfmadd231ps ((\\IC + 16 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm16\n .endif\n"
        ".if U_S > 17\n vfmadd231ps ((\\IC + 17 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm17\n .endif\n"
        ".if U_S > 18\n vfmadd231ps ((\\IC + 18 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm18\n .endif\n"
        ".if U_S > 19\n vfmadd231ps ((\\IC + 19 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm19\n .endif\n"
        ".if U_S > 20\n vfmadd231ps ((\\IC + 20 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm20\n .endif\n"
        ".if U_S > 21\n vfmadd231ps ((\\IC + 21 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm21\n .endif\n"
        ".if U_S > 22\n vfmadd231ps ((\\IC + 22 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm22\n .endif\n"
        ".if U_S > 23\n vfmadd231ps ((\\IC + 23 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm23\n .endif\n"
        ".if U_S > 24\n vfmadd231ps ((\\IC + 24 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm24\n .endif\n"
        ".if U_S > 25\n vfmadd231ps ((\\IC + 25 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm25\n .endif\n"
        ".if U_S > 26\n vfmadd231ps ((\\IC + 26 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm26\n .endif\n"
        ".if U_S > 27\n vfmadd231ps ((\\IC + 27 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm27\n .endif\n"
        ".if U_S > 28\n vfmadd231ps ((\\IC + 28 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm28\n .endif\n"
        ".if U_S > 29\n vfmadd231ps ((\\IC + 29 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm29\n .endif\n"
        ".if U_S > 30\n vfmadd231ps ((\\IC + 30 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm30\n .endif\n"
        ".endr\n"
        "lea (IC_DATA_BLK * OC_DATA_BLK * D_BYTES)(%%rbx), %%rbx\n"
        "lea (%%rax, %%r8, D_BYTES), %%rax\n"
        ".else\n" // .if U_S < 9
        "mov $IC_DATA_BLK, %%r9\n"
"9:\n" // label_ic
        "vmovups (0 * OC_DATA_BLK * D_BYTES)(%%rbx), %%zmm31\n"
        "prefetcht0 ((0 * OC_DATA_BLK + IC_DATA_BLK * OC_DATA_BLK) * D_BYTES)(%%rbx)\n"
        "lea (OC_DATA_BLK * D_BYTES)(%%rbx), %%rbx\n"
        ".if U_S > 0\n vfmadd231ps ((0 + 0 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm0\n .endif\n"
        ".if U_S > 1\n vfmadd231ps ((0 + 1 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm1\n .endif\n"
        ".if U_S > 2\n vfmadd231ps ((0 + 2 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm2\n .endif\n"
        ".if U_S > 3\n vfmadd231ps ((0 + 3 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm3\n .endif\n"
        ".if U_S > 4\n vfmadd231ps ((0 + 4 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm4\n .endif\n"
        ".if U_S > 5\n vfmadd231ps ((0 + 5 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm5\n .endif\n"
        ".if U_S > 6\n vfmadd231ps ((0 + 6 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm6\n .endif\n"
        ".if U_S > 7\n vfmadd231ps ((0 + 7 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm7\n .endif\n"
        ".if U_S > 8\n vfmadd231ps ((0 + 8 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm8\n .endif\n"
        ".if U_S > 9\n vfmadd231ps ((0 + 9 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm9\n .endif\n"
        ".if U_S > 10\n vfmadd231ps ((0 + 10 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm10\n .endif\n"
        ".if U_S > 11\n vfmadd231ps ((0 + 11 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm11\n .endif\n"
        ".if U_S > 12\n vfmadd231ps ((0 + 12 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm12\n .endif\n"
        ".if U_S > 13\n vfmadd231ps ((0 + 13 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm13\n .endif\n"
        ".if U_S > 14\n vfmadd231ps ((0 + 14 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm14\n .endif\n"
        ".if U_S > 15\n vfmadd231ps ((0 + 15 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm15\n .endif\n"
        ".if U_S > 16\n vfmadd231ps ((0 + 16 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm16\n .endif\n"
        ".if U_S > 17\n vfmadd231ps ((0 + 17 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm17\n .endif\n"
        ".if U_S > 18\n vfmadd231ps ((0 + 18 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm18\n .endif\n"
        ".if U_S > 19\n vfmadd231ps ((0 + 19 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm19\n .endif\n"
        ".if U_S > 20\n vfmadd231ps ((0 + 20 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm20\n .endif\n"
        ".if U_S > 21\n vfmadd231ps ((0 + 21 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm21\n .endif\n"
        ".if U_S > 22\n vfmadd231ps ((0 + 22 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm22\n .endif\n"
        ".if U_S > 23\n vfmadd231ps ((0 + 23 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm23\n .endif\n"
        ".if U_S > 24\n vfmadd231ps ((0 + 24 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm24\n .endif\n"
        ".if U_S > 25\n vfmadd231ps ((0 + 25 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm25\n .endif\n"
        ".if U_S > 26\n vfmadd231ps ((0 + 26 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm26\n .endif\n"
        ".if U_S > 27\n vfmadd231ps ((0 + 27 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm27\n .endif\n"
        ".if U_S > 28\n vfmadd231ps ((0 + 28 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm28\n .endif\n"
        ".if U_S > 29\n vfmadd231ps ((0 + 29 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm29\n .endif\n"
        ".if U_S > 30\n vfmadd231ps ((0 + 30 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm30\n .endif\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%r9\n"
        "cmp $0, %%r9\n"
        "jne 9b\n" // label_ic
        "lea (-IC_DATA_BLK * D_BYTES)(%%rax, %%r8, D_BYTES), %%rax\n"
        ".endif\n" // .if U_S < 9
        "sub $IC_DATA_BLK, %%r10\n"
        "cmp $IC_DATA_BLK, %%r10\n"
        "jge 5b\n"// label_ic_body
        "cmp $0, %%r10\n"
        "je 7f\n" // label_finalize_session
"6:\n" // label_ic_remain
        "vmovups (0 * OC_DATA_BLK * D_BYTES)(%%rbx), %%zmm31\n"
        "lea (OC_DATA_BLK * D_BYTES)(%%rbx), %%rbx\n"
        ".if U_S > 0\n vfmadd231ps ((0 + 0 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm0\n .endif\n"
        ".if U_S > 1\n vfmadd231ps ((0 + 1 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm1\n .endif\n"
        ".if U_S > 2\n vfmadd231ps ((0 + 2 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm2\n .endif\n"
        ".if U_S > 3\n vfmadd231ps ((0 + 3 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm3\n .endif\n"
        ".if U_S > 4\n vfmadd231ps ((0 + 4 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm4\n .endif\n"
        ".if U_S > 5\n vfmadd231ps ((0 + 5 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm5\n .endif\n"
        ".if U_S > 6\n vfmadd231ps ((0 + 6 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm6\n .endif\n"
        ".if U_S > 7\n vfmadd231ps ((0 + 7 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm7\n .endif\n"
        ".if U_S > 8\n vfmadd231ps ((0 + 8 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm8\n .endif\n"
        ".if U_S > 9\n vfmadd231ps ((0 + 9 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm9\n .endif\n"
        ".if U_S > 10\n vfmadd231ps ((0 + 10 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm10\n .endif\n"
        ".if U_S > 11\n vfmadd231ps ((0 + 11 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm11\n .endif\n"
        ".if U_S > 12\n vfmadd231ps ((0 + 12 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm12\n .endif\n"
        ".if U_S > 13\n vfmadd231ps ((0 + 13 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm13\n .endif\n"
        ".if U_S > 14\n vfmadd231ps ((0 + 14 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm14\n .endif\n"
        ".if U_S > 15\n vfmadd231ps ((0 + 15 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm15\n .endif\n"
        ".if U_S > 16\n vfmadd231ps ((0 + 16 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm16\n .endif\n"
        ".if U_S > 17\n vfmadd231ps ((0 + 17 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm17\n .endif\n"
        ".if U_S > 18\n vfmadd231ps ((0 + 18 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm18\n .endif\n"
        ".if U_S > 19\n vfmadd231ps ((0 + 19 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm19\n .endif\n"
        ".if U_S > 20\n vfmadd231ps ((0 + 20 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm20\n .endif\n"
        ".if U_S > 21\n vfmadd231ps ((0 + 21 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm21\n .endif\n"
        ".if U_S > 22\n vfmadd231ps ((0 + 22 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm22\n .endif\n"
        ".if U_S > 23\n vfmadd231ps ((0 + 23 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm23\n .endif\n"
        ".if U_S > 24\n vfmadd231ps ((0 + 24 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm24\n .endif\n"
        ".if U_S > 25\n vfmadd231ps ((0 + 25 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm25\n .endif\n"
        ".if U_S > 26\n vfmadd231ps ((0 + 26 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm26\n .endif\n"
        ".if U_S > 27\n vfmadd231ps ((0 + 27 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm27\n .endif\n"
        ".if U_S > 28\n vfmadd231ps ((0 + 28 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm28\n .endif\n"
        ".if U_S > 29\n vfmadd231ps ((0 + 29 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm29\n .endif\n"
        ".if U_S > 30\n vfmadd231ps ((0 + 30 * IC_DATA_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm30\n .endif\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%r10\n"
        "cmp $0, %%r10\n"
        "jne 6b\n" // label_ic_remain

"7:\n" // label_finalize_session
        "test $(KERNEL_FLAG_RELU | KERNEL_FLAG_RELU6), %%r11\n"
        "jz 8f\n" // label_relu_end
        "vpxord %%zmm31, %%zmm31, %%zmm31\n"
        ".if U_S > 0\n vmaxps %%zmm31, %%zmm0, %%zmm0\n .endif\n"
        ".if U_S > 1\n vmaxps %%zmm31, %%zmm1, %%zmm1\n .endif\n"
        ".if U_S > 2\n vmaxps %%zmm31, %%zmm2, %%zmm2\n .endif\n"
        ".if U_S > 3\n vmaxps %%zmm31, %%zmm3, %%zmm3\n .endif\n"
        ".if U_S > 4\n vmaxps %%zmm31, %%zmm4, %%zmm4\n .endif\n"
        ".if U_S > 5\n vmaxps %%zmm31, %%zmm5, %%zmm5\n .endif\n"
        ".if U_S > 6\n vmaxps %%zmm31, %%zmm6, %%zmm6\n .endif\n"
        ".if U_S > 7\n vmaxps %%zmm31, %%zmm7, %%zmm7\n .endif\n"
        ".if U_S > 8\n vmaxps %%zmm31, %%zmm8, %%zmm8\n .endif\n"
        ".if U_S > 9\n vmaxps %%zmm31, %%zmm9, %%zmm9\n .endif\n"
        ".if U_S > 10\n vmaxps %%zmm31, %%zmm10, %%zmm10\n .endif\n"
        ".if U_S > 11\n vmaxps %%zmm31, %%zmm11, %%zmm11\n .endif\n"
        ".if U_S > 12\n vmaxps %%zmm31, %%zmm12, %%zmm12\n .endif\n"
        ".if U_S > 13\n vmaxps %%zmm31, %%zmm13, %%zmm13\n .endif\n"
        ".if U_S > 14\n vmaxps %%zmm31, %%zmm14, %%zmm14\n .endif\n"
        ".if U_S > 15\n vmaxps %%zmm31, %%zmm15, %%zmm15\n .endif\n"
        ".if U_S > 16\n vmaxps %%zmm31, %%zmm16, %%zmm16\n .endif\n"
        ".if U_S > 17\n vmaxps %%zmm31, %%zmm17, %%zmm17\n .endif\n"
        ".if U_S > 18\n vmaxps %%zmm31, %%zmm18, %%zmm18\n .endif\n"
        ".if U_S > 19\n vmaxps %%zmm31, %%zmm19, %%zmm19\n .endif\n"
        ".if U_S > 20\n vmaxps %%zmm31, %%zmm20, %%zmm20\n .endif\n"
        ".if U_S > 21\n vmaxps %%zmm31, %%zmm21, %%zmm21\n .endif\n"
        ".if U_S > 22\n vmaxps %%zmm31, %%zmm22, %%zmm22\n .endif\n"
        ".if U_S > 23\n vmaxps %%zmm31, %%zmm23, %%zmm23\n .endif\n"
        ".if U_S > 24\n vmaxps %%zmm31, %%zmm24, %%zmm24\n .endif\n"
        ".if U_S > 25\n vmaxps %%zmm31, %%zmm25, %%zmm25\n .endif\n"
        ".if U_S > 26\n vmaxps %%zmm31, %%zmm26, %%zmm26\n .endif\n"
        ".if U_S > 27\n vmaxps %%zmm31, %%zmm27, %%zmm27\n .endif\n"
        ".if U_S > 28\n vmaxps %%zmm31, %%zmm28, %%zmm28\n .endif\n"
        ".if U_S > 29\n vmaxps %%zmm31, %%zmm29, %%zmm29\n .endif\n"
        ".if U_S > 30\n vmaxps %%zmm31, %%zmm30, %%zmm30\n .endif\n"
        "test $KERNEL_FLAG_RELU6, %%r11\n"
        "jz 8f\n" // label_relu_end
        "mov $0x40c00000, %%ecx\n"
        "vmovd %%ecx, %%xmm31\n"
        "vbroadcastss %%xmm31, %%zmm31\n" // 6.0
        ".if U_S > 0\n vminps %%zmm31, %%zmm0, %%zmm0\n .endif\n"
        ".if U_S > 1\n vminps %%zmm31, %%zmm1, %%zmm1\n .endif\n"
        ".if U_S > 2\n vminps %%zmm31, %%zmm2, %%zmm2\n .endif\n"
        ".if U_S > 3\n vminps %%zmm31, %%zmm3, %%zmm3\n .endif\n"
        ".if U_S > 4\n vminps %%zmm31, %%zmm4, %%zmm4\n .endif\n"
        ".if U_S > 5\n vminps %%zmm31, %%zmm5, %%zmm5\n .endif\n"
        ".if U_S > 6\n vminps %%zmm31, %%zmm6, %%zmm6\n .endif\n"
        ".if U_S > 7\n vminps %%zmm31, %%zmm7, %%zmm7\n .endif\n"
        ".if U_S > 8\n vminps %%zmm31, %%zmm8, %%zmm8\n .endif\n"
        ".if U_S > 9\n vminps %%zmm31, %%zmm9, %%zmm9\n .endif\n"
        ".if U_S > 10\n vminps %%zmm31, %%zmm10, %%zmm10\n .endif\n"
        ".if U_S > 11\n vminps %%zmm31, %%zmm11, %%zmm11\n .endif\n"
        ".if U_S > 12\n vminps %%zmm31, %%zmm12, %%zmm12\n .endif\n"
        ".if U_S > 13\n vminps %%zmm31, %%zmm13, %%zmm13\n .endif\n"
        ".if U_S > 14\n vminps %%zmm31, %%zmm14, %%zmm14\n .endif\n"
        ".if U_S > 15\n vminps %%zmm31, %%zmm15, %%zmm15\n .endif\n"
        ".if U_S > 16\n vminps %%zmm31, %%zmm16, %%zmm16\n .endif\n"
        ".if U_S > 17\n vminps %%zmm31, %%zmm17, %%zmm17\n .endif\n"
        ".if U_S > 18\n vminps %%zmm31, %%zmm18, %%zmm18\n .endif\n"
        ".if U_S > 19\n vminps %%zmm31, %%zmm19, %%zmm19\n .endif\n"
        ".if U_S > 20\n vminps %%zmm31, %%zmm20, %%zmm20\n .endif\n"
        ".if U_S > 21\n vminps %%zmm31, %%zmm21, %%zmm21\n .endif\n"
        ".if U_S > 22\n vminps %%zmm31, %%zmm22, %%zmm22\n .endif\n"
        ".if U_S > 23\n vminps %%zmm31, %%zmm23, %%zmm23\n .endif\n"
        ".if U_S > 24\n vminps %%zmm31, %%zmm24, %%zmm24\n .endif\n"
        ".if U_S > 25\n vminps %%zmm31, %%zmm25, %%zmm25\n .endif\n"
        ".if U_S > 26\n vminps %%zmm31, %%zmm26, %%zmm26\n .endif\n"
        ".if U_S > 27\n vminps %%zmm31, %%zmm27, %%zmm27\n .endif\n"
        ".if U_S > 28\n vminps %%zmm31, %%zmm28, %%zmm28\n .endif\n"
        ".if U_S > 29\n vminps %%zmm31, %%zmm29, %%zmm29\n .endif\n"
        ".if U_S > 30\n vminps %%zmm31, %%zmm30, %%zmm30\n .endif\n"
"8:\n"
        ".if NT_STORE\n"
        ".if U_S > 0\n vmovntps %%zmm0, (0 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 1\n vmovntps %%zmm1, (1 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 2\n vmovntps %%zmm2, (2 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 3\n vmovntps %%zmm3, (3 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 4\n vmovntps %%zmm4, (4 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 5\n vmovntps %%zmm5, (5 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 6\n vmovntps %%zmm6, (6 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 7\n vmovntps %%zmm7, (7 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 8\n vmovntps %%zmm8, (8 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 9\n vmovntps %%zmm9, (9 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 10\n vmovntps %%zmm10, (10 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 11\n vmovntps %%zmm11, (11 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 12\n vmovntps %%zmm12, (12 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 13\n vmovntps %%zmm13, (13 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 14\n vmovntps %%zmm14, (14 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 15\n vmovntps %%zmm15, (15 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 16\n vmovntps %%zmm16, (16 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 17\n vmovntps %%zmm17, (17 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 18\n vmovntps %%zmm18, (18 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 19\n vmovntps %%zmm19, (19 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 20\n vmovntps %%zmm20, (20 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 21\n vmovntps %%zmm21, (21 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 22\n vmovntps %%zmm22, (22 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 23\n vmovntps %%zmm23, (23 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 24\n vmovntps %%zmm24, (24 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 25\n vmovntps %%zmm25, (25 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 26\n vmovntps %%zmm26, (26 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 27\n vmovntps %%zmm27, (27 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 28\n vmovntps %%zmm28, (28 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 29\n vmovntps %%zmm29, (29 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 30\n vmovntps %%zmm30, (30 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".else\n"
        ".if U_S > 0\n vmovups %%zmm0, (0 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 1\n vmovups %%zmm1, (1 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 2\n vmovups %%zmm2, (2 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 3\n vmovups %%zmm3, (3 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 4\n vmovups %%zmm4, (4 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 5\n vmovups %%zmm5, (5 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 6\n vmovups %%zmm6, (6 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 7\n vmovups %%zmm7, (7 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 8\n vmovups %%zmm8, (8 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 9\n vmovups %%zmm9, (9 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 10\n vmovups %%zmm10, (10 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 11\n vmovups %%zmm11, (11 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 12\n vmovups %%zmm12, (12 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 13\n vmovups %%zmm13, (13 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 14\n vmovups %%zmm14, (14 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 15\n vmovups %%zmm15, (15 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 16\n vmovups %%zmm16, (16 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 17\n vmovups %%zmm17, (17 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 18\n vmovups %%zmm18, (18 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 19\n vmovups %%zmm19, (19 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 20\n vmovups %%zmm20, (20 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 21\n vmovups %%zmm21, (21 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 22\n vmovups %%zmm22, (22 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 23\n vmovups %%zmm23, (23 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 24\n vmovups %%zmm24, (24 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 25\n vmovups %%zmm25, (25 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 26\n vmovups %%zmm26, (26 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 27\n vmovups %%zmm27, (27 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 28\n vmovups %%zmm28, (28 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 29\n vmovups %%zmm29, (29 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
        ".if U_S > 30\n vmovups %%zmm30, (30 * OC_DATA_BLK * D_BYTES)(%%r14)\n .endif\n"
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
        [IC_DATA_BLK]                 "i" (conv2d_n16cx_gemm_direct_kernel_fp32_avx512::config::IC_DATA_BLK),
        [OC_DATA_BLK]                 "i" (conv2d_n16cx_gemm_direct_kernel_fp32_avx512::config::OC_DATA_BLK),
        [KERNEL_FLAG_LD_BIAS]         "i" (conv2d_n16cx_gemm_direct_kernel_fp32_avx512::flag::LOAD_BIAS),
        [KERNEL_FLAG_AD_BIAS]         "i" (conv2d_n16cx_gemm_direct_kernel_fp32_avx512::flag::ADD_BIAS),
        [KERNEL_FLAG_RELU]            "i" (conv2d_n16cx_gemm_direct_kernel_fp32_avx512::flag::RELU),
        [KERNEL_FLAG_RELU6]           "i" (conv2d_n16cx_gemm_direct_kernel_fp32_avx512::flag::RELU6)
        :
        "cc",
        "rax", "rbx", "rcx", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
        "zmm0" , "zmm1" , "zmm2" , "zmm3" , "zmm4" , "zmm5" , "zmm6" , "zmm7" ,
        "zmm8" , "zmm9" , "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
        "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
        "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
        "memory"
    );
}

#endif

template <bool nt_store, int32_t u_oc, int32_t u_s>
void conv2d_n16cx_gemm_direct_fp32_avx512_blk1x31_kernel(int64_t *param)
{
#ifdef PPL_USE_X86_INLINE_ASM
    if (u_oc == 1 * conv2d_n16cx_gemm_direct_kernel_fp32_avx512::config::OC_DATA_BLK) {
        conv2d_n16cx_gemm_direct_fp32_avx512_blk1x31_kernel_core<nt_store, u_s>(param);
        return;
    }
#endif

#define IC_COMPUTE_STEP(IC) do {\
    zmm31 = _mm512_loadu_ps(icb_flt + (IC) * OC_DATA_BLK);\
    if (u_s > 0) zmm0 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 0 * IC_DATA_BLK]), zmm31, zmm0);\
    if (u_s > 1) zmm1 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 1 * IC_DATA_BLK]), zmm31, zmm1);\
    if (u_s > 2) zmm2 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 2 * IC_DATA_BLK]), zmm31, zmm2);\
    if (u_s > 3) zmm3 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 3 * IC_DATA_BLK]), zmm31, zmm3);\
    if (u_s > 4) zmm4 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 4 * IC_DATA_BLK]), zmm31, zmm4);\
    if (u_s > 5) zmm5 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 5 * IC_DATA_BLK]), zmm31, zmm5);\
    if (u_s > 6) zmm6 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 6 * IC_DATA_BLK]), zmm31, zmm6);\
    if (u_s > 7) zmm7 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 7 * IC_DATA_BLK]), zmm31, zmm7);\
    if (u_s > 8) zmm8 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 8 * IC_DATA_BLK]), zmm31, zmm8);\
    if (u_s > 9) zmm9 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 9 * IC_DATA_BLK]), zmm31, zmm9);\
    if (u_s > 10) zmm10 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 10 * IC_DATA_BLK]), zmm31, zmm10);\
    if (u_s > 11) zmm11 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 11 * IC_DATA_BLK]), zmm31, zmm11);\
    if (u_s > 12) zmm12 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 12 * IC_DATA_BLK]), zmm31, zmm12);\
    if (u_s > 13) zmm13 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 13 * IC_DATA_BLK]), zmm31, zmm13);\
    if (u_s > 14) zmm14 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 14 * IC_DATA_BLK]), zmm31, zmm14);\
    if (u_s > 15) zmm15 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 15 * IC_DATA_BLK]), zmm31, zmm15);\
    if (u_s > 16) zmm16 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 16 * IC_DATA_BLK]), zmm31, zmm16);\
    if (u_s > 17) zmm17 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 17 * IC_DATA_BLK]), zmm31, zmm17);\
    if (u_s > 18) zmm18 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 18 * IC_DATA_BLK]), zmm31, zmm18);\
    if (u_s > 19) zmm19 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 19 * IC_DATA_BLK]), zmm31, zmm19);\
    if (u_s > 20) zmm20 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 20 * IC_DATA_BLK]), zmm31, zmm20);\
    if (u_s > 21) zmm21 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 21 * IC_DATA_BLK]), zmm31, zmm21);\
    if (u_s > 22) zmm22 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 22 * IC_DATA_BLK]), zmm31, zmm22);\
    if (u_s > 23) zmm23 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 23 * IC_DATA_BLK]), zmm31, zmm23);\
    if (u_s > 24) zmm24 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 24 * IC_DATA_BLK]), zmm31, zmm24);\
    if (u_s > 25) zmm25 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 25 * IC_DATA_BLK]), zmm31, zmm25);\
    if (u_s > 26) zmm26 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 26 * IC_DATA_BLK]), zmm31, zmm26);\
    if (u_s > 27) zmm27 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 27 * IC_DATA_BLK]), zmm31, zmm27);\
    if (u_s > 28) zmm28 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 28 * IC_DATA_BLK]), zmm31, zmm28);\
    if (u_s > 29) zmm29 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 29 * IC_DATA_BLK]), zmm31, zmm29);\
    if (u_s > 30) zmm30 = _mm512_fmadd_ps(_mm512_set1_ps(icb_src[(IC) + 30 * IC_DATA_BLK]), zmm31, zmm30);\
} while (0)

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    const int64_t IC_DATA_BLK = conv2d_n16cx_gemm_direct_kernel_fp32_avx512::config::IC_DATA_BLK;
    const int64_t OC_DATA_BLK = conv2d_n16cx_gemm_direct_kernel_fp32_avx512::config::OC_DATA_BLK;
    const int64_t u_ocb = div_up(u_oc, OC_DATA_BLK);

    array_param_helper ker_p(param);

    const int64_t src_icb_stride = ker_p.pick<const int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_avx512::param_def::SRC_ICB_STRIDE_IDX);
    const int64_t kernel_flags = ker_p.pick<const int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_avx512::param_def::FLAGS_IDX);

    const float *src = ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_avx512::param_def::SRC_PTR_IDX);
    const float *his = ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_avx512::param_def::HIS_PTR_IDX);
    float *dst       = ker_p.pick<float*>(conv2d_n16cx_gemm_direct_kernel_fp32_avx512::param_def::DST_PTR_IDX);
    int64_t space    = ker_p.pick<const int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_avx512::param_def::SPACE_IDX);
    do {
        if (kernel_flags & conv2d_n16cx_gemm_direct_kernel_fp32_avx512::flag::LOAD_BIAS) {
            const float* bias = ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_avx512::param_def::BIAS_PTR_IDX);
            if (u_ocb > 0) {
                if (u_s > 0) zmm0 = _mm512_loadu_ps(bias + 0 * OC_DATA_BLK);
                if (u_s > 1) zmm1 = zmm0;
                if (u_s > 2) zmm2 = zmm0;
                if (u_s > 3) zmm3 = zmm0;
                if (u_s > 4) zmm4 = zmm0;
                if (u_s > 5) zmm5 = zmm0;
                if (u_s > 6) zmm6 = zmm0;
                if (u_s > 7) zmm7 = zmm0;
                if (u_s > 8) zmm8 = zmm0;
                if (u_s > 9) zmm9 = zmm0;
                if (u_s > 10) zmm10 = zmm0;
                if (u_s > 11) zmm11 = zmm0;
                if (u_s > 12) zmm12 = zmm0;
                if (u_s > 13) zmm13 = zmm0;
                if (u_s > 14) zmm14 = zmm0;
                if (u_s > 15) zmm15 = zmm0;
                if (u_s > 16) zmm16 = zmm0;
                if (u_s > 17) zmm17 = zmm0;
                if (u_s > 18) zmm18 = zmm0;
                if (u_s > 19) zmm19 = zmm0;
                if (u_s > 20) zmm20 = zmm0;
                if (u_s > 21) zmm21 = zmm0;
                if (u_s > 22) zmm22 = zmm0;
                if (u_s > 23) zmm23 = zmm0;
                if (u_s > 24) zmm24 = zmm0;
                if (u_s > 25) zmm25 = zmm0;
                if (u_s > 26) zmm26 = zmm0;
                if (u_s > 27) zmm27 = zmm0;
                if (u_s > 28) zmm28 = zmm0;
                if (u_s > 29) zmm29 = zmm0;
                if (u_s > 30) zmm30 = zmm0;
            }
        } else {
            const float *l_his = his;
            if (u_ocb > 0) {
                if (u_s > 0) zmm0 = _mm512_loadu_ps(l_his + 0 * OC_DATA_BLK);
                if (u_s > 1) zmm1 = _mm512_loadu_ps(l_his + 1 * OC_DATA_BLK);
                if (u_s > 2) zmm2 = _mm512_loadu_ps(l_his + 2 * OC_DATA_BLK);
                if (u_s > 3) zmm3 = _mm512_loadu_ps(l_his + 3 * OC_DATA_BLK);
                if (u_s > 4) zmm4 = _mm512_loadu_ps(l_his + 4 * OC_DATA_BLK);
                if (u_s > 5) zmm5 = _mm512_loadu_ps(l_his + 5 * OC_DATA_BLK);
                if (u_s > 6) zmm6 = _mm512_loadu_ps(l_his + 6 * OC_DATA_BLK);
                if (u_s > 7) zmm7 = _mm512_loadu_ps(l_his + 7 * OC_DATA_BLK);
                if (u_s > 8) zmm8 = _mm512_loadu_ps(l_his + 8 * OC_DATA_BLK);
                if (u_s > 9) zmm9 = _mm512_loadu_ps(l_his + 9 * OC_DATA_BLK);
                if (u_s > 10) zmm10 = _mm512_loadu_ps(l_his + 10 * OC_DATA_BLK);
                if (u_s > 11) zmm11 = _mm512_loadu_ps(l_his + 11 * OC_DATA_BLK);
                if (u_s > 12) zmm12 = _mm512_loadu_ps(l_his + 12 * OC_DATA_BLK);
                if (u_s > 13) zmm13 = _mm512_loadu_ps(l_his + 13 * OC_DATA_BLK);
                if (u_s > 14) zmm14 = _mm512_loadu_ps(l_his + 14 * OC_DATA_BLK);
                if (u_s > 15) zmm15 = _mm512_loadu_ps(l_his + 15 * OC_DATA_BLK);
                if (u_s > 16) zmm16 = _mm512_loadu_ps(l_his + 16 * OC_DATA_BLK);
                if (u_s > 17) zmm17 = _mm512_loadu_ps(l_his + 17 * OC_DATA_BLK);
                if (u_s > 18) zmm18 = _mm512_loadu_ps(l_his + 18 * OC_DATA_BLK);
                if (u_s > 19) zmm19 = _mm512_loadu_ps(l_his + 19 * OC_DATA_BLK);
                if (u_s > 20) zmm20 = _mm512_loadu_ps(l_his + 20 * OC_DATA_BLK);
                if (u_s > 21) zmm21 = _mm512_loadu_ps(l_his + 21 * OC_DATA_BLK);
                if (u_s > 22) zmm22 = _mm512_loadu_ps(l_his + 22 * OC_DATA_BLK);
                if (u_s > 23) zmm23 = _mm512_loadu_ps(l_his + 23 * OC_DATA_BLK);
                if (u_s > 24) zmm24 = _mm512_loadu_ps(l_his + 24 * OC_DATA_BLK);
                if (u_s > 25) zmm25 = _mm512_loadu_ps(l_his + 25 * OC_DATA_BLK);
                if (u_s > 26) zmm26 = _mm512_loadu_ps(l_his + 26 * OC_DATA_BLK);
                if (u_s > 27) zmm27 = _mm512_loadu_ps(l_his + 27 * OC_DATA_BLK);
                if (u_s > 28) zmm28 = _mm512_loadu_ps(l_his + 28 * OC_DATA_BLK);
                if (u_s > 29) zmm29 = _mm512_loadu_ps(l_his + 29 * OC_DATA_BLK);
                if (u_s > 30) zmm30 = _mm512_loadu_ps(l_his + 30 * OC_DATA_BLK);
            }
        }

        if (kernel_flags & conv2d_n16cx_gemm_direct_kernel_fp32_avx512::flag::ADD_BIAS) {
            const float* bias = ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_avx512::param_def::BIAS_PTR_IDX);
            if (u_ocb > 0) {
                zmm31 = _mm512_loadu_ps(bias + 0 * OC_DATA_BLK);
                if (u_s > 0) zmm0 = _mm512_add_ps(zmm31, zmm0);
                if (u_s > 1) zmm1 = _mm512_add_ps(zmm31, zmm1);
                if (u_s > 2) zmm2 = _mm512_add_ps(zmm31, zmm2);
                if (u_s > 3) zmm3 = _mm512_add_ps(zmm31, zmm3);
                if (u_s > 4) zmm4 = _mm512_add_ps(zmm31, zmm4);
                if (u_s > 5) zmm5 = _mm512_add_ps(zmm31, zmm5);
                if (u_s > 6) zmm6 = _mm512_add_ps(zmm31, zmm6);
                if (u_s > 7) zmm7 = _mm512_add_ps(zmm31, zmm7);
                if (u_s > 8) zmm8 = _mm512_add_ps(zmm31, zmm8);
                if (u_s > 9) zmm9 = _mm512_add_ps(zmm31, zmm9);
                if (u_s > 10) zmm10 = _mm512_add_ps(zmm31, zmm10);
                if (u_s > 11) zmm11 = _mm512_add_ps(zmm31, zmm11);
                if (u_s > 12) zmm12 = _mm512_add_ps(zmm31, zmm12);
                if (u_s > 13) zmm13 = _mm512_add_ps(zmm31, zmm13);
                if (u_s > 14) zmm14 = _mm512_add_ps(zmm31, zmm14);
                if (u_s > 15) zmm15 = _mm512_add_ps(zmm31, zmm15);
                if (u_s > 16) zmm16 = _mm512_add_ps(zmm31, zmm16);
                if (u_s > 17) zmm17 = _mm512_add_ps(zmm31, zmm17);
                if (u_s > 18) zmm18 = _mm512_add_ps(zmm31, zmm18);
                if (u_s > 19) zmm19 = _mm512_add_ps(zmm31, zmm19);
                if (u_s > 20) zmm20 = _mm512_add_ps(zmm31, zmm20);
                if (u_s > 21) zmm21 = _mm512_add_ps(zmm31, zmm21);
                if (u_s > 22) zmm22 = _mm512_add_ps(zmm31, zmm22);
                if (u_s > 23) zmm23 = _mm512_add_ps(zmm31, zmm23);
                if (u_s > 24) zmm24 = _mm512_add_ps(zmm31, zmm24);
                if (u_s > 25) zmm25 = _mm512_add_ps(zmm31, zmm25);
                if (u_s > 26) zmm26 = _mm512_add_ps(zmm31, zmm26);
                if (u_s > 27) zmm27 = _mm512_add_ps(zmm31, zmm27);
                if (u_s > 28) zmm28 = _mm512_add_ps(zmm31, zmm28);
                if (u_s > 29) zmm29 = _mm512_add_ps(zmm31, zmm29);
                if (u_s > 30) zmm30 = _mm512_add_ps(zmm31, zmm30);
            }
        }
        
        const float *icb_src = src;
        const float *icb_flt = ker_p.pick<const float*>(conv2d_n16cx_gemm_direct_kernel_fp32_avx512::param_def::FLT_PTR_IDX);
        int64_t channels     = ker_p.pick<int64_t>(conv2d_n16cx_gemm_direct_kernel_fp32_avx512::param_def::CHANNELS_IDX);
        while (channels >= IC_DATA_BLK) {
            channels -= IC_DATA_BLK;
            for (int64_t ic = 0; ic < IC_DATA_BLK; ++ic) {
                IC_COMPUTE_STEP(0);
                icb_src += 1;
                icb_src += OC_DATA_BLK;
            }
            icb_src += src_icb_stride - IC_DATA_BLK;
        }
        if (channels > 0) {
            for (int64_t ic = 0; ic < channels; ++ic) {
                IC_COMPUTE_STEP(0);
                icb_src += 1;
                icb_src += OC_DATA_BLK;
            }
        }
        
        if (kernel_flags & (conv2d_n16cx_gemm_direct_kernel_fp32_avx512::flag::RELU | conv2d_n16cx_gemm_direct_kernel_fp32_avx512::flag::RELU6)) {
            zmm31 = _mm512_setzero_ps();
            if (u_ocb > 0) {
                if (u_s > 0) zmm0 = _mm512_max_ps(zmm0, zmm31);
                if (u_s > 1) zmm1 = _mm512_max_ps(zmm1, zmm31);
                if (u_s > 2) zmm2 = _mm512_max_ps(zmm2, zmm31);
                if (u_s > 3) zmm3 = _mm512_max_ps(zmm3, zmm31);
                if (u_s > 4) zmm4 = _mm512_max_ps(zmm4, zmm31);
                if (u_s > 5) zmm5 = _mm512_max_ps(zmm5, zmm31);
                if (u_s > 6) zmm6 = _mm512_max_ps(zmm6, zmm31);
                if (u_s > 7) zmm7 = _mm512_max_ps(zmm7, zmm31);
                if (u_s > 8) zmm8 = _mm512_max_ps(zmm8, zmm31);
                if (u_s > 9) zmm9 = _mm512_max_ps(zmm9, zmm31);
                if (u_s > 10) zmm10 = _mm512_max_ps(zmm10, zmm31);
                if (u_s > 11) zmm11 = _mm512_max_ps(zmm11, zmm31);
                if (u_s > 12) zmm12 = _mm512_max_ps(zmm12, zmm31);
                if (u_s > 13) zmm13 = _mm512_max_ps(zmm13, zmm31);
                if (u_s > 14) zmm14 = _mm512_max_ps(zmm14, zmm31);
                if (u_s > 15) zmm15 = _mm512_max_ps(zmm15, zmm31);
                if (u_s > 16) zmm16 = _mm512_max_ps(zmm16, zmm31);
                if (u_s > 17) zmm17 = _mm512_max_ps(zmm17, zmm31);
                if (u_s > 18) zmm18 = _mm512_max_ps(zmm18, zmm31);
                if (u_s > 19) zmm19 = _mm512_max_ps(zmm19, zmm31);
                if (u_s > 20) zmm20 = _mm512_max_ps(zmm20, zmm31);
                if (u_s > 21) zmm21 = _mm512_max_ps(zmm21, zmm31);
                if (u_s > 22) zmm22 = _mm512_max_ps(zmm22, zmm31);
                if (u_s > 23) zmm23 = _mm512_max_ps(zmm23, zmm31);
                if (u_s > 24) zmm24 = _mm512_max_ps(zmm24, zmm31);
                if (u_s > 25) zmm25 = _mm512_max_ps(zmm25, zmm31);
                if (u_s > 26) zmm26 = _mm512_max_ps(zmm26, zmm31);
                if (u_s > 27) zmm27 = _mm512_max_ps(zmm27, zmm31);
                if (u_s > 28) zmm28 = _mm512_max_ps(zmm28, zmm31);
                if (u_s > 29) zmm29 = _mm512_max_ps(zmm29, zmm31);
                if (u_s > 30) zmm30 = _mm512_max_ps(zmm30, zmm31);
            }
        }
        if (kernel_flags & conv2d_n16cx_gemm_direct_kernel_fp32_avx512::flag::RELU6) {
            zmm31 = _mm512_set1_ps(6.0f);
            if (u_ocb > 0) {
                if (u_s > 0) zmm0 = _mm512_min_ps(zmm0, zmm31);
                if (u_s > 1) zmm1 = _mm512_min_ps(zmm1, zmm31);
                if (u_s > 2) zmm2 = _mm512_min_ps(zmm2, zmm31);
                if (u_s > 3) zmm3 = _mm512_min_ps(zmm3, zmm31);
                if (u_s > 4) zmm4 = _mm512_min_ps(zmm4, zmm31);
                if (u_s > 5) zmm5 = _mm512_min_ps(zmm5, zmm31);
                if (u_s > 6) zmm6 = _mm512_min_ps(zmm6, zmm31);
                if (u_s > 7) zmm7 = _mm512_min_ps(zmm7, zmm31);
                if (u_s > 8) zmm8 = _mm512_min_ps(zmm8, zmm31);
                if (u_s > 9) zmm9 = _mm512_min_ps(zmm9, zmm31);
                if (u_s > 10) zmm10 = _mm512_min_ps(zmm10, zmm31);
                if (u_s > 11) zmm11 = _mm512_min_ps(zmm11, zmm31);
                if (u_s > 12) zmm12 = _mm512_min_ps(zmm12, zmm31);
                if (u_s > 13) zmm13 = _mm512_min_ps(zmm13, zmm31);
                if (u_s > 14) zmm14 = _mm512_min_ps(zmm14, zmm31);
                if (u_s > 15) zmm15 = _mm512_min_ps(zmm15, zmm31);
                if (u_s > 16) zmm16 = _mm512_min_ps(zmm16, zmm31);
                if (u_s > 17) zmm17 = _mm512_min_ps(zmm17, zmm31);
                if (u_s > 18) zmm18 = _mm512_min_ps(zmm18, zmm31);
                if (u_s > 19) zmm19 = _mm512_min_ps(zmm19, zmm31);
                if (u_s > 20) zmm20 = _mm512_min_ps(zmm20, zmm31);
                if (u_s > 21) zmm21 = _mm512_min_ps(zmm21, zmm31);
                if (u_s > 22) zmm22 = _mm512_min_ps(zmm22, zmm31);
                if (u_s > 23) zmm23 = _mm512_min_ps(zmm23, zmm31);
                if (u_s > 24) zmm24 = _mm512_min_ps(zmm24, zmm31);
                if (u_s > 25) zmm25 = _mm512_min_ps(zmm25, zmm31);
                if (u_s > 26) zmm26 = _mm512_min_ps(zmm26, zmm31);
                if (u_s > 27) zmm27 = _mm512_min_ps(zmm27, zmm31);
                if (u_s > 28) zmm28 = _mm512_min_ps(zmm28, zmm31);
                if (u_s > 29) zmm29 = _mm512_min_ps(zmm29, zmm31);
                if (u_s > 30) zmm30 = _mm512_min_ps(zmm30, zmm31);
            }
        }

        if (nt_store) {
            float* l_dst = dst;
            if (u_ocb > 0) {
                if (u_s > 0) _mm512_stream_ps(l_dst + 0 * OC_DATA_BLK, zmm0);
                if (u_s > 1) _mm512_stream_ps(l_dst + 1 * OC_DATA_BLK, zmm1);
                if (u_s > 2) _mm512_stream_ps(l_dst + 2 * OC_DATA_BLK, zmm2);
                if (u_s > 3) _mm512_stream_ps(l_dst + 3 * OC_DATA_BLK, zmm3);
                if (u_s > 4) _mm512_stream_ps(l_dst + 4 * OC_DATA_BLK, zmm4);
                if (u_s > 5) _mm512_stream_ps(l_dst + 5 * OC_DATA_BLK, zmm5);
                if (u_s > 6) _mm512_stream_ps(l_dst + 6 * OC_DATA_BLK, zmm6);
                if (u_s > 7) _mm512_stream_ps(l_dst + 7 * OC_DATA_BLK, zmm7);
                if (u_s > 8) _mm512_stream_ps(l_dst + 8 * OC_DATA_BLK, zmm8);
                if (u_s > 9) _mm512_stream_ps(l_dst + 9 * OC_DATA_BLK, zmm9);
                if (u_s > 10) _mm512_stream_ps(l_dst + 10 * OC_DATA_BLK, zmm10);
                if (u_s > 11) _mm512_stream_ps(l_dst + 11 * OC_DATA_BLK, zmm11);
                if (u_s > 12) _mm512_stream_ps(l_dst + 12 * OC_DATA_BLK, zmm12);
                if (u_s > 13) _mm512_stream_ps(l_dst + 13 * OC_DATA_BLK, zmm13);
                if (u_s > 14) _mm512_stream_ps(l_dst + 14 * OC_DATA_BLK, zmm14);
                if (u_s > 15) _mm512_stream_ps(l_dst + 15 * OC_DATA_BLK, zmm15);
                if (u_s > 16) _mm512_stream_ps(l_dst + 16 * OC_DATA_BLK, zmm16);
                if (u_s > 17) _mm512_stream_ps(l_dst + 17 * OC_DATA_BLK, zmm17);
                if (u_s > 18) _mm512_stream_ps(l_dst + 18 * OC_DATA_BLK, zmm18);
                if (u_s > 19) _mm512_stream_ps(l_dst + 19 * OC_DATA_BLK, zmm19);
                if (u_s > 20) _mm512_stream_ps(l_dst + 20 * OC_DATA_BLK, zmm20);
                if (u_s > 21) _mm512_stream_ps(l_dst + 21 * OC_DATA_BLK, zmm21);
                if (u_s > 22) _mm512_stream_ps(l_dst + 22 * OC_DATA_BLK, zmm22);
                if (u_s > 23) _mm512_stream_ps(l_dst + 23 * OC_DATA_BLK, zmm23);
                if (u_s > 24) _mm512_stream_ps(l_dst + 24 * OC_DATA_BLK, zmm24);
                if (u_s > 25) _mm512_stream_ps(l_dst + 25 * OC_DATA_BLK, zmm25);
                if (u_s > 26) _mm512_stream_ps(l_dst + 26 * OC_DATA_BLK, zmm26);
                if (u_s > 27) _mm512_stream_ps(l_dst + 27 * OC_DATA_BLK, zmm27);
                if (u_s > 28) _mm512_stream_ps(l_dst + 28 * OC_DATA_BLK, zmm28);
                if (u_s > 29) _mm512_stream_ps(l_dst + 29 * OC_DATA_BLK, zmm29);
                if (u_s > 30) _mm512_stream_ps(l_dst + 30 * OC_DATA_BLK, zmm30);
            }
        } else {
            float* l_dst = dst;
            if (u_ocb > 0) {
                if (u_s > 0) _mm512_storeu_ps(l_dst + 0 * OC_DATA_BLK, zmm0);
                if (u_s > 1) _mm512_storeu_ps(l_dst + 1 * OC_DATA_BLK, zmm1);
                if (u_s > 2) _mm512_storeu_ps(l_dst + 2 * OC_DATA_BLK, zmm2);
                if (u_s > 3) _mm512_storeu_ps(l_dst + 3 * OC_DATA_BLK, zmm3);
                if (u_s > 4) _mm512_storeu_ps(l_dst + 4 * OC_DATA_BLK, zmm4);
                if (u_s > 5) _mm512_storeu_ps(l_dst + 5 * OC_DATA_BLK, zmm5);
                if (u_s > 6) _mm512_storeu_ps(l_dst + 6 * OC_DATA_BLK, zmm6);
                if (u_s > 7) _mm512_storeu_ps(l_dst + 7 * OC_DATA_BLK, zmm7);
                if (u_s > 8) _mm512_storeu_ps(l_dst + 8 * OC_DATA_BLK, zmm8);
                if (u_s > 9) _mm512_storeu_ps(l_dst + 9 * OC_DATA_BLK, zmm9);
                if (u_s > 10) _mm512_storeu_ps(l_dst + 10 * OC_DATA_BLK, zmm10);
                if (u_s > 11) _mm512_storeu_ps(l_dst + 11 * OC_DATA_BLK, zmm11);
                if (u_s > 12) _mm512_storeu_ps(l_dst + 12 * OC_DATA_BLK, zmm12);
                if (u_s > 13) _mm512_storeu_ps(l_dst + 13 * OC_DATA_BLK, zmm13);
                if (u_s > 14) _mm512_storeu_ps(l_dst + 14 * OC_DATA_BLK, zmm14);
                if (u_s > 15) _mm512_storeu_ps(l_dst + 15 * OC_DATA_BLK, zmm15);
                if (u_s > 16) _mm512_storeu_ps(l_dst + 16 * OC_DATA_BLK, zmm16);
                if (u_s > 17) _mm512_storeu_ps(l_dst + 17 * OC_DATA_BLK, zmm17);
                if (u_s > 18) _mm512_storeu_ps(l_dst + 18 * OC_DATA_BLK, zmm18);
                if (u_s > 19) _mm512_storeu_ps(l_dst + 19 * OC_DATA_BLK, zmm19);
                if (u_s > 20) _mm512_storeu_ps(l_dst + 20 * OC_DATA_BLK, zmm20);
                if (u_s > 21) _mm512_storeu_ps(l_dst + 21 * OC_DATA_BLK, zmm21);
                if (u_s > 22) _mm512_storeu_ps(l_dst + 22 * OC_DATA_BLK, zmm22);
                if (u_s > 23) _mm512_storeu_ps(l_dst + 23 * OC_DATA_BLK, zmm23);
                if (u_s > 24) _mm512_storeu_ps(l_dst + 24 * OC_DATA_BLK, zmm24);
                if (u_s > 25) _mm512_storeu_ps(l_dst + 25 * OC_DATA_BLK, zmm25);
                if (u_s > 26) _mm512_storeu_ps(l_dst + 26 * OC_DATA_BLK, zmm26);
                if (u_s > 27) _mm512_storeu_ps(l_dst + 27 * OC_DATA_BLK, zmm27);
                if (u_s > 28) _mm512_storeu_ps(l_dst + 28 * OC_DATA_BLK, zmm28);
                if (u_s > 29) _mm512_storeu_ps(l_dst + 29 * OC_DATA_BLK, zmm29);
                if (u_s > 30) _mm512_storeu_ps(l_dst + 30 * OC_DATA_BLK, zmm30);
            }
        }
        src += u_s * IC_DATA_BLK;
        his += u_s * OC_DATA_BLK;
        dst += u_s * OC_DATA_BLK;
        space -= u_s;
    } while (space > 0);
#undef IC_COMPUTE_STEP
}

}}};

#endif
