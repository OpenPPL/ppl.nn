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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_IM2COL_GEMM_AVX512_CONV2D_IM2COL_GEMM_O31_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_IM2COL_GEMM_AVX512_CONV2D_IM2COL_GEMM_O31_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/im2col_gemm/avx512/conv2d_im2col_gemm_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template <int64_t o_len>
inline void conv2d_im2col_gemm_o31_kernel_fp32_avx512_core(
    const int64_t *param)
{
    __asm__ __volatile__ (
        ".equ P_BYTES, 8\n"
        ".equ D_BYTES, 4\n"

        ".equ NK_DT_BLK, 16\n"

        ".equ SRC_IDX,      (0 * P_BYTES)\n"
        ".equ DST_IDX,      (1 * P_BYTES)\n"
        ".equ FLT_IDX,      (2 * P_BYTES)\n"
        ".equ BIAS_DIX,     (3 * P_BYTES)\n"
        ".equ OC_IDX,       (4 * P_BYTES)\n"
        ".equ K_IDX,        (5 * P_BYTES)\n"
        ".equ FLAGS_IDX,    (6 * P_BYTES)\n"
        ".equ SRC_HWB_STRIDE_IDX, (7 * P_BYTES)\n"
        ".equ DST_HWB_STRIDE_IDX, (8 * P_BYTES)\n"
        ".equ FLT_KB_STRIDE_IDX,  (9 * P_BYTES)\n"
        ".equ SIX_IDX,            (10 * P_BYTES)\n"

        ".equ KERNEL_FLAG_LD_BIAS, %c[KERNEL_FLAG_LD_BIAS]\n"
        ".equ KERNEL_FLAG_RELU, %c[KERNEL_FLAG_RELU]\n"
        ".equ KERNEL_FLAG_RELU6, %c[KERNEL_FLAG_RELU6]\n"
        ".equ O_LEN, %c[O_LEN]\n"

        "mov FLT_KB_STRIDE_IDX(%[param]), %%r8\n"
        "mov FLAGS_IDX(%[param]), %%r11\n"
        "mov BIAS_IDX(%[param]), %%r12\n"
        "mov DST_IDX(%[param]), %%r13\n"
        "mov FLT_IDX(%[param]), %%r14\n"
        "mov OC_IDX(%[param]), %%r15\n"
"1:\n" // label_init_session
        "test $KERNEL_FLAG_LD_BIAS, %%r11\n"
        "jnz 2f\n" // label_load_dst
        ".if O_LEN > 0\n vboradcastss (0 * D_BYTES)(%%r12), %%zmm0\n .endif\n"
        ".if O_LEN > 1\n vboradcastss (1 * D_BYTES)(%%r12), %%zmm1\n .endif\n"
        ".if O_LEN > 2\n vboradcastss (2 * D_BYTES)(%%r12), %%zmm2\n .endif\n"
        ".if O_LEN > 3\n vboradcastss (3 * D_BYTES)(%%r12), %%zmm3\n .endif\n"
        ".if O_LEN > 4\n vboradcastss (4 * D_BYTES)(%%r12), %%zmm4\n .endif\n"
        ".if O_LEN > 5\n vboradcastss (5 * D_BYTES)(%%r12), %%zmm5\n .endif\n"
        ".if O_LEN > 6\n vboradcastss (6 * D_BYTES)(%%r12), %%zmm6\n .endif\n"
        ".if O_LEN > 7\n vboradcastss (7 * D_BYTES)(%%r12), %%zmm7\n .endif\n"
        ".if O_LEN > 8\n vboradcastss (8 * D_BYTES)(%%r12), %%zmm8\n .endif\n"
        ".if O_LEN > 9\n vboradcastss (9 * D_BYTES)(%%r12), %%zmm9\n .endif\n"
        ".if O_LEN > 10\n vboradcastss (10 * D_BYTES)(%%r12), %%zmm10\n .endif\n"
        ".if O_LEN > 11\n vboradcastss (11 * D_BYTES)(%%r12), %%zmm11\n .endif\n"
        ".if O_LEN > 12\n vboradcastss (12 * D_BYTES)(%%r12), %%zmm12\n .endif\n"
        ".if O_LEN > 13\n vboradcastss (13 * D_BYTES)(%%r12), %%zmm13\n .endif\n"
        ".if O_LEN > 14\n vboradcastss (14 * D_BYTES)(%%r12), %%zmm14\n .endif\n"
        ".if O_LEN > 15\n vboradcastss (15 * D_BYTES)(%%r12), %%zmm15\n .endif\n"
        ".if O_LEN > 16\n vboradcastss (16 * D_BYTES)(%%r12), %%zmm16\n .endif\n"
        ".if O_LEN > 17\n vboradcastss (17 * D_BYTES)(%%r12), %%zmm17\n .endif\n"
        ".if O_LEN > 18\n vboradcastss (18 * D_BYTES)(%%r12), %%zmm18\n .endif\n"
        ".if O_LEN > 19\n vboradcastss (19 * D_BYTES)(%%r12), %%zmm19\n .endif\n"
        ".if O_LEN > 20\n vboradcastss (20 * D_BYTES)(%%r12), %%zmm20\n .endif\n"
        ".if O_LEN > 21\n vboradcastss (21 * D_BYTES)(%%r12), %%zmm21\n .endif\n"
        ".if O_LEN > 22\n vboradcastss (22 * D_BYTES)(%%r12), %%zmm22\n .endif\n"
        ".if O_LEN > 23\n vboradcastss (23 * D_BYTES)(%%r12), %%zmm23\n .endif\n"
        ".if O_LEN > 24\n vboradcastss (24 * D_BYTES)(%%r12), %%zmm24\n .endif\n"
        ".if O_LEN > 25\n vboradcastss (25 * D_BYTES)(%%r12), %%zmm25\n .endif\n"
        ".if O_LEN > 26\n vboradcastss (26 * D_BYTES)(%%r12), %%zmm26\n .endif\n"
        ".if O_LEN > 27\n vboradcastss (27 * D_BYTES)(%%r12), %%zmm27\n .endif\n"
        ".if O_LEN > 28\n vboradcastss (28 * D_BYTES)(%%r12), %%zmm28\n .endif\n"
        ".if O_LEN > 29\n vboradcastss (29 * D_BYTES)(%%r12), %%zmm29\n .endif\n"
        ".if O_LEN > 30\n vboradcastss (30 * D_BYTES)(%%r12), %%zmm30\n .endif\n"
        "lea O_LEN(%%r12), %%r12\n"
        "jmp 3f\n"
"2:\n" // label_load_dst
        ".if O_LEN > 0\n vmovups (0 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm0\n .endif\n"
        ".if O_LEN > 1\n vmovups (1 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm1\n .endif\n"
        ".if O_LEN > 2\n vmovups (2 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm2\n .endif\n"
        ".if O_LEN > 3\n vmovups (3 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm3\n .endif\n"
        ".if O_LEN > 4\n vmovups (4 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm4\n .endif\n"
        ".if O_LEN > 5\n vmovups (5 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm5\n .endif\n"
        ".if O_LEN > 6\n vmovups (6 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm6\n .endif\n"
        ".if O_LEN > 7\n vmovups (7 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm7\n .endif\n"
        ".if O_LEN > 8\n vmovups (8 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm8\n .endif\n"
        ".if O_LEN > 9\n vmovups (9 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm9\n .endif\n"
        ".if O_LEN > 10\n vmovups (10 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm10\n .endif\n"
        ".if O_LEN > 11\n vmovups (11 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm11\n .endif\n"
        ".if O_LEN > 12\n vmovups (12 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm12\n .endif\n"
        ".if O_LEN > 13\n vmovups (13 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm13\n .endif\n"
        ".if O_LEN > 14\n vmovups (14 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm14\n .endif\n"
        ".if O_LEN > 15\n vmovups (15 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm15\n .endif\n"
        ".if O_LEN > 16\n vmovups (16 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm16\n .endif\n"
        ".if O_LEN > 17\n vmovups (17 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm17\n .endif\n"
        ".if O_LEN > 18\n vmovups (18 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm18\n .endif\n"
        ".if O_LEN > 19\n vmovups (19 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm19\n .endif\n"
        ".if O_LEN > 20\n vmovups (20 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm20\n .endif\n"
        ".if O_LEN > 21\n vmovups (21 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm21\n .endif\n"
        ".if O_LEN > 22\n vmovups (22 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm22\n .endif\n"
        ".if O_LEN > 23\n vmovups (23 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm23\n .endif\n"
        ".if O_LEN > 24\n vmovups (24 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm24\n .endif\n"
        ".if O_LEN > 25\n vmovups (25 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm25\n .endif\n"
        ".if O_LEN > 26\n vmovups (26 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm26\n .endif\n"
        ".if O_LEN > 27\n vmovups (27 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm27\n .endif\n"
        ".if O_LEN > 28\n vmovups (28 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm28\n .endif\n"
        ".if O_LEN > 29\n vmovups (29 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm29\n .endif\n"
        ".if O_LEN > 30\n vmovups (30 * NK_DT_BLK * D_BYTES)(%%r13), %%zmm30\n .endif\n"

"3:\n" // label_compute_session
        "mov %%r14, %%rax\n"
        "mov SRC_IDX(%[param]), %%rbx\n"
        "mov K_IDX(%[param]), %%r10\n"
        "cmp $NK_DT_BLK, %%r10\n"
        "jl 6f\n" // label_ic_remain
"4:\n" // label_ic_body
        PPL_X86_INLINE_ASM_ALIGN()
        ".if O_LEN < 9\n"
        ".irp IC,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
        "vmovups (\\IC * NK_DT_BLK * D_BYTES)(%%rbx), %%zmm31\n"
        ".if O_LEN > 7\n prefetcht0 ((\\IC * NK_DT_BLK + NK_DT_BLK * NK_DT_BLK) * D_BYTES)(%%rbx)\n .endif\n"
        ".if O_LEN > 0\n vfmadd231ps ((\\IC + 0 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm0\n .endif\n"
        ".if O_LEN > 1\n vfmadd231ps ((\\IC + 1 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm1\n .endif\n"
        ".if O_LEN > 2\n vfmadd231ps ((\\IC + 2 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm2\n .endif\n"
        ".if O_LEN > 3\n vfmadd231ps ((\\IC + 3 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm3\n .endif\n"
        ".if O_LEN > 4\n vfmadd231ps ((\\IC + 4 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm4\n .endif\n"
        ".if O_LEN > 5\n vfmadd231ps ((\\IC + 5 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm5\n .endif\n"
        ".if O_LEN > 6\n vfmadd231ps ((\\IC + 6 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm6\n .endif\n"
        ".if O_LEN > 7\n vfmadd231ps ((\\IC + 7 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm7\n .endif\n"
        ".if O_LEN > 8\n vfmadd231ps ((\\IC + 8 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm8\n .endif\n"
        ".if O_LEN > 9\n vfmadd231ps ((\\IC + 9 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm9\n .endif\n"
        ".if O_LEN > 10\n vfmadd231ps ((\\IC + 10 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm10\n .endif\n"
        ".if O_LEN > 11\n vfmadd231ps ((\\IC + 11 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm11\n .endif\n"
        ".if O_LEN > 12\n vfmadd231ps ((\\IC + 12 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm12\n .endif\n"
        ".if O_LEN > 13\n vfmadd231ps ((\\IC + 13 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm13\n .endif\n"
        ".if O_LEN > 14\n vfmadd231ps ((\\IC + 14 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm14\n .endif\n"
        ".if O_LEN > 15\n vfmadd231ps ((\\IC + 15 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm15\n .endif\n"
        ".if O_LEN > 16\n vfmadd231ps ((\\IC + 16 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm16\n .endif\n"
        ".if O_LEN > 17\n vfmadd231ps ((\\IC + 17 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm17\n .endif\n"
        ".if O_LEN > 18\n vfmadd231ps ((\\IC + 18 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm18\n .endif\n"
        ".if O_LEN > 19\n vfmadd231ps ((\\IC + 19 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm19\n .endif\n"
        ".if O_LEN > 20\n vfmadd231ps ((\\IC + 20 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm20\n .endif\n"
        ".if O_LEN > 21\n vfmadd231ps ((\\IC + 21 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm21\n .endif\n"
        ".if O_LEN > 22\n vfmadd231ps ((\\IC + 22 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm22\n .endif\n"
        ".if O_LEN > 23\n vfmadd231ps ((\\IC + 23 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm23\n .endif\n"
        ".if O_LEN > 24\n vfmadd231ps ((\\IC + 24 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm24\n .endif\n"
        ".if O_LEN > 25\n vfmadd231ps ((\\IC + 25 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm25\n .endif\n"
        ".if O_LEN > 26\n vfmadd231ps ((\\IC + 26 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm26\n .endif\n"
        ".if O_LEN > 27\n vfmadd231ps ((\\IC + 27 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm27\n .endif\n"
        ".if O_LEN > 28\n vfmadd231ps ((\\IC + 28 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm28\n .endif\n"
        ".if O_LEN > 29\n vfmadd231ps ((\\IC + 29 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm29\n .endif\n"
        ".if O_LEN > 30\n vfmadd231ps ((\\IC + 30 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm30\n .endif\n"
        ".endr\n"
        "lea (NK_DT_BLK * NK_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        ".else\n" // .if O_LEN < 9
        "mov %%rax, %%rcx\n"
        "mov $NK_DT_BLK, %%r9\n"
"9:\n" // label_ic
        "vmovups (0 * NK_DT_BLK * D_BYTES)(%%rbx), %%zmm31\n"
        "prefetcht0 ((0 * NK_DT_BLK + NK_DT_BLK * NK_DT_BLK) * D_BYTES)(%%rbx)\n"
        "lea (NK_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        ".if O_LEN > 0\n vfmadd231ps ((0 + 0 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm0\n .endif\n"
        ".if O_LEN > 1\n vfmadd231ps ((0 + 1 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm1\n .endif\n"
        ".if O_LEN > 2\n vfmadd231ps ((0 + 2 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm2\n .endif\n"
        ".if O_LEN > 3\n vfmadd231ps ((0 + 3 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm3\n .endif\n"
        ".if O_LEN > 4\n vfmadd231ps ((0 + 4 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm4\n .endif\n"
        ".if O_LEN > 5\n vfmadd231ps ((0 + 5 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm5\n .endif\n"
        ".if O_LEN > 6\n vfmadd231ps ((0 + 6 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm6\n .endif\n"
        ".if O_LEN > 7\n vfmadd231ps ((0 + 7 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm7\n .endif\n"
        ".if O_LEN > 8\n vfmadd231ps ((0 + 8 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm8\n .endif\n"
        ".if O_LEN > 9\n vfmadd231ps ((0 + 9 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm9\n .endif\n"
        ".if O_LEN > 10\n vfmadd231ps ((0 + 10 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm10\n .endif\n"
        ".if O_LEN > 11\n vfmadd231ps ((0 + 11 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm11\n .endif\n"
        ".if O_LEN > 12\n vfmadd231ps ((0 + 12 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm12\n .endif\n"
        ".if O_LEN > 13\n vfmadd231ps ((0 + 13 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm13\n .endif\n"
        ".if O_LEN > 14\n vfmadd231ps ((0 + 14 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm14\n .endif\n"
        ".if O_LEN > 15\n vfmadd231ps ((0 + 15 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm15\n .endif\n"
        ".if O_LEN > 16\n vfmadd231ps ((0 + 16 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm16\n .endif\n"
        ".if O_LEN > 17\n vfmadd231ps ((0 + 17 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm17\n .endif\n"
        ".if O_LEN > 18\n vfmadd231ps ((0 + 18 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm18\n .endif\n"
        ".if O_LEN > 19\n vfmadd231ps ((0 + 19 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm19\n .endif\n"
        ".if O_LEN > 20\n vfmadd231ps ((0 + 20 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm20\n .endif\n"
        ".if O_LEN > 21\n vfmadd231ps ((0 + 21 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm21\n .endif\n"
        ".if O_LEN > 22\n vfmadd231ps ((0 + 22 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm22\n .endif\n"
        ".if O_LEN > 23\n vfmadd231ps ((0 + 23 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm23\n .endif\n"
        ".if O_LEN > 24\n vfmadd231ps ((0 + 24 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm24\n .endif\n"
        ".if O_LEN > 25\n vfmadd231ps ((0 + 25 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm25\n .endif\n"
        ".if O_LEN > 26\n vfmadd231ps ((0 + 26 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm26\n .endif\n"
        ".if O_LEN > 27\n vfmadd231ps ((0 + 27 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm27\n .endif\n"
        ".if O_LEN > 28\n vfmadd231ps ((0 + 28 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm28\n .endif\n"
        ".if O_LEN > 29\n vfmadd231ps ((0 + 29 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm29\n .endif\n"
        ".if O_LEN > 30\n vfmadd231ps ((0 + 30 * NK_DT_BLK) * D_BYTES)(%%rcx)%{1to16}, %%zmm31, %%zmm30\n .endif\n"
        "lea D_BYTES(%%rcx), %%rcx\n"
        "sub $1, %%r9\n"
        "cmp $0, %%r9\n"
        "jne 9b\n" // label_ic
        ".endif\n" // .if O_LEN < 9
        "lea (%%rax, %%r8, D_BYTES), %%rax\n"
        "sub $NK_DT_BLK, %%r10\n"
        "cmp $NK_DT_BLK, %%r10\n"
        "jge 4b\n"// label_ic_body
        "cmp $0, %%r10\n"
        "je 6f\n" // label_finalize_session
"5:\n" // label_ic_remain
        "vmovups (0 * NK_DT_BLK * D_BYTES)(%%rbx), %%zmm31\n"
        "lea (NK_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        ".if O_LEN > 0\n vfmadd231ps ((0 + 0 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm0\n .endif\n"
        ".if O_LEN > 1\n vfmadd231ps ((0 + 1 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm1\n .endif\n"
        ".if O_LEN > 2\n vfmadd231ps ((0 + 2 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm2\n .endif\n"
        ".if O_LEN > 3\n vfmadd231ps ((0 + 3 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm3\n .endif\n"
        ".if O_LEN > 4\n vfmadd231ps ((0 + 4 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm4\n .endif\n"
        ".if O_LEN > 5\n vfmadd231ps ((0 + 5 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm5\n .endif\n"
        ".if O_LEN > 6\n vfmadd231ps ((0 + 6 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm6\n .endif\n"
        ".if O_LEN > 7\n vfmadd231ps ((0 + 7 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm7\n .endif\n"
        ".if O_LEN > 8\n vfmadd231ps ((0 + 8 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm8\n .endif\n"
        ".if O_LEN > 9\n vfmadd231ps ((0 + 9 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm9\n .endif\n"
        ".if O_LEN > 10\n vfmadd231ps ((0 + 10 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm10\n .endif\n"
        ".if O_LEN > 11\n vfmadd231ps ((0 + 11 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm11\n .endif\n"
        ".if O_LEN > 12\n vfmadd231ps ((0 + 12 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm12\n .endif\n"
        ".if O_LEN > 13\n vfmadd231ps ((0 + 13 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm13\n .endif\n"
        ".if O_LEN > 14\n vfmadd231ps ((0 + 14 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm14\n .endif\n"
        ".if O_LEN > 15\n vfmadd231ps ((0 + 15 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm15\n .endif\n"
        ".if O_LEN > 16\n vfmadd231ps ((0 + 16 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm16\n .endif\n"
        ".if O_LEN > 17\n vfmadd231ps ((0 + 17 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm17\n .endif\n"
        ".if O_LEN > 18\n vfmadd231ps ((0 + 18 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm18\n .endif\n"
        ".if O_LEN > 19\n vfmadd231ps ((0 + 19 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm19\n .endif\n"
        ".if O_LEN > 20\n vfmadd231ps ((0 + 20 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm20\n .endif\n"
        ".if O_LEN > 21\n vfmadd231ps ((0 + 21 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm21\n .endif\n"
        ".if O_LEN > 22\n vfmadd231ps ((0 + 22 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm22\n .endif\n"
        ".if O_LEN > 23\n vfmadd231ps ((0 + 23 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm23\n .endif\n"
        ".if O_LEN > 24\n vfmadd231ps ((0 + 24 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm24\n .endif\n"
        ".if O_LEN > 25\n vfmadd231ps ((0 + 25 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm25\n .endif\n"
        ".if O_LEN > 26\n vfmadd231ps ((0 + 26 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm26\n .endif\n"
        ".if O_LEN > 27\n vfmadd231ps ((0 + 27 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm27\n .endif\n"
        ".if O_LEN > 28\n vfmadd231ps ((0 + 28 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm28\n .endif\n"
        ".if O_LEN > 29\n vfmadd231ps ((0 + 29 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm29\n .endif\n"
        ".if O_LEN > 30\n vfmadd231ps ((0 + 30 * NK_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm30\n .endif\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%r10\n"
        "cmp $0, %%r10\n"
        "jne 5b\n" // label_ic_remain

"6:\n" // label_finalize_session
        ".if O_LEN > 0\n vmovups %%zmm0, (0 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 1\n vmovups %%zmm1, (1 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 2\n vmovups %%zmm2, (2 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 3\n vmovups %%zmm3, (3 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 4\n vmovups %%zmm4, (4 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 5\n vmovups %%zmm5, (5 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 6\n vmovups %%zmm6, (6 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 7\n vmovups %%zmm7, (7 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 8\n vmovups %%zmm8, (8 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 9\n vmovups %%zmm9, (9 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 10\n vmovups %%zmm10, (10 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 11\n vmovups %%zmm11, (11 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 12\n vmovups %%zmm12, (12 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 13\n vmovups %%zmm13, (13 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 14\n vmovups %%zmm14, (14 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 15\n vmovups %%zmm15, (15 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 16\n vmovups %%zmm16, (16 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 17\n vmovups %%zmm17, (17 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 18\n vmovups %%zmm18, (18 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 19\n vmovups %%zmm19, (19 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 20\n vmovups %%zmm20, (20 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 21\n vmovups %%zmm21, (21 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 22\n vmovups %%zmm22, (22 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 23\n vmovups %%zmm23, (23 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 24\n vmovups %%zmm24, (24 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 25\n vmovups %%zmm25, (25 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 26\n vmovups %%zmm26, (26 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 27\n vmovups %%zmm27, (27 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 28\n vmovups %%zmm28, (28 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 29\n vmovups %%zmm29, (29 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if O_LEN > 30\n vmovups %%zmm30, (30 * NK_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        "sub $O_LEN, %%r15\n"
        "cmp $0, %%r15\n"
        "lea (O_LEN * NK_DT_BLK * D_BYTES)(%%r13), %%r13\n"
        "lea (O_LEN * NK_DT_BLK * D_BYTES)(%%r14), %%r14\n"
        "jne 1b\n" // label_init_session
        :
        :
        [param]                 "r" (param),
        [O_LEN]                 "i" (o_len),
        [KERNEL_FLAG_LD_BIAS]   "i" (KERNEL_FLAG_LD_BIAS()),
        [KERNEL_FLAG_RELU]      "i" (KERNEL_FLAG_RELU()),
        [KERNEL_FLAG_RELU6]     "i" (KERNEL_FLAG_RELU6())
        :
        "cc",
        "rax", "rbx", "rcx",
        "r8" , "r9" , "r10", "r11", "r12", "r13", "r14", "r15",
        "zmm0" , "zmm1" , "zmm2" , "zmm3" , "zmm4" , "zmm5" , "zmm6" , "zmm7" ,
        "zmm8" , "zmm9" , "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
        "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
        "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
        "memory"
    );
}

#endif

template <int64_t hw_len, int64_t o_len>
void conv2d_im2col_gemm_o31_kernel_fp32_avx512(
    const int64_t *param)
{

#ifdef PPL_USE_X86_INLINE_ASM
    if (hw_len == 1 * NK_DT_BLK()) {
        conv2d_im2col_gemm_o31_kernel_fp32_avx512_core<o_len>(param);
        return;
    }
#endif

#define IC_COMPUTE_STEP(IC) do {\
    zmm31 = _mm512_loadu_ps(k_src + (IC) * NK_DT_BLK());\
    if (o_len > 0) zmm0 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 0 * NK_DT_BLK()]), zmm31, zmm0);\
    if (o_len > 1) zmm1 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 1 * NK_DT_BLK()]), zmm31, zmm1);\
    if (o_len > 2) zmm2 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 2 * NK_DT_BLK()]), zmm31, zmm2);\
    if (o_len > 3) zmm3 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 3 * NK_DT_BLK()]), zmm31, zmm3);\
    if (o_len > 4) zmm4 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 4 * NK_DT_BLK()]), zmm31, zmm4);\
    if (o_len > 5) zmm5 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 5 * NK_DT_BLK()]), zmm31, zmm5);\
    if (o_len > 6) zmm6 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 6 * NK_DT_BLK()]), zmm31, zmm6);\
    if (o_len > 7) zmm7 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 7 * NK_DT_BLK()]), zmm31, zmm7);\
    if (o_len > 8) zmm8 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 8 * NK_DT_BLK()]), zmm31, zmm8);\
    if (o_len > 9) zmm9 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 9 * NK_DT_BLK()]), zmm31, zmm9);\
    if (o_len > 10) zmm10 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 10 * NK_DT_BLK()]), zmm31, zmm10);\
    if (o_len > 11) zmm11 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 11 * NK_DT_BLK()]), zmm31, zmm11);\
    if (o_len > 12) zmm12 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 12 * NK_DT_BLK()]), zmm31, zmm12);\
    if (o_len > 13) zmm13 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 13 * NK_DT_BLK()]), zmm31, zmm13);\
    if (o_len > 14) zmm14 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 14 * NK_DT_BLK()]), zmm31, zmm14);\
    if (o_len > 15) zmm15 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 15 * NK_DT_BLK()]), zmm31, zmm15);\
    if (o_len > 16) zmm16 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 16 * NK_DT_BLK()]), zmm31, zmm16);\
    if (o_len > 17) zmm17 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 17 * NK_DT_BLK()]), zmm31, zmm17);\
    if (o_len > 18) zmm18 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 18 * NK_DT_BLK()]), zmm31, zmm18);\
    if (o_len > 19) zmm19 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 19 * NK_DT_BLK()]), zmm31, zmm19);\
    if (o_len > 20) zmm20 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 20 * NK_DT_BLK()]), zmm31, zmm20);\
    if (o_len > 21) zmm21 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 21 * NK_DT_BLK()]), zmm31, zmm21);\
    if (o_len > 22) zmm22 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 22 * NK_DT_BLK()]), zmm31, zmm22);\
    if (o_len > 23) zmm23 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 23 * NK_DT_BLK()]), zmm31, zmm23);\
    if (o_len > 24) zmm24 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 24 * NK_DT_BLK()]), zmm31, zmm24);\
    if (o_len > 25) zmm25 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 25 * NK_DT_BLK()]), zmm31, zmm25);\
    if (o_len > 26) zmm26 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 26 * NK_DT_BLK()]), zmm31, zmm26);\
    if (o_len > 27) zmm27 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 27 * NK_DT_BLK()]), zmm31, zmm27);\
    if (o_len > 28) zmm28 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 28 * NK_DT_BLK()]), zmm31, zmm28);\
    if (o_len > 29) zmm29 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 29 * NK_DT_BLK()]), zmm31, zmm29);\
    if (o_len > 30) zmm30 = _mm512_fmadd_ps(_mm512_set1_ps(k_flt[(IC) + 30 * NK_DT_BLK()]), zmm31, zmm30);\
} while (0)

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    const int64_t flt_kb_stride  = param[FLT_KB_STRIDE_IDX()];
    const int64_t kernel_flags   = param[FLAGS_IDX()];

    const float *o_flt  = PICK_PARAM(const float*, param, FLT_IDX());
    const float *o_bias = PICK_PARAM(const float*, param, BIAS_IDX());
    float *o_dst        = PICK_PARAM(float*, param, DST_IDX());
    int64_t o           = param[OC_IDX()];
    do {
        if (kernel_flags & KERNEL_FLAG_LD_BIAS()) {
            if (hw_len > 0 * NK_DT_BLK()) {
                if (o_len > 0) zmm0 = _mm512_set1_ps(o_bias[0]);
                if (o_len > 1) zmm1 = _mm512_set1_ps(o_bias[1]);
                if (o_len > 2) zmm2 = _mm512_set1_ps(o_bias[2]);
                if (o_len > 3) zmm3 = _mm512_set1_ps(o_bias[3]);
                if (o_len > 4) zmm4 = _mm512_set1_ps(o_bias[4]);
                if (o_len > 5) zmm5 = _mm512_set1_ps(o_bias[5]);
                if (o_len > 6) zmm6 = _mm512_set1_ps(o_bias[6]);
                if (o_len > 7) zmm7 = _mm512_set1_ps(o_bias[7]);
                if (o_len > 8) zmm8 = _mm512_set1_ps(o_bias[8]);
                if (o_len > 9) zmm9 = _mm512_set1_ps(o_bias[9]);
                if (o_len > 10) zmm10 = _mm512_set1_ps(o_bias[10]);
                if (o_len > 11) zmm11 = _mm512_set1_ps(o_bias[11]);
                if (o_len > 12) zmm12 = _mm512_set1_ps(o_bias[12]);
                if (o_len > 13) zmm13 = _mm512_set1_ps(o_bias[13]);
                if (o_len > 14) zmm14 = _mm512_set1_ps(o_bias[14]);
                if (o_len > 15) zmm15 = _mm512_set1_ps(o_bias[15]);
                if (o_len > 16) zmm16 = _mm512_set1_ps(o_bias[16]);
                if (o_len > 17) zmm17 = _mm512_set1_ps(o_bias[17]);
                if (o_len > 18) zmm18 = _mm512_set1_ps(o_bias[18]);
                if (o_len > 19) zmm19 = _mm512_set1_ps(o_bias[19]);
                if (o_len > 20) zmm20 = _mm512_set1_ps(o_bias[20]);
                if (o_len > 21) zmm21 = _mm512_set1_ps(o_bias[21]);
                if (o_len > 22) zmm22 = _mm512_set1_ps(o_bias[22]);
                if (o_len > 23) zmm23 = _mm512_set1_ps(o_bias[23]);
                if (o_len > 24) zmm24 = _mm512_set1_ps(o_bias[24]);
                if (o_len > 25) zmm25 = _mm512_set1_ps(o_bias[25]);
                if (o_len > 26) zmm26 = _mm512_set1_ps(o_bias[26]);
                if (o_len > 27) zmm27 = _mm512_set1_ps(o_bias[27]);
                if (o_len > 28) zmm28 = _mm512_set1_ps(o_bias[28]);
                if (o_len > 29) zmm29 = _mm512_set1_ps(o_bias[29]);
                if (o_len > 30) zmm30 = _mm512_set1_ps(o_bias[30]);
                o_bias += o_len;
            }
        } else {
            const float *l_dst = o_dst;
            if (hw_len > 0 * NK_DT_BLK()) {
                if (o_len > 0) zmm0 = _mm512_loadu_ps(l_dst + 0 * NK_DT_BLK());
                if (o_len > 1) zmm1 = _mm512_loadu_ps(l_dst + 1 * NK_DT_BLK());
                if (o_len > 2) zmm2 = _mm512_loadu_ps(l_dst + 2 * NK_DT_BLK());
                if (o_len > 3) zmm3 = _mm512_loadu_ps(l_dst + 3 * NK_DT_BLK());
                if (o_len > 4) zmm4 = _mm512_loadu_ps(l_dst + 4 * NK_DT_BLK());
                if (o_len > 5) zmm5 = _mm512_loadu_ps(l_dst + 5 * NK_DT_BLK());
                if (o_len > 6) zmm6 = _mm512_loadu_ps(l_dst + 6 * NK_DT_BLK());
                if (o_len > 7) zmm7 = _mm512_loadu_ps(l_dst + 7 * NK_DT_BLK());
                if (o_len > 8) zmm8 = _mm512_loadu_ps(l_dst + 8 * NK_DT_BLK());
                if (o_len > 9) zmm9 = _mm512_loadu_ps(l_dst + 9 * NK_DT_BLK());
                if (o_len > 10) zmm10 = _mm512_loadu_ps(l_dst + 10 * NK_DT_BLK());
                if (o_len > 11) zmm11 = _mm512_loadu_ps(l_dst + 11 * NK_DT_BLK());
                if (o_len > 12) zmm12 = _mm512_loadu_ps(l_dst + 12 * NK_DT_BLK());
                if (o_len > 13) zmm13 = _mm512_loadu_ps(l_dst + 13 * NK_DT_BLK());
                if (o_len > 14) zmm14 = _mm512_loadu_ps(l_dst + 14 * NK_DT_BLK());
                if (o_len > 15) zmm15 = _mm512_loadu_ps(l_dst + 15 * NK_DT_BLK());
                if (o_len > 16) zmm16 = _mm512_loadu_ps(l_dst + 16 * NK_DT_BLK());
                if (o_len > 17) zmm17 = _mm512_loadu_ps(l_dst + 17 * NK_DT_BLK());
                if (o_len > 18) zmm18 = _mm512_loadu_ps(l_dst + 18 * NK_DT_BLK());
                if (o_len > 19) zmm19 = _mm512_loadu_ps(l_dst + 19 * NK_DT_BLK());
                if (o_len > 20) zmm20 = _mm512_loadu_ps(l_dst + 20 * NK_DT_BLK());
                if (o_len > 21) zmm21 = _mm512_loadu_ps(l_dst + 21 * NK_DT_BLK());
                if (o_len > 22) zmm22 = _mm512_loadu_ps(l_dst + 22 * NK_DT_BLK());
                if (o_len > 23) zmm23 = _mm512_loadu_ps(l_dst + 23 * NK_DT_BLK());
                if (o_len > 24) zmm24 = _mm512_loadu_ps(l_dst + 24 * NK_DT_BLK());
                if (o_len > 25) zmm25 = _mm512_loadu_ps(l_dst + 25 * NK_DT_BLK());
                if (o_len > 26) zmm26 = _mm512_loadu_ps(l_dst + 26 * NK_DT_BLK());
                if (o_len > 27) zmm27 = _mm512_loadu_ps(l_dst + 27 * NK_DT_BLK());
                if (o_len > 28) zmm28 = _mm512_loadu_ps(l_dst + 28 * NK_DT_BLK());
                if (o_len > 29) zmm29 = _mm512_loadu_ps(l_dst + 29 * NK_DT_BLK());
                if (o_len > 30) zmm30 = _mm512_loadu_ps(l_dst + 30 * NK_DT_BLK());
            }
        }
        
        // src: Nk16n
        // flt: Km16k
        const float *kb_src = PICK_PARAM(const float*, param, SRC_IDX());
        const float *kb_flt = o_flt;
        int64_t kb          = param[K_IDX()];
        while (kb >= NK_DT_BLK()) {
            kb -= NK_DT_BLK();
            const float *k_src = kb_src;
            const float *k_flt = kb_flt;
            for (int64_t ic = 0; ic < NK_DT_BLK(); ++ic) {
                IC_COMPUTE_STEP(0);
                k_flt += 1;
                k_src += NK_DT_BLK();
            }
            kb_src += NK_DT_BLK() * NK_DT_BLK();
            kb_flt += flt_kb_stride;
        }
        if (kb > 0) {
            const float *k_src = kb_src;
            const float *k_flt = kb_flt;
            for (int64_t ic = 0; ic < NK_DT_BLK(); ++ic) {
                IC_COMPUTE_STEP(0);
                k_flt += 1;
                k_src += NK_DT_BLK();
            }
        }
        
        {
            float* l_dst = o_dst;
            if (hw_len > 0 * NK_DT_BLK()) {
                if (o_len > 0) _mm512_storeu_ps(l_dst + 0 * NK_DT_BLK(), zmm0);
                if (o_len > 1) _mm512_storeu_ps(l_dst + 1 * NK_DT_BLK(), zmm1);
                if (o_len > 2) _mm512_storeu_ps(l_dst + 2 * NK_DT_BLK(), zmm2);
                if (o_len > 3) _mm512_storeu_ps(l_dst + 3 * NK_DT_BLK(), zmm3);
                if (o_len > 4) _mm512_storeu_ps(l_dst + 4 * NK_DT_BLK(), zmm4);
                if (o_len > 5) _mm512_storeu_ps(l_dst + 5 * NK_DT_BLK(), zmm5);
                if (o_len > 6) _mm512_storeu_ps(l_dst + 6 * NK_DT_BLK(), zmm6);
                if (o_len > 7) _mm512_storeu_ps(l_dst + 7 * NK_DT_BLK(), zmm7);
                if (o_len > 8) _mm512_storeu_ps(l_dst + 8 * NK_DT_BLK(), zmm8);
                if (o_len > 9) _mm512_storeu_ps(l_dst + 9 * NK_DT_BLK(), zmm9);
                if (o_len > 10) _mm512_storeu_ps(l_dst + 10 * NK_DT_BLK(), zmm10);
                if (o_len > 11) _mm512_storeu_ps(l_dst + 11 * NK_DT_BLK(), zmm11);
                if (o_len > 12) _mm512_storeu_ps(l_dst + 12 * NK_DT_BLK(), zmm12);
                if (o_len > 13) _mm512_storeu_ps(l_dst + 13 * NK_DT_BLK(), zmm13);
                if (o_len > 14) _mm512_storeu_ps(l_dst + 14 * NK_DT_BLK(), zmm14);
                if (o_len > 15) _mm512_storeu_ps(l_dst + 15 * NK_DT_BLK(), zmm15);
                if (o_len > 16) _mm512_storeu_ps(l_dst + 16 * NK_DT_BLK(), zmm16);
                if (o_len > 17) _mm512_storeu_ps(l_dst + 17 * NK_DT_BLK(), zmm17);
                if (o_len > 18) _mm512_storeu_ps(l_dst + 18 * NK_DT_BLK(), zmm18);
                if (o_len > 19) _mm512_storeu_ps(l_dst + 19 * NK_DT_BLK(), zmm19);
                if (o_len > 20) _mm512_storeu_ps(l_dst + 20 * NK_DT_BLK(), zmm20);
                if (o_len > 21) _mm512_storeu_ps(l_dst + 21 * NK_DT_BLK(), zmm21);
                if (o_len > 22) _mm512_storeu_ps(l_dst + 22 * NK_DT_BLK(), zmm22);
                if (o_len > 23) _mm512_storeu_ps(l_dst + 23 * NK_DT_BLK(), zmm23);
                if (o_len > 24) _mm512_storeu_ps(l_dst + 24 * NK_DT_BLK(), zmm24);
                if (o_len > 25) _mm512_storeu_ps(l_dst + 25 * NK_DT_BLK(), zmm25);
                if (o_len > 26) _mm512_storeu_ps(l_dst + 26 * NK_DT_BLK(), zmm26);
                if (o_len > 27) _mm512_storeu_ps(l_dst + 27 * NK_DT_BLK(), zmm27);
                if (o_len > 28) _mm512_storeu_ps(l_dst + 28 * NK_DT_BLK(), zmm28);
                if (o_len > 29) _mm512_storeu_ps(l_dst + 29 * NK_DT_BLK(), zmm29);
                if (o_len > 30) _mm512_storeu_ps(l_dst + 30 * NK_DT_BLK(), zmm30);
            }
        }
        o_dst += o_len * NK_DT_BLK();
        o_flt += o_len * NK_DT_BLK();
        o -= o_len;
    } while (t > 0);
#undef IC_COMPUTE_STEP
}

}}};

#endif
