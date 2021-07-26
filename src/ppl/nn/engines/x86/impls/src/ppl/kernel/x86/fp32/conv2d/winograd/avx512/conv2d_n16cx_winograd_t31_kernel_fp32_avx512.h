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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_AVX512_CONV2D_N16CX_WINOGRAD_T31_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_AVX512_CONV2D_N16CX_WINOGRAD_T31_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/winograd/avx512/conv2d_n16cx_winograd_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template <int64_t t_len>
inline void conv2d_n16cx_winograd_t31_kernel_fp32_avx512_core(
    const int64_t *param)
{
    __asm__ __volatile__ (
        ".equ P_BYTES, 8\n"
        ".equ D_BYTES, 4\n"

        ".equ CH_DT_BLK, 16\n"

        ".equ SRC_IDX,      (0 * P_BYTES)\n"
        ".equ DST_IDX,      (1 * P_BYTES)\n"
        ".equ FLT_IDX,      (2 * P_BYTES)\n"
        ".equ TILES_IDX,    (3 * P_BYTES)\n"
        ".equ CHANNELS_IDX, (4 * P_BYTES)\n"
        ".equ SRC_TKB_STRIDE_IDX, (5 * P_BYTES)\n"
        ".equ DST_OCB_STRIDE_IDX, (6 * P_BYTES)\n"
        ".equ FLT_OCB_STRIDE_IDX, (7 * P_BYTES)\n"
        ".equ LOAD_DST_IDX,       (8 * P_BYTES)\n"

        ".equ T_LEN, %c[T_LEN]\n"

        "mov SRC_TKB_STRIDE_IDX(%[param]), %%r8\n"
        "mov LOAD_DST_IDX(%[param]), %%r12\n"
        "mov DST_IDX(%[param]), %%r13\n"
        "mov SRC_IDX(%[param]), %%r14\n"
        "mov TILES_IDX(%[param]), %%r15\n"
"1:\n" // label_init_session
        "test %%r12, %%r12\n"
        "jnz 2f\n" // label_load_dst
        ".if T_LEN > 0\n vpxord %%zmm0, %%zmm0, %%zmm0\n .endif\n"
        ".if T_LEN > 1\n vpxord %%zmm1, %%zmm1, %%zmm1\n .endif\n"
        ".if T_LEN > 2\n vpxord %%zmm2, %%zmm2, %%zmm2\n .endif\n"
        ".if T_LEN > 3\n vpxord %%zmm3, %%zmm3, %%zmm3\n .endif\n"
        ".if T_LEN > 4\n vpxord %%zmm4, %%zmm4, %%zmm4\n .endif\n"
        ".if T_LEN > 5\n vpxord %%zmm5, %%zmm5, %%zmm5\n .endif\n"
        ".if T_LEN > 6\n vpxord %%zmm6, %%zmm6, %%zmm6\n .endif\n"
        ".if T_LEN > 7\n vpxord %%zmm7, %%zmm7, %%zmm7\n .endif\n"
        ".if T_LEN > 8\n vpxord %%zmm8, %%zmm8, %%zmm8\n .endif\n"
        ".if T_LEN > 9\n vpxord %%zmm9, %%zmm9, %%zmm9\n .endif\n"
        ".if T_LEN > 10\n vpxord %%zmm10, %%zmm10, %%zmm10\n .endif\n"
        ".if T_LEN > 11\n vpxord %%zmm11, %%zmm11, %%zmm11\n .endif\n"
        ".if T_LEN > 12\n vpxord %%zmm12, %%zmm12, %%zmm12\n .endif\n"
        ".if T_LEN > 13\n vpxord %%zmm13, %%zmm13, %%zmm13\n .endif\n"
        ".if T_LEN > 14\n vpxord %%zmm14, %%zmm14, %%zmm14\n .endif\n"
        ".if T_LEN > 15\n vpxord %%zmm15, %%zmm15, %%zmm15\n .endif\n"
        ".if T_LEN > 16\n vpxord %%zmm16, %%zmm16, %%zmm16\n .endif\n"
        ".if T_LEN > 17\n vpxord %%zmm17, %%zmm17, %%zmm17\n .endif\n"
        ".if T_LEN > 18\n vpxord %%zmm18, %%zmm18, %%zmm18\n .endif\n"
        ".if T_LEN > 19\n vpxord %%zmm19, %%zmm19, %%zmm19\n .endif\n"
        ".if T_LEN > 20\n vpxord %%zmm20, %%zmm20, %%zmm20\n .endif\n"
        ".if T_LEN > 21\n vpxord %%zmm21, %%zmm21, %%zmm21\n .endif\n"
        ".if T_LEN > 22\n vpxord %%zmm22, %%zmm22, %%zmm22\n .endif\n"
        ".if T_LEN > 23\n vpxord %%zmm23, %%zmm23, %%zmm23\n .endif\n"
        ".if T_LEN > 24\n vpxord %%zmm24, %%zmm24, %%zmm24\n .endif\n"
        ".if T_LEN > 25\n vpxord %%zmm25, %%zmm25, %%zmm25\n .endif\n"
        ".if T_LEN > 26\n vpxord %%zmm26, %%zmm26, %%zmm26\n .endif\n"
        ".if T_LEN > 27\n vpxord %%zmm27, %%zmm27, %%zmm27\n .endif\n"
        ".if T_LEN > 28\n vpxord %%zmm28, %%zmm28, %%zmm28\n .endif\n"
        ".if T_LEN > 29\n vpxord %%zmm29, %%zmm29, %%zmm29\n .endif\n"
        ".if T_LEN > 30\n vpxord %%zmm30, %%zmm30, %%zmm30\n .endif\n"
        "jmp 3f\n"
"2:\n" // label_load_dst
        ".if T_LEN > 0\n vmovups (0 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm0\n .endif\n"
        ".if T_LEN > 1\n vmovups (1 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm1\n .endif\n"
        ".if T_LEN > 2\n vmovups (2 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm2\n .endif\n"
        ".if T_LEN > 3\n vmovups (3 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm3\n .endif\n"
        ".if T_LEN > 4\n vmovups (4 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm4\n .endif\n"
        ".if T_LEN > 5\n vmovups (5 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm5\n .endif\n"
        ".if T_LEN > 6\n vmovups (6 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm6\n .endif\n"
        ".if T_LEN > 7\n vmovups (7 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm7\n .endif\n"
        ".if T_LEN > 8\n vmovups (8 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm8\n .endif\n"
        ".if T_LEN > 9\n vmovups (9 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm9\n .endif\n"
        ".if T_LEN > 10\n vmovups (10 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm10\n .endif\n"
        ".if T_LEN > 11\n vmovups (11 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm11\n .endif\n"
        ".if T_LEN > 12\n vmovups (12 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm12\n .endif\n"
        ".if T_LEN > 13\n vmovups (13 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm13\n .endif\n"
        ".if T_LEN > 14\n vmovups (14 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm14\n .endif\n"
        ".if T_LEN > 15\n vmovups (15 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm15\n .endif\n"
        ".if T_LEN > 16\n vmovups (16 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm16\n .endif\n"
        ".if T_LEN > 17\n vmovups (17 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm17\n .endif\n"
        ".if T_LEN > 18\n vmovups (18 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm18\n .endif\n"
        ".if T_LEN > 19\n vmovups (19 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm19\n .endif\n"
        ".if T_LEN > 20\n vmovups (20 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm20\n .endif\n"
        ".if T_LEN > 21\n vmovups (21 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm21\n .endif\n"
        ".if T_LEN > 22\n vmovups (22 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm22\n .endif\n"
        ".if T_LEN > 23\n vmovups (23 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm23\n .endif\n"
        ".if T_LEN > 24\n vmovups (24 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm24\n .endif\n"
        ".if T_LEN > 25\n vmovups (25 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm25\n .endif\n"
        ".if T_LEN > 26\n vmovups (26 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm26\n .endif\n"
        ".if T_LEN > 27\n vmovups (27 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm27\n .endif\n"
        ".if T_LEN > 28\n vmovups (28 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm28\n .endif\n"
        ".if T_LEN > 29\n vmovups (29 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm29\n .endif\n"
        ".if T_LEN > 30\n vmovups (30 * CH_DT_BLK * D_BYTES)(%%r13), %%zmm30\n .endif\n"

"3:\n" // label_compute_session
        "mov %%r14, %%rax\n"
        "mov FLT_IDX(%[param]), %%rbx\n"
        "mov CHANNELS_IDX(%[param]), %%r10\n"
        "cmp $CH_DT_BLK, %%r10\n"
        "jl 6f\n" // label_ic_remain
"4:\n" // label_ic_body
        PPL_X86_INLINE_ASM_ALIGN()
        ".if T_LEN < 9\n"
        ".irp IC,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
        "vmovups (\\IC * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm31\n"
        ".if T_LEN > 7\n prefetcht0 ((\\IC * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * D_BYTES)(%%rbx)\n .endif\n"
        ".if T_LEN > 0\n vfmadd231ps ((\\IC + 0 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm0\n .endif\n"
        ".if T_LEN > 1\n vfmadd231ps ((\\IC + 1 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm1\n .endif\n"
        ".if T_LEN > 2\n vfmadd231ps ((\\IC + 2 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm2\n .endif\n"
        ".if T_LEN > 3\n vfmadd231ps ((\\IC + 3 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm3\n .endif\n"
        ".if T_LEN > 4\n vfmadd231ps ((\\IC + 4 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm4\n .endif\n"
        ".if T_LEN > 5\n vfmadd231ps ((\\IC + 5 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm5\n .endif\n"
        ".if T_LEN > 6\n vfmadd231ps ((\\IC + 6 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm6\n .endif\n"
        ".if T_LEN > 7\n vfmadd231ps ((\\IC + 7 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm7\n .endif\n"
        ".if T_LEN > 8\n vfmadd231ps ((\\IC + 8 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm8\n .endif\n"
        ".if T_LEN > 9\n vfmadd231ps ((\\IC + 9 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm9\n .endif\n"
        ".if T_LEN > 10\n vfmadd231ps ((\\IC + 10 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm10\n .endif\n"
        ".if T_LEN > 11\n vfmadd231ps ((\\IC + 11 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm11\n .endif\n"
        ".if T_LEN > 12\n vfmadd231ps ((\\IC + 12 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm12\n .endif\n"
        ".if T_LEN > 13\n vfmadd231ps ((\\IC + 13 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm13\n .endif\n"
        ".if T_LEN > 14\n vfmadd231ps ((\\IC + 14 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm14\n .endif\n"
        ".if T_LEN > 15\n vfmadd231ps ((\\IC + 15 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm15\n .endif\n"
        ".if T_LEN > 16\n vfmadd231ps ((\\IC + 16 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm16\n .endif\n"
        ".if T_LEN > 17\n vfmadd231ps ((\\IC + 17 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm17\n .endif\n"
        ".if T_LEN > 18\n vfmadd231ps ((\\IC + 18 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm18\n .endif\n"
        ".if T_LEN > 19\n vfmadd231ps ((\\IC + 19 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm19\n .endif\n"
        ".if T_LEN > 20\n vfmadd231ps ((\\IC + 20 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm20\n .endif\n"
        ".if T_LEN > 21\n vfmadd231ps ((\\IC + 21 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm21\n .endif\n"
        ".if T_LEN > 22\n vfmadd231ps ((\\IC + 22 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm22\n .endif\n"
        ".if T_LEN > 23\n vfmadd231ps ((\\IC + 23 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm23\n .endif\n"
        ".if T_LEN > 24\n vfmadd231ps ((\\IC + 24 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm24\n .endif\n"
        ".if T_LEN > 25\n vfmadd231ps ((\\IC + 25 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm25\n .endif\n"
        ".if T_LEN > 26\n vfmadd231ps ((\\IC + 26 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm26\n .endif\n"
        ".if T_LEN > 27\n vfmadd231ps ((\\IC + 27 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm27\n .endif\n"
        ".if T_LEN > 28\n vfmadd231ps ((\\IC + 28 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm28\n .endif\n"
        ".if T_LEN > 29\n vfmadd231ps ((\\IC + 29 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm29\n .endif\n"
        ".if T_LEN > 30\n vfmadd231ps ((\\IC + 30 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm30\n .endif\n"
        ".endr\n"
        "lea (CH_DT_BLK * CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        "lea (T_LEN * CH_DT_BLK * D_BYTES)(%%rax), %%rax\n"
        ".else\n" // .if T_LEN < 9
        "mov $CH_DT_BLK, %%r9\n"
"9:\n" // label_ic
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm31\n"
        "prefetcht0 ((0 * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * D_BYTES)(%%rbx)\n"
        "lea (CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        ".if T_LEN > 0\n vfmadd231ps ((0 + 0 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm0\n .endif\n"
        ".if T_LEN > 1\n vfmadd231ps ((0 + 1 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm1\n .endif\n"
        ".if T_LEN > 2\n vfmadd231ps ((0 + 2 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm2\n .endif\n"
        ".if T_LEN > 3\n vfmadd231ps ((0 + 3 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm3\n .endif\n"
        ".if T_LEN > 4\n vfmadd231ps ((0 + 4 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm4\n .endif\n"
        ".if T_LEN > 5\n vfmadd231ps ((0 + 5 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm5\n .endif\n"
        ".if T_LEN > 6\n vfmadd231ps ((0 + 6 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm6\n .endif\n"
        ".if T_LEN > 7\n vfmadd231ps ((0 + 7 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm7\n .endif\n"
        ".if T_LEN > 8\n vfmadd231ps ((0 + 8 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm8\n .endif\n"
        ".if T_LEN > 9\n vfmadd231ps ((0 + 9 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm9\n .endif\n"
        ".if T_LEN > 10\n vfmadd231ps ((0 + 10 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm10\n .endif\n"
        ".if T_LEN > 11\n vfmadd231ps ((0 + 11 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm11\n .endif\n"
        ".if T_LEN > 12\n vfmadd231ps ((0 + 12 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm12\n .endif\n"
        ".if T_LEN > 13\n vfmadd231ps ((0 + 13 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm13\n .endif\n"
        ".if T_LEN > 14\n vfmadd231ps ((0 + 14 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm14\n .endif\n"
        ".if T_LEN > 15\n vfmadd231ps ((0 + 15 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm15\n .endif\n"
        ".if T_LEN > 16\n vfmadd231ps ((0 + 16 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm16\n .endif\n"
        ".if T_LEN > 17\n vfmadd231ps ((0 + 17 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm17\n .endif\n"
        ".if T_LEN > 18\n vfmadd231ps ((0 + 18 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm18\n .endif\n"
        ".if T_LEN > 19\n vfmadd231ps ((0 + 19 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm19\n .endif\n"
        ".if T_LEN > 20\n vfmadd231ps ((0 + 20 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm20\n .endif\n"
        ".if T_LEN > 21\n vfmadd231ps ((0 + 21 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm21\n .endif\n"
        ".if T_LEN > 22\n vfmadd231ps ((0 + 22 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm22\n .endif\n"
        ".if T_LEN > 23\n vfmadd231ps ((0 + 23 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm23\n .endif\n"
        ".if T_LEN > 24\n vfmadd231ps ((0 + 24 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm24\n .endif\n"
        ".if T_LEN > 25\n vfmadd231ps ((0 + 25 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm25\n .endif\n"
        ".if T_LEN > 26\n vfmadd231ps ((0 + 26 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm26\n .endif\n"
        ".if T_LEN > 27\n vfmadd231ps ((0 + 27 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm27\n .endif\n"
        ".if T_LEN > 28\n vfmadd231ps ((0 + 28 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm28\n .endif\n"
        ".if T_LEN > 29\n vfmadd231ps ((0 + 29 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm29\n .endif\n"
        ".if T_LEN > 30\n vfmadd231ps ((0 + 30 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm30\n .endif\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%r9\n"
        "cmp $0, %%r9\n"
        "jne 9b\n" // label_ic
        "lea ((T_LEN - 1) * CH_DT_BLK * D_BYTES)(%%rax), %%rax\n"
        ".endif\n" // .if T_LEN < 9
        "sub $CH_DT_BLK, %%r10\n"
        "cmp $CH_DT_BLK, %%r10\n"
        "jge 4b\n"// label_ic_body
        "cmp $0, %%r10\n"
        "je 6f\n" // label_finalize_session
"5:\n" // label_ic_remain
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm31\n"
        "lea (CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        ".if T_LEN > 0\n vfmadd231ps ((0 + 0 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm0\n .endif\n"
        ".if T_LEN > 1\n vfmadd231ps ((0 + 1 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm1\n .endif\n"
        ".if T_LEN > 2\n vfmadd231ps ((0 + 2 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm2\n .endif\n"
        ".if T_LEN > 3\n vfmadd231ps ((0 + 3 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm3\n .endif\n"
        ".if T_LEN > 4\n vfmadd231ps ((0 + 4 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm4\n .endif\n"
        ".if T_LEN > 5\n vfmadd231ps ((0 + 5 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm5\n .endif\n"
        ".if T_LEN > 6\n vfmadd231ps ((0 + 6 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm6\n .endif\n"
        ".if T_LEN > 7\n vfmadd231ps ((0 + 7 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm7\n .endif\n"
        ".if T_LEN > 8\n vfmadd231ps ((0 + 8 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm8\n .endif\n"
        ".if T_LEN > 9\n vfmadd231ps ((0 + 9 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm9\n .endif\n"
        ".if T_LEN > 10\n vfmadd231ps ((0 + 10 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm10\n .endif\n"
        ".if T_LEN > 11\n vfmadd231ps ((0 + 11 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm11\n .endif\n"
        ".if T_LEN > 12\n vfmadd231ps ((0 + 12 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm12\n .endif\n"
        ".if T_LEN > 13\n vfmadd231ps ((0 + 13 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm13\n .endif\n"
        ".if T_LEN > 14\n vfmadd231ps ((0 + 14 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm14\n .endif\n"
        ".if T_LEN > 15\n vfmadd231ps ((0 + 15 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm15\n .endif\n"
        ".if T_LEN > 16\n vfmadd231ps ((0 + 16 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm16\n .endif\n"
        ".if T_LEN > 17\n vfmadd231ps ((0 + 17 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm17\n .endif\n"
        ".if T_LEN > 18\n vfmadd231ps ((0 + 18 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm18\n .endif\n"
        ".if T_LEN > 19\n vfmadd231ps ((0 + 19 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm19\n .endif\n"
        ".if T_LEN > 20\n vfmadd231ps ((0 + 20 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm20\n .endif\n"
        ".if T_LEN > 21\n vfmadd231ps ((0 + 21 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm21\n .endif\n"
        ".if T_LEN > 22\n vfmadd231ps ((0 + 22 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm22\n .endif\n"
        ".if T_LEN > 23\n vfmadd231ps ((0 + 23 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm23\n .endif\n"
        ".if T_LEN > 24\n vfmadd231ps ((0 + 24 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm24\n .endif\n"
        ".if T_LEN > 25\n vfmadd231ps ((0 + 25 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm25\n .endif\n"
        ".if T_LEN > 26\n vfmadd231ps ((0 + 26 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm26\n .endif\n"
        ".if T_LEN > 27\n vfmadd231ps ((0 + 27 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm27\n .endif\n"
        ".if T_LEN > 28\n vfmadd231ps ((0 + 28 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm28\n .endif\n"
        ".if T_LEN > 29\n vfmadd231ps ((0 + 29 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm29\n .endif\n"
        ".if T_LEN > 30\n vfmadd231ps ((0 + 30 * CH_DT_BLK) * D_BYTES)(%%rax)%{1to16}, %%zmm31, %%zmm30\n .endif\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%r10\n"
        "cmp $0, %%r10\n"
        "jne 5b\n" // label_ic_remain

"6:\n" // label_finalize_session
        ".if T_LEN > 0\n vmovups %%zmm0, (0 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 1\n vmovups %%zmm1, (1 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 2\n vmovups %%zmm2, (2 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 3\n vmovups %%zmm3, (3 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 4\n vmovups %%zmm4, (4 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 5\n vmovups %%zmm5, (5 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 6\n vmovups %%zmm6, (6 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 7\n vmovups %%zmm7, (7 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 8\n vmovups %%zmm8, (8 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 9\n vmovups %%zmm9, (9 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 10\n vmovups %%zmm10, (10 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 11\n vmovups %%zmm11, (11 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 12\n vmovups %%zmm12, (12 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 13\n vmovups %%zmm13, (13 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 14\n vmovups %%zmm14, (14 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 15\n vmovups %%zmm15, (15 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 16\n vmovups %%zmm16, (16 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 17\n vmovups %%zmm17, (17 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 18\n vmovups %%zmm18, (18 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 19\n vmovups %%zmm19, (19 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 20\n vmovups %%zmm20, (20 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 21\n vmovups %%zmm21, (21 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 22\n vmovups %%zmm22, (22 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 23\n vmovups %%zmm23, (23 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 24\n vmovups %%zmm24, (24 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 25\n vmovups %%zmm25, (25 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 26\n vmovups %%zmm26, (26 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 27\n vmovups %%zmm27, (27 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 28\n vmovups %%zmm28, (28 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 29\n vmovups %%zmm29, (29 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        ".if T_LEN > 30\n vmovups %%zmm30, (30 * CH_DT_BLK * D_BYTES)(%%r13)\n .endif\n"
        "sub $T_LEN, %%r15\n"
        "cmp $0, %%r15\n"
        "lea (T_LEN * CH_DT_BLK * D_BYTES)(%%r13), %%r13\n"
        "lea (%%r14, %%r8, D_BYTES), %%r14\n"
        "jne 1b\n" // label_init_session
        :
        :
        [param] "r" (param),
        [T_LEN] "i" (t_len)
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

template <int64_t oc_len, int64_t t_len>
void conv2d_n16cx_winograd_t31_kernel_fp32_avx512(
    const int64_t *param)
{

#ifdef PPL_USE_X86_INLINE_ASM
    if (oc_len == 1 * CH_DT_BLK()) {
        conv2d_n16cx_winograd_t31_kernel_fp32_avx512_core<t_len>(param);
        return;
    }
#endif

#define IC_COMPUTE_STEP(IC) do {\
    zmm31 = _mm512_loadu_ps(ic_flt + (IC) * CH_DT_BLK());\
    if (t_len > 0) zmm0 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 0 * CH_DT_BLK()]), zmm31, zmm0);\
    if (t_len > 1) zmm1 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 1 * CH_DT_BLK()]), zmm31, zmm1);\
    if (t_len > 2) zmm2 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 2 * CH_DT_BLK()]), zmm31, zmm2);\
    if (t_len > 3) zmm3 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 3 * CH_DT_BLK()]), zmm31, zmm3);\
    if (t_len > 4) zmm4 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 4 * CH_DT_BLK()]), zmm31, zmm4);\
    if (t_len > 5) zmm5 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 5 * CH_DT_BLK()]), zmm31, zmm5);\
    if (t_len > 6) zmm6 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 6 * CH_DT_BLK()]), zmm31, zmm6);\
    if (t_len > 7) zmm7 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 7 * CH_DT_BLK()]), zmm31, zmm7);\
    if (t_len > 8) zmm8 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 8 * CH_DT_BLK()]), zmm31, zmm8);\
    if (t_len > 9) zmm9 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 9 * CH_DT_BLK()]), zmm31, zmm9);\
    if (t_len > 10) zmm10 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 10 * CH_DT_BLK()]), zmm31, zmm10);\
    if (t_len > 11) zmm11 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 11 * CH_DT_BLK()]), zmm31, zmm11);\
    if (t_len > 12) zmm12 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 12 * CH_DT_BLK()]), zmm31, zmm12);\
    if (t_len > 13) zmm13 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 13 * CH_DT_BLK()]), zmm31, zmm13);\
    if (t_len > 14) zmm14 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 14 * CH_DT_BLK()]), zmm31, zmm14);\
    if (t_len > 15) zmm15 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 15 * CH_DT_BLK()]), zmm31, zmm15);\
    if (t_len > 16) zmm16 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 16 * CH_DT_BLK()]), zmm31, zmm16);\
    if (t_len > 17) zmm17 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 17 * CH_DT_BLK()]), zmm31, zmm17);\
    if (t_len > 18) zmm18 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 18 * CH_DT_BLK()]), zmm31, zmm18);\
    if (t_len > 19) zmm19 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 19 * CH_DT_BLK()]), zmm31, zmm19);\
    if (t_len > 20) zmm20 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 20 * CH_DT_BLK()]), zmm31, zmm20);\
    if (t_len > 21) zmm21 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 21 * CH_DT_BLK()]), zmm31, zmm21);\
    if (t_len > 22) zmm22 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 22 * CH_DT_BLK()]), zmm31, zmm22);\
    if (t_len > 23) zmm23 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 23 * CH_DT_BLK()]), zmm31, zmm23);\
    if (t_len > 24) zmm24 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 24 * CH_DT_BLK()]), zmm31, zmm24);\
    if (t_len > 25) zmm25 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 25 * CH_DT_BLK()]), zmm31, zmm25);\
    if (t_len > 26) zmm26 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 26 * CH_DT_BLK()]), zmm31, zmm26);\
    if (t_len > 27) zmm27 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 27 * CH_DT_BLK()]), zmm31, zmm27);\
    if (t_len > 28) zmm28 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 28 * CH_DT_BLK()]), zmm31, zmm28);\
    if (t_len > 29) zmm29 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 29 * CH_DT_BLK()]), zmm31, zmm29);\
    if (t_len > 30) zmm30 = _mm512_fmadd_ps(_mm512_set1_ps(ic_src[(IC) + 30 * CH_DT_BLK()]), zmm31, zmm30);\
} while (0)

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    const int64_t src_tkb_stride = param[SRC_TKB_STRIDE_IDX()];
    const int64_t load_dst       = param[LOAD_DST_IDX()];

    const float *t_src = PICK_PARAM(const float*, param, SRC_IDX());
    float *t_dst       = PICK_PARAM(float*, param, DST_IDX());
    int64_t t          = param[TILES_IDX()];
    do {
        if (load_dst) {
            const float *l_dst = t_dst;
            if (oc_len > 0 * CH_DT_BLK()) {
                if (t_len > 0) zmm0 = _mm512_loadu_ps(l_dst + 0 * CH_DT_BLK());
                if (t_len > 1) zmm1 = _mm512_loadu_ps(l_dst + 1 * CH_DT_BLK());
                if (t_len > 2) zmm2 = _mm512_loadu_ps(l_dst + 2 * CH_DT_BLK());
                if (t_len > 3) zmm3 = _mm512_loadu_ps(l_dst + 3 * CH_DT_BLK());
                if (t_len > 4) zmm4 = _mm512_loadu_ps(l_dst + 4 * CH_DT_BLK());
                if (t_len > 5) zmm5 = _mm512_loadu_ps(l_dst + 5 * CH_DT_BLK());
                if (t_len > 6) zmm6 = _mm512_loadu_ps(l_dst + 6 * CH_DT_BLK());
                if (t_len > 7) zmm7 = _mm512_loadu_ps(l_dst + 7 * CH_DT_BLK());
                if (t_len > 8) zmm8 = _mm512_loadu_ps(l_dst + 8 * CH_DT_BLK());
                if (t_len > 9) zmm9 = _mm512_loadu_ps(l_dst + 9 * CH_DT_BLK());
                if (t_len > 10) zmm10 = _mm512_loadu_ps(l_dst + 10 * CH_DT_BLK());
                if (t_len > 11) zmm11 = _mm512_loadu_ps(l_dst + 11 * CH_DT_BLK());
                if (t_len > 12) zmm12 = _mm512_loadu_ps(l_dst + 12 * CH_DT_BLK());
                if (t_len > 13) zmm13 = _mm512_loadu_ps(l_dst + 13 * CH_DT_BLK());
                if (t_len > 14) zmm14 = _mm512_loadu_ps(l_dst + 14 * CH_DT_BLK());
                if (t_len > 15) zmm15 = _mm512_loadu_ps(l_dst + 15 * CH_DT_BLK());
                if (t_len > 16) zmm16 = _mm512_loadu_ps(l_dst + 16 * CH_DT_BLK());
                if (t_len > 17) zmm17 = _mm512_loadu_ps(l_dst + 17 * CH_DT_BLK());
                if (t_len > 18) zmm18 = _mm512_loadu_ps(l_dst + 18 * CH_DT_BLK());
                if (t_len > 19) zmm19 = _mm512_loadu_ps(l_dst + 19 * CH_DT_BLK());
                if (t_len > 20) zmm20 = _mm512_loadu_ps(l_dst + 20 * CH_DT_BLK());
                if (t_len > 21) zmm21 = _mm512_loadu_ps(l_dst + 21 * CH_DT_BLK());
                if (t_len > 22) zmm22 = _mm512_loadu_ps(l_dst + 22 * CH_DT_BLK());
                if (t_len > 23) zmm23 = _mm512_loadu_ps(l_dst + 23 * CH_DT_BLK());
                if (t_len > 24) zmm24 = _mm512_loadu_ps(l_dst + 24 * CH_DT_BLK());
                if (t_len > 25) zmm25 = _mm512_loadu_ps(l_dst + 25 * CH_DT_BLK());
                if (t_len > 26) zmm26 = _mm512_loadu_ps(l_dst + 26 * CH_DT_BLK());
                if (t_len > 27) zmm27 = _mm512_loadu_ps(l_dst + 27 * CH_DT_BLK());
                if (t_len > 28) zmm28 = _mm512_loadu_ps(l_dst + 28 * CH_DT_BLK());
                if (t_len > 29) zmm29 = _mm512_loadu_ps(l_dst + 29 * CH_DT_BLK());
                if (t_len > 30) zmm30 = _mm512_loadu_ps(l_dst + 30 * CH_DT_BLK());
            }
        } else {
            if (oc_len > 0 * CH_DT_BLK()) {
                if (t_len > 0) zmm0 = _mm512_setzero_ps();
                if (t_len > 1) zmm1 = _mm512_setzero_ps();
                if (t_len > 2) zmm2 = _mm512_setzero_ps();
                if (t_len > 3) zmm3 = _mm512_setzero_ps();
                if (t_len > 4) zmm4 = _mm512_setzero_ps();
                if (t_len > 5) zmm5 = _mm512_setzero_ps();
                if (t_len > 6) zmm6 = _mm512_setzero_ps();
                if (t_len > 7) zmm7 = _mm512_setzero_ps();
                if (t_len > 8) zmm8 = _mm512_setzero_ps();
                if (t_len > 9) zmm9 = _mm512_setzero_ps();
                if (t_len > 10) zmm10 = _mm512_setzero_ps();
                if (t_len > 11) zmm11 = _mm512_setzero_ps();
                if (t_len > 12) zmm12 = _mm512_setzero_ps();
                if (t_len > 13) zmm13 = _mm512_setzero_ps();
                if (t_len > 14) zmm14 = _mm512_setzero_ps();
                if (t_len > 15) zmm15 = _mm512_setzero_ps();
                if (t_len > 16) zmm16 = _mm512_setzero_ps();
                if (t_len > 17) zmm17 = _mm512_setzero_ps();
                if (t_len > 18) zmm18 = _mm512_setzero_ps();
                if (t_len > 19) zmm19 = _mm512_setzero_ps();
                if (t_len > 20) zmm20 = _mm512_setzero_ps();
                if (t_len > 21) zmm21 = _mm512_setzero_ps();
                if (t_len > 22) zmm22 = _mm512_setzero_ps();
                if (t_len > 23) zmm23 = _mm512_setzero_ps();
                if (t_len > 24) zmm24 = _mm512_setzero_ps();
                if (t_len > 25) zmm25 = _mm512_setzero_ps();
                if (t_len > 26) zmm26 = _mm512_setzero_ps();
                if (t_len > 27) zmm27 = _mm512_setzero_ps();
                if (t_len > 28) zmm28 = _mm512_setzero_ps();
                if (t_len > 29) zmm29 = _mm512_setzero_ps();
                if (t_len > 30) zmm30 = _mm512_setzero_ps();
            }
        }
        
        const float *icb_src = t_src;
        const float *icb_flt = PICK_PARAM(const float*, param, FLT_IDX());
        int64_t icb          = param[CHANNELS_IDX()];
        while (icb >= CH_DT_BLK()) {
            icb -= CH_DT_BLK();
            const float *ic_src = icb_src;
            const float *ic_flt = icb_flt;
            for (int64_t ic = 0; ic < CH_DT_BLK(); ++ic) {
                IC_COMPUTE_STEP(0);
                ic_src += 1;
                ic_flt += CH_DT_BLK();
            }
            icb_flt += CH_DT_BLK() * CH_DT_BLK();
            icb_src += t_len * CH_DT_BLK();
        }
        if (icb > 0) {
            const float *ic_src = icb_src;
            const float *ic_flt = icb_flt;
            for (int64_t ic = 0; ic < icb; ++ic) {
                IC_COMPUTE_STEP(0);
                ic_src += 1;
                ic_flt += CH_DT_BLK();
            }
        }
        
        {
            float* l_dst = t_dst;
            if (oc_len > 0 * CH_DT_BLK()) {
                if (t_len > 0) _mm512_storeu_ps(l_dst + 0 * CH_DT_BLK(), zmm0);
                if (t_len > 1) _mm512_storeu_ps(l_dst + 1 * CH_DT_BLK(), zmm1);
                if (t_len > 2) _mm512_storeu_ps(l_dst + 2 * CH_DT_BLK(), zmm2);
                if (t_len > 3) _mm512_storeu_ps(l_dst + 3 * CH_DT_BLK(), zmm3);
                if (t_len > 4) _mm512_storeu_ps(l_dst + 4 * CH_DT_BLK(), zmm4);
                if (t_len > 5) _mm512_storeu_ps(l_dst + 5 * CH_DT_BLK(), zmm5);
                if (t_len > 6) _mm512_storeu_ps(l_dst + 6 * CH_DT_BLK(), zmm6);
                if (t_len > 7) _mm512_storeu_ps(l_dst + 7 * CH_DT_BLK(), zmm7);
                if (t_len > 8) _mm512_storeu_ps(l_dst + 8 * CH_DT_BLK(), zmm8);
                if (t_len > 9) _mm512_storeu_ps(l_dst + 9 * CH_DT_BLK(), zmm9);
                if (t_len > 10) _mm512_storeu_ps(l_dst + 10 * CH_DT_BLK(), zmm10);
                if (t_len > 11) _mm512_storeu_ps(l_dst + 11 * CH_DT_BLK(), zmm11);
                if (t_len > 12) _mm512_storeu_ps(l_dst + 12 * CH_DT_BLK(), zmm12);
                if (t_len > 13) _mm512_storeu_ps(l_dst + 13 * CH_DT_BLK(), zmm13);
                if (t_len > 14) _mm512_storeu_ps(l_dst + 14 * CH_DT_BLK(), zmm14);
                if (t_len > 15) _mm512_storeu_ps(l_dst + 15 * CH_DT_BLK(), zmm15);
                if (t_len > 16) _mm512_storeu_ps(l_dst + 16 * CH_DT_BLK(), zmm16);
                if (t_len > 17) _mm512_storeu_ps(l_dst + 17 * CH_DT_BLK(), zmm17);
                if (t_len > 18) _mm512_storeu_ps(l_dst + 18 * CH_DT_BLK(), zmm18);
                if (t_len > 19) _mm512_storeu_ps(l_dst + 19 * CH_DT_BLK(), zmm19);
                if (t_len > 20) _mm512_storeu_ps(l_dst + 20 * CH_DT_BLK(), zmm20);
                if (t_len > 21) _mm512_storeu_ps(l_dst + 21 * CH_DT_BLK(), zmm21);
                if (t_len > 22) _mm512_storeu_ps(l_dst + 22 * CH_DT_BLK(), zmm22);
                if (t_len > 23) _mm512_storeu_ps(l_dst + 23 * CH_DT_BLK(), zmm23);
                if (t_len > 24) _mm512_storeu_ps(l_dst + 24 * CH_DT_BLK(), zmm24);
                if (t_len > 25) _mm512_storeu_ps(l_dst + 25 * CH_DT_BLK(), zmm25);
                if (t_len > 26) _mm512_storeu_ps(l_dst + 26 * CH_DT_BLK(), zmm26);
                if (t_len > 27) _mm512_storeu_ps(l_dst + 27 * CH_DT_BLK(), zmm27);
                if (t_len > 28) _mm512_storeu_ps(l_dst + 28 * CH_DT_BLK(), zmm28);
                if (t_len > 29) _mm512_storeu_ps(l_dst + 29 * CH_DT_BLK(), zmm29);
                if (t_len > 30) _mm512_storeu_ps(l_dst + 30 * CH_DT_BLK(), zmm30);
            }
        }
        t_src += src_tkb_stride;
        t_dst += t_len * CH_DT_BLK();
        t -= t_len;
    } while (t > 0);
#undef IC_COMPUTE_STEP
}

}}};

#endif
