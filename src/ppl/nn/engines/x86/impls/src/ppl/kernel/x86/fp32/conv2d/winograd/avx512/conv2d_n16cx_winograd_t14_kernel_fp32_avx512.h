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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_AVX512_CONV2D_N16CX_WINOGRAD_T14_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_AVX512_CONV2D_N16CX_WINOGRAD_T14_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/winograd/avx512/conv2d_n16cx_winograd_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template <int64_t t_len>
void conv2d_n16cx_winograd_t14_kernel_fp32_avx512_core(
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
        "mov FLT_OCB_STRIDE_IDX(%[param]), %%r9\n"
        "mov DST_OCB_STRIDE_IDX(%[param]), %%r11\n"
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
        ".if T_LEN > 0\n vpxord %%zmm14, %%zmm14, %%zmm14\n .endif\n"
        ".if T_LEN > 1\n vpxord %%zmm15, %%zmm15, %%zmm15\n .endif\n"
        ".if T_LEN > 2\n vpxord %%zmm16, %%zmm16, %%zmm16\n .endif\n"
        ".if T_LEN > 3\n vpxord %%zmm17, %%zmm17, %%zmm17\n .endif\n"
        ".if T_LEN > 4\n vpxord %%zmm18, %%zmm18, %%zmm18\n .endif\n"
        ".if T_LEN > 5\n vpxord %%zmm19, %%zmm19, %%zmm19\n .endif\n"
        ".if T_LEN > 6\n vpxord %%zmm20, %%zmm20, %%zmm20\n .endif\n"
        ".if T_LEN > 7\n vpxord %%zmm21, %%zmm21, %%zmm21\n .endif\n"
        ".if T_LEN > 8\n vpxord %%zmm22, %%zmm22, %%zmm22\n .endif\n"
        ".if T_LEN > 9\n vpxord %%zmm23, %%zmm23, %%zmm23\n .endif\n"
        ".if T_LEN > 10\n vpxord %%zmm24, %%zmm24, %%zmm24\n .endif\n"
        ".if T_LEN > 11\n vpxord %%zmm25, %%zmm25, %%zmm25\n .endif\n"
        ".if T_LEN > 12\n vpxord %%zmm26, %%zmm26, %%zmm26\n .endif\n"
        ".if T_LEN > 13\n vpxord %%zmm27, %%zmm27, %%zmm27\n .endif\n"
        "jmp 3f\n" // label_compute_session
"2:\n" // label_load_dst
        "lea (%%r13, %%r11, D_BYTES), %%r10\n"
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
        ".if T_LEN > 0\n vmovups (0 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm14\n .endif\n"
        ".if T_LEN > 1\n vmovups (1 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm15\n .endif\n"
        ".if T_LEN > 2\n vmovups (2 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm16\n .endif\n"
        ".if T_LEN > 3\n vmovups (3 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm17\n .endif\n"
        ".if T_LEN > 4\n vmovups (4 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm18\n .endif\n"
        ".if T_LEN > 5\n vmovups (5 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm19\n .endif\n"
        ".if T_LEN > 6\n vmovups (6 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm20\n .endif\n"
        ".if T_LEN > 7\n vmovups (7 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm21\n .endif\n"
        ".if T_LEN > 8\n vmovups (8 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm22\n .endif\n"
        ".if T_LEN > 9\n vmovups (9 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm23\n .endif\n"
        ".if T_LEN > 10\n vmovups (10 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm24\n .endif\n"
        ".if T_LEN > 11\n vmovups (11 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm25\n .endif\n"
        ".if T_LEN > 12\n vmovups (12 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm26\n .endif\n"
        ".if T_LEN > 13\n vmovups (13 * CH_DT_BLK * D_BYTES)(%%r10), %%zmm27\n .endif\n"

"3:\n" // label_compute_session
        "mov %%r14, %%rax\n"
        "mov FLT_IDX(%[param]), %%rbx\n"
        "mov CHANNELS_IDX(%[param]), %%r10\n"
        "lea (%%rbx, %%r9, D_BYTES), %%rcx\n"
        "cmp $CH_DT_BLK, %%r10\n"
        "jl 5f\n" // label_ic_remain
"4:\n" // label_ic_body
        ".align 16\n"
        ".if T_LEN < 6\n"
        ".irp IC,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
        "vmovups (\\IC * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm28\n"
        "vmovups (\\IC * CH_DT_BLK * D_BYTES)(%%rcx), %%zmm29\n"
        "prefetcht0 ((\\IC * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * D_BYTES)(%%rbx)\n"
        "prefetcht0 ((\\IC * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * D_BYTES)(%%rcx)\n"
        ".if T_LEN > 0\n"
        "vbroadcastss ((\\IC + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm0\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm14\n"
        ".endif\n"
        ".if T_LEN > 1\n"
        "vbroadcastss ((\\IC + 1 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm1\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm15\n"
        ".endif\n"
        ".if T_LEN > 2\n"
        "vbroadcastss ((\\IC + 2 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm2\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm16\n"
        ".endif\n"
        ".if T_LEN > 3\n"
        "vbroadcastss ((\\IC + 3 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm3\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm17\n"
        ".endif\n"
        ".if T_LEN > 4\n"
        "vbroadcastss ((\\IC + 4 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm4\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm18\n"
        ".endif\n"
        ".if T_LEN > 5\n"
        "vbroadcastss ((\\IC + 5 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm5\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm19\n"
        ".endif\n"
        ".if T_LEN > 6\n"
        "vbroadcastss ((\\IC + 6 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm6\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm20\n"
        ".endif\n"
        ".if T_LEN > 7\n"
        "vbroadcastss ((\\IC + 7 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm7\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm21\n"
        ".endif\n"
        ".if T_LEN > 8\n"
        "vbroadcastss ((\\IC + 8 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm8\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm22\n"
        ".endif\n"
        ".if T_LEN > 9\n"
        "vbroadcastss ((\\IC + 9 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm9\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm23\n"
        ".endif\n"
        ".if T_LEN > 10\n"
        "vbroadcastss ((\\IC + 10 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm10\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm24\n"
        ".endif\n"
        ".if T_LEN > 11\n"
        "vbroadcastss ((\\IC + 11 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm11\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm25\n"
        ".endif\n"
        ".if T_LEN > 12\n"
        "vbroadcastss ((\\IC + 12 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm12\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm26\n"
        ".endif\n"
        ".if T_LEN > 13\n"
        "vbroadcastss ((\\IC + 13 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm13\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm27\n"
        ".endif\n"
        ".endr\n"
        "lea (CH_DT_BLK * CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        "lea (CH_DT_BLK * CH_DT_BLK * D_BYTES)(%%rcx), %%rcx\n"
        "lea (T_LEN * CH_DT_BLK * D_BYTES)(%%rax), %%rax\n"
        ".else\n" // .if T_LEN < 6
        "mov $CH_DT_BLK, %%rsi\n"
"9:\n" // label_ic
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm28\n"
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rcx), %%zmm29\n"
        "prefetcht0 ((0 * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * D_BYTES)(%%rbx)\n"
        "prefetcht0 ((0 * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * D_BYTES)(%%rcx)\n"
        "lea (CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        "lea (CH_DT_BLK * D_BYTES)(%%rcx), %%rcx\n"
        ".if T_LEN > 0\n"
        "vbroadcastss ((0 + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm0\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm14\n"
        ".endif\n"
        ".if T_LEN > 1\n"
        "vbroadcastss ((0 + 1 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm1\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm15\n"
        ".endif\n"
        ".if T_LEN > 2\n"
        "vbroadcastss ((0 + 2 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm2\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm16\n"
        ".endif\n"
        ".if T_LEN > 3\n"
        "vbroadcastss ((0 + 3 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm3\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm17\n"
        ".endif\n"
        ".if T_LEN > 4\n"
        "vbroadcastss ((0 + 4 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm4\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm18\n"
        ".endif\n"
        ".if T_LEN > 5\n"
        "vbroadcastss ((0 + 5 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm5\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm19\n"
        ".endif\n"
        ".if T_LEN > 6\n"
        "vbroadcastss ((0 + 6 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm6\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm20\n"
        ".endif\n"
        ".if T_LEN > 7\n"
        "vbroadcastss ((0 + 7 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm7\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm21\n"
        ".endif\n"
        ".if T_LEN > 8\n"
        "vbroadcastss ((0 + 8 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm8\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm22\n"
        ".endif\n"
        ".if T_LEN > 9\n"
        "vbroadcastss ((0 + 9 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm9\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm23\n"
        ".endif\n"
        ".if T_LEN > 10\n"
        "vbroadcastss ((0 + 10 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm10\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm24\n"
        ".endif\n"
        ".if T_LEN > 11\n"
        "vbroadcastss ((0 + 11 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm11\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm25\n"
        ".endif\n"
        ".if T_LEN > 12\n"
        "vbroadcastss ((0 + 12 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm12\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm26\n"
        ".endif\n"
        ".if T_LEN > 13\n"
        "vbroadcastss ((0 + 13 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm13\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm27\n"
        ".endif\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%rsi\n"
        "cmp $0, %%rsi\n"
        "jne 9b\n" // label_ic
        "lea ((T_LEN - 1) * CH_DT_BLK * D_BYTES)(%%rax), %%rax\n"
        ".endif\n"
        "sub $CH_DT_BLK, %%r10\n"
        "cmp $CH_DT_BLK, %%r10\n"
        "jge 4b\n" // label_ic_body
        "cmp $0, %%r10\n"
        "je 6f\n" // label_finalize_session
"5:\n" // label_ic_remain
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rbx), %%zmm28\n"
        "vmovups (0 * CH_DT_BLK * D_BYTES)(%%rcx), %%zmm29\n"
        ".if T_LEN > 0\n"
        "vbroadcastss ((0 + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm0\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm14\n"
        ".endif\n"
        ".if T_LEN > 1\n"
        "vbroadcastss ((0 + 1 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm1\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm15\n"
        ".endif\n"
        ".if T_LEN > 2\n"
        "vbroadcastss ((0 + 2 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm2\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm16\n"
        ".endif\n"
        ".if T_LEN > 3\n"
        "vbroadcastss ((0 + 3 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm3\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm17\n"
        ".endif\n"
        ".if T_LEN > 4\n"
        "vbroadcastss ((0 + 4 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm4\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm18\n"
        ".endif\n"
        ".if T_LEN > 5\n"
        "vbroadcastss ((0 + 5 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm5\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm19\n"
        ".endif\n"
        ".if T_LEN > 6\n"
        "vbroadcastss ((0 + 6 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm6\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm20\n"
        ".endif\n"
        ".if T_LEN > 7\n"
        "vbroadcastss ((0 + 7 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm7\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm21\n"
        ".endif\n"
        ".if T_LEN > 8\n"
        "vbroadcastss ((0 + 8 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm8\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm22\n"
        ".endif\n"
        ".if T_LEN > 9\n"
        "vbroadcastss ((0 + 9 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm9\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm23\n"
        ".endif\n"
        ".if T_LEN > 10\n"
        "vbroadcastss ((0 + 10 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm10\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm24\n"
        ".endif\n"
        ".if T_LEN > 11\n"
        "vbroadcastss ((0 + 11 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm11\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm25\n"
        ".endif\n"
        ".if T_LEN > 12\n"
        "vbroadcastss ((0 + 12 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm30\n"
        "vfmadd231ps %%zmm28, %%zmm30, %%zmm12\n"
        "vfmadd231ps %%zmm29, %%zmm30, %%zmm26\n"
        ".endif\n"
        ".if T_LEN > 13\n"
        "vbroadcastss ((0 + 13 * CH_DT_BLK) * D_BYTES)(%%rax), %%zmm31\n"
        "vfmadd231ps %%zmm28, %%zmm31, %%zmm13\n"
        "vfmadd231ps %%zmm29, %%zmm31, %%zmm27\n"
        ".endif\n"
        "lea (CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n"
        "lea (CH_DT_BLK * D_BYTES)(%%rcx), %%rcx\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%r10\n"
        "cmp $0, %%r10\n"
        "jne 5b\n" // label_ic_remain

"6:\n" // label_finalize_session
        "lea (%%r13, %%r11, D_BYTES), %%r10\n"
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
        ".if T_LEN > 0\n vmovups %%zmm14, (0 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 1\n vmovups %%zmm15, (1 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 2\n vmovups %%zmm16, (2 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 3\n vmovups %%zmm17, (3 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 4\n vmovups %%zmm18, (4 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 5\n vmovups %%zmm19, (5 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 6\n vmovups %%zmm20, (6 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 7\n vmovups %%zmm21, (7 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 8\n vmovups %%zmm22, (8 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 9\n vmovups %%zmm23, (9 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 10\n vmovups %%zmm24, (10 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 11\n vmovups %%zmm25, (11 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 12\n vmovups %%zmm26, (12 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
        ".if T_LEN > 13\n vmovups %%zmm27, (13 * CH_DT_BLK * D_BYTES)(%%r10)\n .endif\n"
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
        "rax", "rbx", "rcx", "rdx", "rsi",
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
void conv2d_n16cx_winograd_t14_kernel_fp32_avx512(
    const int64_t *param)
{

#ifdef PPL_USE_X86_INLINE_ASM
    if (oc_len == 2 * CH_DT_BLK() && t_len > 3) {
        conv2d_n16cx_winograd_t14_kernel_fp32_avx512_core<t_len>(param);
        return;
    }
#endif

#define IC_COMPUTE_STEP(IC) do {\
    if (oc_len > 0 * CH_DT_BLK()) zmm28 = _mm512_loadu_ps(ic_flt + 0 * flt_ocb_stride + (IC) * CH_DT_BLK());\
    if (oc_len > 1 * CH_DT_BLK()) zmm29 = _mm512_loadu_ps(ic_flt + 1 * flt_ocb_stride + (IC) * CH_DT_BLK());\
    if (t_len > 12) {\
        if (oc_len > 0 * CH_DT_BLK()) _mm_prefetch((const char*)(ic_flt + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK()), _MM_HINT_T0);\
        if (oc_len > 1 * CH_DT_BLK()) _mm_prefetch((const char*)(ic_flt + 1 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK()), _MM_HINT_T0);\
    }\
    if (t_len > 0) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 0 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm0  = _mm512_fmadd_ps(zmm28, zmm30, zmm0);\
        if (oc_len > 1 * CH_DT_BLK()) zmm14 = _mm512_fmadd_ps(zmm29, zmm30, zmm14);\
    }\
    if (t_len > 1) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 1 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm1  = _mm512_fmadd_ps(zmm28, zmm31, zmm1);\
        if (oc_len > 1 * CH_DT_BLK()) zmm15 = _mm512_fmadd_ps(zmm29, zmm31, zmm15);\
    }\
    if (t_len > 2) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 2 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm2  = _mm512_fmadd_ps(zmm28, zmm30, zmm2);\
        if (oc_len > 1 * CH_DT_BLK()) zmm16 = _mm512_fmadd_ps(zmm29, zmm30, zmm16);\
    }\
    if (t_len > 3) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 3 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm3  = _mm512_fmadd_ps(zmm28, zmm31, zmm3);\
        if (oc_len > 1 * CH_DT_BLK()) zmm17 = _mm512_fmadd_ps(zmm29, zmm31, zmm17);\
    }\
    if (t_len > 4) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 4 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm4  = _mm512_fmadd_ps(zmm28, zmm30, zmm4);\
        if (oc_len > 1 * CH_DT_BLK()) zmm18 = _mm512_fmadd_ps(zmm29, zmm30, zmm18);\
    }\
    if (t_len > 5) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 5 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm5  = _mm512_fmadd_ps(zmm28, zmm31, zmm5);\
        if (oc_len > 1 * CH_DT_BLK()) zmm19 = _mm512_fmadd_ps(zmm29, zmm31, zmm19);\
    }\
    if (t_len > 6) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 6 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm6  = _mm512_fmadd_ps(zmm28, zmm30, zmm6);\
        if (oc_len > 1 * CH_DT_BLK()) zmm20 = _mm512_fmadd_ps(zmm29, zmm30, zmm20);\
    }\
    if (t_len > 6) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 7 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm7  = _mm512_fmadd_ps(zmm28, zmm31, zmm7);\
        if (oc_len > 1 * CH_DT_BLK()) zmm21 = _mm512_fmadd_ps(zmm29, zmm31, zmm21);\
    }\
    if (t_len > 8) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 8 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm8  = _mm512_fmadd_ps(zmm28, zmm30, zmm8);\
        if (oc_len > 1 * CH_DT_BLK()) zmm22 = _mm512_fmadd_ps(zmm29, zmm30, zmm22);\
    }\
    if (t_len > 9) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 9 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm9  = _mm512_fmadd_ps(zmm28, zmm31, zmm9);\
        if (oc_len > 1 * CH_DT_BLK()) zmm23 = _mm512_fmadd_ps(zmm29, zmm31, zmm23);\
    }\
    if (t_len > 10) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 10 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm10 = _mm512_fmadd_ps(zmm28, zmm30, zmm10);\
        if (oc_len > 1 * CH_DT_BLK()) zmm24 = _mm512_fmadd_ps(zmm29, zmm30, zmm24);\
    }\
    if (t_len > 11) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 11 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm11 = _mm512_fmadd_ps(zmm28, zmm31, zmm11);\
        if (oc_len > 1 * CH_DT_BLK()) zmm25 = _mm512_fmadd_ps(zmm29, zmm31, zmm25);\
    }\
    if (t_len > 12) {\
        zmm30 = _mm512_set1_ps(ic_src[(IC) + 12 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm12 = _mm512_fmadd_ps(zmm28, zmm30, zmm12);\
        if (oc_len > 1 * CH_DT_BLK()) zmm26 = _mm512_fmadd_ps(zmm29, zmm30, zmm26);\
    }\
    if (t_len > 13) {\
        zmm31 = _mm512_set1_ps(ic_src[(IC) + 13 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) zmm13 = _mm512_fmadd_ps(zmm28, zmm31, zmm13);\
        if (oc_len > 1 * CH_DT_BLK()) zmm27 = _mm512_fmadd_ps(zmm29, zmm31, zmm27);\
    }\
} while (0)

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    const int64_t src_tkb_stride = param[SRC_TKB_STRIDE_IDX()];
    const int64_t flt_ocb_stride = param[FLT_OCB_STRIDE_IDX()];
    const int64_t dst_ocb_stride = param[DST_OCB_STRIDE_IDX()];
    const int64_t load_dst = param[LOAD_DST_IDX()];

    const float *t_src = PICK_PARAM(const float*, param, SRC_IDX());
    float *t_dst       = PICK_PARAM(float *, param, DST_IDX());
    int64_t t          = param[TILES_IDX()];
    do {
        if (load_dst) {
            float *l_dst = t_dst;
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
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (t_len > 0) zmm14 = _mm512_loadu_ps(l_dst + 0 * CH_DT_BLK());
                if (t_len > 1) zmm15 = _mm512_loadu_ps(l_dst + 1 * CH_DT_BLK());
                if (t_len > 2) zmm16 = _mm512_loadu_ps(l_dst + 2 * CH_DT_BLK());
                if (t_len > 3) zmm17 = _mm512_loadu_ps(l_dst + 3 * CH_DT_BLK());
                if (t_len > 4) zmm18 = _mm512_loadu_ps(l_dst + 4 * CH_DT_BLK());
                if (t_len > 5) zmm19 = _mm512_loadu_ps(l_dst + 5 * CH_DT_BLK());
                if (t_len > 6) zmm20 = _mm512_loadu_ps(l_dst + 6 * CH_DT_BLK());
                if (t_len > 7) zmm21 = _mm512_loadu_ps(l_dst + 7 * CH_DT_BLK());
                if (t_len > 8) zmm22 = _mm512_loadu_ps(l_dst + 8 * CH_DT_BLK());
                if (t_len > 9) zmm23 = _mm512_loadu_ps(l_dst + 9 * CH_DT_BLK());
                if (t_len > 10) zmm24 = _mm512_loadu_ps(l_dst + 10 * CH_DT_BLK());
                if (t_len > 11) zmm25 = _mm512_loadu_ps(l_dst + 11 * CH_DT_BLK());
                if (t_len > 12) zmm26 = _mm512_loadu_ps(l_dst + 12 * CH_DT_BLK());
                if (t_len > 13) zmm27 = _mm512_loadu_ps(l_dst + 13 * CH_DT_BLK());
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
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (t_len > 0) zmm14 = _mm512_setzero_ps();
                if (t_len > 1) zmm15 = _mm512_setzero_ps();
                if (t_len > 2) zmm16 = _mm512_setzero_ps();
                if (t_len > 3) zmm17 = _mm512_setzero_ps();
                if (t_len > 4) zmm18 = _mm512_setzero_ps();
                if (t_len > 5) zmm19 = _mm512_setzero_ps();
                if (t_len > 6) zmm20 = _mm512_setzero_ps();
                if (t_len > 7) zmm21 = _mm512_setzero_ps();
                if (t_len > 8) zmm22 = _mm512_setzero_ps();
                if (t_len > 9) zmm23 = _mm512_setzero_ps();
                if (t_len > 10) zmm24 = _mm512_setzero_ps();
                if (t_len > 11) zmm25 = _mm512_setzero_ps();
                if (t_len > 12) zmm26 = _mm512_setzero_ps();
                if (t_len > 13) zmm27 = _mm512_setzero_ps();
            }
        }
        
        const float *icb_src = t_src;
        const float *icb_flt = PICK_PARAM(const float*, param, FLT_IDX());
        int64_t icb = param[CHANNELS_IDX()];
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
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (t_len > 0) _mm512_storeu_ps(l_dst + 0 * CH_DT_BLK(), zmm14);
                if (t_len > 1) _mm512_storeu_ps(l_dst + 1 * CH_DT_BLK(), zmm15);
                if (t_len > 2) _mm512_storeu_ps(l_dst + 2 * CH_DT_BLK(), zmm16);
                if (t_len > 3) _mm512_storeu_ps(l_dst + 3 * CH_DT_BLK(), zmm17);
                if (t_len > 4) _mm512_storeu_ps(l_dst + 4 * CH_DT_BLK(), zmm18);
                if (t_len > 5) _mm512_storeu_ps(l_dst + 5 * CH_DT_BLK(), zmm19);
                if (t_len > 6) _mm512_storeu_ps(l_dst + 6 * CH_DT_BLK(), zmm20);
                if (t_len > 7) _mm512_storeu_ps(l_dst + 7 * CH_DT_BLK(), zmm21);
                if (t_len > 8) _mm512_storeu_ps(l_dst + 8 * CH_DT_BLK(), zmm22);
                if (t_len > 9) _mm512_storeu_ps(l_dst + 9 * CH_DT_BLK(), zmm23);
                if (t_len > 10) _mm512_storeu_ps(l_dst + 10 * CH_DT_BLK(), zmm24);
                if (t_len > 11) _mm512_storeu_ps(l_dst + 11 * CH_DT_BLK(), zmm25);
                if (t_len > 12) _mm512_storeu_ps(l_dst + 12 * CH_DT_BLK(), zmm26);
                if (t_len > 13) _mm512_storeu_ps(l_dst + 13 * CH_DT_BLK(), zmm27);
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
