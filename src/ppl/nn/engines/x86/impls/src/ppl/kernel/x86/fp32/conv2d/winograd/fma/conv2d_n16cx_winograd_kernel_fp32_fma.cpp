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

#include "ppl/kernel/x86/fp32/conv2d/winograd/fma/conv2d_n16cx_winograd_kernel_fp32_fma.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template <int64_t t_len>
void conv2d_n16cx_winograd_kernel_fp32_fma_core(
    const float *src,
    const float *filter,
    const int64_t tiles,
    const int64_t channels,
    const int64_t src_tblk_stride,
    const int64_t load_dst,
    float *dst)
{
    __asm__ __volatile__(
        ".equ T_LEN, %c[T_LEN]\n"
        ".equ CH_DT_BLK, 16\n"
        ".equ CH_RF_BLK, 8\n"
        ".equ DBYTES, 4\n"

        "mov %[src], %%r8\n"
        "mov %[dst], %%r9\n"
        "mov %[tiles], %%r10\n"
"1:\n"
        "cmp $0, %[load_dst]\n"
        "je 5f\n"
        ".if T_LEN > 0\n"
        "vmovups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm0\n"
        "vmovups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm1\n"
        ".endif\n"
        ".if T_LEN > 1\n"
        "vmovups ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm2\n"
        "vmovups ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm3\n"
        ".endif\n"
        ".if T_LEN > 2\n"
        "vmovups ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm4\n"
        "vmovups ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm5\n"
        ".endif\n"
        ".if T_LEN > 3\n"
        "vmovups ((3 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm6\n"
        "vmovups ((3 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm7\n"
        ".endif\n"
        ".if T_LEN > 4\n"
        "vmovups ((4 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm8\n"
        "vmovups ((4 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm9\n"
        ".endif\n"
        ".if T_LEN > 5\n"
        "vmovups ((5 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm10\n"
        "vmovups ((5 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9), %%ymm11\n"
        ".endif\n"
        "jmp 6f\n"
"5:\n"
        ".if T_LEN > 0\n"
        "vxorps %%ymm0, %%ymm0, %%ymm0\n"
        "vxorps %%ymm1, %%ymm1, %%ymm1\n"
        ".endif\n"
        ".if T_LEN > 1\n"
        "vxorps %%ymm2, %%ymm2, %%ymm2\n"
        "vxorps %%ymm3, %%ymm3, %%ymm3\n"
        ".endif\n"
        ".if T_LEN > 2\n"
        "vxorps %%ymm4, %%ymm4, %%ymm4\n"
        "vxorps %%ymm5, %%ymm5, %%ymm5\n"
        ".endif\n"
        ".if T_LEN > 3\n"
        "vxorps %%ymm6, %%ymm6, %%ymm6\n"
        "vxorps %%ymm7, %%ymm7, %%ymm7\n"
        ".endif\n"
        ".if T_LEN > 4\n"
        "vxorps %%ymm8, %%ymm8, %%ymm8\n"
        "vxorps %%ymm9, %%ymm9, %%ymm9\n"
        ".endif\n"
        ".if T_LEN > 5\n"
        "vxorps %%ymm10, %%ymm10, %%ymm10\n"
        "vxorps %%ymm11, %%ymm11, %%ymm11\n"
        ".endif\n"
"6:\n"

        "mov %[channels], %%r13\n"
        "mov %[filter], %%r14\n"
        "mov %%r8, %%r15\n"

        "cmp $CH_DT_BLK, %%r13\n"
        "jl 3f\n" // label_loop_c

        PPL_X86_INLINE_ASM_ALIGN()
"2:\n" // label_loop_16c
        ".irp IC,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n"
        "vmovups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r14), %%ymm14\n"
        "vmovups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r14), %%ymm15\n"
        ".if T_LEN > 0\n"
        "vbroadcastss ((\\IC + 0 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm1\n"
        ".endif\n"
        ".if T_LEN > 1\n"
        "vbroadcastss ((\\IC + 1 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm2\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm3\n"
        ".endif\n"
        ".if T_LEN > 2\n"
        "vbroadcastss ((\\IC + 2 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm5\n"
        ".endif\n"
        ".if T_LEN > 3\n"
        "vbroadcastss ((\\IC + 3 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm6\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n"
        ".endif\n"
        ".if T_LEN > 4\n"
        "vbroadcastss ((\\IC + 4 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n"
        ".endif\n"
        ".if T_LEN > 5\n"
        "vbroadcastss ((\\IC + 5 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm10\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm11\n"
        ".endif\n"
        ".if T_LEN > 1\n"
        "prefetcht0 ((\\IC * CH_DT_BLK + CH_DT_BLK * CH_DT_BLK) * DBYTES)(%%r14)\n"
        ".endif\n"
        ".endr\n"
        "sub $CH_DT_BLK, %%r13\n"
        "cmp $CH_DT_BLK, %%r13\n"
        "lea (T_LEN * CH_DT_BLK * DBYTES)(%%r15), %%r15\n"
        "lea (CH_DT_BLK * CH_DT_BLK * DBYTES)(%%r14), %%r14\n"
        "jge 2b\n" // label_loop_16c

        "cmp $0, %%r13\n"
        "je 4f\n"

"3:\n" // label_loop_c
        "vmovups ((0 * CH_RF_BLK) * DBYTES)(%%r14), %%ymm14\n"
        "vmovups ((1 * CH_RF_BLK) * DBYTES)(%%r14), %%ymm15\n"
        ".if T_LEN > 0\n"
        "vbroadcastss ((0 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm0\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm1\n"
        ".endif\n"
        ".if T_LEN > 1\n"
        "vbroadcastss ((1 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm2\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm3\n"
        ".endif\n"
        ".if T_LEN > 2\n"
        "vbroadcastss ((2 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm4\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm5\n"
        ".endif\n"
        ".if T_LEN > 3\n"
        "vbroadcastss ((3 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm6\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7\n"
        ".endif\n"
        ".if T_LEN > 4\n"
        "vbroadcastss ((4 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm12\n"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm8\n"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9\n"
        ".endif\n"
        ".if T_LEN > 5\n"
        "vbroadcastss ((5 * CH_DT_BLK) * DBYTES)(%%r15), %%ymm13\n"
        "vfmadd231ps %%ymm14, %%ymm13, %%ymm10\n"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm11\n"
        ".endif\n"
        "lea DBYTES(%%r15), %%r15\n"
        "lea (CH_DT_BLK * DBYTES)(%%r14), %%r14\n"
        "sub $1, %%r13\n"
        "cmp $0, %%r13\n"
        "jne 3b\n" // label_loop_c

"4:\n" // label_store
        ".if T_LEN > 0\n"
        "vmovups %%ymm0, ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        "vmovups %%ymm1, ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        ".endif\n"
        ".if T_LEN > 1\n"
        "vmovups %%ymm2, ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        "vmovups %%ymm3, ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        ".endif\n"
        ".if T_LEN > 2\n"
        "vmovups %%ymm4, ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        "vmovups %%ymm5, ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        ".endif\n"
        ".if T_LEN > 3\n"
        "vmovups %%ymm6, ((3 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        "vmovups %%ymm7, ((3 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        ".endif\n"
        ".if T_LEN > 4\n"
        "vmovups %%ymm8, ((4 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        "vmovups %%ymm9, ((4 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        ".endif\n"
        ".if T_LEN > 5\n"
        "vmovups %%ymm10, ((5 * CH_DT_BLK + 0 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        "vmovups %%ymm11, ((5 * CH_DT_BLK + 1 * CH_RF_BLK) * DBYTES)(%%r9)\n"
        ".endif\n"

        "sub $T_LEN, %%r10\n"
        "cmp $0, %%r10\n"
        "lea (%%r8, %[src_tblk_stride], DBYTES), %%r8\n"
        "lea (T_LEN * CH_DT_BLK * DBYTES)(%%r9), %%r9\n"
        "jne 1b\n"
        :
        :
        [src]               "r" (src),
        [filter]            "r" (filter),
        [tiles]             "r" (tiles),
        [channels]          "r" (channels),
        [src_tblk_stride]   "r" (src_tblk_stride),
        [load_dst]          "r" (load_dst),
        [dst]               "r" (dst),
        [T_LEN]             "i" (t_len)
        :
        "cc", "r8", "r9", "r10", "r13", "r14", "r15",
        "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
        "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
        "memory");
}

#endif

template <int64_t oc_len, int64_t t_len>
void conv2d_n16cx_winograd_kernel_fp32_fma(
    const float *src,
    const float *filter,
    const int64_t tiles,
    const int64_t channels,
    const int64_t src_tblk_stride,
    const int64_t load_dst,
    float *dst)
{

#ifdef PPL_USE_X86_INLINE_ASM
    if (oc_len == 2 * CH_RF_BLK()) {
        conv2d_n16cx_winograd_kernel_fp32_fma_core<t_len>(src, filter, tiles, channels, src_tblk_stride, load_dst, dst);
        return;
    }
#endif

#define IC_COMPUTE_STEP(IC) do {\
        if (oc_len > 0 * CH_RF_BLK()) ymm14 = _mm256_loadu_ps(l_flt + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
        if (oc_len > 1 * CH_RF_BLK()) ymm15 = _mm256_loadu_ps(l_flt + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
        if (t_len > 1) _mm_prefetch((const char *)(l_flt + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK()), _MM_HINT_T0);\
        if (t_len > 0) {\
            ymm12 = _mm256_broadcast_ss(l_src + (IC) + 0 * CH_DT_BLK());\
            if (oc_len > 0 * CH_RF_BLK()) ymm0  = _mm256_fmadd_ps(ymm12, ymm14, ymm0);\
            if (oc_len > 1 * CH_RF_BLK()) ymm1  = _mm256_fmadd_ps(ymm12, ymm15, ymm1);\
        }\
        if (t_len > 1) {\
            ymm13 = _mm256_broadcast_ss(l_src + (IC) + 1 * CH_DT_BLK());\
            if (oc_len > 0 * CH_RF_BLK()) ymm2  = _mm256_fmadd_ps(ymm13, ymm14, ymm2);\
            if (oc_len > 1 * CH_RF_BLK()) ymm3  = _mm256_fmadd_ps(ymm13, ymm15, ymm3);\
        }\
        if (t_len > 2) {\
            ymm12 = _mm256_broadcast_ss(l_src + (IC) + 2 * CH_DT_BLK());\
            if (oc_len > 0 * CH_RF_BLK()) ymm4  = _mm256_fmadd_ps(ymm12, ymm14, ymm4);\
            if (oc_len > 1 * CH_RF_BLK()) ymm5  = _mm256_fmadd_ps(ymm12, ymm15, ymm5);\
        }\
        if (t_len > 3) {\
            ymm13 = _mm256_broadcast_ss(l_src + (IC) + 3 * CH_DT_BLK());\
            if (oc_len > 0 * CH_RF_BLK()) ymm6  = _mm256_fmadd_ps(ymm13, ymm14, ymm6);\
            if (oc_len > 1 * CH_RF_BLK()) ymm7  = _mm256_fmadd_ps(ymm13, ymm15, ymm7);\
        }\
        if (t_len > 4) {\
            ymm12 = _mm256_broadcast_ss(l_src + (IC) + 4 * CH_DT_BLK());\
            if (oc_len > 0 * CH_RF_BLK()) ymm8  = _mm256_fmadd_ps(ymm12, ymm14, ymm8);\
            if (oc_len > 1 * CH_RF_BLK()) ymm9  = _mm256_fmadd_ps(ymm12, ymm15, ymm9);\
        }\
        if (t_len > 5) {\
            ymm13 = _mm256_broadcast_ss(l_src + (IC) + 5 * CH_DT_BLK());\
            if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_fmadd_ps(ymm13, ymm14, ymm10);\
            if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_fmadd_ps(ymm13, ymm15, ymm11);\
        }\
    } while (0)

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    const float *t_src = src;
    float *t_dst       = dst;
    int64_t t          = tiles;
    do {
        if (load_dst) {
            if (t_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_loadu_ps(t_dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_loadu_ps(t_dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (t_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_loadu_ps(t_dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_loadu_ps(t_dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (t_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_loadu_ps(t_dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_loadu_ps(t_dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (t_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_loadu_ps(t_dst + 3 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_loadu_ps(t_dst + 3 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (t_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_loadu_ps(t_dst + 4 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_loadu_ps(t_dst + 4 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (t_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_loadu_ps(t_dst + 5 * CH_DT_BLK() + 0 * CH_RF_BLK());
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_loadu_ps(t_dst + 5 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
        } else {
            if (t_len > 0) {
                if (oc_len > 0 * CH_RF_BLK()) ymm0 = _mm256_setzero_ps();
                if (oc_len > 1 * CH_RF_BLK()) ymm1 = _mm256_setzero_ps();
            }
            if (t_len > 1) {
                if (oc_len > 0 * CH_RF_BLK()) ymm2 = _mm256_setzero_ps();
                if (oc_len > 1 * CH_RF_BLK()) ymm3 = _mm256_setzero_ps();
            }
            if (t_len > 2) {
                if (oc_len > 0 * CH_RF_BLK()) ymm4 = _mm256_setzero_ps();
                if (oc_len > 1 * CH_RF_BLK()) ymm5 = _mm256_setzero_ps();
            }
            if (t_len > 3) {
                if (oc_len > 0 * CH_RF_BLK()) ymm6 = _mm256_setzero_ps();
                if (oc_len > 1 * CH_RF_BLK()) ymm7 = _mm256_setzero_ps();
            }
            if (t_len > 4) {
                if (oc_len > 0 * CH_RF_BLK()) ymm8 = _mm256_setzero_ps();
                if (oc_len > 1 * CH_RF_BLK()) ymm9 = _mm256_setzero_ps();
            }
            if (t_len > 5) {
                if (oc_len > 0 * CH_RF_BLK()) ymm10 = _mm256_setzero_ps();
                if (oc_len > 1 * CH_RF_BLK()) ymm11 = _mm256_setzero_ps();
            }
        }

        int64_t ic         = channels;
        const float *l_src = t_src;
        const float *l_flt = filter;
        while (ic >= CH_DT_BLK()) {
            ic -= CH_DT_BLK();
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
            l_flt += CH_DT_BLK() * CH_DT_BLK();
            l_src += t_len * CH_DT_BLK();
        }
        while (ic > 0) {
            --ic;
            IC_COMPUTE_STEP(0);
            l_flt += CH_DT_BLK();
            l_src += 1;
        }

        if (t_len > 0) {
            if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm0);
            if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm1);
        }
        if (t_len > 1) {
            if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm2);
            if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm3);
        }
        if (t_len > 2) {
            if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm4);
            if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm5);
        }
        if (t_len > 3) {
            if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 3 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm6);
            if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 3 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm7);
        }
        if (t_len > 4) {
            if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 4 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm8);
            if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 4 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm9);
        }
        if (t_len > 5) {
            if (oc_len > 0 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 5 * CH_DT_BLK() + 0 * CH_RF_BLK(), ymm10);
            if (oc_len > 1 * CH_RF_BLK()) _mm256_storeu_ps(t_dst + 5 * CH_DT_BLK() + 1 * CH_RF_BLK(), ymm11);
        }
        t_src += src_tblk_stride;
        t_dst += t_len * CH_DT_BLK();
        t -= t_len;
    } while (t > 0);
#undef IC_COMPUTE_LOOP
}


conv2d_n16cx_winograd_kernel_fp32_fma_func_t
    conv2d_n16cx_winograd_kernel_fp32_fma_table[TILE_RF_CNT()] =
{
    conv2d_n16cx_winograd_kernel_fp32_fma<2 * CH_RF_BLK(), 1>,
    conv2d_n16cx_winograd_kernel_fp32_fma<2 * CH_RF_BLK(), 2>,
    conv2d_n16cx_winograd_kernel_fp32_fma<2 * CH_RF_BLK(), 3>,
    conv2d_n16cx_winograd_kernel_fp32_fma<2 * CH_RF_BLK(), 4>,
    conv2d_n16cx_winograd_kernel_fp32_fma<2 * CH_RF_BLK(), 5>,
    conv2d_n16cx_winograd_kernel_fp32_fma<2 * CH_RF_BLK(), 6>,
};

}}};
