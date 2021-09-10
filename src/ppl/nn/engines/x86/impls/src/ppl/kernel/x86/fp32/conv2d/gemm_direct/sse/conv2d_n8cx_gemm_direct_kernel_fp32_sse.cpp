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

#include "ppl/kernel/x86/fp32/conv2d/gemm_direct/sse/conv2d_n8cx_gemm_direct_kernel_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template <bool nt_store, int32_t oc_len>
void conv2d_n8cx_gemm_direct_fp32_sse_blk1x3_kernel_core(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
    static float six[4] = {6.0f, 6.0f, 6.0f, 6.0f};
    __asm__ __volatile__ (
        ".equ P_BYTES, 8\n"
        ".equ D_BYTES, 4\n"

        ".equ CH_DT_BLK, 8\n"
        ".equ CH_RF_BLK, 4\n"

        ".equ SRC_IDX,  (0 * P_BYTES)\n"
        ".equ HIS_IDX,  (1 * P_BYTES)\n"
        ".equ DST_IDX,  (2 * P_BYTES)\n"
        ".equ FLT_IDX,  (3 * P_BYTES)\n"
        ".equ BIAS_IDX, (4 * P_BYTES)\n"
        ".equ HW_IDX,   (5 * P_BYTES)\n"

        ".equ CHANNELS_IDX, (0 * P_BYTES)\n"
        ".equ SRC_ICB_STRIDE_IDX, (1 * P_BYTES)\n"
        ".equ HIS_OCB_STRIDE_IDX, (2 * P_BYTES)\n"
        ".equ DST_OCB_STRIDE_IDX, (3 * P_BYTES)\n"
        ".equ FLT_OCB_STRIDE_IDX, (4 * P_BYTES)\n"
        ".equ FLAGS_IDX, (5 * P_BYTES)\n"

        ".equ NT_STORE, %c[NT_STORE]\n"
        ".equ OC_LEN, %c[OC_LEN]\n"
        ".equ HW_LEN, 3\n"
        ".equ KERNEL_FLAG_LD_BIAS, %c[KERNEL_FLAG_LD_BIAS]\n"
        ".equ KERNEL_FLAG_AD_BIAS, %c[KERNEL_FLAG_AD_BIAS]\n"
        ".equ KERNEL_FLAG_RELU, %c[KERNEL_FLAG_RELU]\n"
        ".equ KERNEL_FLAG_RELU6, %c[KERNEL_FLAG_RELU6]\n"

        "mov SRC_ICB_STRIDE_IDX(%[shar_param]), %%r8\n"
        "mov FLT_OCB_STRIDE_IDX(%[shar_param]), %%r9\n"
        "mov FLAGS_IDX(%[shar_param]), %%r11\n"
        "mov SRC_IDX(%[priv_param]), %%r12\n"
        "mov HIS_IDX(%[priv_param]), %%r13\n"
        "mov DST_IDX(%[priv_param]), %%r14\n"
        "mov HW_IDX(%[priv_param]), %%r15\n"
"1:\n" // label_init_session
        "test $KERNEL_FLAG_LD_BIAS, %%r11\n"
        "jz 2f\n" // label_load_h
        "mov BIAS_IDX(%[priv_param]), %%r10\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm0\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm1\n"
        "movaps %%xmm0, %%xmm2\n"
        "movaps %%xmm1, %%xmm3\n"
        "movaps %%xmm0, %%xmm4\n"
        "movaps %%xmm1, %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm6\n"
        "movups ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm7\n"
        "movaps %%xmm6, %%xmm8\n"
        "movaps %%xmm7, %%xmm9\n"
        "movaps %%xmm6, %%xmm10\n"
        "movaps %%xmm7, %%xmm11\n"
        ".endif\n"
        "jmp 3f\n" // label_load_h_end
"2:\n" // label_load_h
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "mov HIS_OCB_STRIDE_IDX(%[shar_param]), %%r10\n"
        "lea (%%r13, %%r10, D_BYTES), %%r10\n"
        ".endif\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r13), %%xmm0\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r13), %%xmm1\n"
        "movups ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r13), %%xmm2\n"
        "movups ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r13), %%xmm3\n"
        "movups ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r13), %%xmm4\n"
        "movups ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r13), %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm6\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm7\n"
        "movups ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm8\n"
        "movups ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm9\n"
        "movups ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm10\n"
        "movups ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm11\n"
        ".endif\n"
"3:\n" // label_load_h_end
        "test $KERNEL_FLAG_AD_BIAS, %%r11\n"
        "jz 4f\n" // label_compute_session
        "mov BIAS_IDX(%[priv_param]), %%r10\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm12\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm13\n"
        "addps %%xmm12, %%xmm0\n"
        "addps %%xmm13, %%xmm1\n"
        "addps %%xmm12, %%xmm2\n"
        "addps %%xmm13, %%xmm3\n"
        "addps %%xmm12, %%xmm4\n"
        "addps %%xmm13, %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm14\n"
        "movups ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm15\n"
        "addps %%xmm14, %%xmm6\n"
        "addps %%xmm15, %%xmm7\n"
        "addps %%xmm14, %%xmm8\n"
        "addps %%xmm15, %%xmm9\n"
        "addps %%xmm14, %%xmm10\n"
        "addps %%xmm15, %%xmm11\n"
        ".endif\n"

"4:\n" // label_compute_session
        "mov %%r12, %%rax\n"
        "mov FLT_IDX(%[priv_param]), %%rbx\n"
        "mov CHANNELS_IDX(%[shar_param]), %%r10\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n lea (%%rbx, %%r9, D_BYTES), %%rcx\n .endif\n"
        "cmp $CH_DT_BLK, %%r10\n"
        "jl 6f\n" // label_ic_remain
        PPL_X86_INLINE_ASM_ALIGN()
"5:\n" // label_ic_body
        ".irp IC,0,1,2,3,4,5,6,7\n"
        "movss ((\\IC + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%xmm12\n"
        "shufps $0, %%xmm12, %%xmm12\n"
        "movss ((\\IC + 1 * CH_DT_BLK) * D_BYTES)(%%rax), %%xmm13\n"
        "shufps $0, %%xmm13, %%xmm13\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm0\n"
        "addps %%xmm15, %%xmm1\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm6\n"
        "addps %%xmm15, %%xmm7\n"
        ".endif\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm15\n"
        "mulps %%xmm13, %%xmm14\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm14, %%xmm2\n"
        "addps %%xmm15, %%xmm3\n"
        ".endif\n"
        "movss ((\\IC + 2 * CH_DT_BLK) * D_BYTES)(%%rax), %%xmm12\n"
        "shufps $0, %%xmm12, %%xmm12\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm15\n"
        "mulps %%xmm13, %%xmm14\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm14, %%xmm8\n"
        "addps %%xmm15, %%xmm9\n"
        ".endif\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm4\n"
        "addps %%xmm15, %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm10\n"
        "addps %%xmm15, %%xmm11\n"
        ".endif\n"
        ".endr\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n lea (CH_DT_BLK * CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n .endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n lea (CH_DT_BLK * CH_DT_BLK * D_BYTES)(%%rcx), %%rcx\n .endif\n"
        "lea (%%rax, %%r8, D_BYTES), %%rax\n"
        "sub $CH_DT_BLK, %%r10\n"
        "cmp $CH_DT_BLK, %%r10\n"
        "jge 5b\n" // label_ic_body
        "cmp $0, %%r10\n"
        "je 7f\n" // label_finalize_session
"6:\n" // label_ic_remain
        "movss ((0 + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%xmm12\n"
        "shufps $0, %%xmm12, %%xmm12\n"
        "movss ((0 + 1 * CH_DT_BLK) * D_BYTES)(%%rax), %%xmm13\n"
        "shufps $0, %%xmm13, %%xmm13\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm0\n"
        "addps %%xmm15, %%xmm1\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm6\n"
        "addps %%xmm15, %%xmm7\n"
        ".endif\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm15\n"
        "mulps %%xmm13, %%xmm14\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm14, %%xmm2\n"
        "addps %%xmm15, %%xmm3\n"
        ".endif\n"
        "movss ((0 + 2 * CH_DT_BLK) * D_BYTES)(%%rax), %%xmm12\n"
        "shufps $0, %%xmm12, %%xmm12\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm15\n"
        "mulps %%xmm13, %%xmm14\n"
        "mulps %%xmm13, %%xmm15\n"
        "addps %%xmm14, %%xmm8\n"
        "addps %%xmm15, %%xmm9\n"
        ".endif\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm4\n"
        "addps %%xmm15, %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm10\n"
        "addps %%xmm15, %%xmm11\n"
        ".endif\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n lea (CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n .endif\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n lea (CH_DT_BLK * D_BYTES)(%%rcx), %%rcx\n .endif\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%r10\n"
        "cmp $0, %%r10\n"
        "jne 6b\n" // label_ic_remain

"7:\n" // label_finalize_session
        "test $(KERNEL_FLAG_RELU | KERNEL_FLAG_RELU6), %%r11\n"
        "jz 8f\n" // label_relu_end
        "xorps %%xmm12, %%xmm12\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "maxps %%xmm12, %%xmm0\n"
        "maxps %%xmm12, %%xmm1\n"
        "maxps %%xmm12, %%xmm2\n"
        "maxps %%xmm12, %%xmm3\n"
        "maxps %%xmm12, %%xmm4\n"
        "maxps %%xmm12, %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "maxps %%xmm12, %%xmm6\n"
        "maxps %%xmm12, %%xmm7\n"
        "maxps %%xmm12, %%xmm8\n"
        "maxps %%xmm12, %%xmm9\n"
        "maxps %%xmm12, %%xmm10\n"
        "maxps %%xmm12, %%xmm11\n"
        ".endif\n"
        "test $KERNEL_FLAG_RELU6, %%r11\n"
        "jz 8f\n" // label_relu_end
        "movups (%[six]), %%xmm13\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "minps %%xmm13, %%xmm0\n"
        "minps %%xmm13, %%xmm1\n"
        "minps %%xmm13, %%xmm2\n"
        "minps %%xmm13, %%xmm3\n"
        "minps %%xmm13, %%xmm4\n"
        "minps %%xmm13, %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "minps %%xmm13, %%xmm6\n"
        "minps %%xmm13, %%xmm7\n"
        "minps %%xmm13, %%xmm8\n"
        "minps %%xmm13, %%xmm9\n"
        "minps %%xmm13, %%xmm10\n"
        "minps %%xmm13, %%xmm11\n"
        ".endif\n"
"8:\n" // label_relu_end
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "mov DST_OCB_STRIDE_IDX(%[shar_param]), %%r10\n"
        "lea (%%r14, %%r10, D_BYTES), %%r10\n"
        ".endif\n"
        ".if NT_STORE\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movntps %%xmm0, ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        "movntps %%xmm1, ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        "movntps %%xmm2, ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        "movntps %%xmm3, ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        "movntps %%xmm4, ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        "movntps %%xmm5, ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movntps %%xmm6, ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
        "movntps %%xmm7, ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
        "movntps %%xmm8, ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
        "movntps %%xmm9, ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
        "movntps %%xmm10, ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
        "movntps %%xmm11, ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
        ".endif\n"
        ".else\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups %%xmm0, ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        "movups %%xmm1, ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        "movups %%xmm2, ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        "movups %%xmm3, ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        "movups %%xmm4, ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        "movups %%xmm5, ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups %%xmm6, ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
        "movups %%xmm7, ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
        "movups %%xmm8, ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
        "movups %%xmm9, ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
        "movups %%xmm10, ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
        "movups %%xmm11, ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10)\n"
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
        [six]                         "r" (six),
        [NT_STORE]                    "i" (nt_store),
        [OC_LEN]                      "i" (oc_len),
        [KERNEL_FLAG_LD_BIAS]         "i" (KERNEL_FLAG_LD_BIAS()),
        [KERNEL_FLAG_AD_BIAS]         "i" (KERNEL_FLAG_AD_BIAS()),
        [KERNEL_FLAG_RELU]            "i" (KERNEL_FLAG_RELU()),
        [KERNEL_FLAG_RELU6]           "i" (KERNEL_FLAG_RELU6())
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

template <bool nt_store, int32_t oc_len, int32_t hw_len>
void conv2d_n8cx_gemm_direct_fp32_sse_blk1x3_kernel(
    const int64_t *priv_param,
    const int64_t *shar_param)
{

#ifdef PPL_USE_X86_INLINE_ASM
    if (hw_len == 3) {
        conv2d_n8cx_gemm_direct_fp32_sse_blk1x3_kernel_core<nt_store, oc_len>(priv_param, shar_param);
        return;
    }
#endif

#define IC_COMPUTE_STEP(IC) do {\
    if (hw_len > 0) {\
        xmm12 = _mm_set1_ps(ic_src[(IC) + 0 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm12);\
            xmm0 = _mm_add_ps(xmm0, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm12);\
            xmm1 = _mm_add_ps(xmm1, xmm15);\
        }\
        if (oc_len > 1 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm12);\
            xmm6 = _mm_add_ps(xmm6, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm12);\
            xmm7 = _mm_add_ps(xmm7, xmm15);\
        }\
    }\
    if (hw_len > 1) {\
        xmm13 = _mm_set1_ps(ic_src[(IC) + 1 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm13);\
            xmm2 = _mm_add_ps(xmm2, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm13);\
            xmm3 = _mm_add_ps(xmm3, xmm15);\
        }\
        if (oc_len > 1 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm13);\
            xmm8 = _mm_add_ps(xmm8, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm13);\
            xmm9 = _mm_add_ps(xmm9, xmm15);\
        }\
    }\
    if (hw_len > 2) {\
        xmm14 = _mm_set1_ps(ic_src[(IC) + 2 * CH_DT_BLK()]);\
        if (oc_len > 0 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm14);\
            xmm4 = _mm_add_ps(xmm4, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o8 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm14);\
            xmm5 = _mm_add_ps(xmm5, xmm15);\
        }\
        if (oc_len > 1 * CH_DT_BLK()) {\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm14);\
            xmm10 = _mm_add_ps(xmm10, xmm15);\
            xmm15 = _mm_loadu_ps(ic_flt_o16 + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
            xmm15 = _mm_mul_ps(xmm15, xmm14);\
            xmm11 = _mm_add_ps(xmm11, xmm15);\
        }\
    }\
} while (0)

#define IC_PREFETCH_STEP(IC)  do {\
    if (oc_len > 1 * CH_DT_BLK() && hw_len > 1) {\
        if (oc_len > 0 * CH_DT_BLK()) _mm_prefetch((const char*)ic_flt_o8 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
        if (oc_len > 1 * CH_DT_BLK()) _mm_prefetch((const char*)ic_flt_o16 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
    }\
} while (0)

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;

    const int64_t src_icb_stride = shar_param[SRC_ICB_STRIDE_IDX()];
    const int64_t flt_ocb_stride = shar_param[FLT_OCB_STRIDE_IDX()];
    const int64_t kernel_flags = shar_param[FLAGS_IDX()];

    const float *src = PICK_PARAM(const float*, priv_param, SRC_IDX());
    const float *his = PICK_PARAM(const float*, priv_param, HIS_IDX());
    float *dst       = PICK_PARAM(float*, priv_param, DST_IDX());
    int64_t hw       = priv_param[HW_IDX()];
    do {
        if (kernel_flags & KERNEL_FLAG_LD_BIAS()) {
            const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
            if (oc_len > 0 * CH_DT_BLK()) {
                if (hw_len > 0) {
                    xmm0 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm1 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (hw_len > 1) {
                    xmm2 = xmm0;
                    xmm3 = xmm1;
                }
                if (hw_len > 2) {
                    xmm4 = xmm0;
                    xmm5 = xmm1;
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (hw_len > 0) {
                    xmm6 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm7 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (hw_len > 1) {
                    xmm8 = xmm6;
                    xmm9 = xmm7;
                }
                if (hw_len > 2) {
                    xmm10 = xmm6;
                    xmm11 = xmm7;
                }
            }
        } else {
            const float *l_his = his;
            const int64_t his_ocb_stride = shar_param[HIS_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                if (hw_len > 0) {
                    xmm0 = _mm_loadu_ps(l_his + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm1 = _mm_loadu_ps(l_his + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (hw_len > 1) {
                    xmm2 = _mm_loadu_ps(l_his + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm3 = _mm_loadu_ps(l_his + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (hw_len > 2) {
                    xmm4 = _mm_loadu_ps(l_his + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm5 = _mm_loadu_ps(l_his + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_his += his_ocb_stride;
                if (hw_len > 0) {
                    xmm6 = _mm_loadu_ps(l_his + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm7 = _mm_loadu_ps(l_his + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (hw_len > 1) {
                    xmm8 = _mm_loadu_ps(l_his + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm9 = _mm_loadu_ps(l_his + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
                if (hw_len > 2) {
                    xmm10 = _mm_loadu_ps(l_his + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
                    xmm11 = _mm_loadu_ps(l_his + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
                }
            }
        }

        if (kernel_flags & KERNEL_FLAG_AD_BIAS()) {
            const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
            if (oc_len > 0 * CH_DT_BLK()) {
                xmm12 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm13 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
                if (hw_len > 0) {
                    xmm0 = _mm_add_ps(xmm0, xmm12);
                    xmm1 = _mm_add_ps(xmm1, xmm13);
                }
                if (hw_len > 1) {
                    xmm2 = _mm_add_ps(xmm2, xmm12);
                    xmm3 = _mm_add_ps(xmm3, xmm13);
                }
                if (hw_len > 2) {
                    xmm4 = _mm_add_ps(xmm4, xmm12);
                    xmm5 = _mm_add_ps(xmm5, xmm13);
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                xmm14 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm15 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
                if (hw_len > 0) {
                    xmm6 = _mm_add_ps(xmm6, xmm14);
                    xmm7 = _mm_add_ps(xmm7, xmm15);
                }
                if (hw_len > 1) {
                    xmm8 = _mm_add_ps(xmm8, xmm14);
                    xmm9 = _mm_add_ps(xmm9, xmm15);
                }
                if (hw_len > 2) {
                    xmm10 = _mm_add_ps(xmm10, xmm14);
                    xmm11 = _mm_add_ps(xmm11, xmm15);
                }
            }
        }
        
        const float *icb_src = src;
        const float *icb_flt = PICK_PARAM(const float*, priv_param, FLT_IDX());
        int64_t channels     = shar_param[CHANNELS_IDX()];
        while (channels >= CH_DT_BLK()) {
            channels -= CH_DT_BLK();
            const float *ic_src = icb_src;
            const float *ic_flt_o8 = icb_flt + 0 * flt_ocb_stride;
            const float *ic_flt_o16 = icb_flt + 1 * flt_ocb_stride;
            IC_COMPUTE_STEP(0);
            IC_COMPUTE_STEP(1);
            IC_COMPUTE_STEP(2);
            IC_COMPUTE_STEP(3);
            IC_COMPUTE_STEP(4);
            IC_COMPUTE_STEP(5);
            IC_COMPUTE_STEP(6);
            IC_COMPUTE_STEP(7);
            icb_flt += CH_DT_BLK() * CH_DT_BLK();
            icb_src += src_icb_stride;
        }
        if (channels > 0) {
            const float *ic_src = icb_src;
            const float *ic_flt_o8 = icb_flt + 0 * flt_ocb_stride;
            const float *ic_flt_o16 = icb_flt + 1 * flt_ocb_stride;
            for (int64_t ic = 0; ic < channels; ++ic) {
                IC_COMPUTE_STEP(0);
                ic_src += 1;
                ic_flt_o8 += CH_DT_BLK();
                ic_flt_o16 += CH_DT_BLK();
            }
        }
        
        if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
            xmm12 = _mm_setzero_ps();
            if (oc_len > 0 * CH_DT_BLK()) {
                if (hw_len > 0) {
                    xmm0 = _mm_max_ps(xmm0, xmm12);
                    xmm1 = _mm_max_ps(xmm1, xmm12);
                }
                if (hw_len > 1) {
                    xmm2 = _mm_max_ps(xmm2, xmm12);
                    xmm3 = _mm_max_ps(xmm3, xmm12);
                }
                if (hw_len > 2) {
                    xmm4 = _mm_max_ps(xmm4, xmm12);
                    xmm5 = _mm_max_ps(xmm5, xmm12);
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (hw_len > 0) {
                    xmm6 = _mm_max_ps(xmm6, xmm12);
                    xmm7 = _mm_max_ps(xmm7, xmm12);
                }
                if (hw_len > 1) {
                    xmm8 = _mm_max_ps(xmm8, xmm12);
                    xmm9 = _mm_max_ps(xmm9, xmm12);
                }
                if (hw_len > 2) {
                    xmm10 = _mm_max_ps(xmm10, xmm12);
                    xmm11 = _mm_max_ps(xmm11, xmm12);
                }
            }
        }
        if (kernel_flags & KERNEL_FLAG_RELU6()) {
            xmm13 = _mm_set1_ps(6.0f);
            if (oc_len > 0 * CH_DT_BLK()) {
                if (hw_len > 0) {
                    xmm0 = _mm_min_ps(xmm0, xmm13);
                    xmm1 = _mm_min_ps(xmm1, xmm13);
                }
                if (hw_len > 1) {
                    xmm2 = _mm_min_ps(xmm2, xmm13);
                    xmm3 = _mm_min_ps(xmm3, xmm13);
                }
                if (hw_len > 2) {
                    xmm4 = _mm_min_ps(xmm4, xmm13);
                    xmm5 = _mm_min_ps(xmm5, xmm13);
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                if (hw_len > 0) {
                    xmm6 = _mm_min_ps(xmm6, xmm13);
                    xmm7 = _mm_min_ps(xmm7, xmm13);
                }
                if (hw_len > 1) {
                    xmm8 = _mm_min_ps(xmm8, xmm13);
                    xmm9 = _mm_min_ps(xmm9, xmm13);
                }
                if (hw_len > 2) {
                    xmm10 = _mm_min_ps(xmm10, xmm13);
                    xmm11 = _mm_min_ps(xmm11, xmm13);
                }
            }
        }

        if (nt_store) {
            float* l_dst = dst;
            const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                if (hw_len > 0) {
                    _mm_stream_ps(l_dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm0);
                    _mm_stream_ps(l_dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm1);
                }
                if (hw_len > 1) {
                    _mm_stream_ps(l_dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm2);
                    _mm_stream_ps(l_dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm3);
                }
                if (hw_len > 2) {
                    _mm_stream_ps(l_dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm4);
                    _mm_stream_ps(l_dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm5);
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (hw_len > 0) {
                    _mm_stream_ps(l_dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm6);
                    _mm_stream_ps(l_dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm7);
                }
                if (hw_len > 1) {
                    _mm_stream_ps(l_dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);
                    _mm_stream_ps(l_dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm9);
                }
                if (hw_len > 2) {
                    _mm_stream_ps(l_dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm10);
                    _mm_stream_ps(l_dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm11);
                }
            }
        } else {
            float* l_dst = dst;
            const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                if (hw_len > 0) {
                    _mm_storeu_ps(l_dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm0);
                    _mm_storeu_ps(l_dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm1);
                }
                if (hw_len > 1) {
                    _mm_storeu_ps(l_dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm2);
                    _mm_storeu_ps(l_dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm3);
                }
                if (hw_len > 2) {
                    _mm_storeu_ps(l_dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm4);
                    _mm_storeu_ps(l_dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm5);
                }
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                if (hw_len > 0) {
                    _mm_storeu_ps(l_dst + 0 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm6);
                    _mm_storeu_ps(l_dst + 0 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm7);
                }
                if (hw_len > 1) {
                    _mm_storeu_ps(l_dst + 1 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm8);
                    _mm_storeu_ps(l_dst + 1 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm9);
                }
                if (hw_len > 2) {
                    _mm_storeu_ps(l_dst + 2 * CH_DT_BLK() + 0 * CH_RF_BLK(), xmm10);
                    _mm_storeu_ps(l_dst + 2 * CH_DT_BLK() + 1 * CH_RF_BLK(), xmm11);
                }
            }
        }
        src += hw_len * CH_DT_BLK();
        his += hw_len * CH_DT_BLK();
        dst += hw_len * CH_DT_BLK();
        hw -= hw_len;
    } while (hw > 0);
#undef IC_COMPUTE_STEP
#undef IC_PREFETCH_STEP
}

#ifdef PPL_USE_X86_INLINE_ASM

template <bool nt_store, int32_t oc_len>
void conv2d_n8cx_gemm_direct_fp32_sse_blk1x1_kernel_core(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
    static float six[4] = {6.0f, 6.0f, 6.0f, 6.0f};
    __asm__ __volatile__ (
        ".equ P_BYTES, 8\n"
        ".equ D_BYTES, 4\n"

        ".equ CH_DT_BLK, 8\n"
        ".equ CH_RF_BLK, 4\n"

        ".equ SRC_IDX,  (0 * P_BYTES)\n"
        ".equ HIS_IDX,  (1 * P_BYTES)\n"
        ".equ DST_IDX,  (2 * P_BYTES)\n"
        ".equ FLT_IDX,  (3 * P_BYTES)\n"
        ".equ BIAS_IDX, (4 * P_BYTES)\n"
        ".equ HW_IDX,   (5 * P_BYTES)\n"

        ".equ CHANNELS_IDX, (0 * P_BYTES)\n"
        ".equ SRC_ICB_STRIDE_IDX, (1 * P_BYTES)\n"
        ".equ HIS_OCB_STRIDE_IDX, (2 * P_BYTES)\n"
        ".equ DST_OCB_STRIDE_IDX, (3 * P_BYTES)\n"
        ".equ FLT_OCB_STRIDE_IDX, (4 * P_BYTES)\n"
        ".equ FLAGS_IDX, (5 * P_BYTES)\n"

        ".equ NT_STORE, %c[NT_STORE]\n"
        ".equ OC_LEN, %c[OC_LEN]\n"
        ".equ KERNEL_FLAG_LD_BIAS, %c[KERNEL_FLAG_LD_BIAS]\n"
        ".equ KERNEL_FLAG_AD_BIAS, %c[KERNEL_FLAG_AD_BIAS]\n"
        ".equ KERNEL_FLAG_RELU, %c[KERNEL_FLAG_RELU]\n"
        ".equ KERNEL_FLAG_RELU6, %c[KERNEL_FLAG_RELU6]\n"

        "mov SRC_ICB_STRIDE_IDX(%[shar_param]), %%r8\n"
        "mov FLT_OCB_STRIDE_IDX(%[shar_param]), %%r9\n"
        "mov FLAGS_IDX(%[shar_param]), %%r11\n"
        "mov SRC_IDX(%[priv_param]), %%r12\n"
        "mov HIS_IDX(%[priv_param]), %%r13\n"
        "mov DST_IDX(%[priv_param]), %%r14\n"
        "mov HW_IDX(%[priv_param]), %%r15\n"
"1:\n" // label_init_session
        "test $KERNEL_FLAG_LD_BIAS, %%r11\n"
        "jz 2f\n" // label_load_h
        "mov BIAS_IDX(%[priv_param]), %%r10\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm0\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm1\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm2\n"
        "movups ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm3\n"
        ".endif\n"
        ".if OC_LEN > 2 * CH_DT_BLK\n"
        "movups ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm4\n"
        "movups ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n"
        "movups ((3 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm6\n"
        "movups ((3 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm7\n"
        ".endif\n"
        ".if OC_LEN > 4 * CH_DT_BLK\n"
        "movups ((4 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm8\n"
        "movups ((4 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm9\n"
        ".endif\n"
        ".if OC_LEN > 5 * CH_DT_BLK\n"
        "movups ((5 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm10\n"
        "movups ((5 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm11\n"
        ".endif\n"
        "jmp 4f\n" // label_compute_session
"2:\n" // label_load_h
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "mov HIS_OCB_STRIDE_IDX(%[shar_param]), %%r10\n"
        ".endif\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups (0 * CH_RF_BLK * D_BYTES)(%%r13), %%xmm0\n"
        "movups (1 * CH_RF_BLK * D_BYTES)(%%r13), %%xmm1\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "lea (%%r13, %%r10, D_BYTES), %%rax\n"
        "movups (0 * CH_RF_BLK * D_BYTES)(%%rax), %%xmm2\n"
        "movups (1 * CH_RF_BLK * D_BYTES)(%%rax), %%xmm3\n"
        ".endif\n"
        ".if OC_LEN > 2 * CH_DT_BLK\n"
        "lea (%%r13, %%r10, 2 * D_BYTES), %%rbx\n"
        "movups (0 * CH_RF_BLK * D_BYTES)(%%rbx), %%xmm4\n"
        "movups (1 * CH_RF_BLK * D_BYTES)(%%rbx), %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n"
        "lea (%%rax, %%r10, 2 * D_BYTES), %%rax\n"
        "movups (0 * CH_RF_BLK * D_BYTES)(%%rax), %%xmm6\n"
        "movups (1 * CH_RF_BLK * D_BYTES)(%%rax), %%xmm7\n"
        ".endif\n"
        ".if OC_LEN > 4 * CH_DT_BLK\n"
        "lea (%%rbx, %%r10, 2 * D_BYTES), %%rbx\n"
        "movups (0 * CH_RF_BLK * D_BYTES)(%%rbx), %%xmm8\n"
        "movups (1 * CH_RF_BLK * D_BYTES)(%%rbx), %%xmm9\n"
        ".endif\n"
        ".if OC_LEN > 5 * CH_DT_BLK\n"
        "lea (%%rax, %%r10, 2 * D_BYTES), %%rax\n"
        "movups (0 * CH_RF_BLK * D_BYTES)(%%rax), %%xmm10\n"
        "movups (1 * CH_RF_BLK * D_BYTES)(%%rax), %%xmm11\n"
        ".endif\n"
        "test $KERNEL_FLAG_AD_BIAS, %%r11\n"
        "jz 4f\n" // label_compute_session
        "mov BIAS_IDX(%[priv_param]), %%r10\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "addps ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm0\n"
        "addps ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm1\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "addps ((1 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm2\n"
        "addps ((1 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm3\n"
        ".endif\n"
        ".if OC_LEN > 2 * CH_DT_BLK\n"
        "addps ((2 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm4\n"
        "addps ((2 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n"
        "addps ((3 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm6\n"
        "addps ((3 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm7\n"
        ".endif\n"
        ".if OC_LEN > 4 * CH_DT_BLK\n"
        "addps ((4 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm8\n"
        "addps ((4 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm9\n"
        ".endif\n"
        ".if OC_LEN > 5 * CH_DT_BLK\n"
        "addps ((5 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm10\n"
        "addps ((5 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%r10), %%xmm11\n"
        ".endif\n"

"4:\n" // label_compute_session
        "mov %%r12, %%rax\n"
        "mov FLT_IDX(%[priv_param]), %%rbx\n"
        "mov CHANNELS_IDX(%[shar_param]), %%r10\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n"
        "lea (%%rbx, %%r9, 2 * D_BYTES), %%rcx\n"
        "lea (%%rcx, %%r9, 1 * D_BYTES), %%rcx\n"
        ".endif\n"
        "cmp $CH_DT_BLK, %%r10\n"
        "jl 6f\n" // label_ic_remain
        PPL_X86_INLINE_ASM_ALIGN()
"5:\n" // label_ic_body
        ".irp IC,0,1,2,3,4,5,6,7\n"
        "movss ((\\IC + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%xmm12\n"
        "shufps $0, %%xmm12, %%xmm12\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm0\n"
        "addps %%xmm15, %%xmm1\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx, %%r9, 1 * D_BYTES), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx, %%r9, 1 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm2\n"
        "addps %%xmm15, %%xmm3\n"
        ".endif\n"
        ".if OC_LEN > 2 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx, %%r9, 2 * D_BYTES), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx, %%r9, 2 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm4\n"
        "addps %%xmm15, %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm6\n"
        "addps %%xmm15, %%xmm7\n"
        ".endif\n"
        ".if OC_LEN > 4 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx, %%r9, 1 * D_BYTES), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx, %%r9, 1 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm8\n"
        "addps %%xmm15, %%xmm9\n"
        ".endif\n"
        ".if OC_LEN > 5 * CH_DT_BLK\n"
        "movups ((\\IC * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx, %%r9, 2 * D_BYTES), %%xmm14\n"
        "movups ((\\IC * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx, %%r9, 2 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm10\n"
        "addps %%xmm15, %%xmm11\n"
        ".endif\n"
        ".endr\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n lea (CH_DT_BLK * CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n .endif\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n lea (CH_DT_BLK * CH_DT_BLK * D_BYTES)(%%rcx), %%rcx\n .endif\n"
        "lea (%%rax, %%r8, D_BYTES), %%rax\n"
        "sub $CH_DT_BLK, %%r10\n"
        "cmp $CH_DT_BLK, %%r10\n"
        "jge 5b\n" // label_ic_body
        "cmp $0, %%r10\n"
        "je 7f\n" // label_finalize_session
"6:\n" // label_ic_remain
        "movss ((0 + 0 * CH_DT_BLK) * D_BYTES)(%%rax), %%xmm12\n"
        "shufps $0, %%xmm12, %%xmm12\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm0\n"
        "addps %%xmm15, %%xmm1\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx, %%r9, 1 * D_BYTES), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx, %%r9, 1 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm2\n"
        "addps %%xmm15, %%xmm3\n"
        ".endif\n"
        ".if OC_LEN > 2 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rbx, %%r9, 2 * D_BYTES), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rbx, %%r9, 2 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm4\n"
        "addps %%xmm15, %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm6\n"
        "addps %%xmm15, %%xmm7\n"
        ".endif\n"
        ".if OC_LEN > 4 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx, %%r9, 1 * D_BYTES), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx, %%r9, 1 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm8\n"
        "addps %%xmm15, %%xmm9\n"
        ".endif\n"
        ".if OC_LEN > 5 * CH_DT_BLK\n"
        "movups ((0 * CH_DT_BLK + 0 * CH_RF_BLK) * D_BYTES)(%%rcx, %%r9, 2 * D_BYTES), %%xmm14\n"
        "movups ((0 * CH_DT_BLK + 1 * CH_RF_BLK) * D_BYTES)(%%rcx, %%r9, 2 * D_BYTES), %%xmm15\n"
        "mulps %%xmm12, %%xmm14\n"
        "mulps %%xmm12, %%xmm15\n"
        "addps %%xmm14, %%xmm10\n"
        "addps %%xmm15, %%xmm11\n"
        ".endif\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n lea (CH_DT_BLK * D_BYTES)(%%rbx), %%rbx\n .endif\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n lea (CH_DT_BLK * D_BYTES)(%%rcx), %%rcx\n .endif\n"
        "lea D_BYTES(%%rax), %%rax\n"
        "sub $1, %%r10\n"
        "cmp $0, %%r10\n"
        "jne 6b\n" // label_ic_remain

"7:\n" // label_finalize_session
        "test $(KERNEL_FLAG_RELU | KERNEL_FLAG_RELU6), %%r11\n"
        "jz 8f\n" // label_relu_end
        "xorps %%xmm12, %%xmm12\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "maxps %%xmm12, %%xmm0\n"
        "maxps %%xmm12, %%xmm1\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "maxps %%xmm12, %%xmm2\n"
        "maxps %%xmm12, %%xmm3\n"
        ".endif\n"
        ".if OC_LEN > 2 * CH_DT_BLK\n"
        "maxps %%xmm12, %%xmm4\n"
        "maxps %%xmm12, %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n"
        "maxps %%xmm12, %%xmm6\n"
        "maxps %%xmm12, %%xmm7\n"
        ".endif\n"
        ".if OC_LEN > 4 * CH_DT_BLK\n"
        "maxps %%xmm12, %%xmm8\n"
        "maxps %%xmm12, %%xmm9\n"
        ".endif\n"
        ".if OC_LEN > 5 * CH_DT_BLK\n"
        "maxps %%xmm12, %%xmm10\n"
        "maxps %%xmm12, %%xmm11\n"
        ".endif\n"
        "test $KERNEL_FLAG_RELU6, %%r11\n"
        "jz 8f\n" // label_relu_end
        "movups (%[six]), %%xmm13\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "minps %%xmm13, %%xmm0\n"
        "minps %%xmm13, %%xmm1\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "minps %%xmm13, %%xmm2\n"
        "minps %%xmm13, %%xmm3\n"
        ".endif\n"
        ".if OC_LEN > 2 * CH_DT_BLK\n"
        "minps %%xmm13, %%xmm4\n"
        "minps %%xmm13, %%xmm5\n"
        ".endif\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n"
        "minps %%xmm13, %%xmm6\n"
        "minps %%xmm13, %%xmm7\n"
        ".endif\n"
        ".if OC_LEN > 4 * CH_DT_BLK\n"
        "minps %%xmm13, %%xmm8\n"
        "minps %%xmm13, %%xmm9\n"
        ".endif\n"
        ".if OC_LEN > 5 * CH_DT_BLK\n"
        "minps %%xmm13, %%xmm10\n"
        "minps %%xmm13, %%xmm11\n"
        ".endif\n"
"8:\n" // label_relu_end
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "mov DST_OCB_STRIDE_IDX(%[shar_param]), %%r10\n"
        ".endif\n"
        ".if NT_STORE\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movntps %%xmm0, (0 * CH_RF_BLK * D_BYTES)(%%r14)\n"
        "movntps %%xmm1, (1 * CH_RF_BLK * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "lea (%%r14, %%r10, D_BYTES), %%rax\n"
        "movntps %%xmm2, (0 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        "movntps %%xmm3, (1 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        ".endif\n"
        ".if OC_LEN > 2 * CH_DT_BLK\n"
        "lea (%%r14, %%r10, 2 * D_BYTES), %%rbx\n"
        "movntps %%xmm4, (0 * CH_RF_BLK * D_BYTES)(%%rbx)\n"
        "movntps %%xmm5, (1 * CH_RF_BLK * D_BYTES)(%%rbx)\n"
        ".endif\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n"
        "lea (%%rax, %%r10, 2 * D_BYTES), %%rax\n"
        "movntps %%xmm6, (0 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        "movntps %%xmm7, (1 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        ".endif\n"
        ".if OC_LEN > 4 * CH_DT_BLK\n"
        "lea (%%rbx, %%r10, 2 * D_BYTES), %%rbx\n"
        "movntps %%xmm8, (0 * CH_RF_BLK * D_BYTES)(%%rbx)\n"
        "movntps %%xmm9, (1 * CH_RF_BLK * D_BYTES)(%%rbx)\n"
        ".endif\n"
        ".if OC_LEN > 5 * CH_DT_BLK\n"
        "lea (%%rax, %%r10, 2 * D_BYTES), %%rax\n"
        "movntps %%xmm10, (0 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        "movntps %%xmm11, (1 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        ".endif\n"
        ".else\n"
        ".if OC_LEN > 0 * CH_DT_BLK\n"
        "movups %%xmm0, (0 * CH_RF_BLK * D_BYTES)(%%r14)\n"
        "movups %%xmm1, (1 * CH_RF_BLK * D_BYTES)(%%r14)\n"
        ".endif\n"
        ".if OC_LEN > 1 * CH_DT_BLK\n"
        "lea (%%r14, %%r10, D_BYTES), %%rax\n"
        "movups %%xmm2, (0 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        "movups %%xmm3, (1 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        ".endif\n"
        ".if OC_LEN > 2 * CH_DT_BLK\n"
        "lea (%%r14, %%r10, 2 * D_BYTES), %%rbx\n"
        "movups %%xmm4, (0 * CH_RF_BLK * D_BYTES)(%%rbx)\n"
        "movups %%xmm5, (1 * CH_RF_BLK * D_BYTES)(%%rbx)\n"
        ".endif\n"
        ".if OC_LEN > 3 * CH_DT_BLK\n"
        "lea (%%rax, %%r10, 2 * D_BYTES), %%rax\n"
        "movups %%xmm6, (0 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        "movups %%xmm7, (1 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        ".endif\n"
        ".if OC_LEN > 4 * CH_DT_BLK\n"
        "lea (%%rbx, %%r10, 2 * D_BYTES), %%rbx\n"
        "movups %%xmm8, (0 * CH_RF_BLK * D_BYTES)(%%rbx)\n"
        "movups %%xmm9, (1 * CH_RF_BLK * D_BYTES)(%%rbx)\n"
        ".endif\n"
        ".if OC_LEN > 5 * CH_DT_BLK\n"
        "lea (%%rax, %%r10, 2 * D_BYTES), %%rax\n"
        "movups %%xmm10, (0 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        "movups %%xmm11, (1 * CH_RF_BLK * D_BYTES)(%%rax)\n"
        ".endif\n"
        ".endif\n"
        "sub $1, %%r15\n"
        "cmp $0, %%r15\n"
        "lea (CH_DT_BLK * D_BYTES)(%%r12), %%r12\n"
        "lea (CH_DT_BLK * D_BYTES)(%%r13), %%r13\n"
        "lea (CH_DT_BLK * D_BYTES)(%%r14), %%r14\n"
        "jne 1b\n" // label_init_session
        :
        :
        [priv_param]                  "r" (priv_param),
        [shar_param]                  "r" (shar_param),
        [six]                         "r" (six),
        [NT_STORE]                    "i" (nt_store),
        [OC_LEN]                      "i" (oc_len),
        [KERNEL_FLAG_LD_BIAS]         "i" (KERNEL_FLAG_LD_BIAS()),
        [KERNEL_FLAG_AD_BIAS]         "i" (KERNEL_FLAG_AD_BIAS()),
        [KERNEL_FLAG_RELU]            "i" (KERNEL_FLAG_RELU()),
        [KERNEL_FLAG_RELU6]           "i" (KERNEL_FLAG_RELU6())
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

template <bool nt_store, int32_t oc_len>
void conv2d_n8cx_gemm_direct_fp32_sse_blk1x1_kernel(
    const int64_t *priv_param,
    const int64_t *shar_param)
{

#ifdef PPL_USE_X86_INLINE_ASM
    conv2d_n8cx_gemm_direct_fp32_sse_blk1x1_kernel_core<nt_store, oc_len>(priv_param, shar_param);
    return;
#endif

#define IC_COMPUTE_STEP(IC) do {\
    xmm12 = _mm_set1_ps(ic_src[(IC)]);\
    if (oc_len > 0 * CH_DT_BLK()) {\
        xmm14 = _mm_loadu_ps(ic_flt_o24 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
        xmm15 = _mm_loadu_ps(ic_flt_o24 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm0 = _mm_add_ps(xmm0, xmm14);\
        xmm1 = _mm_add_ps(xmm1, xmm15);\
    }\
    if (oc_len > 1 * CH_DT_BLK()) {\
        xmm14 = _mm_loadu_ps(ic_flt_o24 + 1 * flt_ocb_stride + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
        xmm15 = _mm_loadu_ps(ic_flt_o24 + 1 * flt_ocb_stride + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm2 = _mm_add_ps(xmm2, xmm14);\
        xmm3 = _mm_add_ps(xmm3, xmm15);\
    }\
    if (oc_len > 2 * CH_DT_BLK()) {\
        xmm14 = _mm_loadu_ps(ic_flt_o24 + 2 * flt_ocb_stride + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
        xmm15 = _mm_loadu_ps(ic_flt_o24 + 2 * flt_ocb_stride + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm4 = _mm_add_ps(xmm4, xmm14);\
        xmm5 = _mm_add_ps(xmm5, xmm15);\
    }\
    if (oc_len > 3 * CH_DT_BLK()) {\
        xmm14 = _mm_loadu_ps(ic_flt_o48 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
        xmm15 = _mm_loadu_ps(ic_flt_o48 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm6 = _mm_add_ps(xmm6, xmm14);\
        xmm7 = _mm_add_ps(xmm7, xmm15);\
    }\
    if (oc_len > 4 * CH_DT_BLK()) {\
        xmm14 = _mm_loadu_ps(ic_flt_o48 + 1 * flt_ocb_stride + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
        xmm15 = _mm_loadu_ps(ic_flt_o48 + 1 * flt_ocb_stride + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm8 = _mm_add_ps(xmm8, xmm14);\
        xmm9 = _mm_add_ps(xmm9, xmm15);\
    }\
    if (oc_len > 5 * CH_DT_BLK()) {\
        xmm14 = _mm_loadu_ps(ic_flt_o48 + 2 * flt_ocb_stride + (IC) * CH_DT_BLK() + 0 * CH_RF_BLK());\
        xmm15 = _mm_loadu_ps(ic_flt_o48 + 2 * flt_ocb_stride + (IC) * CH_DT_BLK() + 1 * CH_RF_BLK());\
        xmm14 = _mm_mul_ps(xmm14, xmm12);\
        xmm15 = _mm_mul_ps(xmm15, xmm12);\
        xmm10 = _mm_add_ps(xmm10, xmm14);\
        xmm11 = _mm_add_ps(xmm11, xmm15);\
    }\
} while (0)

#define IC_PREFETCH_STEP(IC)  do {\
    if (oc_len > 1 * CH_DT_BLK()) {\
        if (oc_len > 0 * CH_DT_BLK()) _mm_prefetch((const char*)ic_flt_o24 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
        if (oc_len > 1 * CH_DT_BLK()) _mm_prefetch((const char*)ic_flt_o24 + 1 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
        if (oc_len > 2 * CH_DT_BLK()) _mm_prefetch((const char*)ic_flt_o24 + 2 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
        if (oc_len > 3 * CH_DT_BLK()) _mm_prefetch((const char*)ic_flt_o48 + 0 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
        if (oc_len > 4 * CH_DT_BLK()) _mm_prefetch((const char*)ic_flt_o48 + 1 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
        if (oc_len > 5 * CH_DT_BLK()) _mm_prefetch((const char*)ic_flt_o48 + 2 * flt_ocb_stride + (IC) * CH_DT_BLK() + CH_DT_BLK() * CH_DT_BLK(), _MM_HINT_T0);\
    }\
} while (0)

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;

    const int64_t src_icb_stride = shar_param[SRC_ICB_STRIDE_IDX()];
    const int64_t flt_ocb_stride = shar_param[FLT_OCB_STRIDE_IDX()];
    const int64_t kernel_flags = shar_param[FLAGS_IDX()];

    const float *src = PICK_PARAM(const float*, priv_param, SRC_IDX());
    const float *his = PICK_PARAM(const float*, priv_param, HIS_IDX());
    float *dst       = PICK_PARAM(float*, priv_param, DST_IDX());
    int64_t hw       = priv_param[HW_IDX()];
    do {
        if (kernel_flags & KERNEL_FLAG_LD_BIAS()) {
            const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
            if (oc_len > 0 * CH_DT_BLK()) {
                xmm0 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm1 = _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                xmm2 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm3 = _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                xmm4 = _mm_loadu_ps(bias + 2 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm5 = _mm_loadu_ps(bias + 2 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (oc_len > 3 * CH_DT_BLK()) {
                xmm6 = _mm_loadu_ps(bias + 3 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm7 = _mm_loadu_ps(bias + 3 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (oc_len > 4 * CH_DT_BLK()) {
                xmm8 = _mm_loadu_ps(bias + 4 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm9 = _mm_loadu_ps(bias + 4 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
            if (oc_len > 5 * CH_DT_BLK()) {
                xmm10 = _mm_loadu_ps(bias + 5 * CH_DT_BLK() + 0 * CH_RF_BLK());
                xmm11 = _mm_loadu_ps(bias + 5 * CH_DT_BLK() + 1 * CH_RF_BLK());
            }
        } else {
            const float *l_his = his;
            const int64_t his_ocb_stride = shar_param[HIS_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                xmm0 = _mm_loadu_ps(l_his + 0 * CH_RF_BLK());
                xmm1 = _mm_loadu_ps(l_his + 1 * CH_RF_BLK());
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_his += his_ocb_stride;
                xmm2 = _mm_loadu_ps(l_his + 0 * CH_RF_BLK());
                xmm3 = _mm_loadu_ps(l_his + 1 * CH_RF_BLK());
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                l_his += his_ocb_stride;
                xmm4 = _mm_loadu_ps(l_his + 0 * CH_RF_BLK());
                xmm5 = _mm_loadu_ps(l_his + 1 * CH_RF_BLK());
            }
            if (oc_len > 3 * CH_DT_BLK()) {
                l_his += his_ocb_stride;
                xmm6 = _mm_loadu_ps(l_his + 0 * CH_RF_BLK());
                xmm7 = _mm_loadu_ps(l_his + 1 * CH_RF_BLK());
            }
            if (oc_len > 4 * CH_DT_BLK()) {
                l_his += his_ocb_stride;
                xmm8 = _mm_loadu_ps(l_his + 0 * CH_RF_BLK());
                xmm9 = _mm_loadu_ps(l_his + 1 * CH_RF_BLK());
            }
            if (oc_len > 5 * CH_DT_BLK()) {
                l_his += his_ocb_stride;
                xmm10 = _mm_loadu_ps(l_his + 0 * CH_RF_BLK());
                xmm11 = _mm_loadu_ps(l_his + 1 * CH_RF_BLK());
            }
        }

        if (kernel_flags & KERNEL_FLAG_AD_BIAS()) {
            const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
            if (oc_len > 0 * CH_DT_BLK()) {
                xmm0 = _mm_add_ps(xmm0, _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 0 * CH_RF_BLK()));
                xmm1 = _mm_add_ps(xmm1, _mm_loadu_ps(bias + 0 * CH_DT_BLK() + 1 * CH_RF_BLK()));
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                xmm2 = _mm_add_ps(xmm2, _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 0 * CH_RF_BLK()));
                xmm3 = _mm_add_ps(xmm3, _mm_loadu_ps(bias + 1 * CH_DT_BLK() + 1 * CH_RF_BLK()));
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                xmm4 = _mm_add_ps(xmm4, _mm_loadu_ps(bias + 2 * CH_DT_BLK() + 0 * CH_RF_BLK()));
                xmm5 = _mm_add_ps(xmm5, _mm_loadu_ps(bias + 2 * CH_DT_BLK() + 1 * CH_RF_BLK()));
            }
            if (oc_len > 3 * CH_DT_BLK()) {
                xmm6 = _mm_add_ps(xmm6, _mm_loadu_ps(bias + 3 * CH_DT_BLK() + 0 * CH_RF_BLK()));
                xmm7 = _mm_add_ps(xmm7, _mm_loadu_ps(bias + 3 * CH_DT_BLK() + 1 * CH_RF_BLK()));
            }
            if (oc_len > 4 * CH_DT_BLK()) {
                xmm8 = _mm_add_ps(xmm8, _mm_loadu_ps(bias + 4 * CH_DT_BLK() + 0 * CH_RF_BLK()));
                xmm9 = _mm_add_ps(xmm9, _mm_loadu_ps(bias + 4 * CH_DT_BLK() + 1 * CH_RF_BLK()));
            }
            if (oc_len > 5 * CH_DT_BLK()) {
                xmm10 = _mm_add_ps(xmm10, _mm_loadu_ps(bias + 5 * CH_DT_BLK() + 0 * CH_RF_BLK()));
                xmm11 = _mm_add_ps(xmm11, _mm_loadu_ps(bias + 5 * CH_DT_BLK() + 1 * CH_RF_BLK()));
            }
        }
        
        const float *icb_src = src;
        const float *icb_flt = PICK_PARAM(const float*, priv_param, FLT_IDX());
        int64_t channels     = shar_param[CHANNELS_IDX()];
        while (channels >= CH_DT_BLK()) {
            channels -= CH_DT_BLK();
            const float *ic_src = icb_src;
            const float *ic_flt_o24 = icb_flt + 0 * flt_ocb_stride;
            const float *ic_flt_o48 = icb_flt + 3 * flt_ocb_stride;
            IC_COMPUTE_STEP(0);
            IC_COMPUTE_STEP(1);
            IC_COMPUTE_STEP(2);
            IC_COMPUTE_STEP(3);
            IC_COMPUTE_STEP(4);
            IC_COMPUTE_STEP(5);
            IC_COMPUTE_STEP(6);
            IC_COMPUTE_STEP(7);
            icb_flt += CH_DT_BLK() * CH_DT_BLK();
            icb_src += src_icb_stride;
        }
        if (channels > 0) {
            const float *ic_src = icb_src;
            const float *ic_flt_o24 = icb_flt + 0 * flt_ocb_stride;
            const float *ic_flt_o48 = icb_flt + 3 * flt_ocb_stride;
            for (int64_t ic = 0; ic < channels; ++ic) {
                IC_COMPUTE_STEP(0);
                ic_src += 1;
                ic_flt_o24 += CH_DT_BLK();
                ic_flt_o48 += CH_DT_BLK();
            }
        }
        
        if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
            xmm14 = _mm_setzero_ps();
            if (oc_len > 0 * CH_DT_BLK()) {
                xmm0 = _mm_max_ps(xmm0, xmm14);
                xmm1 = _mm_max_ps(xmm1, xmm14);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                xmm2 = _mm_max_ps(xmm2, xmm14);
                xmm3 = _mm_max_ps(xmm3, xmm14);
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                xmm4 = _mm_max_ps(xmm4, xmm14);
                xmm5 = _mm_max_ps(xmm5, xmm14);
            }
            if (oc_len > 3 * CH_DT_BLK()) {
                xmm6 = _mm_max_ps(xmm6, xmm14);
                xmm7 = _mm_max_ps(xmm7, xmm14);
            }
            if (oc_len > 4 * CH_DT_BLK()) {
                xmm8 = _mm_max_ps(xmm8, xmm14);
                xmm9 = _mm_max_ps(xmm9, xmm14);
            }
            if (oc_len > 5 * CH_DT_BLK()) {
                xmm10 = _mm_max_ps(xmm10, xmm14);
                xmm11 = _mm_max_ps(xmm11, xmm14);
            }
        }
        if (kernel_flags & KERNEL_FLAG_RELU6()) {
            xmm13 = _mm_set1_ps(6.0f);
            if (oc_len > 0 * CH_DT_BLK()) {
                xmm0 = _mm_min_ps(xmm0, xmm13);
                xmm1 = _mm_min_ps(xmm1, xmm13);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                xmm2 = _mm_min_ps(xmm2, xmm13);
                xmm3 = _mm_min_ps(xmm3, xmm13);
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                xmm4 = _mm_min_ps(xmm4, xmm13);
                xmm5 = _mm_min_ps(xmm5, xmm13);
            }
            if (oc_len > 3 * CH_DT_BLK()) {
                xmm6 = _mm_min_ps(xmm6, xmm13);
                xmm7 = _mm_min_ps(xmm7, xmm13);
            }
            if (oc_len > 4 * CH_DT_BLK()) {
                xmm8 = _mm_min_ps(xmm8, xmm13);
                xmm9 = _mm_min_ps(xmm9, xmm13);
            }
            if (oc_len > 5 * CH_DT_BLK()) {
                xmm10 = _mm_min_ps(xmm10, xmm13);
                xmm11 = _mm_min_ps(xmm11, xmm13);
            }
        }

        if (nt_store) {
            float* l_dst = dst;
            const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                _mm_stream_ps(l_dst + 0 * CH_RF_BLK(), xmm0);
                _mm_stream_ps(l_dst + 1 * CH_RF_BLK(), xmm1);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                _mm_stream_ps(l_dst + 0 * CH_RF_BLK(), xmm2);
                _mm_stream_ps(l_dst + 1 * CH_RF_BLK(), xmm3);
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                _mm_stream_ps(l_dst + 0 * CH_RF_BLK(), xmm4);
                _mm_stream_ps(l_dst + 1 * CH_RF_BLK(), xmm5);
            }
            if (oc_len > 3 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                _mm_stream_ps(l_dst + 0 * CH_RF_BLK(), xmm6);
                _mm_stream_ps(l_dst + 1 * CH_RF_BLK(), xmm7);
            }
            if (oc_len > 4 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                _mm_stream_ps(l_dst + 0 * CH_RF_BLK(), xmm8);
                _mm_stream_ps(l_dst + 1 * CH_RF_BLK(), xmm9);
            }
            if (oc_len > 5 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                _mm_stream_ps(l_dst + 0 * CH_RF_BLK(), xmm10);
                _mm_stream_ps(l_dst + 1 * CH_RF_BLK(), xmm11);
            }
        } else {
            float* l_dst = dst;
            const int64_t dst_ocb_stride = shar_param[DST_OCB_STRIDE_IDX()];
            if (oc_len > 0 * CH_DT_BLK()) {
                _mm_storeu_ps(l_dst + 0 * CH_RF_BLK(), xmm0);
                _mm_storeu_ps(l_dst + 1 * CH_RF_BLK(), xmm1);
            }
            if (oc_len > 1 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                _mm_storeu_ps(l_dst + 0 * CH_RF_BLK(), xmm2);
                _mm_storeu_ps(l_dst + 1 * CH_RF_BLK(), xmm3);
            }
            if (oc_len > 2 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                _mm_storeu_ps(l_dst + 0 * CH_RF_BLK(), xmm4);
                _mm_storeu_ps(l_dst + 1 * CH_RF_BLK(), xmm5);
            }
            if (oc_len > 3 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                _mm_storeu_ps(l_dst + 0 * CH_RF_BLK(), xmm6);
                _mm_storeu_ps(l_dst + 1 * CH_RF_BLK(), xmm7);
            }
            if (oc_len > 4 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                _mm_storeu_ps(l_dst + 0 * CH_RF_BLK(), xmm8);
                _mm_storeu_ps(l_dst + 1 * CH_RF_BLK(), xmm9);
            }
            if (oc_len > 5 * CH_DT_BLK()) {
                l_dst += dst_ocb_stride;
                _mm_storeu_ps(l_dst + 0 * CH_RF_BLK(), xmm10);
                _mm_storeu_ps(l_dst + 1 * CH_RF_BLK(), xmm11);
            }
        }
        src += CH_DT_BLK();
        his += CH_DT_BLK();
        dst += CH_DT_BLK();
        hw -= 1;
    } while (hw > 0);
#undef IC_COMPUTE_STEP
#undef IC_PREFETCH_STEP
}

#define GEMM_DIRECT_HW3_KERNEL_TABLE_BLK(NT_STORE) \
    {\
        conv2d_n8cx_gemm_direct_fp32_sse_blk1x3_kernel<NT_STORE, 1 * CH_DT_BLK(), 3>,\
        conv2d_n8cx_gemm_direct_fp32_sse_blk1x3_kernel<NT_STORE, 2 * CH_DT_BLK(), 3>,\
    }

#define GEMM_DIRECT_HW1_KERNEL_TABLE_BLK(NT_STORE) \
    {\
        conv2d_n8cx_gemm_direct_fp32_sse_blk1x1_kernel<NT_STORE, 1 * CH_DT_BLK()>,\
        conv2d_n8cx_gemm_direct_fp32_sse_blk1x1_kernel<NT_STORE, 2 * CH_DT_BLK()>,\
        conv2d_n8cx_gemm_direct_fp32_sse_blk1x1_kernel<NT_STORE, 3 * CH_DT_BLK()>,\
        conv2d_n8cx_gemm_direct_fp32_sse_blk1x1_kernel<NT_STORE, 4 * CH_DT_BLK()>,\
        conv2d_n8cx_gemm_direct_fp32_sse_blk1x1_kernel<NT_STORE, 5 * CH_DT_BLK()>,\
        conv2d_n8cx_gemm_direct_fp32_sse_blk1x1_kernel<NT_STORE, 6 * CH_DT_BLK()>,\
    }

conv2d_n8cx_gemm_direct_kernel_fp32_sse_func_t
conv2d_n8cx_gemm_direct_kernel_fp32_sse_hw3_table[NT_STORE_OPT()][BLK1X3_OC_RF() / 2] =
{
    GEMM_DIRECT_HW3_KERNEL_TABLE_BLK(false),
    GEMM_DIRECT_HW3_KERNEL_TABLE_BLK(true),
};

conv2d_n8cx_gemm_direct_kernel_fp32_sse_func_t
conv2d_n8cx_gemm_direct_kernel_fp32_sse_hw1_table[NT_STORE_OPT()][BLK1X1_OC_RF() / 2] =
{
    GEMM_DIRECT_HW1_KERNEL_TABLE_BLK(false),
    GEMM_DIRECT_HW1_KERNEL_TABLE_BLK(true),
};

}}};
