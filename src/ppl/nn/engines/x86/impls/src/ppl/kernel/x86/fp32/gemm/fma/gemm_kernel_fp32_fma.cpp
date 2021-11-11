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

#include "ppl/kernel/x86/fp32/gemm/fma/gemm_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

#ifdef PPL_USE_X86_INLINE_ASM

template<int64_t u_m, int64_t u_n>
void gemm_fp32_fma_kernel_core(int64_t *param)
{
    __asm__ __volatile__ (
        ".equ P_BYTES, 8\n"
        ".equ D_BYTES, 4\n"
        ".equ LOG2_D_BYTES, 2\n"

        ".equ A_PTR_IDX,        (0 * P_BYTES)\n"
        ".equ PACKED_B_PTR_IDX, (1 * P_BYTES)\n"
        ".equ C_PTR_IDX,        (2 * P_BYTES)\n"
        ".equ N_IDX,            (3 * P_BYTES)\n"
        ".equ K_IDX,            (4 * P_BYTES)\n"
        ".equ LDA_IDX,          (5 * P_BYTES)\n"
        ".equ LDPACKED_B_IDX,   (6 * P_BYTES)\n"
        ".equ LDC_IDX,          (7 * P_BYTES)\n"
        ".equ ALPHA_IDX,        (8 * P_BYTES)\n"
        ".equ FLAGS_IDX,        (9 * P_BYTES)\n"

        ".equ N_REG_ELTS, %c[N_REG_ELTS]\n"
        ".equ U_M, %c[U_M]\n"
        ".equ U_N, %c[U_N]\n"
        ".equ U_NR, ((U_N + N_REG_ELTS - 1) / N_REG_ELTS)\n"
        ".equ U_K, %c[U_K]\n"
        ".equ KERNEL_FLAG_LOAD_C, %c[KERNEL_FLAG_LOAD_C]\n"
        ".equ KERNEL_FLAG_RELU, %c[KERNEL_FLAG_RELU]\n"
        ".equ KERNEL_FLAG_RELU6, %c[KERNEL_FLAG_RELU6]\n"

        "mov PACKED_B_PTR_IDX(%[param]), %%r15\n"
        "mov C_PTR_IDX(%[param]),        %%r14\n"
        "mov N_IDX(%[param]),            %%r13\n"
        "mov LDPACKED_B_IDX(%[param]),   %%r12\n"
        "shl $LOG2_D_BYTES, %%r12\n"

"1:\n" // label_init_session
        ".if U_M > 0\n"
        ".if U_NR > 0\n vxorps %%ymm0, %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm1, %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm2, %%ymm2, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vxorps %%ymm3, %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm4, %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm5, %%ymm5, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vxorps %%ymm6, %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm7, %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm8, %%ymm8, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vxorps %%ymm9, %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vxorps %%ymm10, %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vxorps %%ymm11, %%ymm11, %%ymm11\n .endif\n"
        ".endif\n"

"3:\n" // label_compute_session
        "mov %%r15, %%rbx\n" // packed_b_ptr -> kpacked_b
        "mov LDA_IDX(%[param]), %%r11\n"
        "shl $LOG2_D_BYTES, %%r11\n"
        "mov K_IDX(%[param]), %%r10\n"
        ".if U_M > 0\n mov A_PTR_IDX(%[param]), %%r9\n .endif\n" // ka_m0
        ".if U_M > 2\n lea (%%r9, %%r11, 2), %%r8\n .endif\n" // ka_m2
        "cmp $U_K, %%r10\n"
        "jl 5f\n" // label_k_remain
        PPL_X86_INLINE_ASM_ALIGN()
"4:\n" // label_k_body
        ".irp K,0,1,2,3,4,5,6,7\n"
        ".if U_NR > 0\n vmovups (\\K * N_REG_ELTS * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vmovups (\\K * N_REG_ELTS * D_BYTES)(%%rbx, %%r12, 1), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vmovups (\\K * N_REG_ELTS * D_BYTES)(%%rbx, %%r12, 2), %%ymm14\n .endif\n"
        ".if U_M > 0\n"
        "vbroadcastss (\\K * D_BYTES)(%%r9), %%ymm15\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm12, %%ymm15, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm13, %%ymm15, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm14, %%ymm15, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        "vbroadcastss (\\K * D_BYTES)(%%r9, %%r11, 1), %%ymm15\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm12, %%ymm15, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm13, %%ymm15, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm14, %%ymm15, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        "vbroadcastss (\\K * D_BYTES)(%%r8), %%ymm15\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm13, %%ymm15, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm14, %%ymm15, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        "vbroadcastss (\\K * D_BYTES)(%%r8, %%r11, 1), %%ymm15\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm12, %%ymm15, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm13, %%ymm15, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm14, %%ymm15, %%ymm11\n .endif\n"
        ".endif\n"
        ".endr\n" // .irp K
        ".if U_M > 0\n lea (U_K * D_BYTES)(%%r9), %%r9\n .endif\n"
        ".if U_M > 2\n lea (U_K * D_BYTES)(%%r8), %%r8\n .endif\n"
        "lea (U_K * N_REG_ELTS * D_BYTES)(%%rbx), %%rbx\n"
        "sub $U_K, %%r10\n"
        "cmp $U_K, %%r10\n"
        "jge 4b\n" // label_k_body
        "cmp $0, %%r10\n"
        "je 6f\n" // label_finalize_session
"5:\n" // label_k_remain
        ".if U_NR > 0\n vmovups (0 * N_REG_ELTS * D_BYTES)(%%rbx), %%ymm12\n .endif\n"
        ".if U_NR > 1\n vmovups (0 * N_REG_ELTS * D_BYTES)(%%rbx, %%r12, 1), %%ymm13\n .endif\n"
        ".if U_NR > 2\n vmovups (0 * N_REG_ELTS * D_BYTES)(%%rbx, %%r12, 2), %%ymm14\n .endif\n"
        ".if U_M > 0\n"
        "vbroadcastss (0 * D_BYTES)(%%r9), %%ymm15\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm12, %%ymm15, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm13, %%ymm15, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm14, %%ymm15, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        "vbroadcastss (0 * D_BYTES)(%%r9, %%r11), %%ymm15\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm12, %%ymm15, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm13, %%ymm15, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm14, %%ymm15, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        "vbroadcastss (0 * D_BYTES)(%%r8), %%ymm15\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm13, %%ymm15, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm14, %%ymm15, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        "vbroadcastss (0 * D_BYTES)(%%r8, %%r11), %%ymm15\n"
        ".if U_NR > 0\n vfmadd231ps %%ymm12, %%ymm15, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vfmadd231ps %%ymm13, %%ymm15, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vfmadd231ps %%ymm14, %%ymm15, %%ymm11\n .endif\n"
        ".endif\n"
        ".if U_M > 0\n lea (D_BYTES)(%%r9), %%r9\n .endif\n"
        ".if U_M > 2\n lea (D_BYTES)(%%r8), %%r8\n .endif\n"
        "lea (N_REG_ELTS * D_BYTES)(%%rbx), %%rbx\n"
        "sub $1, %%r10\n"
        "cmp $0, %%r10\n"
        "jne 5b\n" // label_k_remain

"6:\n" // label_finalize_session
        "mov ALPHA_IDX(%[param]), %%ecx\n"
        "cmp $0x3f800000, %%ecx\n" // alpha == 1.0f
        "je 7f\n" // label_apply_alpha_end
        "vbroadcastss ALPHA_IDX(%[param]), %%ymm15\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vmulps %%ymm15, %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vmulps %%ymm15, %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vmulps %%ymm15, %%ymm2, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vmulps %%ymm15, %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vmulps %%ymm15, %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vmulps %%ymm15, %%ymm5, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vmulps %%ymm15, %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vmulps %%ymm15, %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vmulps %%ymm15, %%ymm8, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vmulps %%ymm15, %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vmulps %%ymm15, %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vmulps %%ymm15, %%ymm11, %%ymm11\n .endif\n"
        ".endif\n"
"7:\n" // label_apply_alpha_end
        "mov FLAGS_IDX(%[param]),  %%r11\n"
        "test $KERNEL_FLAG_LOAD_C, %%r11\n"
        "jz 8f\n" // label_load_c_end
        "mov %%r14, %%r10\n" // c_ptr -> l_c
        ".if U_M > 0\n"
        ".if U_NR > 0\n vaddps (0 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vaddps (1 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vaddps (2 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm2, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        "mov LDC_IDX(%[param]), %%r9\n"
        "lea (%%r10, %%r9, D_BYTES), %%r10\n"
        ".if U_NR > 0\n vaddps (0 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vaddps (1 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vaddps (2 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm5, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        "lea (%%r10, %%r9, D_BYTES), %%r10\n"
        ".if U_NR > 0\n vaddps (0 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vaddps (1 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vaddps (2 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm8, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        "lea (%%r10, %%r9, D_BYTES), %%r10\n"
        ".if U_NR > 0\n vaddps (0 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vaddps (1 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vaddps (2 * N_REG_ELTS * D_BYTES)(%%r10), %%ymm11, %%ymm11\n .endif\n"
        ".endif\n"
"8:\n" // label_load_c_end
        "test $(KERNEL_FLAG_RELU | KERNEL_FLAG_RELU6), %%r11\n"
        "jz 9f\n" // label_relu_end
        "vxorps %%ymm15, %%ymm15, %%ymm15\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vmaxps %%ymm15, %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vmaxps %%ymm15, %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vmaxps %%ymm15, %%ymm2, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vmaxps %%ymm15, %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vmaxps %%ymm15, %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vmaxps %%ymm15, %%ymm5, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vmaxps %%ymm15, %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vmaxps %%ymm15, %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vmaxps %%ymm15, %%ymm8, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vmaxps %%ymm15, %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vmaxps %%ymm15, %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vmaxps %%ymm15, %%ymm11, %%ymm11\n .endif\n"
        ".endif\n"
        "test $KERNEL_FLAG_RELU6, %%r11\n"
        "jz 9f\n" // label_relu_end
        "mov $0x40c00000, %%ecx\n"
        "vmovd %%ecx, %%xmm15\n"
        "vbroadcastss %%xmm15, %%ymm15\n"
        ".if U_M > 0\n"
        ".if U_NR > 0\n vminps %%ymm15, %%ymm0, %%ymm0\n .endif\n"
        ".if U_NR > 1\n vminps %%ymm15, %%ymm1, %%ymm1\n .endif\n"
        ".if U_NR > 2\n vminps %%ymm15, %%ymm2, %%ymm2\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        ".if U_NR > 0\n vminps %%ymm15, %%ymm3, %%ymm3\n .endif\n"
        ".if U_NR > 1\n vminps %%ymm15, %%ymm4, %%ymm4\n .endif\n"
        ".if U_NR > 2\n vminps %%ymm15, %%ymm5, %%ymm5\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        ".if U_NR > 0\n vminps %%ymm15, %%ymm6, %%ymm6\n .endif\n"
        ".if U_NR > 1\n vminps %%ymm15, %%ymm7, %%ymm7\n .endif\n"
        ".if U_NR > 2\n vminps %%ymm15, %%ymm8, %%ymm8\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        ".if U_NR > 0\n vminps %%ymm15, %%ymm9, %%ymm9\n .endif\n"
        ".if U_NR > 1\n vminps %%ymm15, %%ymm10, %%ymm10\n .endif\n"
        ".if U_NR > 2\n vminps %%ymm15, %%ymm11, %%ymm11\n .endif\n"
        ".endif\n"
"9:\n" // label_relu_end

        "mov %%r14, %%r10\n" // c_ptr -> l_c
        ".if U_M > 0\n"
        ".if U_NR > 0\n vmovups %%ymm0, (0 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".if U_NR > 1\n vmovups %%ymm1, (1 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".if U_NR > 2\n vmovups %%ymm2, (2 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".endif\n"
        ".if U_M > 1\n"
        "mov LDC_IDX(%[param]), %%r9\n"
        "lea (%%r10, %%r9, D_BYTES), %%r10\n"
        ".if U_NR > 0\n vmovups %%ymm3, (0 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".if U_NR > 1\n vmovups %%ymm4, (1 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".if U_NR > 2\n vmovups %%ymm5, (2 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".endif\n"
        ".if U_M > 2\n"
        "lea (%%r10, %%r9, D_BYTES), %%r10\n"
        ".if U_NR > 0\n vmovups %%ymm6, (0 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".if U_NR > 1\n vmovups %%ymm7, (1 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".if U_NR > 2\n vmovups %%ymm8, (2 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".endif\n"
        ".if U_M > 3\n"
        "lea (%%r10, %%r9, D_BYTES), %%r10\n"
        ".if U_NR > 0\n vmovups %%ymm9, (0 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".if U_NR > 1\n vmovups %%ymm10, (1 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".if U_NR > 2\n vmovups %%ymm11, (2 * N_REG_ELTS * D_BYTES)(%%r10)\n .endif\n"
        ".endif\n"
        ".if U_NR == 1 || U_NR == 3\n lea (%%r15, %%r12, 1), %%r15\n .endif\n" // packed_b_ptr += ldpacked_b
        ".if U_NR == 2 || U_NR == 3\n lea (%%r15, %%r12, 2), %%r15\n .endif\n" // packed_b_ptr += 2 * ldpacked_b
        "lea (U_N * D_BYTES)(%%r14), %%r14\n" // c_ptr += u_n
        "sub $U_N, %%r13\n" // n -= u_n
        "cmp $0, %%r13\n"
        "jne 1b\n" // label_init_session
        :
        :
        [param]                         "r" (param),
        [N_REG_ELTS]                    "i" (gemm_kernel_fp32_fma::config::N_REG_ELTS),
        [U_M]                           "i" (u_m),
        [U_N]                           "i" (u_n),
        [U_K]                           "i" (8),
        [KERNEL_FLAG_LOAD_C]            "i" (gemm_kernel_fp32_fma::flag::LOAD_C),
        [KERNEL_FLAG_RELU]              "i" (gemm_kernel_fp32_fma::flag::RELU),
        [KERNEL_FLAG_RELU6]             "i" (gemm_kernel_fp32_fma::flag::RELU6)
        :
        "cc",
        "rax", "rbx", "rcx", "rdx",
        "r8" , "r9" , "r10", "r11",
        "r12", "r13", "r14", "r15",
        "ymm0" , "ymm1" , "ymm2" , "ymm3" , "ymm4" , "ymm5" , "ymm6" , "ymm7" ,
        "ymm8" , "ymm9" , "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
        "memory"
    );
}

#endif

template<int64_t u_m, int64_t u_n>
void gemm_fp32_fma_kernel(int64_t *param)
{

#ifdef PPL_USE_X86_INLINE_ASM
    gemm_fp32_fma_kernel_core<u_m, u_n>(param);
    return;
#endif

#define K_COMPUTE_STEP(K) do {\
    if (u_nr > 0) ymm13 = _mm256_loadu_ps(kpacked_b + 0 * ldpacked_b + K * n_reg_elts);\
    if (u_nr > 1) ymm14 = _mm256_loadu_ps(kpacked_b + 1 * ldpacked_b + K * n_reg_elts);\
    if (u_nr > 2) ymm15 = _mm256_loadu_ps(kpacked_b + 2 * ldpacked_b + K * n_reg_elts);\
    if (u_m > 0) {\
        ymm12 = _mm256_set1_ps(ka_m0[0 * lda + K]);\
        if (u_nr > 0) ymm0 = _mm256_fmadd_ps(ymm13, ymm12, ymm0);\
        if (u_nr > 1) ymm1 = _mm256_fmadd_ps(ymm14, ymm12, ymm1);\
        if (u_nr > 2) ymm2 = _mm256_fmadd_ps(ymm15, ymm12, ymm2);\
    }\
    if (u_m > 1) {\
        ymm12 = _mm256_set1_ps(ka_m0[1 * lda + K]);\
        if (u_nr > 0) ymm3 = _mm256_fmadd_ps(ymm13, ymm12, ymm3);\
        if (u_nr > 1) ymm4 = _mm256_fmadd_ps(ymm14, ymm12, ymm4);\
        if (u_nr > 2) ymm5 = _mm256_fmadd_ps(ymm15, ymm12, ymm5);\
    }\
    if (u_m > 2) {\
        ymm12 = _mm256_set1_ps(ka_m2[0 * lda + K]);\
        if (u_nr > 0) ymm6 = _mm256_fmadd_ps(ymm13, ymm12, ymm6);\
        if (u_nr > 1) ymm7 = _mm256_fmadd_ps(ymm14, ymm12, ymm7);\
        if (u_nr > 2) ymm8 = _mm256_fmadd_ps(ymm15, ymm12, ymm8);\
    }\
    if (u_m > 3) {\
        ymm12 = _mm256_set1_ps(ka_m2[1 * lda + K]);\
        if (u_nr > 0) ymm9 = _mm256_fmadd_ps(ymm13, ymm12, ymm9);\
        if (u_nr > 1) ymm10 = _mm256_fmadd_ps(ymm14, ymm12, ymm10);\
        if (u_nr > 2) ymm11 = _mm256_fmadd_ps(ymm15, ymm12, ymm11);\
    }\
} while (0)

    array_param_helper kp(param);
    const int64_t n_reg_elts = gemm_kernel_fp32_fma::config::N_REG_ELTS;
    const int64_t u_nr = div_up(u_n, n_reg_elts);
    const int64_t u_k = gemm_kernel_fp32_fma::config::UNROLL_K;

    const gemm_kernel_fp32_fma::flag_t flags = kp.pick<const gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::FLAGS_IDX);
    const float *packed_b_ptr = kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::PACKED_B_PTR_IDX);
    float *c_ptr = kp.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX);
    const int64_t ldpacked_b = kp.pick<const int64_t>(gemm_kernel_fp32_fma::param_def::LDPACKED_B_IDX);
    int64_t n = kp.pick<int64_t>(gemm_kernel_fp32_fma::param_def::N_IDX);
    do {
        __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
        __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
        { // session - initialize
            if (u_m > 0) {
                if (u_nr > 0) ymm0 = _mm256_setzero_ps();
                if (u_nr > 1) ymm1 = _mm256_setzero_ps();
                if (u_nr > 2) ymm2 = _mm256_setzero_ps();
            }
            if (u_m > 1) {
                if (u_nr > 0) ymm3 = _mm256_setzero_ps();
                if (u_nr > 1) ymm4 = _mm256_setzero_ps();
                if (u_nr > 2) ymm5 = _mm256_setzero_ps();
            }
            if (u_m > 2) {
                if (u_nr > 0) ymm6 = _mm256_setzero_ps();
                if (u_nr > 1) ymm7 = _mm256_setzero_ps();
                if (u_nr > 2) ymm8 = _mm256_setzero_ps();
            }
            if (u_m > 3) {
                if (u_nr > 0) ymm9 = _mm256_setzero_ps();
                if (u_nr > 1) ymm10 = _mm256_setzero_ps();
                if (u_nr > 2) ymm11 = _mm256_setzero_ps();
            }
        }

        { // session - compute
            int64_t k = kp.pick<int64_t>(gemm_kernel_fp32_fma::param_def::K_IDX);
            const int64_t lda = kp.pick<const int64_t>(gemm_kernel_fp32_fma::param_def::LDA_IDX);
            const float *kpacked_b = packed_b_ptr;
            const float *ka_m0;
            const float *ka_m2;
            if (u_m > 0) ka_m0 = kp.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX);
            if (u_m > 2) ka_m2 = ka_m0 + 2 * lda;
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
                if (u_m > 0) ka_m0 += u_k;
                if (u_m > 2) ka_m2 += u_k;
                kpacked_b += u_k * n_reg_elts;
            }
            while (k > 0) {
                --k;
                K_COMPUTE_STEP(0);
                if (u_m > 0) ka_m0 += 1;
                if (u_m > 2) ka_m2 += 1;
                kpacked_b += n_reg_elts;
            }
        }
        
        { // session - finalize
            const float alpha = kp.pick<const float>(gemm_kernel_fp32_fma::param_def::ALPHA_IDX);
            if (alpha != 1.0f) {
                ymm12 = _mm256_set1_ps(alpha);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_mul_ps(ymm12, ymm0);
                    if (u_nr > 1) ymm1 = _mm256_mul_ps(ymm12, ymm1);
                    if (u_nr > 2) ymm2 = _mm256_mul_ps(ymm12, ymm2);
                }
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_mul_ps(ymm12, ymm3);
                    if (u_nr > 1) ymm4 = _mm256_mul_ps(ymm12, ymm4);
                    if (u_nr > 2) ymm5 = _mm256_mul_ps(ymm12, ymm5);
                }
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_mul_ps(ymm12, ymm6);
                    if (u_nr > 1) ymm7 = _mm256_mul_ps(ymm12, ymm7);
                    if (u_nr > 2) ymm8 = _mm256_mul_ps(ymm12, ymm8);
                }
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_mul_ps(ymm12, ymm9);
                    if (u_nr > 1) ymm10 = _mm256_mul_ps(ymm12, ymm10);
                    if (u_nr > 2) ymm11 = _mm256_mul_ps(ymm12, ymm11);
                }
            }

            if (flags & gemm_kernel_fp32_fma::flag::LOAD_C) {
                const int64_t ldc = kp.pick<const int64_t>(gemm_kernel_fp32_fma::param_def::LDC_IDX);
                const float *l_c = c_ptr;
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_add_ps(_mm256_loadu_ps(l_c + 0 * n_reg_elts), ymm0);
                    if (u_nr > 1) ymm1 = _mm256_add_ps(_mm256_loadu_ps(l_c + 1 * n_reg_elts), ymm1);
                    if (u_nr > 2) ymm2 = _mm256_add_ps(_mm256_loadu_ps(l_c + 2 * n_reg_elts), ymm2);
                }
                if (u_m > 1) {
                    l_c += ldc;
                    if (u_nr > 0) ymm3 = _mm256_add_ps(_mm256_loadu_ps(l_c + 0 * n_reg_elts), ymm3);
                    if (u_nr > 1) ymm4 = _mm256_add_ps(_mm256_loadu_ps(l_c + 1 * n_reg_elts), ymm4);
                    if (u_nr > 2) ymm5 = _mm256_add_ps(_mm256_loadu_ps(l_c + 2 * n_reg_elts), ymm5);
                }
                if (u_m > 2) {
                    l_c += ldc;
                    if (u_nr > 0) ymm6 = _mm256_add_ps(_mm256_loadu_ps(l_c + 0 * n_reg_elts), ymm6);
                    if (u_nr > 1) ymm7 = _mm256_add_ps(_mm256_loadu_ps(l_c + 1 * n_reg_elts), ymm7);
                    if (u_nr > 2) ymm8 = _mm256_add_ps(_mm256_loadu_ps(l_c + 2 * n_reg_elts), ymm8);
                }
                if (u_m > 3) {
                    l_c += ldc;
                    if (u_nr > 0) ymm9 = _mm256_add_ps(_mm256_loadu_ps(l_c + 0 * n_reg_elts), ymm9);
                    if (u_nr > 1) ymm10 = _mm256_add_ps(_mm256_loadu_ps(l_c + 1 * n_reg_elts), ymm10);
                    if (u_nr > 2) ymm11 = _mm256_add_ps(_mm256_loadu_ps(l_c + 2 * n_reg_elts), ymm11);
                }
            }

            if (flags & (gemm_kernel_fp32_fma::flag::RELU | gemm_kernel_fp32_fma::flag::RELU6)) {
                ymm13 = _mm256_setzero_ps();
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_max_ps(ymm13, ymm0);
                    if (u_nr > 1) ymm1 = _mm256_max_ps(ymm13, ymm1);
                    if (u_nr > 2) ymm2 = _mm256_max_ps(ymm13, ymm2);
                }
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_max_ps(ymm13, ymm3);
                    if (u_nr > 1) ymm4 = _mm256_max_ps(ymm13, ymm4);
                    if (u_nr > 2) ymm5 = _mm256_max_ps(ymm13, ymm5);
                }
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_max_ps(ymm13, ymm6);
                    if (u_nr > 1) ymm7 = _mm256_max_ps(ymm13, ymm7);
                    if (u_nr > 2) ymm8 = _mm256_max_ps(ymm13, ymm8);
                }
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_max_ps(ymm13, ymm9);
                    if (u_nr > 1) ymm10 = _mm256_max_ps(ymm13, ymm10);
                    if (u_nr > 2) ymm11 = _mm256_max_ps(ymm13, ymm11);
                }
            }

            if (flags & gemm_kernel_fp32_fma::flag::RELU6) {
                ymm12 = _mm256_set1_ps(6.0f);
                if (u_m > 0) {
                    if (u_nr > 0) ymm0 = _mm256_min_ps(ymm12, ymm0);
                    if (u_nr > 1) ymm1 = _mm256_min_ps(ymm12, ymm1);
                    if (u_nr > 2) ymm2 = _mm256_min_ps(ymm12, ymm2);
                }
                if (u_m > 1) {
                    if (u_nr > 0) ymm3 = _mm256_min_ps(ymm12, ymm3);
                    if (u_nr > 1) ymm4 = _mm256_min_ps(ymm12, ymm4);
                    if (u_nr > 2) ymm5 = _mm256_min_ps(ymm12, ymm5);
                }
                if (u_m > 2) {
                    if (u_nr > 0) ymm6 = _mm256_min_ps(ymm12, ymm6);
                    if (u_nr > 1) ymm7 = _mm256_min_ps(ymm12, ymm7);
                    if (u_nr > 2) ymm8 = _mm256_min_ps(ymm12, ymm8);
                }
                if (u_m > 3) {
                    if (u_nr > 0) ymm9 = _mm256_min_ps(ymm12, ymm9);
                    if (u_nr > 1) ymm10 = _mm256_min_ps(ymm12, ymm10);
                    if (u_nr > 2) ymm11 = _mm256_min_ps(ymm12, ymm11);
                }
            }

            const int64_t ldc = kp.pick<const int64_t>(gemm_kernel_fp32_fma::param_def::LDC_IDX);
            float *l_c = c_ptr;
            if (u_m > 0) {
                if (u_nr > 0) _mm256_storeu_ps(l_c + 0 * n_reg_elts, ymm0);
                if (u_nr > 1) _mm256_storeu_ps(l_c + 1 * n_reg_elts, ymm1);
                if (u_nr > 2) _mm256_storeu_ps(l_c + 2 * n_reg_elts, ymm2);
            }
            if (u_m > 1) {
                l_c += ldc;
                if (u_nr > 0) _mm256_storeu_ps(l_c + 0 * n_reg_elts, ymm3);
                if (u_nr > 1) _mm256_storeu_ps(l_c + 1 * n_reg_elts, ymm4);
                if (u_nr > 2) _mm256_storeu_ps(l_c + 2 * n_reg_elts, ymm5);
            }
            if (u_m > 2) {
                l_c += ldc;
                if (u_nr > 0) _mm256_storeu_ps(l_c + 0 * n_reg_elts, ymm6);
                if (u_nr > 1) _mm256_storeu_ps(l_c + 1 * n_reg_elts, ymm7);
                if (u_nr > 2) _mm256_storeu_ps(l_c + 2 * n_reg_elts, ymm8);
            }
            if (u_m > 3) {
                l_c += ldc;
                if (u_nr > 0) _mm256_storeu_ps(l_c + 0 * n_reg_elts, ymm9);
                if (u_nr > 1) _mm256_storeu_ps(l_c + 1 * n_reg_elts, ymm10);
                if (u_nr > 2) _mm256_storeu_ps(l_c + 2 * n_reg_elts, ymm11);
            }
        }
        { // next n block
            packed_b_ptr += u_nr * ldpacked_b;
            c_ptr += u_n;
            n -= u_n;
        }
    } while (n > 0);
#undef K_COMPUTE_STEP
}

const gemm_kernel_fp32_fma::func_t
    gemm_kernel_fp32_fma::table_[config::MAX_N_REGS][config::MAX_M_REGS] =
{
    {
        gemm_fp32_fma_kernel<1, 1 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
        gemm_fp32_fma_kernel<2, 1 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
        gemm_fp32_fma_kernel<3, 1 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
        gemm_fp32_fma_kernel<4, 1 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
    },
    {
        gemm_fp32_fma_kernel<1, 2 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
        gemm_fp32_fma_kernel<2, 2 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
        gemm_fp32_fma_kernel<3, 2 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
        gemm_fp32_fma_kernel<4, 2 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
    },
    {
        gemm_fp32_fma_kernel<1, 3 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
        gemm_fp32_fma_kernel<2, 3 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
        gemm_fp32_fma_kernel<3, 3 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
        gemm_fp32_fma_kernel<4, 3 * gemm_kernel_fp32_fma::config::N_REG_ELTS>,
    },
};

}}}; // namespace ppl::kernel::x86

