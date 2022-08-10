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

#ifndef PPL_ARM_SERVER_KERNEL_PP_STR
#define PPL_ARM_SERVER_KERNEL_PP_STR(X) #X
#endif
#ifndef PPL_ARM_SERVER_KERNEL_STR
#define PPL_ARM_SERVER_KERNEL_STR(X) PPL_ARM_SERVER_KERNEL_PP_STR(X)
#endif

template<>
void hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A(), PREFETCH_B(), INIT_T(), M_BLOCK(), N_BLOCK()>(
    const __fp16* A, 
    const __fp16* B, 
    const int64_t K, 
    const int64_t lda, 
    const int64_t ldb, 
    const int64_t ldc, 
    __fp16* C)
{
#define IF_M_GT(M, INSTRUCTION) ".if " PPL_ARM_SERVER_KERNEL_STR(M_BLOCK()) " > " #M "\n\t" INSTRUCTION ".endif\n\t"
#define IF_N_GT(N, INSTRUCTION) ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK()) " > " #N "\n\t" INSTRUCTION ".endif\n\t"
#define IF_MN_GT(M, N, INSTRUCTION) \
    ".if " PPL_ARM_SERVER_KERNEL_STR(M_BLOCK()) " > " #M "\n\t" \
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK()) " > " #N "\n\t" \
    INSTRUCTION \
    ".endif\n\t" \
    ".endif\n\t"
#define IF_PFC_A_M_GT(M, INSTRUCTION) \
    ".if " PPL_ARM_SERVER_KERNEL_STR(PREFETCH_A()) " > 0\n\t" \
    ".if " PPL_ARM_SERVER_KERNEL_STR(M_BLOCK()) " > " #M "\n\t" \
    INSTRUCTION \
    ".endif\n\t" \
    ".endif\n\t"
#define IF_PFC_B_N_GT(N, INSTRUCTION) \
    ".if " PPL_ARM_SERVER_KERNEL_STR(PREFETCH_B()) " > 0\n\t" \
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK()) " > " #N "\n\t" \
    INSTRUCTION \
    ".endif\n\t" \
    ".endif\n\t"

    const int64_t head = K % 2;
    const int64_t body = K - head;

    const int64_t prefetch_line_a = 8;
    const int64_t prefetch_line_b = 8;
    const __fp16* prefetch_a_ptr = A + prefetch_line_a * lda;
    const __fp16* prefetch_b_ptr = B + prefetch_line_b * ldb;

    asm volatile(
        // init C
".if " PPL_ARM_SERVER_KERNEL_STR(INIT_T()) " == 0\n\t"  // init C as 0
        IF_MN_GT(0,  0,  "movi      v8.8h,      #0          \n\t")
        IF_MN_GT(0,  1,  "movi      v9.8h,      #0          \n\t")
        IF_MN_GT(0,  2,  "movi      v10.8h,     #0          \n\t")
        IF_MN_GT(1,  0,  "movi      v11.8h,     #0          \n\t")
        IF_MN_GT(1,  1,  "movi      v12.8h,     #0          \n\t")
        IF_MN_GT(1,  2,  "movi      v13.8h,     #0          \n\t")
        IF_MN_GT(2,  0,  "movi      v14.8h,     #0          \n\t")
        IF_MN_GT(2,  1,  "movi      v15.8h,     #0          \n\t")
        IF_MN_GT(2,  2,  "movi      v16.8h,     #0          \n\t")
        IF_MN_GT(3,  0,  "movi      v17.8h,     #0          \n\t")
        IF_MN_GT(3,  1,  "movi      v18.8h,     #0          \n\t")
        IF_MN_GT(3,  2,  "movi      v19.8h,     #0          \n\t")
        IF_MN_GT(4,  0,  "movi      v20.8h,     #0          \n\t")
        IF_MN_GT(4,  1,  "movi      v21.8h,     #0          \n\t")
        IF_MN_GT(4,  2,  "movi      v22.8h,     #0          \n\t")
        IF_MN_GT(5,  0,  "movi      v23.8h,     #0          \n\t")
        IF_MN_GT(5,  1,  "movi      v24.8h,     #0          \n\t")
        IF_MN_GT(5,  2,  "movi      v25.8h,     #0          \n\t")
        IF_MN_GT(6,  0,  "movi      v26.8h,     #0          \n\t")
        IF_MN_GT(6,  1,  "movi      v27.8h,     #0          \n\t")
        IF_MN_GT(6,  2,  "movi      v28.8h,     #0          \n\t")
        IF_MN_GT(7,  0,  "movi      v29.8h,     #0          \n\t")
        IF_MN_GT(7,  1,  "movi      v30.8h,     #0          \n\t")
        IF_MN_GT(7,  2,  "movi      v31.8h,     #0          \n\t")
".endif\n\t"
".if " PPL_ARM_SERVER_KERNEL_STR(INIT_T()) " == 1\n\t"  // init C by load
                         "mov       x15,        %[ptr_c]    \n\t"
        IF_MN_GT(0,  0,  "ldr       q8,         [x15,       #0]         \n\t")
        IF_MN_GT(0,  1,  "ldr       q9,         [x15,       #16]        \n\t")
        IF_MN_GT(0,  2,  "ldr       q10,        [x15,       #32]        \n\t")
        IF_M_GT( 1,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(1,  0,  "ldr       q11,        [x15,       #0]         \n\t")
        IF_MN_GT(1,  1,  "ldr       q12,        [x15,       #16]        \n\t")
        IF_MN_GT(1,  2,  "ldr       q13,        [x15,       #32]        \n\t")
        IF_M_GT( 2,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(2,  0,  "ldr       q14,        [x15,       #0]         \n\t")
        IF_MN_GT(2,  1,  "ldr       q15,        [x15,       #16]        \n\t")
        IF_MN_GT(2,  2,  "ldr       q16,        [x15,       #32]        \n\t")
        IF_M_GT( 3,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(3,  0,  "ldr       q17,        [x15,       #0]         \n\t")
        IF_MN_GT(3,  1,  "ldr       q18,        [x15,       #16]        \n\t")
        IF_MN_GT(3,  2,  "ldr       q19,        [x15,       #32]        \n\t")
        IF_M_GT( 4,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(4,  0,  "ldr       q20,        [x15,       #0]         \n\t")
        IF_MN_GT(4,  1,  "ldr       q21,        [x15,       #16]        \n\t")
        IF_MN_GT(4,  2,  "ldr       q22,        [x15,       #32]        \n\t")
        IF_M_GT( 5,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(5,  0,  "ldr       q23,        [x15,       #0]         \n\t")
        IF_MN_GT(5,  1,  "ldr       q24,        [x15,       #16]        \n\t")
        IF_MN_GT(5,  2,  "ldr       q25,        [x15,       #32]        \n\t")
        IF_M_GT( 6,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(6,  0,  "ldr       q26,        [x15,       #0]         \n\t")
        IF_MN_GT(6,  1,  "ldr       q27,        [x15,       #16]        \n\t")
        IF_MN_GT(6,  2,  "ldr       q28,        [x15,       #32]        \n\t")
        IF_M_GT( 7,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(7,  0,  "ldr       q29,        [x15,       #0]         \n\t")
        IF_MN_GT(7,  1,  "ldr       q30,        [x15,       #16]        \n\t")
        IF_MN_GT(7,  2,  "ldr       q31,        [x15,       #32]        \n\t")
".endif\n\t"

        "1: \n\t"   // calc head
                         "cbz       %[head],    2f                      \n\t"   // no head, calc body

        IF_M_GT( 0,      "ldr       q0,         [%[ptr_a],  #0]         \n\t")
                         "add       %[ptr_a],   %[ptr_a],   %[lda]      \n\t"
        IF_N_GT( 0,      "ldr       q2,         [%[ptr_b],  #0]         \n\t")
        IF_N_GT( 1,      "ldr       q3,         [%[ptr_b],  #16]        \n\t")
        IF_N_GT( 2,      "ldr       q4,         [%[ptr_b],  #32]        \n\t")
                         "add       %[ptr_b],   %[ptr_b],   %[ldb]      \n\t"

        IF_MN_GT(0,  0,  "fmla      v8.8h,      v2.8h,      v0.h[0]     \n\t")
        IF_MN_GT(1,  0,  "fmla      v11.8h,     v2.8h,      v0.h[1]     \n\t")
        IF_MN_GT(2,  0,  "fmla      v14.8h,     v2.8h,      v0.h[2]     \n\t")
        IF_MN_GT(3,  0,  "fmla      v17.8h,     v2.8h,      v0.h[3]     \n\t")
        IF_MN_GT(4,  0,  "fmla      v20.8h,     v2.8h,      v0.h[4]     \n\t")
        IF_MN_GT(5,  0,  "fmla      v23.8h,     v2.8h,      v0.h[5]     \n\t")
        IF_MN_GT(6,  0,  "fmla      v26.8h,     v2.8h,      v0.h[6]     \n\t")
        IF_MN_GT(7,  0,  "fmla      v29.8h,     v2.8h,      v0.h[7]     \n\t")

        IF_PFC_A_M_GT(0, "prfm      pldl1strm,  [%[pfc_a],  #0]         \n\t")
        IF_PFC_A_M_GT(0, "add       %[pfc_a],   %[pfc_a],   %[lda]      \n\t")

        IF_MN_GT(0,  1,  "fmla      v9.8h,      v3.8h,      v0.h[0]     \n\t")
        IF_MN_GT(1,  1,  "fmla      v12.8h,     v3.8h,      v0.h[1]     \n\t")
        IF_MN_GT(2,  1,  "fmla      v15.8h,     v3.8h,      v0.h[2]     \n\t")
        IF_MN_GT(3,  1,  "fmla      v18.8h,     v3.8h,      v0.h[3]     \n\t")
        IF_MN_GT(4,  1,  "fmla      v21.8h,     v3.8h,      v0.h[4]     \n\t")
        IF_MN_GT(5,  1,  "fmla      v24.8h,     v3.8h,      v0.h[5]     \n\t")
        IF_MN_GT(6,  1,  "fmla      v27.8h,     v3.8h,      v0.h[6]     \n\t")
        IF_MN_GT(7,  1,  "fmla      v30.8h,     v3.8h,      v0.h[7]     \n\t")

        IF_PFC_B_N_GT(0, "prfm      pldl1strm,  [%[pfc_b],  #0]         \n\t")
        IF_PFC_B_N_GT(0, "add       %[pfc_b],   %[pfc_b],   %[ldb]      \n\t")

        IF_MN_GT(0,  2,  "fmla      v10.8h,     v4.8h,      v0.h[0]     \n\t")
        IF_MN_GT(1,  2,  "fmla      v13.8h,     v4.8h,      v0.h[1]     \n\t")
        IF_MN_GT(2,  2,  "fmla      v16.8h,     v4.8h,      v0.h[2]     \n\t")
        IF_MN_GT(3,  2,  "fmla      v19.8h,     v4.8h,      v0.h[3]     \n\t")
        IF_MN_GT(4,  2,  "fmla      v22.8h,     v4.8h,      v0.h[4]     \n\t")
        IF_MN_GT(5,  2,  "fmla      v25.8h,     v4.8h,      v0.h[5]     \n\t")
        IF_MN_GT(6,  2,  "fmla      v28.8h,     v4.8h,      v0.h[6]     \n\t")
        IF_MN_GT(7,  2,  "fmla      v31.8h,     v4.8h,      v0.h[7]     \n\t")

        "2: \n\t"   // prepare data
                         "cbz       %[body],    5f                      \n\t"   // no body, store C

        IF_M_GT( 0,      "ldr       q0,         [%[ptr_a],  #0]         \n\t")
                         "add       %[ptr_a],   %[ptr_a],   %[lda]      \n\t"
        IF_N_GT( 0,      "ldr       q2,         [%[ptr_b],  #0]         \n\t")
        IF_N_GT( 1,      "ldr       q3,         [%[ptr_b],  #16]        \n\t")
        IF_N_GT( 2,      "ldr       q4,         [%[ptr_b],  #32]        \n\t")
                         "add       %[ptr_b],   %[ptr_b],   %[ldb]      \n\t"
        IF_M_GT( 0,      "ldr       q1,         [%[ptr_a],  #0]         \n\t")
                         "add       %[ptr_a],   %[ptr_a],   %[lda]      \n\t"
        IF_N_GT( 0,      "ldr       q5,         [%[ptr_b],  #0]         \n\t")
        IF_N_GT( 1,      "ldr       q6,         [%[ptr_b],  #16]        \n\t")
        IF_N_GT( 2,      "ldr       q7,         [%[ptr_b],  #32]        \n\t")
                         "add       %[ptr_b],   %[ptr_b],   %[ldb]      \n\t"

        "3: \n\t"   // calc body
                         "subs      %[body],    %[body],    #2          \n\t"
                         "cbz       %[body],    4f                      \n\t"

        IF_MN_GT(0,  0,  "fmla      v8.8h,      v2.8h,      v0.h[0]     \n\t")
        IF_MN_GT(1,  0,  "fmla      v11.8h,     v2.8h,      v0.h[1]     \n\t")
        IF_MN_GT(2,  0,  "fmla      v14.8h,     v2.8h,      v0.h[2]     \n\t")
        IF_MN_GT(3,  0,  "fmla      v17.8h,     v2.8h,      v0.h[3]     \n\t")

        IF_PFC_A_M_GT(0, "prfm      pldl1strm,  [%[pfc_a],  #0]         \n\t")
        IF_PFC_A_M_GT(0, "add       %[pfc_a],   %[pfc_a],   %[lda]      \n\t")

        IF_MN_GT(4,  0,  "fmla      v20.8h,     v2.8h,      v0.h[4]     \n\t")
        IF_MN_GT(5,  0,  "fmla      v23.8h,     v2.8h,      v0.h[5]     \n\t")
        IF_MN_GT(6,  0,  "fmla      v26.8h,     v2.8h,      v0.h[6]     \n\t")
        IF_MN_GT(7,  0,  "fmla      v29.8h,     v2.8h,      v0.h[7]     \n\t")

        IF_N_GT( 0,      "ldr       q2,         [%[ptr_b],  #0]         \n\t")

        IF_MN_GT(0,  1,  "fmla      v9.8h,      v3.8h,      v0.h[0]     \n\t")
        IF_MN_GT(1,  1,  "fmla      v12.8h,     v3.8h,      v0.h[1]     \n\t")
        IF_MN_GT(2,  1,  "fmla      v15.8h,     v3.8h,      v0.h[2]     \n\t")
        IF_MN_GT(3,  1,  "fmla      v18.8h,     v3.8h,      v0.h[3]     \n\t")

        IF_PFC_B_N_GT(0, "prfm      pldl1strm,  [%[pfc_b],  #0]         \n\t")
        IF_PFC_B_N_GT(0, "add       %[pfc_b],   %[pfc_b],   %[ldb]      \n\t")

        IF_MN_GT(4,  1,  "fmla      v21.8h,     v3.8h,      v0.h[4]     \n\t")
        IF_MN_GT(5,  1,  "fmla      v24.8h,     v3.8h,      v0.h[5]     \n\t")
        IF_MN_GT(6,  1,  "fmla      v27.8h,     v3.8h,      v0.h[6]     \n\t")
        IF_MN_GT(7,  1,  "fmla      v30.8h,     v3.8h,      v0.h[7]     \n\t")

        IF_N_GT( 1,      "ldr       q3,         [%[ptr_b],  #16]        \n\t")

        IF_MN_GT(0,  2,  "fmla      v10.8h,     v4.8h,      v0.h[0]     \n\t")
        IF_MN_GT(1,  2,  "fmla      v13.8h,     v4.8h,      v0.h[1]     \n\t")
        IF_MN_GT(2,  2,  "fmla      v16.8h,     v4.8h,      v0.h[2]     \n\t")
        IF_MN_GT(3,  2,  "fmla      v19.8h,     v4.8h,      v0.h[3]     \n\t")

        IF_PFC_A_M_GT(0, "prfm      pldl1strm,  [%[pfc_a],  #0]         \n\t")
        IF_PFC_A_M_GT(0, "add       %[pfc_a],   %[pfc_a],   %[lda]      \n\t")

        IF_MN_GT(4,  2,  "fmla      v22.8h,     v4.8h,      v0.h[4]     \n\t")
        IF_MN_GT(5,  2,  "fmla      v25.8h,     v4.8h,      v0.h[5]     \n\t")
        IF_MN_GT(6,  2,  "fmla      v28.8h,     v4.8h,      v0.h[6]     \n\t")
        IF_MN_GT(7,  2,  "fmla      v31.8h,     v4.8h,      v0.h[7]     \n\t")

        IF_M_GT( 0,      "ldr       q0,         [%[ptr_a],  #0]         \n\t")
                         "add       %[ptr_a],   %[ptr_a],   %[lda]      \n\t"
        IF_N_GT( 2,      "ldr       q4,         [%[ptr_b],  #32]        \n\t")
                         "add       %[ptr_b],   %[ptr_b],   %[ldb]      \n\t"

        IF_MN_GT(0,  0,  "fmla      v8.8h,      v5.8h,      v1.h[0]     \n\t")
        IF_MN_GT(1,  0,  "fmla      v11.8h,     v5.8h,      v1.h[1]     \n\t")
        IF_MN_GT(2,  0,  "fmla      v14.8h,     v5.8h,      v1.h[2]     \n\t")
        IF_MN_GT(3,  0,  "fmla      v17.8h,     v5.8h,      v1.h[3]     \n\t")

        IF_PFC_B_N_GT(0, "prfm      pldl1strm,  [%[pfc_b],  #0]         \n\t")
        IF_PFC_B_N_GT(0, "add       %[pfc_b],   %[pfc_b],   %[ldb]      \n\t")

        IF_MN_GT(4,  0,  "fmla      v20.8h,     v5.8h,      v1.h[4]     \n\t")
        IF_MN_GT(5,  0,  "fmla      v23.8h,     v5.8h,      v1.h[5]     \n\t")
        IF_MN_GT(6,  0,  "fmla      v26.8h,     v5.8h,      v1.h[6]     \n\t")
        IF_MN_GT(7,  0,  "fmla      v29.8h,     v5.8h,      v1.h[7]     \n\t")

        IF_N_GT( 0,      "ldr       q5,         [%[ptr_b],  #0]         \n\t")

        IF_MN_GT(0,  1,  "fmla      v9.8h,      v6.8h,      v1.h[0]     \n\t")
        IF_MN_GT(1,  1,  "fmla      v12.8h,     v6.8h,      v1.h[1]     \n\t")
        IF_MN_GT(2,  1,  "fmla      v15.8h,     v6.8h,      v1.h[2]     \n\t")
        IF_MN_GT(3,  1,  "fmla      v18.8h,     v6.8h,      v1.h[3]     \n\t")
        IF_MN_GT(4,  1,  "fmla      v21.8h,     v6.8h,      v1.h[4]     \n\t")
        IF_MN_GT(5,  1,  "fmla      v24.8h,     v6.8h,      v1.h[5]     \n\t")
        IF_MN_GT(6,  1,  "fmla      v27.8h,     v6.8h,      v1.h[6]     \n\t")
        IF_MN_GT(7,  1,  "fmla      v30.8h,     v6.8h,      v1.h[7]     \n\t")

        IF_N_GT( 1,      "ldr       q6,         [%[ptr_b],  #16]        \n\t")

        IF_MN_GT(0,  2,  "fmla      v10.8h,     v7.8h,      v1.h[0]     \n\t")
        IF_MN_GT(1,  2,  "fmla      v13.8h,     v7.8h,      v1.h[1]     \n\t")
        IF_MN_GT(2,  2,  "fmla      v16.8h,     v7.8h,      v1.h[2]     \n\t")
        IF_MN_GT(3,  2,  "fmla      v19.8h,     v7.8h,      v1.h[3]     \n\t")
        IF_MN_GT(4,  2,  "fmla      v22.8h,     v7.8h,      v1.h[4]     \n\t")
        IF_MN_GT(5,  2,  "fmla      v25.8h,     v7.8h,      v1.h[5]     \n\t")
        IF_MN_GT(6,  2,  "fmla      v28.8h,     v7.8h,      v1.h[6]     \n\t")
        IF_MN_GT(7,  2,  "fmla      v31.8h,     v7.8h,      v1.h[7]     \n\t")

        IF_M_GT( 0,      "ldr       q1,         [%[ptr_a],  #0]         \n\t")
                         "add       %[ptr_a],   %[ptr_a],   %[lda]      \n\t"
        IF_N_GT( 2,      "ldr       q7,         [%[ptr_b],  #32]        \n\t")
                         "add       %[ptr_b],   %[ptr_b],   %[ldb]      \n\t"

                         "b         3b                                  \n\t"

        "4: \n\t"   // calc tail
        IF_MN_GT(0,  0,  "fmla      v8.8h,      v2.8h,      v0.h[0]     \n\t")
        IF_MN_GT(1,  0,  "fmla      v11.8h,     v2.8h,      v0.h[1]     \n\t")
        IF_MN_GT(2,  0,  "fmla      v14.8h,     v2.8h,      v0.h[2]     \n\t")
        IF_MN_GT(3,  0,  "fmla      v17.8h,     v2.8h,      v0.h[3]     \n\t")
        IF_MN_GT(4,  0,  "fmla      v20.8h,     v2.8h,      v0.h[4]     \n\t")
        IF_MN_GT(5,  0,  "fmla      v23.8h,     v2.8h,      v0.h[5]     \n\t")
        IF_MN_GT(6,  0,  "fmla      v26.8h,     v2.8h,      v0.h[6]     \n\t")
        IF_MN_GT(7,  0,  "fmla      v29.8h,     v2.8h,      v0.h[7]     \n\t")

        IF_MN_GT(0,  1,  "fmla      v9.8h,      v3.8h,      v0.h[0]     \n\t")
        IF_MN_GT(1,  1,  "fmla      v12.8h,     v3.8h,      v0.h[1]     \n\t")
        IF_MN_GT(2,  1,  "fmla      v15.8h,     v3.8h,      v0.h[2]     \n\t")
        IF_MN_GT(3,  1,  "fmla      v18.8h,     v3.8h,      v0.h[3]     \n\t")
        IF_MN_GT(4,  1,  "fmla      v21.8h,     v3.8h,      v0.h[4]     \n\t")
        IF_MN_GT(5,  1,  "fmla      v24.8h,     v3.8h,      v0.h[5]     \n\t")
        IF_MN_GT(6,  1,  "fmla      v27.8h,     v3.8h,      v0.h[6]     \n\t")
        IF_MN_GT(7,  1,  "fmla      v30.8h,     v3.8h,      v0.h[7]     \n\t")

        IF_MN_GT(0,  2,  "fmla      v10.8h,     v4.8h,      v0.h[0]     \n\t")
        IF_MN_GT(1,  2,  "fmla      v13.8h,     v4.8h,      v0.h[1]     \n\t")
        IF_MN_GT(2,  2,  "fmla      v16.8h,     v4.8h,      v0.h[2]     \n\t")
        IF_MN_GT(3,  2,  "fmla      v19.8h,     v4.8h,      v0.h[3]     \n\t")
        IF_MN_GT(4,  2,  "fmla      v22.8h,     v4.8h,      v0.h[4]     \n\t")
        IF_MN_GT(5,  2,  "fmla      v25.8h,     v4.8h,      v0.h[5]     \n\t")
        IF_MN_GT(6,  2,  "fmla      v28.8h,     v4.8h,      v0.h[6]     \n\t")
        IF_MN_GT(7,  2,  "fmla      v31.8h,     v4.8h,      v0.h[7]     \n\t")

        IF_MN_GT(0,  0,  "fmla      v8.8h,      v5.8h,      v1.h[0]     \n\t")
        IF_MN_GT(1,  0,  "fmla      v11.8h,     v5.8h,      v1.h[1]     \n\t")
        IF_MN_GT(2,  0,  "fmla      v14.8h,     v5.8h,      v1.h[2]     \n\t")
        IF_MN_GT(3,  0,  "fmla      v17.8h,     v5.8h,      v1.h[3]     \n\t")
        IF_MN_GT(4,  0,  "fmla      v20.8h,     v5.8h,      v1.h[4]     \n\t")
        IF_MN_GT(5,  0,  "fmla      v23.8h,     v5.8h,      v1.h[5]     \n\t")
        IF_MN_GT(6,  0,  "fmla      v26.8h,     v5.8h,      v1.h[6]     \n\t")
        IF_MN_GT(7,  0,  "fmla      v29.8h,     v5.8h,      v1.h[7]     \n\t")

        IF_MN_GT(0,  1,  "fmla      v9.8h,      v6.8h,      v1.h[0]     \n\t")
        IF_MN_GT(1,  1,  "fmla      v12.8h,     v6.8h,      v1.h[1]     \n\t")
        IF_MN_GT(2,  1,  "fmla      v15.8h,     v6.8h,      v1.h[2]     \n\t")
        IF_MN_GT(3,  1,  "fmla      v18.8h,     v6.8h,      v1.h[3]     \n\t")
        IF_MN_GT(4,  1,  "fmla      v21.8h,     v6.8h,      v1.h[4]     \n\t")
        IF_MN_GT(5,  1,  "fmla      v24.8h,     v6.8h,      v1.h[5]     \n\t")
        IF_MN_GT(6,  1,  "fmla      v27.8h,     v6.8h,      v1.h[6]     \n\t")
        IF_MN_GT(7,  1,  "fmla      v30.8h,     v6.8h,      v1.h[7]     \n\t")

        IF_MN_GT(0,  2,  "fmla      v10.8h,     v7.8h,      v1.h[0]     \n\t")
        IF_MN_GT(1,  2,  "fmla      v13.8h,     v7.8h,      v1.h[1]     \n\t")
        IF_MN_GT(2,  2,  "fmla      v16.8h,     v7.8h,      v1.h[2]     \n\t")
        IF_MN_GT(3,  2,  "fmla      v19.8h,     v7.8h,      v1.h[3]     \n\t")
        IF_MN_GT(4,  2,  "fmla      v22.8h,     v7.8h,      v1.h[4]     \n\t")
        IF_MN_GT(5,  2,  "fmla      v25.8h,     v7.8h,      v1.h[5]     \n\t")
        IF_MN_GT(6,  2,  "fmla      v28.8h,     v7.8h,      v1.h[6]     \n\t")
        IF_MN_GT(7,  2,  "fmla      v31.8h,     v7.8h,      v1.h[7]     \n\t")

        "5: \n\t"   // store C
                         "mov       x15,        %[ptr_c]    \n\t"
        IF_MN_GT(0,  0,  "str       q8,         [x15,       #0]         \n\t")
        IF_MN_GT(0,  1,  "str       q9,         [x15,       #16]        \n\t")
        IF_MN_GT(0,  2,  "str       q10,        [x15,       #32]        \n\t")
        IF_M_GT( 1,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(1,  0,  "str       q11,        [x15,       #0]         \n\t")
        IF_MN_GT(1,  1,  "str       q12,        [x15,       #16]        \n\t")
        IF_MN_GT(1,  2,  "str       q13,        [x15,       #32]        \n\t")
        IF_M_GT( 2,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(2,  0,  "str       q14,        [x15,       #0]         \n\t")
        IF_MN_GT(2,  1,  "str       q15,        [x15,       #16]        \n\t")
        IF_MN_GT(2,  2,  "str       q16,        [x15,       #32]        \n\t")
        IF_M_GT( 3,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(3,  0,  "str       q17,        [x15,       #0]         \n\t")
        IF_MN_GT(3,  1,  "str       q18,        [x15,       #16]        \n\t")
        IF_MN_GT(3,  2,  "str       q19,        [x15,       #32]        \n\t")
        IF_M_GT( 4,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(4,  0,  "str       q20,        [x15,       #0]         \n\t")
        IF_MN_GT(4,  1,  "str       q21,        [x15,       #16]        \n\t")
        IF_MN_GT(4,  2,  "str       q22,        [x15,       #32]        \n\t")
        IF_M_GT( 5,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(5,  0,  "str       q23,        [x15,       #0]         \n\t")
        IF_MN_GT(5,  1,  "str       q24,        [x15,       #16]        \n\t")
        IF_MN_GT(5,  2,  "str       q25,        [x15,       #32]        \n\t")
        IF_M_GT( 6,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(6,  0,  "str       q26,        [x15,       #0]         \n\t")
        IF_MN_GT(6,  1,  "str       q27,        [x15,       #16]        \n\t")
        IF_MN_GT(6,  2,  "str       q28,        [x15,       #32]        \n\t")
        IF_M_GT( 7,      "add       x15,        x15,        %[ldc]      \n\t")
        IF_MN_GT(7,  0,  "str       q29,        [x15,       #0]         \n\t")
        IF_MN_GT(7,  1,  "str       q30,        [x15,       #16]        \n\t")
        IF_MN_GT(7,  2,  "str       q31,        [x15,       #32]        \n\t")
        :
        : [ptr_a]"r"(A)
        , [ptr_b]"r"(B)
        , [ptr_c]"r"(C)
        , [lda]"r"(lda * sizeof(__fp16))
        , [ldb]"r"(ldb * sizeof(__fp16))
        , [ldc]"r"(ldc * sizeof(__fp16))
        , [head]"r"(head)
        , [body]"r"(body)
        , [pfc_a]"r"(prefetch_a_ptr)
        , [pfc_b]"r"(prefetch_b_ptr)
        :"memory", "cc"
        , "x15" // temp store ptr_c
        , "v0", "v1"    // A data (double buffer)
        , "v2", "v3", "v4", "v5", "v6", "v7"    // B data (double buffer)
        , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"      // C data
        , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
        , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
    );

#undef IF_M_GT
#undef IF_N_GT
#undef IF_MN_GT
}
