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

#ifndef PP_PPL_ARM_SERVER_KERNEL_FP16_CONV_N8CX_HGEMM_KERNEL_M8NX_FUNC
#define PP_PPL_ARM_SERVER_KERNEL_FP16_CONV_N8CX_HGEMM_KERNEL_M8NX_FUNC(N_BLOCK0, INIT_T, FUSE_T) \
    hgemm_n8cx_kernel_m8nx_fp16_func<N_BLOCK0, INIT_T,FUSE_T>
#endif
#ifndef PPL_ARM_SERVER_KERNEL_FP16_CONV_N8CX_HGEMM_KERNEL_M8NX_FUNC
#define PPL_ARM_SERVER_KERNEL_FP16_CONV_N8CX_HGEMM_KERNEL_M8NX_FUNC(N_BLOCK0, INIT_T, FUSE_T) \
    PP_PPL_ARM_SERVER_KERNEL_FP16_CONV_N8CX_HGEMM_KERNEL_M8NX_FUNC(N_BLOCK0, INIT_T, FUSE_T)
#endif

#ifndef PPL_ARM_SERVER_KERNEL_PP_STR
#define PPL_ARM_SERVER_KERNEL_PP_STR(X) #X
#endif
#ifndef PPL_ARM_SERVER_KERNEL_STR
#define PPL_ARM_SERVER_KERNEL_STR(X) PPL_ARM_SERVER_KERNEL_PP_STR(X)
#endif

template<>
void PPL_ARM_SERVER_KERNEL_FP16_CONV_N8CX_HGEMM_KERNEL_M8NX_FUNC(N_BLOCK0(), INIT_T(), FUSE_T()) (
    const __fp16* A,
    const __fp16* B,
    const __fp16* constant,
    const __fp16* DX,
    __fp16* C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t lddx,
    const int64_t ldc)
{
#define IF_N_GT(N0, INSTRUCTION) ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " > " #N0 "\n\t  " INSTRUCTION ".endif\n\t"

    asm volatile (
        ".align 3\n\t"
        "mov x9, %[Bptr]\n\t"
        "mov x10, %[Aptr]\n\t"
        "mov x11, %[Cptr]\n\t"
        
        "ldr q12, [x9]\n\t\n\t"
        "ldr q13, [x9, #16]\n\t\n\t"
        "ldr  q0, [x10]\n\t"
        "subs %[k], %[k], #8\n\t"
        
".if " PPL_ARM_SERVER_KERNEL_STR(INIT_T()) " == 0\n\t"
                    "fmul v20.8h, v12.8h, v0.h[0]\n\t"
        IF_N_GT(1 , "ldr  q1,     [x10, #16]\n\t")
        IF_N_GT(1 , "fmul v21.8h, v12.8h, v1.h[0]\n\t")
        IF_N_GT(2 , "ldr  q2,     [x10, #32]\n\t")
        IF_N_GT(2 , "fmul v22.8h, v12.8h, v2.h[0]\n\t")
        IF_N_GT(3 , "ldr  q3,     [x10, #48]\n\t")
        IF_N_GT(3 , "fmul v23.8h, v12.8h, v3.h[0]\n\t")
        IF_N_GT(4 , "ldr  q4,     [x10, #64]\n\t")
        IF_N_GT(4 , "fmul v24.8h, v12.8h, v4.h[0]\n\t")
        IF_N_GT(5 , "ldr  q5,     [x10, #80]\n\t")
        IF_N_GT(5 , "fmul v25.8h, v12.8h, v5.h[0]\n\t")
        IF_N_GT(6 , "ldr  q6,     [x10, #96]\n\t")
        IF_N_GT(6 , "fmul v26.8h, v12.8h, v6.h[0]\n\t")
        IF_N_GT(7 , "ldr  q7,     [x10, #112]\n\t")
        IF_N_GT(7 , "fmul v27.8h, v12.8h, v7.h[0]\n\t")
        IF_N_GT(8 , "ldr  q8,     [x10, #128]\n\t")
        IF_N_GT(8 , "fmul v28.8h, v12.8h, v8.h[0]\n\t")
        IF_N_GT(9 , "ldr  q9,     [x10, #144]\n\t")
        IF_N_GT(9 , "fmul v29.8h, v12.8h, v9.h[0]\n\t")
        IF_N_GT(10, "ldr  q10,    [x10, #160]\n\t")
        IF_N_GT(10, "fmul v30.8h, v12.8h, v10.h[0]\n\t")
        IF_N_GT(11, "ldr  q11,    [x10, #176]\n\t")
        IF_N_GT(11, "fmul v31.8h, v12.8h, v11.h[0]\n\t")
".else\n\t.if " PPL_ARM_SERVER_KERNEL_STR(INIT_T()) " == 1\n\t"
                    "ldr  q20,    [%[const_ptr]]\n\t"
        IF_N_GT(1 , "ldr  q1,     [x10, #16]\n\t")
        IF_N_GT(1 , "mov  v21.16b, v20.16b\n\t")
        IF_N_GT(1 , "fmla v21.8h, v12.8h, v1.h[0]\n\t")
        IF_N_GT(2 , "ldr  q2,     [x10, #32]\n\t")
        IF_N_GT(2 , "mov  v22.16b, v20.16b\n\t")
        IF_N_GT(2 , "fmla v22.8h, v12.8h, v2.h[0]\n\t")
        IF_N_GT(3 , "ldr  q3,     [x10, #48]\n\t")
        IF_N_GT(3 , "mov  v23.16b, v20.16b\n\t")
        IF_N_GT(3 , "fmla v23.8h, v12.8h, v3.h[0]\n\t")
        IF_N_GT(4 , "ldr  q4,     [x10, #64]\n\t")
        IF_N_GT(4 , "mov  v24.16b, v20.16b\n\t")
        IF_N_GT(4 , "fmla v24.8h, v12.8h, v4.h[0]\n\t")
        IF_N_GT(5 , "ldr  q5,     [x10, #80]\n\t")
        IF_N_GT(5 , "mov  v25.16b, v20.16b\n\t")
        IF_N_GT(5 , "fmla v25.8h, v12.8h, v5.h[0]\n\t")
        IF_N_GT(6 , "ldr  q6,     [x10, #96]\n\t")
        IF_N_GT(6 , "mov  v26.16b, v20.16b\n\t")
        IF_N_GT(6 , "fmla v26.8h, v12.8h, v6.h[0]\n\t")
        IF_N_GT(7 , "ldr  q7,     [x10, #112]\n\t")
        IF_N_GT(7 , "mov  v27.16b, v20.16b\n\t")
        IF_N_GT(7 , "fmla v27.8h, v12.8h, v7.h[0]\n\t")
        IF_N_GT(8 , "ldr  q8,     [x10, #128]\n\t")
        IF_N_GT(8 , "mov  v28.16b, v20.16b\n\t")
        IF_N_GT(8 , "fmla v28.8h, v12.8h, v8.h[0]\n\t")
        IF_N_GT(9 , "ldr  q9,     [x10, #144]\n\t")
        IF_N_GT(9 , "mov  v29.16b, v20.16b\n\t")
        IF_N_GT(9 , "fmla v29.8h, v12.8h, v9.h[0]\n\t")
        IF_N_GT(10, "ldr  q10,    [x10, #160]\n\t")
        IF_N_GT(10, "mov  v30.16b, v20.16b\n\t")
        IF_N_GT(10, "fmla v30.8h, v12.8h, v10.h[0]\n\t")
        IF_N_GT(11, "ldr  q11,    [x10, #176]\n\t")
        IF_N_GT(11, "mov  v31.16b, v20.16b\n\t")
        IF_N_GT(11, "fmla v31.8h, v12.8h, v11.h[0]\n\t")
                    "fmla v20.8h, v12.8h, v0.h[0]\n\t"
".else\n\t.if " PPL_ARM_SERVER_KERNEL_STR(INIT_T()) " == 2\n\t"
                    "ldr  q20,    [x11]\n\t"
                    "fmla v20.8h, v12.8h, v0.h[0]\n\t"
        IF_N_GT(1 , "ldr  q1,     [x10, #16]\n\t")
        IF_N_GT(1 , "ldr  q21,    [x11, #16]\n\t")
        IF_N_GT(1 , "fmla v21.8h, v12.8h, v1.h[0]\n\t")
        IF_N_GT(2 , "ldr  q2,     [x10, #32]\n\t")
        IF_N_GT(2 , "ldr  q22,    [x11, #32]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v12.8h, v2.h[0]\n\t")
        IF_N_GT(3 , "ldr  q3,     [x10, #48]\n\t")
        IF_N_GT(3 , "ldr  q23,    [x11, #48]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v12.8h, v3.h[0]\n\t")
        IF_N_GT(4 , "ldr  q4,     [x10, #64]\n\t")
        IF_N_GT(4 , "ldr  q24,    [x11, #64]\n\t")
        IF_N_GT(4 , "fmla v24.8h, v12.8h, v4.h[0]\n\t")
        IF_N_GT(5 , "ldr  q5,     [x10, #80]\n\t")
        IF_N_GT(5 , "ldr  q25,    [x11, #80]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v12.8h, v5.h[0]\n\t")
        IF_N_GT(6 , "ldr  q6,     [x10, #96]\n\t")
        IF_N_GT(6 , "ldr  q26,    [x11, #96]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v12.8h, v6.h[0]\n\t")
        IF_N_GT(7 , "ldr  q7,     [x10, #112]\n\t")
        IF_N_GT(7 , "ldr  q27,    [x11, #112]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v12.8h, v7.h[0]\n\t")
        IF_N_GT(8 , "ldr  q8,     [x10, #128]\n\t")
        IF_N_GT(8 , "ldr  q28,    [x11, #128]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v12.8h, v8.h[0]\n\t")
        IF_N_GT(9 , "ldr  q9,     [x10, #144]\n\t")
        IF_N_GT(9 , "ldr  q29,    [x11, #144]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v12.8h, v9.h[0]\n\t")
        IF_N_GT(10, "ldr  q10,    [x10, #160]\n\t")
        IF_N_GT(10, "ldr  q30,    [x11, #160]\n\t")
        IF_N_GT(10, "fmla v30.8h, v12.8h, v10.h[0]\n\t")
        IF_N_GT(11, "ldr  q11,    [x10, #176]\n\t")
        IF_N_GT(11, "ldr  q31,    [x11, #176]\n\t")
        IF_N_GT(11, "fmla v31.8h, v12.8h, v11.h[0]\n\t")
".endif\n\t.endif\n\t.endif\n\t"

        "ldr q14, [x9, #32]\n\t\n\t"
                    "fmla v20.8h, v13.8h, v0.h[1]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v13.8h, v1.h[1]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v13.8h, v2.h[1]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v13.8h, v3.h[1]\n\t")
        "ldr q15, [x9, #48]\n\t\n\t"
        IF_N_GT(4 , "fmla v24.8h, v13.8h, v4.h[1]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v13.8h, v5.h[1]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v13.8h, v6.h[1]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v13.8h, v7.h[1]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v13.8h, v8.h[1]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v13.8h, v9.h[1]\n\t")
        IF_N_GT(10, "fmla v30.8h, v13.8h, v10.h[1]\n\t")
        IF_N_GT(11, "fmla v31.8h, v13.8h, v11.h[1]\n\t")

        "ldr q16, [x9, #64]\n\t\n\t"
                    "fmla v20.8h, v14.8h, v0.h[2]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v14.8h, v1.h[2]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v14.8h, v2.h[2]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v14.8h, v3.h[2]\n\t")
        "ldr q17, [x9, #80]\n\t\n\t"
        IF_N_GT(4 , "fmla v24.8h, v14.8h, v4.h[2]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v14.8h, v5.h[2]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v14.8h, v6.h[2]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v14.8h, v7.h[2]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v14.8h, v8.h[2]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v14.8h, v9.h[2]\n\t")
        IF_N_GT(10, "fmla v30.8h, v14.8h, v10.h[2]\n\t")
        IF_N_GT(11, "fmla v31.8h, v14.8h, v11.h[2]\n\t")

        "ldr q18, [x9, #96]\n\t\n\t"
                    "fmla v20.8h, v15.8h, v0.h[3]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v15.8h, v1.h[3]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v15.8h, v2.h[3]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v15.8h, v3.h[3]\n\t")
        "ldr q19, [x9, #112]\n\t\n\t"
        IF_N_GT(4 , "fmla v24.8h, v15.8h, v4.h[3]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v15.8h, v5.h[3]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v15.8h, v6.h[3]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v15.8h, v7.h[3]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v15.8h, v8.h[3]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v15.8h, v9.h[3]\n\t")
        IF_N_GT(10, "fmla v30.8h, v15.8h, v10.h[3]\n\t")
        IF_N_GT(11, "fmla v31.8h, v15.8h, v11.h[3]\n\t")

        "beq 2f\n\t"

        ".align 3\n\t"
    "1:\n\t"
        "add x9, x9, %[Bstride]\n\t"
        "ldr q12, [x9]\n\t\n\t"
                    "fmla v20.8h, v16.8h, v0.h[4]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v16.8h, v1.h[4]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v16.8h, v2.h[4]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v16.8h, v3.h[4]\n\t")
        "add x10, x10, %[Astride]\n\t"
        IF_N_GT(4 , "fmla v24.8h, v16.8h, v4.h[4]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v16.8h, v5.h[4]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v16.8h, v6.h[4]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v16.8h, v7.h[4]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v16.8h, v8.h[4]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v16.8h, v9.h[4]\n\t")
        IF_N_GT(10, "fmla v30.8h, v16.8h, v10.h[4]\n\t")
        IF_N_GT(11, "fmla v31.8h, v16.8h, v11.h[4]\n\t")

        "ldr q13, [x9, #16]\n\t\n\t"
                    "fmla v20.8h, v17.8h, v0.h[5]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v17.8h, v1.h[5]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v17.8h, v2.h[5]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v17.8h, v3.h[5]\n\t")
        "ldr q14, [x9, #32]\n\t\n\t"
        IF_N_GT(4 , "fmla v24.8h, v17.8h, v4.h[5]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v17.8h, v5.h[5]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v17.8h, v6.h[5]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v17.8h, v7.h[5]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v17.8h, v8.h[5]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v17.8h, v9.h[5]\n\t")
        IF_N_GT(10, "fmla v30.8h, v17.8h, v10.h[5]\n\t")
        IF_N_GT(11, "fmla v31.8h, v17.8h, v11.h[5]\n\t")

        "ldr q15, [x9, #48]\n\t\n\t"
                    "fmla v20.8h, v18.8h, v0.h[6]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v18.8h, v1.h[6]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v18.8h, v2.h[6]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v18.8h, v3.h[6]\n\t")
        IF_N_GT(4 , "fmla v24.8h, v18.8h, v4.h[6]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v18.8h, v5.h[6]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v18.8h, v6.h[6]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v18.8h, v7.h[6]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v18.8h, v8.h[6]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v18.8h, v9.h[6]\n\t")
        IF_N_GT(10, "fmla v30.8h, v18.8h, v10.h[6]\n\t")
        IF_N_GT(11, "fmla v31.8h, v18.8h, v11.h[6]\n\t")

                    "fmla v20.8h, v19.8h, v0.h[7]\n\t"
                    "ldr  q0,     [x10      ]\n\t"
                    "prfm pldl1keep, [x12]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v19.8h, v1.h[7]\n\t")
        IF_N_GT(1 , "ldr  q1,     [x10, #16 ]\n\t")
        IF_N_GT(1 , "prfm pldl1keep, [x12, #16]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v19.8h, v2.h[7]\n\t")
        IF_N_GT(2 , "ldr  q2,     [x10, #32 ]\n\t")
        IF_N_GT(2 , "prfm pldl1keep, [x12, #32]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v19.8h, v3.h[7]\n\t")
        IF_N_GT(3 , "ldr  q3,     [x10, #48 ]\n\t")
        IF_N_GT(3 , "prfm pldl1keep, [x12, #48]\n\t")
        IF_N_GT(4 , "fmla v24.8h, v19.8h, v4.h[7]\n\t")
        IF_N_GT(4 , "ldr  q4,     [x10, #64 ]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v19.8h, v5.h[7]\n\t")
        IF_N_GT(5 , "ldr  q5,     [x10, #80 ]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v19.8h, v6.h[7]\n\t")
        IF_N_GT(6 , "ldr  q6,     [x10, #96 ]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v19.8h, v7.h[7]\n\t")
        IF_N_GT(7 , "ldr  q7,     [x10, #112]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v19.8h, v8.h[7]\n\t")
        IF_N_GT(8 , "ldr  q8,     [x10, #128]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v19.8h, v9.h[7]\n\t")
        IF_N_GT(9 , "ldr  q9,     [x10, #144]\n\t")
        IF_N_GT(10, "fmla v30.8h, v19.8h, v10.h[7]\n\t")
        IF_N_GT(10, "ldr  q10,    [x10, #160]\n\t")
        IF_N_GT(11, "fmla v31.8h, v19.8h, v11.h[7]\n\t")
        IF_N_GT(11, "ldr  q11,    [x10, #176]\n\t")
        "subs %[k], %[k], #8\n\t"

        "ldr q16, [x9, #64]\n\t\n\t"
                    "fmla v20.8h, v12.8h, v0.h[0]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v12.8h, v1.h[0]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v12.8h, v2.h[0]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v12.8h, v3.h[0]\n\t")
        IF_N_GT(4 , "fmla v24.8h, v12.8h, v4.h[0]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v12.8h, v5.h[0]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v12.8h, v6.h[0]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v12.8h, v7.h[0]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v12.8h, v8.h[0]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v12.8h, v9.h[0]\n\t")
        IF_N_GT(10, "fmla v30.8h, v12.8h, v10.h[0]\n\t")
        IF_N_GT(11, "fmla v31.8h, v12.8h, v11.h[0]\n\t")

        "ldr q17, [x9, #80]\n\t\n\t"
                    "fmla v20.8h, v13.8h, v0.h[1]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v13.8h, v1.h[1]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v13.8h, v2.h[1]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v13.8h, v3.h[1]\n\t")
        IF_N_GT(4 , "fmla v24.8h, v13.8h, v4.h[1]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v13.8h, v5.h[1]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v13.8h, v6.h[1]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v13.8h, v7.h[1]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v13.8h, v8.h[1]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v13.8h, v9.h[1]\n\t")
        IF_N_GT(10, "fmla v30.8h, v13.8h, v10.h[1]\n\t")
        IF_N_GT(11, "fmla v31.8h, v13.8h, v11.h[1]\n\t")

        "ldr q18, [x9, #96]\n\t\n\t"
                    "fmla v20.8h, v14.8h, v0.h[2]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v14.8h, v1.h[2]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v14.8h, v2.h[2]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v14.8h, v3.h[2]\n\t")
        IF_N_GT(4 , "fmla v24.8h, v14.8h, v4.h[2]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v14.8h, v5.h[2]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v14.8h, v6.h[2]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v14.8h, v7.h[2]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v14.8h, v8.h[2]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v14.8h, v9.h[2]\n\t")
        IF_N_GT(10, "fmla v30.8h, v14.8h, v10.h[2]\n\t")
        IF_N_GT(11, "fmla v31.8h, v14.8h, v11.h[2]\n\t")

        "ldr q19, [x9, #112]\n\t\n\t"
                    "fmla v20.8h, v15.8h, v0.h[3]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v15.8h, v1.h[3]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v15.8h, v2.h[3]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v15.8h, v3.h[3]\n\t")
        IF_N_GT(4 , "fmla v24.8h, v15.8h, v4.h[3]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v15.8h, v5.h[3]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v15.8h, v6.h[3]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v15.8h, v7.h[3]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v15.8h, v8.h[3]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v15.8h, v9.h[3]\n\t")
        IF_N_GT(10, "fmla v30.8h, v15.8h, v10.h[3]\n\t")
        IF_N_GT(11, "fmla v31.8h, v15.8h, v11.h[3]\n\t")

        "bgt 1b\n\t"
        
        ".align 3\n\t"
    "2:\n\t"
        "prfm pstl1strm, [x11]\n\t"
        IF_N_GT(4 , "prfm pstl1strm, [x11, #64]\n\t")
        IF_N_GT(8 , "prfm pstl1strm, [x11, #128]\n\t")

                    "fmla v20.8h, v16.8h, v0.h[4]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v16.8h, v1.h[4]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v16.8h, v2.h[4]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v16.8h, v3.h[4]\n\t")
        IF_N_GT(4 , "fmla v24.8h, v16.8h, v4.h[4]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v16.8h, v5.h[4]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v16.8h, v6.h[4]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v16.8h, v7.h[4]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v16.8h, v8.h[4]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v16.8h, v9.h[4]\n\t")
        IF_N_GT(10, "fmla v30.8h, v16.8h, v10.h[4]\n\t")
        IF_N_GT(11, "fmla v31.8h, v16.8h, v11.h[4]\n\t")

                    "fmla v20.8h, v17.8h, v0.h[5]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v17.8h, v1.h[5]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v17.8h, v2.h[5]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v17.8h, v3.h[5]\n\t")
        IF_N_GT(4 , "fmla v24.8h, v17.8h, v4.h[5]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v17.8h, v5.h[5]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v17.8h, v6.h[5]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v17.8h, v7.h[5]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v17.8h, v8.h[5]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v17.8h, v9.h[5]\n\t")
        IF_N_GT(10, "fmla v30.8h, v17.8h, v10.h[5]\n\t")
        IF_N_GT(11, "fmla v31.8h, v17.8h, v11.h[5]\n\t")

                    "fmla v20.8h, v18.8h, v0.h[6]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v18.8h, v1.h[6]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v18.8h, v2.h[6]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v18.8h, v3.h[6]\n\t")
        IF_N_GT(4 , "fmla v24.8h, v18.8h, v4.h[6]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v18.8h, v5.h[6]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v18.8h, v6.h[6]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v18.8h, v7.h[6]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v18.8h, v8.h[6]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v18.8h, v9.h[6]\n\t")
        IF_N_GT(10, "fmla v30.8h, v18.8h, v10.h[6]\n\t")
        IF_N_GT(11, "fmla v31.8h, v18.8h, v11.h[6]\n\t")

                    "fmla v20.8h, v19.8h, v0.h[7]\n\t"
        IF_N_GT(1 , "fmla v21.8h, v19.8h, v1.h[7]\n\t")
        IF_N_GT(2 , "fmla v22.8h, v19.8h, v2.h[7]\n\t")
        IF_N_GT(3 , "fmla v23.8h, v19.8h, v3.h[7]\n\t")
        IF_N_GT(4 , "fmla v24.8h, v19.8h, v4.h[7]\n\t")
        IF_N_GT(5 , "fmla v25.8h, v19.8h, v5.h[7]\n\t")
        IF_N_GT(6 , "fmla v26.8h, v19.8h, v6.h[7]\n\t")
        IF_N_GT(7 , "fmla v27.8h, v19.8h, v7.h[7]\n\t")
        IF_N_GT(8 , "fmla v28.8h, v19.8h, v8.h[7]\n\t")
        IF_N_GT(9 , "fmla v29.8h, v19.8h, v9.h[7]\n\t")
        IF_N_GT(10, "fmla v30.8h, v19.8h, v10.h[7]\n\t")
        IF_N_GT(11, "fmla v31.8h, v19.8h, v11.h[7]\n\t")

".if " PPL_ARM_SERVER_KERNEL_STR(FUSE_T()) " == 0\n\t"    // none
                    "str  q20   , [x11      ]\n\t"
        IF_N_GT(1 , "str  q21   , [x11, #16 ]\n\t")
        IF_N_GT(2 , "str  q22   , [x11, #32 ]\n\t")
        IF_N_GT(3 , "str  q23   , [x11, #48 ]\n\t")
        IF_N_GT(4 , "str  q24   , [x11, #64 ]\n\t")
        IF_N_GT(5 , "str  q25   , [x11, #80 ]\n\t")
        IF_N_GT(6 , "str  q26   , [x11, #96 ]\n\t")
        IF_N_GT(7 , "str  q27   , [x11, #112]\n\t")
        IF_N_GT(8 , "str  q28   , [x11, #128]\n\t")
        IF_N_GT(9 , "str  q29   , [x11, #144]\n\t")
        IF_N_GT(10, "str  q30   , [x11, #160]\n\t")
        IF_N_GT(11, "str  q31   , [x11, #176]\n\t")

".else\n\t.if " PPL_ARM_SERVER_KERNEL_STR(FUSE_T()) " == 1\n\t"   // relu
        "eor v16.16b, v16.16b, v16.16b\n\t"
                    "fmax v20.8h, v20.8h, v16.8h\n\t"
                    "str  q20   , [x11      ]\n\t"        
        IF_N_GT(1 , "fmax v21.8h, v21.8h, v16.8h\n\t" )
        IF_N_GT(1 , "str  q21   , [x11, #16 ]\n\t")        
        IF_N_GT(2 , "fmax v22.8h, v22.8h, v16.8h\n\t" )
        IF_N_GT(2 , "str  q22   , [x11, #32 ]\n\t")        
        IF_N_GT(3 , "fmax v23.8h, v23.8h, v16.8h\n\t" )
        IF_N_GT(3 , "str  q23   , [x11, #48 ]\n\t")        
        IF_N_GT(4 , "fmax v24.8h, v24.8h, v16.8h\n\t" )
        IF_N_GT(4 , "str  q24   , [x11, #64 ]\n\t")        
        IF_N_GT(5 , "fmax v25.8h, v25.8h, v16.8h\n\t" )
        IF_N_GT(5 , "str  q25   , [x11, #80 ]\n\t")        
        IF_N_GT(6 , "fmax v26.8h, v26.8h, v16.8h\n\t" )
        IF_N_GT(6 , "str  q26   , [x11, #96 ]\n\t")        
        IF_N_GT(7 , "fmax v27.8h, v27.8h, v16.8h\n\t" )
        IF_N_GT(7 , "str  q27   , [x11, #112]\n\t")        
        IF_N_GT(8 , "fmax v28.8h, v28.8h, v16.8h\n\t" )
        IF_N_GT(8 , "str  q28   , [x11, #128]\n\t")        
        IF_N_GT(9 , "fmax v29.8h, v29.8h, v16.8h\n\t" )
        IF_N_GT(9 , "str  q29   , [x11, #144]\n\t")         
        IF_N_GT(10, "fmax v30.8h, v30.8h, v16.8h\n\t"  )
        IF_N_GT(10, "str  q30   , [x11, #160]\n\t")         
        IF_N_GT(11, "fmax v31.8h, v31.8h, v16.8h\n\t"  )
        IF_N_GT(11, "str  q31   , [x11, #176]\n\t")

".else\n\t.if " PPL_ARM_SERVER_KERNEL_STR(FUSE_T()) " == 2\n\t"   // relu6
        "fmov h16, 6.0e+0\n\t"
        "dup v16.8h, v16.h[0]\n\t"
                    "fmin v20.8h, v20.8h, v16.8h\n\t"
        IF_N_GT(1 , "fmin v21.8h, v21.8h, v16.8h\n\t")
        IF_N_GT(2 , "fmin v22.8h, v22.8h, v16.8h\n\t")
        IF_N_GT(3 , "fmin v23.8h, v23.8h, v16.8h\n\t")
        IF_N_GT(4 , "fmin v24.8h, v24.8h, v16.8h\n\t")
        IF_N_GT(5 , "fmin v25.8h, v25.8h, v16.8h\n\t")
        IF_N_GT(6 , "fmin v26.8h, v26.8h, v16.8h\n\t")
        IF_N_GT(7 , "fmin v27.8h, v27.8h, v16.8h\n\t")
        IF_N_GT(8 , "fmin v28.8h, v28.8h, v16.8h\n\t")
        IF_N_GT(9 , "fmin v29.8h, v29.8h, v16.8h\n\t")
        IF_N_GT(10, "fmin v30.8h, v30.8h, v16.8h\n\t")
        IF_N_GT(11, "fmin v31.8h, v31.8h, v16.8h\n\t")

        "eor v16.16b, v16.16b, v16.16b\n\t"
                    "fmax v20.8h, v20.8h, v16.8h\n\t"
                    "str  q20   , [x11      ]\n\t"        
        IF_N_GT(1 , "fmax v21.8h, v21.8h, v16.8h\n\t" )
        IF_N_GT(1 , "str  q21   , [x11, #16 ]\n\t")        
        IF_N_GT(2 , "fmax v22.8h, v22.8h, v16.8h\n\t" )
        IF_N_GT(2 , "str  q22   , [x11, #32 ]\n\t")        
        IF_N_GT(3 , "fmax v23.8h, v23.8h, v16.8h\n\t" )
        IF_N_GT(3 , "str  q23   , [x11, #48 ]\n\t")        
        IF_N_GT(4 , "fmax v24.8h, v24.8h, v16.8h\n\t" )
        IF_N_GT(4 , "str  q24   , [x11, #64 ]\n\t")        
        IF_N_GT(5 , "fmax v25.8h, v25.8h, v16.8h\n\t" )
        IF_N_GT(5 , "str  q25   , [x11, #80 ]\n\t")        
        IF_N_GT(6 , "fmax v26.8h, v26.8h, v16.8h\n\t" )
        IF_N_GT(6 , "str  q26   , [x11, #96 ]\n\t")        
        IF_N_GT(7 , "fmax v27.8h, v27.8h, v16.8h\n\t" )
        IF_N_GT(7 , "str  q27   , [x11, #112]\n\t")        
        IF_N_GT(8 , "fmax v28.8h, v28.8h, v16.8h\n\t" )
        IF_N_GT(8 , "str  q28   , [x11, #128]\n\t")        
        IF_N_GT(9 , "fmax v29.8h, v29.8h, v16.8h\n\t" )
        IF_N_GT(9 , "str  q29   , [x11, #144]\n\t")         
        IF_N_GT(10, "fmax v30.8h, v30.8h, v16.8h\n\t"  )
        IF_N_GT(10, "str  q30   , [x11, #160]\n\t")         
        IF_N_GT(11, "fmax v31.8h, v31.8h, v16.8h\n\t"  )
        IF_N_GT(11, "str  q31   , [x11, #176]\n\t")

".else\n\t.if " PPL_ARM_SERVER_KERNEL_STR(FUSE_T()) " == 4\n\t"  // sum
                    "ldr  q0    , [%[Dptr]      ]\n\t"
        IF_N_GT(1 , "ldr  q1    , [%[Dptr], #16 ]\n\t")
        IF_N_GT(2 , "ldr  q2    , [%[Dptr], #32 ]\n\t")
        IF_N_GT(3 , "ldr  q3    , [%[Dptr], #48 ]\n\t")
        IF_N_GT(4 , "ldr  q4    , [%[Dptr], #64 ]\n\t")
        IF_N_GT(5 , "ldr  q5    , [%[Dptr], #80 ]\n\t")
        IF_N_GT(6 , "ldr  q6    , [%[Dptr], #96 ]\n\t")
        IF_N_GT(7 , "ldr  q7    , [%[Dptr], #112]\n\t")
        IF_N_GT(8 , "ldr  q8    , [%[Dptr], #128]\n\t")
        IF_N_GT(9 , "ldr  q9    , [%[Dptr], #144]\n\t")
        IF_N_GT(10, "ldr  q10   , [%[Dptr], #160]\n\t")
        IF_N_GT(11, "ldr  q11   , [%[Dptr], #176]\n\t")

                    "fadd  v20.8h, v20.8h, v0.8h \n\t"
        IF_N_GT(1 , "fadd  v21.8h, v21.8h, v1.8h \n\t")
        IF_N_GT(2 , "fadd  v22.8h, v22.8h, v2.8h \n\t")
        IF_N_GT(3 , "fadd  v23.8h, v23.8h, v3.8h \n\t")
        IF_N_GT(4 , "fadd  v24.8h, v24.8h, v4.8h \n\t")
        IF_N_GT(5 , "fadd  v25.8h, v25.8h, v5.8h \n\t")
        IF_N_GT(6 , "fadd  v26.8h, v26.8h, v6.8h \n\t")
        IF_N_GT(7 , "fadd  v27.8h, v27.8h, v7.8h \n\t")
        IF_N_GT(8 , "fadd  v28.8h, v28.8h, v8.8h \n\t")
        IF_N_GT(9 , "fadd  v29.8h, v29.8h, v9.8h \n\t")
        IF_N_GT(10, "fadd  v30.8h, v30.8h, v10.8h\n\t")
        IF_N_GT(11, "fadd  v31.8h, v31.8h, v11.8h\n\t")

                    "str  q20   , [x11      ]\n\t"
        IF_N_GT(1 , "str  q21   , [x11, #16 ]\n\t")
        IF_N_GT(2 , "str  q22   , [x11, #32 ]\n\t")
        IF_N_GT(3 , "str  q23   , [x11, #48 ]\n\t")
        IF_N_GT(4 , "str  q24   , [x11, #64 ]\n\t")
        IF_N_GT(5 , "str  q25   , [x11, #80 ]\n\t")
        IF_N_GT(6 , "str  q26   , [x11, #96 ]\n\t")
        IF_N_GT(7 , "str  q27   , [x11, #112]\n\t")
        IF_N_GT(8 , "str  q28   , [x11, #128]\n\t")
        IF_N_GT(9 , "str  q29   , [x11, #144]\n\t")
        IF_N_GT(10, "str  q30   , [x11, #160]\n\t")
        IF_N_GT(11, "str  q31   , [x11, #176]\n\t")

".else\n\t.if " PPL_ARM_SERVER_KERNEL_STR(FUSE_T()) " == 5\n\t"  // sum & relu
        "eor v16.16b, v16.16b, v16.16b\n\t"
        
                    "ldr  q0    , [%[Dptr]      ]\n\t"
        IF_N_GT(1 , "ldr  q1    , [%[Dptr], #16 ]\n\t")
        IF_N_GT(2 , "ldr  q2    , [%[Dptr], #32 ]\n\t")
        IF_N_GT(3 , "ldr  q3    , [%[Dptr], #48 ]\n\t")
        IF_N_GT(4 , "ldr  q4    , [%[Dptr], #64 ]\n\t")
        IF_N_GT(5 , "ldr  q5    , [%[Dptr], #80 ]\n\t")
        IF_N_GT(6 , "ldr  q6    , [%[Dptr], #96 ]\n\t")
        IF_N_GT(7 , "ldr  q7    , [%[Dptr], #112]\n\t")
        IF_N_GT(8 , "ldr  q8    , [%[Dptr], #128]\n\t")
        IF_N_GT(9 , "ldr  q9    , [%[Dptr], #144]\n\t")
        IF_N_GT(10, "ldr  q10   , [%[Dptr], #160]\n\t")
        IF_N_GT(11, "ldr  q11   , [%[Dptr], #176]\n\t")

                    "fadd  v20.8h, v20.8h, v0.8h \n\t"
        IF_N_GT(1 , "fadd  v21.8h, v21.8h, v1.8h \n\t")
        IF_N_GT(2 , "fadd  v22.8h, v22.8h, v2.8h \n\t")
        IF_N_GT(3 , "fadd  v23.8h, v23.8h, v3.8h \n\t")
        IF_N_GT(4 , "fadd  v24.8h, v24.8h, v4.8h \n\t")
        IF_N_GT(5 , "fadd  v25.8h, v25.8h, v5.8h \n\t")
        IF_N_GT(6 , "fadd  v26.8h, v26.8h, v6.8h \n\t")
        IF_N_GT(7 , "fadd  v27.8h, v27.8h, v7.8h \n\t")
        IF_N_GT(8 , "fadd  v28.8h, v28.8h, v8.8h \n\t")
        IF_N_GT(9 , "fadd  v29.8h, v29.8h, v9.8h \n\t")
        IF_N_GT(10, "fadd  v30.8h, v30.8h, v10.8h\n\t")
        IF_N_GT(11, "fadd  v31.8h, v31.8h, v11.8h\n\t")

                    "fmax v20.8h, v20.8h, v16.8h\n\t"
                    "str  q20   , [x11      ]\n\t"        
        IF_N_GT(1 , "fmax v21.8h, v21.8h, v16.8h\n\t" )
        IF_N_GT(1 , "str  q21   , [x11, #16 ]\n\t")        
        IF_N_GT(2 , "fmax v22.8h, v22.8h, v16.8h\n\t" )
        IF_N_GT(2 , "str  q22   , [x11, #32 ]\n\t")        
        IF_N_GT(3 , "fmax v23.8h, v23.8h, v16.8h\n\t" )
        IF_N_GT(3 , "str  q23   , [x11, #48 ]\n\t")        
        IF_N_GT(4 , "fmax v24.8h, v24.8h, v16.8h\n\t" )
        IF_N_GT(4 , "str  q24   , [x11, #64 ]\n\t")        
        IF_N_GT(5 , "fmax v25.8h, v25.8h, v16.8h\n\t" )
        IF_N_GT(5 , "str  q25   , [x11, #80 ]\n\t")        
        IF_N_GT(6 , "fmax v26.8h, v26.8h, v16.8h\n\t" )
        IF_N_GT(6 , "str  q26   , [x11, #96 ]\n\t")        
        IF_N_GT(7 , "fmax v27.8h, v27.8h, v16.8h\n\t" )
        IF_N_GT(7 , "str  q27   , [x11, #112]\n\t")        
        IF_N_GT(8 , "fmax v28.8h, v28.8h, v16.8h\n\t" )
        IF_N_GT(8 , "str  q28   , [x11, #128]\n\t")        
        IF_N_GT(9 , "fmax v29.8h, v29.8h, v16.8h\n\t" )
        IF_N_GT(9 , "str  q29   , [x11, #144]\n\t")         
        IF_N_GT(10, "fmax v30.8h, v30.8h, v16.8h\n\t"  )
        IF_N_GT(10, "str  q30   , [x11, #160]\n\t")         
        IF_N_GT(11, "fmax v31.8h, v31.8h, v16.8h\n\t"  )
        IF_N_GT(11, "str  q31   , [x11, #176]\n\t")

".else\n\t.if " PPL_ARM_SERVER_KERNEL_STR(FUSE_T()) " == 6\n\t"  // sum & relu6
        "fmov v16.8h, 6.0e+0\n\t"

                    "ldr  q0    , [%[Dptr]      ]\n\t"
        IF_N_GT(1 , "ldr  q1    , [%[Dptr], #16 ]\n\t")
        IF_N_GT(2 , "ldr  q2    , [%[Dptr], #32 ]\n\t")
        IF_N_GT(3 , "ldr  q3    , [%[Dptr], #48 ]\n\t")
        IF_N_GT(4 , "ldr  q4    , [%[Dptr], #64 ]\n\t")
        IF_N_GT(5 , "ldr  q5    , [%[Dptr], #80 ]\n\t")
        IF_N_GT(6 , "ldr  q6    , [%[Dptr], #96 ]\n\t")
        IF_N_GT(7 , "ldr  q7    , [%[Dptr], #112]\n\t")
        IF_N_GT(8 , "ldr  q8    , [%[Dptr], #128]\n\t")
        IF_N_GT(9 , "ldr  q9    , [%[Dptr], #144]\n\t")
        IF_N_GT(10, "ldr  q10   , [%[Dptr], #160]\n\t")
        IF_N_GT(11, "ldr  q11   , [%[Dptr], #176]\n\t")

                    "fadd  v20.8h, v20.8h, v0.8h \n\t"
        IF_N_GT(1 , "fadd  v21.8h, v21.8h, v1.8h \n\t")
        IF_N_GT(2 , "fadd  v22.8h, v22.8h, v2.8h \n\t")
        IF_N_GT(3 , "fadd  v23.8h, v23.8h, v3.8h \n\t")
        IF_N_GT(4 , "fadd  v24.8h, v24.8h, v4.8h \n\t")
        IF_N_GT(5 , "fadd  v25.8h, v25.8h, v5.8h \n\t")
        IF_N_GT(6 , "fadd  v26.8h, v26.8h, v6.8h \n\t")
        IF_N_GT(7 , "fadd  v27.8h, v27.8h, v7.8h \n\t")
        IF_N_GT(8 , "fadd  v28.8h, v28.8h, v8.8h \n\t")
        IF_N_GT(9 , "fadd  v29.8h, v29.8h, v9.8h \n\t")
        IF_N_GT(10, "fadd  v30.8h, v30.8h, v10.8h\n\t")
        IF_N_GT(11, "fadd  v31.8h, v31.8h, v11.8h\n\t")

                    "fmin v20.8h, v20.8h, v16.8h\n\t"
        IF_N_GT(1 , "fmin v21.8h, v21.8h, v16.8h\n\t")
        IF_N_GT(2 , "fmin v22.8h, v22.8h, v16.8h\n\t")
        IF_N_GT(3 , "fmin v23.8h, v23.8h, v16.8h\n\t")
        IF_N_GT(4 , "fmin v24.8h, v24.8h, v16.8h\n\t")
        IF_N_GT(5 , "fmin v25.8h, v25.8h, v16.8h\n\t")
        IF_N_GT(6 , "fmin v26.8h, v26.8h, v16.8h\n\t")
        IF_N_GT(7 , "fmin v27.8h, v27.8h, v16.8h\n\t")
        IF_N_GT(8 , "fmin v28.8h, v28.8h, v16.8h\n\t")
        IF_N_GT(9 , "fmin v29.8h, v29.8h, v16.8h\n\t")
        IF_N_GT(10, "fmin v30.8h, v30.8h, v16.8h\n\t")
        IF_N_GT(11, "fmin v31.8h, v31.8h, v16.8h\n\t")

        "eor v16.16b, v16.16b, v16.16b\n\t"
                    "fmax v20.8h, v20.8h, v16.8h\n\t"
                    "str  q20   , [x11      ]\n\t"        
        IF_N_GT(1 , "fmax v21.8h, v21.8h, v16.8h\n\t" )
        IF_N_GT(1 , "str  q21   , [x11, #16 ]\n\t")        
        IF_N_GT(2 , "fmax v22.8h, v22.8h, v16.8h\n\t" )
        IF_N_GT(2 , "str  q22   , [x11, #32 ]\n\t")        
        IF_N_GT(3 , "fmax v23.8h, v23.8h, v16.8h\n\t" )
        IF_N_GT(3 , "str  q23   , [x11, #48 ]\n\t")        
        IF_N_GT(4 , "fmax v24.8h, v24.8h, v16.8h\n\t" )
        IF_N_GT(4 , "str  q24   , [x11, #64 ]\n\t")        
        IF_N_GT(5 , "fmax v25.8h, v25.8h, v16.8h\n\t" )
        IF_N_GT(5 , "str  q25   , [x11, #80 ]\n\t")        
        IF_N_GT(6 , "fmax v26.8h, v26.8h, v16.8h\n\t" )
        IF_N_GT(6 , "str  q26   , [x11, #96 ]\n\t")        
        IF_N_GT(7 , "fmax v27.8h, v27.8h, v16.8h\n\t" )
        IF_N_GT(7 , "str  q27   , [x11, #112]\n\t")        
        IF_N_GT(8 , "fmax v28.8h, v28.8h, v16.8h\n\t" )
        IF_N_GT(8 , "str  q28   , [x11, #128]\n\t")        
        IF_N_GT(9 , "fmax v29.8h, v29.8h, v16.8h\n\t" )
        IF_N_GT(9 , "str  q29   , [x11, #144]\n\t")         
        IF_N_GT(10, "fmax v30.8h, v30.8h, v16.8h\n\t"  )
        IF_N_GT(10, "str  q30   , [x11, #160]\n\t")         
        IF_N_GT(11, "fmax v31.8h, v31.8h, v16.8h\n\t"  )
        IF_N_GT(11, "str  q31   , [x11, #176]\n\t")
".endif\n\t.endif\n\t.endif\n\t.endif\n\t.endif\n\t.endif\n\t"
    :
    : [Bptr]      "r" ((intptr_t)A), 
      [Aptr]      "r" ((intptr_t)B), 
      [const_ptr] "r" ((intptr_t)constant),
      [Dptr]      "r" ((intptr_t)DX),
      [Cptr]      "r" ((intptr_t)C), 
      [k]         "r" (K), 
      [Bstride]   "i" ((intptr_t)128),        // [Bstride] "i" ((intptr_t)CVL() * 8 * sizeof(__fp16))
      [Astride]   "r" ((intptr_t)ldb * 16)   // [Astride] "r" ((intptr_t)ldb * CVL() * sizeof(__fp16)),
    : "cc", "memory"
#ifndef __INTELLISENSE__
          ,
          "x9", "x10", "x11", "x12", "x13", "x14", "x15",
          
          "v0" , "v1" , "v2" , "v3" , "v4" , "v5" , "v6" , "v7" ,
          "v8" , "v9" , "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
#endif
    );
}

