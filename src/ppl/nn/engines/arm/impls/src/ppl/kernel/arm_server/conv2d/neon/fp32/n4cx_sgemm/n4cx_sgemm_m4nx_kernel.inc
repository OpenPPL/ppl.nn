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
void sgemm_n4cx_kernel_m4nx_fp32_func<N_BLOCK0(), INIT_T(), FUSE_T()> (
    const float* A,
    const float* B,
    const float* constant,
    const float* DX,
    float* C,
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
        
        "ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x9]\n\t"
        "ldr  q0, [x10]\n\t"
        "subs %[k], %[k], #4\n\t"
        
".if " PPL_ARM_SERVER_KERNEL_STR(INIT_T()) " == 0\n\t"
                    "fmul v20.4s, v12.4s, v0.s[0]\n\t"
        IF_N_GT(1 , "ldr  q1,     [x10, #16]\n\t")
        IF_N_GT(1 , "fmul v21.4s, v12.4s, v1.s[0]\n\t")
        IF_N_GT(2 , "ldr  q2,     [x10, #32]\n\t")
        IF_N_GT(2 , "fmul v22.4s, v12.4s, v2.s[0]\n\t")
        IF_N_GT(3 , "ldr  q3,     [x10, #48]\n\t")
        IF_N_GT(3 , "fmul v23.4s, v12.4s, v3.s[0]\n\t")
        IF_N_GT(4 , "ldr  q4,     [x10, #64]\n\t")
        IF_N_GT(4 , "fmul v24.4s, v12.4s, v4.s[0]\n\t")
        IF_N_GT(5 , "ldr  q5,     [x10, #80]\n\t")
        IF_N_GT(5 , "fmul v25.4s, v12.4s, v5.s[0]\n\t")
        IF_N_GT(6 , "ldr  q6,     [x10, #96]\n\t")
        IF_N_GT(6 , "fmul v26.4s, v12.4s, v6.s[0]\n\t")
        IF_N_GT(7 , "ldr  q7,     [x10, #112]\n\t")
        IF_N_GT(7 , "fmul v27.4s, v12.4s, v7.s[0]\n\t")
        IF_N_GT(8 , "ldr  q8,     [x10, #128]\n\t")
        IF_N_GT(8 , "fmul v28.4s, v12.4s, v8.s[0]\n\t")
        IF_N_GT(9 , "ldr  q9,     [x10, #144]\n\t")
        IF_N_GT(9 , "fmul v29.4s, v12.4s, v9.s[0]\n\t")
        IF_N_GT(10, "ldr  q10,    [x10, #160]\n\t")
        IF_N_GT(10, "fmul v30.4s, v12.4s, v10.s[0]\n\t")
        IF_N_GT(11, "ldr  q11,    [x10, #176]\n\t")
        IF_N_GT(11, "fmul v31.4s, v12.4s, v11.s[0]\n\t")
".else\n\t.if " PPL_ARM_SERVER_KERNEL_STR(INIT_T()) " == 1\n\t"
                    "ldr  q20,    [%[const_ptr]]\n\t"
        IF_N_GT(1 , "ldr  q1,     [x10, #16]\n\t")
        IF_N_GT(1 , "mov  v21.16b, v20.16b\n\t")
        IF_N_GT(1 , "fmla v21.4s, v12.4s, v1.s[0]\n\t")
        IF_N_GT(2 , "ldr  q2,     [x10, #32]\n\t")
        IF_N_GT(2 , "mov  v22.16b, v20.16b\n\t")
        IF_N_GT(2 , "fmla v22.4s, v12.4s, v2.s[0]\n\t")
        IF_N_GT(3 , "ldr  q3,     [x10, #48]\n\t")
        IF_N_GT(3 , "mov  v23.16b, v20.16b\n\t")
        IF_N_GT(3 , "fmla v23.4s, v12.4s, v3.s[0]\n\t")
        IF_N_GT(4 , "ldr  q4,     [x10, #64]\n\t")
        IF_N_GT(4 , "mov  v24.16b, v20.16b\n\t")
        IF_N_GT(4 , "fmla v24.4s, v12.4s, v4.s[0]\n\t")
        IF_N_GT(5 , "ldr  q5,     [x10, #80]\n\t")
        IF_N_GT(5 , "mov  v25.16b, v20.16b\n\t")
        IF_N_GT(5 , "fmla v25.4s, v12.4s, v5.s[0]\n\t")
        IF_N_GT(6 , "ldr  q6,     [x10, #96]\n\t")
        IF_N_GT(6 , "mov  v26.16b, v20.16b\n\t")
        IF_N_GT(6 , "fmla v26.4s, v12.4s, v6.s[0]\n\t")
        IF_N_GT(7 , "ldr  q7,     [x10, #112]\n\t")
        IF_N_GT(7 , "mov  v27.16b, v20.16b\n\t")
        IF_N_GT(7 , "fmla v27.4s, v12.4s, v7.s[0]\n\t")
        IF_N_GT(8 , "ldr  q8,     [x10, #128]\n\t")
        IF_N_GT(8 , "mov  v28.16b, v20.16b\n\t")
        IF_N_GT(8 , "fmla v28.4s, v12.4s, v8.s[0]\n\t")
        IF_N_GT(9 , "ldr  q9,     [x10, #144]\n\t")
        IF_N_GT(9 , "mov  v29.16b, v20.16b\n\t")
        IF_N_GT(9 , "fmla v29.4s, v12.4s, v9.s[0]\n\t")
        IF_N_GT(10, "ldr  q10,    [x10, #160]\n\t")
        IF_N_GT(10, "mov  v30.16b, v20.16b\n\t")
        IF_N_GT(10, "fmla v30.4s, v12.4s, v10.s[0]\n\t")
        IF_N_GT(11, "ldr  q11,    [x10, #176]\n\t")
        IF_N_GT(11, "mov  v31.16b, v20.16b\n\t")
        IF_N_GT(11, "fmla v31.4s, v12.4s, v11.s[0]\n\t")
                    "fmla v20.4s, v12.4s, v0.s[0]\n\t"
".else\n\t.if " PPL_ARM_SERVER_KERNEL_STR(INIT_T()) " == 2\n\t"
                    "ldr  q20,    [x11]\n\t"
                    "fmla v20.4s, v12.4s, v0.s[0]\n\t"
        IF_N_GT(1 , "ldr  q1,     [x10, #16]\n\t")
        IF_N_GT(1 , "ldr  q21,    [x11, #16]\n\t")
        IF_N_GT(1 , "fmla v21.4s, v12.4s, v1.s[0]\n\t")
        IF_N_GT(2 , "ldr  q2,     [x10, #32]\n\t")
        IF_N_GT(2 , "ldr  q22,    [x11, #32]\n\t")
        IF_N_GT(2 , "fmla v22.4s, v12.4s, v2.s[0]\n\t")
        IF_N_GT(3 , "ldr  q3,     [x10, #48]\n\t")
        IF_N_GT(3 , "ldr  q23,    [x11, #48]\n\t")
        IF_N_GT(3 , "fmla v23.4s, v12.4s, v3.s[0]\n\t")
        IF_N_GT(4 , "ldr  q4,     [x10, #64]\n\t")
        IF_N_GT(4 , "ldr  q24,    [x11, #64]\n\t")
        IF_N_GT(4 , "fmla v24.4s, v12.4s, v4.s[0]\n\t")
        IF_N_GT(5 , "ldr  q5,     [x10, #80]\n\t")
        IF_N_GT(5 , "ldr  q25,    [x11, #80]\n\t")
        IF_N_GT(5 , "fmla v25.4s, v12.4s, v5.s[0]\n\t")
        IF_N_GT(6 , "ldr  q6,     [x10, #96]\n\t")
        IF_N_GT(6 , "ldr  q26,    [x11, #96]\n\t")
        IF_N_GT(6 , "fmla v26.4s, v12.4s, v6.s[0]\n\t")
        IF_N_GT(7 , "ldr  q7,     [x10, #112]\n\t")
        IF_N_GT(7 , "ldr  q27,    [x11, #112]\n\t")
        IF_N_GT(7 , "fmla v27.4s, v12.4s, v7.s[0]\n\t")
        IF_N_GT(8 , "ldr  q8,     [x10, #128]\n\t")
        IF_N_GT(8 , "ldr  q28,    [x11, #128]\n\t")
        IF_N_GT(8 , "fmla v28.4s, v12.4s, v8.s[0]\n\t")
        IF_N_GT(9 , "ldr  q9,     [x10, #144]\n\t")
        IF_N_GT(9 , "ldr  q29,    [x11, #144]\n\t")
        IF_N_GT(9 , "fmla v29.4s, v12.4s, v9.s[0]\n\t")
        IF_N_GT(10, "ldr  q10,    [x10, #160]\n\t")
        IF_N_GT(10, "ldr  q30,    [x11, #160]\n\t")
        IF_N_GT(10, "fmla v30.4s, v12.4s, v10.s[0]\n\t")
        IF_N_GT(11, "ldr  q11,    [x10, #176]\n\t")
        IF_N_GT(11, "ldr  q31,    [x11, #176]\n\t")
        IF_N_GT(11, "fmla v31.4s, v12.4s, v11.s[0]\n\t")
".endif\n\t.endif\n\t.endif\n\t"

                    "fmla v20.4s, v13.4s, v0.s[1]\n\t"
        IF_N_GT(1 , "fmla v21.4s, v13.4s, v1.s[1]\n\t")
        IF_N_GT(2 , "fmla v22.4s, v13.4s, v2.s[1]\n\t")
        IF_N_GT(3 , "fmla v23.4s, v13.4s, v3.s[1]\n\t")
        IF_N_GT(4 , "fmla v24.4s, v13.4s, v4.s[1]\n\t")
        IF_N_GT(5 , "fmla v25.4s, v13.4s, v5.s[1]\n\t")
        IF_N_GT(6 , "fmla v26.4s, v13.4s, v6.s[1]\n\t")
        IF_N_GT(7 , "fmla v27.4s, v13.4s, v7.s[1]\n\t")
        IF_N_GT(8 , "fmla v28.4s, v13.4s, v8.s[1]\n\t")
        IF_N_GT(9 , "fmla v29.4s, v13.4s, v9.s[1]\n\t")
        IF_N_GT(10, "fmla v30.4s, v13.4s, v10.s[1]\n\t")
        IF_N_GT(11, "fmla v31.4s, v13.4s, v11.s[1]\n\t")

        "beq 2f\n\t"

        ".align 3\n\t"
    "1:\n\t"
        "add x9, x9, %[Bstride]\n\t"
                    "fmla v20.4s, v14.4s, v0.s[2]\n\t"
        IF_N_GT(1 , "fmla v21.4s, v14.4s, v1.s[2]\n\t")
        IF_N_GT(2 , "fmla v22.4s, v14.4s, v2.s[2]\n\t")
        IF_N_GT(3 , "fmla v23.4s, v14.4s, v3.s[2]\n\t")
        "add x10, x10, %[Astride]\n\t"
        IF_N_GT(4 , "fmla v24.4s, v14.4s, v4.s[2]\n\t")
        IF_N_GT(5 , "fmla v25.4s, v14.4s, v5.s[2]\n\t")
        IF_N_GT(6 , "fmla v26.4s, v14.4s, v6.s[2]\n\t")
        IF_N_GT(7 , "fmla v27.4s, v14.4s, v7.s[2]\n\t")
        IF_N_GT(8 , "fmla v28.4s, v14.4s, v8.s[2]\n\t")
        IF_N_GT(9 , "fmla v29.4s, v14.4s, v9.s[2]\n\t")
        IF_N_GT(10, "fmla v30.4s, v14.4s, v10.s[2]\n\t")
        IF_N_GT(11, "fmla v31.4s, v14.4s, v11.s[2]\n\t")

                    "fmla v20.4s, v15.4s, v0.s[3]\n\t"
                    "ldr  q0,     [x10      ]\n\t"
        IF_N_GT(1 , "fmla v21.4s, v15.4s, v1.s[3]\n\t")
        IF_N_GT(1 , "ldr  q1,     [x10, #16 ]\n\t")
        IF_N_GT(2 , "fmla v22.4s, v15.4s, v2.s[3]\n\t")
        IF_N_GT(2 , "ldr  q2,     [x10, #32 ]\n\t")
        IF_N_GT(3 , "fmla v23.4s, v15.4s, v3.s[3]\n\t")
        IF_N_GT(3 , "ldr  q3,     [x10, #48 ]\n\t")
        IF_N_GT(4 , "fmla v24.4s, v15.4s, v4.s[3]\n\t")
        IF_N_GT(4 , "ldr  q4,     [x10, #64 ]\n\t")
        IF_N_GT(5 , "fmla v25.4s, v15.4s, v5.s[3]\n\t")
        IF_N_GT(5 , "ldr  q5,     [x10, #80 ]\n\t")
        IF_N_GT(6 , "fmla v26.4s, v15.4s, v6.s[3]\n\t")
        IF_N_GT(6 , "ldr  q6,     [x10, #96 ]\n\t")
        IF_N_GT(7 , "fmla v27.4s, v15.4s, v7.s[3]\n\t")
        IF_N_GT(7 , "ldr  q7,     [x10, #112]\n\t")
        IF_N_GT(8 , "fmla v28.4s, v15.4s, v8.s[3]\n\t")
        IF_N_GT(8 , "ldr  q8,     [x10, #128]\n\t")
        IF_N_GT(9 , "fmla v29.4s, v15.4s, v9.s[3]\n\t")
        IF_N_GT(9 , "ldr  q9,     [x10, #144]\n\t")
        IF_N_GT(10, "fmla v30.4s, v15.4s, v10.s[3]\n\t")
        IF_N_GT(10, "ldr  q10,    [x10, #160]\n\t")
        IF_N_GT(11, "fmla v31.4s, v15.4s, v11.s[3]\n\t")
        IF_N_GT(11, "ldr  q11,    [x10, #176]\n\t")
        "ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x9]\n\t"
        "subs %[k], %[k], #4\n\t"

                    "fmla v20.4s, v12.4s, v0.s[0]\n\t"
        IF_N_GT(1 , "fmla v21.4s, v12.4s, v1.s[0]\n\t")
        IF_N_GT(2 , "fmla v22.4s, v12.4s, v2.s[0]\n\t")
        IF_N_GT(3 , "fmla v23.4s, v12.4s, v3.s[0]\n\t")
        IF_N_GT(4 , "fmla v24.4s, v12.4s, v4.s[0]\n\t")
        IF_N_GT(5 , "fmla v25.4s, v12.4s, v5.s[0]\n\t")
        IF_N_GT(6 , "fmla v26.4s, v12.4s, v6.s[0]\n\t")
        IF_N_GT(7 , "fmla v27.4s, v12.4s, v7.s[0]\n\t")
        IF_N_GT(8 , "fmla v28.4s, v12.4s, v8.s[0]\n\t")
        IF_N_GT(9 , "fmla v29.4s, v12.4s, v9.s[0]\n\t")
        IF_N_GT(10, "fmla v30.4s, v12.4s, v10.s[0]\n\t")
        IF_N_GT(11, "fmla v31.4s, v12.4s, v11.s[0]\n\t")

                    "fmla v20.4s, v13.4s, v0.s[1]\n\t"
        IF_N_GT(1 , "fmla v21.4s, v13.4s, v1.s[1]\n\t")
        IF_N_GT(2 , "fmla v22.4s, v13.4s, v2.s[1]\n\t")
        IF_N_GT(3 , "fmla v23.4s, v13.4s, v3.s[1]\n\t")
        IF_N_GT(4 , "fmla v24.4s, v13.4s, v4.s[1]\n\t")
        IF_N_GT(5 , "fmla v25.4s, v13.4s, v5.s[1]\n\t")
        IF_N_GT(6 , "fmla v26.4s, v13.4s, v6.s[1]\n\t")
        IF_N_GT(7 , "fmla v27.4s, v13.4s, v7.s[1]\n\t")
        IF_N_GT(8 , "fmla v28.4s, v13.4s, v8.s[1]\n\t")
        IF_N_GT(9 , "fmla v29.4s, v13.4s, v9.s[1]\n\t")
        IF_N_GT(10, "fmla v30.4s, v13.4s, v10.s[1]\n\t")
        IF_N_GT(11, "fmla v31.4s, v13.4s, v11.s[1]\n\t")

        "bgt 1b\n\t"
        
        ".align 3\n\t"
    "2:\n\t"
        "prfm pstl1strm, [x11]\n\t"
        IF_N_GT(4 , "prfm pstl1strm, [x11, #64]\n\t")
        IF_N_GT(8 , "prfm pstl1strm, [x11, #128]\n\t")

                    "fmla v20.4s, v14.4s, v0.s[2]\n\t"
        IF_N_GT(1 , "fmla v21.4s, v14.4s, v1.s[2]\n\t")
        IF_N_GT(2 , "fmla v22.4s, v14.4s, v2.s[2]\n\t")
        IF_N_GT(3 , "fmla v23.4s, v14.4s, v3.s[2]\n\t")
        IF_N_GT(4 , "fmla v24.4s, v14.4s, v4.s[2]\n\t")
        IF_N_GT(5 , "fmla v25.4s, v14.4s, v5.s[2]\n\t")
        IF_N_GT(6 , "fmla v26.4s, v14.4s, v6.s[2]\n\t")
        IF_N_GT(7 , "fmla v27.4s, v14.4s, v7.s[2]\n\t")
        IF_N_GT(8 , "fmla v28.4s, v14.4s, v8.s[2]\n\t")
        IF_N_GT(9 , "fmla v29.4s, v14.4s, v9.s[2]\n\t")
        IF_N_GT(10, "fmla v30.4s, v14.4s, v10.s[2]\n\t")
        IF_N_GT(11, "fmla v31.4s, v14.4s, v11.s[2]\n\t")

                    "fmla v20.4s, v15.4s, v0.s[3]\n\t"
        IF_N_GT(1 , "fmla v21.4s, v15.4s, v1.s[3]\n\t")
        IF_N_GT(2 , "fmla v22.4s, v15.4s, v2.s[3]\n\t")
        IF_N_GT(3 , "fmla v23.4s, v15.4s, v3.s[3]\n\t")
        IF_N_GT(4 , "fmla v24.4s, v15.4s, v4.s[3]\n\t")
        IF_N_GT(5 , "fmla v25.4s, v15.4s, v5.s[3]\n\t")
        IF_N_GT(6 , "fmla v26.4s, v15.4s, v6.s[3]\n\t")
        IF_N_GT(7 , "fmla v27.4s, v15.4s, v7.s[3]\n\t")
        IF_N_GT(8 , "fmla v28.4s, v15.4s, v8.s[3]\n\t")
        IF_N_GT(9 , "fmla v29.4s, v15.4s, v9.s[3]\n\t")
        IF_N_GT(10, "fmla v30.4s, v15.4s, v10.s[3]\n\t")
        IF_N_GT(11, "fmla v31.4s, v15.4s, v11.s[3]\n\t")

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
                    "fmax v20.4s, v20.4s, v16.4s\n\t"
                    "str  q20   , [x11      ]\n\t"        
        IF_N_GT(1 , "fmax v21.4s, v21.4s, v16.4s\n\t" )
        IF_N_GT(1 , "str  q21   , [x11, #16 ]\n\t")        
        IF_N_GT(2 , "fmax v22.4s, v22.4s, v16.4s\n\t" )
        IF_N_GT(2 , "str  q22   , [x11, #32 ]\n\t")        
        IF_N_GT(3 , "fmax v23.4s, v23.4s, v16.4s\n\t" )
        IF_N_GT(3 , "str  q23   , [x11, #48 ]\n\t")        
        IF_N_GT(4 , "fmax v24.4s, v24.4s, v16.4s\n\t" )
        IF_N_GT(4 , "str  q24   , [x11, #64 ]\n\t")        
        IF_N_GT(5 , "fmax v25.4s, v25.4s, v16.4s\n\t" )
        IF_N_GT(5 , "str  q25   , [x11, #80 ]\n\t")        
        IF_N_GT(6 , "fmax v26.4s, v26.4s, v16.4s\n\t" )
        IF_N_GT(6 , "str  q26   , [x11, #96 ]\n\t")        
        IF_N_GT(7 , "fmax v27.4s, v27.4s, v16.4s\n\t" )
        IF_N_GT(7 , "str  q27   , [x11, #112]\n\t")        
        IF_N_GT(8 , "fmax v28.4s, v28.4s, v16.4s\n\t" )
        IF_N_GT(8 , "str  q28   , [x11, #128]\n\t")        
        IF_N_GT(9 , "fmax v29.4s, v29.4s, v16.4s\n\t" )
        IF_N_GT(9 , "str  q29   , [x11, #144]\n\t")         
        IF_N_GT(10, "fmax v30.4s, v30.4s, v16.4s\n\t"  )
        IF_N_GT(10, "str  q30   , [x11, #160]\n\t")         
        IF_N_GT(11, "fmax v31.4s, v31.4s, v16.4s\n\t"  )
        IF_N_GT(11, "str  q31   , [x11, #176]\n\t")

".else\n\t.if " PPL_ARM_SERVER_KERNEL_STR(FUSE_T()) " == 2\n\t"   // relu6
        "fmov s16, 6.0e+0\n\t"
        "dup v16.4s, v16.s[0]\n\t"
                    "fmin v20.4s, v20.4s, v16.4s\n\t"
        IF_N_GT(1 , "fmin v21.4s, v21.4s, v16.4s\n\t")
        IF_N_GT(2 , "fmin v22.4s, v22.4s, v16.4s\n\t")
        IF_N_GT(3 , "fmin v23.4s, v23.4s, v16.4s\n\t")
        IF_N_GT(4 , "fmin v24.4s, v24.4s, v16.4s\n\t")
        IF_N_GT(5 , "fmin v25.4s, v25.4s, v16.4s\n\t")
        IF_N_GT(6 , "fmin v26.4s, v26.4s, v16.4s\n\t")
        IF_N_GT(7 , "fmin v27.4s, v27.4s, v16.4s\n\t")
        IF_N_GT(8 , "fmin v28.4s, v28.4s, v16.4s\n\t")
        IF_N_GT(9 , "fmin v29.4s, v29.4s, v16.4s\n\t")
        IF_N_GT(10, "fmin v30.4s, v30.4s, v16.4s\n\t")
        IF_N_GT(11, "fmin v31.4s, v31.4s, v16.4s\n\t")

        "eor v16.16b, v16.16b, v16.16b\n\t"
                    "fmax v20.4s, v20.4s, v16.4s\n\t"
                    "str  q20   , [x11      ]\n\t"        
        IF_N_GT(1 , "fmax v21.4s, v21.4s, v16.4s\n\t" )
        IF_N_GT(1 , "str  q21   , [x11, #16 ]\n\t")        
        IF_N_GT(2 , "fmax v22.4s, v22.4s, v16.4s\n\t" )
        IF_N_GT(2 , "str  q22   , [x11, #32 ]\n\t")        
        IF_N_GT(3 , "fmax v23.4s, v23.4s, v16.4s\n\t" )
        IF_N_GT(3 , "str  q23   , [x11, #48 ]\n\t")        
        IF_N_GT(4 , "fmax v24.4s, v24.4s, v16.4s\n\t" )
        IF_N_GT(4 , "str  q24   , [x11, #64 ]\n\t")        
        IF_N_GT(5 , "fmax v25.4s, v25.4s, v16.4s\n\t" )
        IF_N_GT(5 , "str  q25   , [x11, #80 ]\n\t")        
        IF_N_GT(6 , "fmax v26.4s, v26.4s, v16.4s\n\t" )
        IF_N_GT(6 , "str  q26   , [x11, #96 ]\n\t")        
        IF_N_GT(7 , "fmax v27.4s, v27.4s, v16.4s\n\t" )
        IF_N_GT(7 , "str  q27   , [x11, #112]\n\t")        
        IF_N_GT(8 , "fmax v28.4s, v28.4s, v16.4s\n\t" )
        IF_N_GT(8 , "str  q28   , [x11, #128]\n\t")        
        IF_N_GT(9 , "fmax v29.4s, v29.4s, v16.4s\n\t" )
        IF_N_GT(9 , "str  q29   , [x11, #144]\n\t")         
        IF_N_GT(10, "fmax v30.4s, v30.4s, v16.4s\n\t"  )
        IF_N_GT(10, "str  q30   , [x11, #160]\n\t")         
        IF_N_GT(11, "fmax v31.4s, v31.4s, v16.4s\n\t"  )
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

                    "fadd  v20.4s, v20.4s, v0.4s \n\t"
        IF_N_GT(1 , "fadd  v21.4s, v21.4s, v1.4s \n\t")
        IF_N_GT(2 , "fadd  v22.4s, v22.4s, v2.4s \n\t")
        IF_N_GT(3 , "fadd  v23.4s, v23.4s, v3.4s \n\t")
        IF_N_GT(4 , "fadd  v24.4s, v24.4s, v4.4s \n\t")
        IF_N_GT(5 , "fadd  v25.4s, v25.4s, v5.4s \n\t")
        IF_N_GT(6 , "fadd  v26.4s, v26.4s, v6.4s \n\t")
        IF_N_GT(7 , "fadd  v27.4s, v27.4s, v7.4s \n\t")
        IF_N_GT(8 , "fadd  v28.4s, v28.4s, v8.4s \n\t")
        IF_N_GT(9 , "fadd  v29.4s, v29.4s, v9.4s \n\t")
        IF_N_GT(10, "fadd  v30.4s, v30.4s, v10.4s\n\t")
        IF_N_GT(11, "fadd  v31.4s, v31.4s, v11.4s\n\t")

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

                    "fadd  v20.4s, v20.4s, v0.4s \n\t"
        IF_N_GT(1 , "fadd  v21.4s, v21.4s, v1.4s \n\t")
        IF_N_GT(2 , "fadd  v22.4s, v22.4s, v2.4s \n\t")
        IF_N_GT(3 , "fadd  v23.4s, v23.4s, v3.4s \n\t")
        IF_N_GT(4 , "fadd  v24.4s, v24.4s, v4.4s \n\t")
        IF_N_GT(5 , "fadd  v25.4s, v25.4s, v5.4s \n\t")
        IF_N_GT(6 , "fadd  v26.4s, v26.4s, v6.4s \n\t")
        IF_N_GT(7 , "fadd  v27.4s, v27.4s, v7.4s \n\t")
        IF_N_GT(8 , "fadd  v28.4s, v28.4s, v8.4s \n\t")
        IF_N_GT(9 , "fadd  v29.4s, v29.4s, v9.4s \n\t")
        IF_N_GT(10, "fadd  v30.4s, v30.4s, v10.4s\n\t")
        IF_N_GT(11, "fadd  v31.4s, v31.4s, v11.4s\n\t")

                    "fmax v20.4s, v20.4s, v16.4s\n\t"
                    "str  q20   , [x11      ]\n\t"        
        IF_N_GT(1 , "fmax v21.4s, v21.4s, v16.4s\n\t" )
        IF_N_GT(1 , "str  q21   , [x11, #16 ]\n\t")        
        IF_N_GT(2 , "fmax v22.4s, v22.4s, v16.4s\n\t" )
        IF_N_GT(2 , "str  q22   , [x11, #32 ]\n\t")        
        IF_N_GT(3 , "fmax v23.4s, v23.4s, v16.4s\n\t" )
        IF_N_GT(3 , "str  q23   , [x11, #48 ]\n\t")        
        IF_N_GT(4 , "fmax v24.4s, v24.4s, v16.4s\n\t" )
        IF_N_GT(4 , "str  q24   , [x11, #64 ]\n\t")        
        IF_N_GT(5 , "fmax v25.4s, v25.4s, v16.4s\n\t" )
        IF_N_GT(5 , "str  q25   , [x11, #80 ]\n\t")        
        IF_N_GT(6 , "fmax v26.4s, v26.4s, v16.4s\n\t" )
        IF_N_GT(6 , "str  q26   , [x11, #96 ]\n\t")        
        IF_N_GT(7 , "fmax v27.4s, v27.4s, v16.4s\n\t" )
        IF_N_GT(7 , "str  q27   , [x11, #112]\n\t")        
        IF_N_GT(8 , "fmax v28.4s, v28.4s, v16.4s\n\t" )
        IF_N_GT(8 , "str  q28   , [x11, #128]\n\t")        
        IF_N_GT(9 , "fmax v29.4s, v29.4s, v16.4s\n\t" )
        IF_N_GT(9 , "str  q29   , [x11, #144]\n\t")         
        IF_N_GT(10, "fmax v30.4s, v30.4s, v16.4s\n\t"  )
        IF_N_GT(10, "str  q30   , [x11, #160]\n\t")         
        IF_N_GT(11, "fmax v31.4s, v31.4s, v16.4s\n\t"  )
        IF_N_GT(11, "str  q31   , [x11, #176]\n\t")

".else\n\t.if " PPL_ARM_SERVER_KERNEL_STR(FUSE_T()) " == 6\n\t"  // sum & relu6
        "fmov v16.4s, 6.0e+0\n\t"

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

                    "fadd  v20.4s, v20.4s, v0.4s \n\t"
        IF_N_GT(1 , "fadd  v21.4s, v21.4s, v1.4s \n\t")
        IF_N_GT(2 , "fadd  v22.4s, v22.4s, v2.4s \n\t")
        IF_N_GT(3 , "fadd  v23.4s, v23.4s, v3.4s \n\t")
        IF_N_GT(4 , "fadd  v24.4s, v24.4s, v4.4s \n\t")
        IF_N_GT(5 , "fadd  v25.4s, v25.4s, v5.4s \n\t")
        IF_N_GT(6 , "fadd  v26.4s, v26.4s, v6.4s \n\t")
        IF_N_GT(7 , "fadd  v27.4s, v27.4s, v7.4s \n\t")
        IF_N_GT(8 , "fadd  v28.4s, v28.4s, v8.4s \n\t")
        IF_N_GT(9 , "fadd  v29.4s, v29.4s, v9.4s \n\t")
        IF_N_GT(10, "fadd  v30.4s, v30.4s, v10.4s\n\t")
        IF_N_GT(11, "fadd  v31.4s, v31.4s, v11.4s\n\t")

                    "fmin v20.4s, v20.4s, v16.4s\n\t"
        IF_N_GT(1 , "fmin v21.4s, v21.4s, v16.4s\n\t")
        IF_N_GT(2 , "fmin v22.4s, v22.4s, v16.4s\n\t")
        IF_N_GT(3 , "fmin v23.4s, v23.4s, v16.4s\n\t")
        IF_N_GT(4 , "fmin v24.4s, v24.4s, v16.4s\n\t")
        IF_N_GT(5 , "fmin v25.4s, v25.4s, v16.4s\n\t")
        IF_N_GT(6 , "fmin v26.4s, v26.4s, v16.4s\n\t")
        IF_N_GT(7 , "fmin v27.4s, v27.4s, v16.4s\n\t")
        IF_N_GT(8 , "fmin v28.4s, v28.4s, v16.4s\n\t")
        IF_N_GT(9 , "fmin v29.4s, v29.4s, v16.4s\n\t")
        IF_N_GT(10, "fmin v30.4s, v30.4s, v16.4s\n\t")
        IF_N_GT(11, "fmin v31.4s, v31.4s, v16.4s\n\t")

        "eor v16.16b, v16.16b, v16.16b\n\t"
                    "fmax v20.4s, v20.4s, v16.4s\n\t"
                    "str  q20   , [x11      ]\n\t"        
        IF_N_GT(1 , "fmax v21.4s, v21.4s, v16.4s\n\t" )
        IF_N_GT(1 , "str  q21   , [x11, #16 ]\n\t")        
        IF_N_GT(2 , "fmax v22.4s, v22.4s, v16.4s\n\t" )
        IF_N_GT(2 , "str  q22   , [x11, #32 ]\n\t")        
        IF_N_GT(3 , "fmax v23.4s, v23.4s, v16.4s\n\t" )
        IF_N_GT(3 , "str  q23   , [x11, #48 ]\n\t")        
        IF_N_GT(4 , "fmax v24.4s, v24.4s, v16.4s\n\t" )
        IF_N_GT(4 , "str  q24   , [x11, #64 ]\n\t")        
        IF_N_GT(5 , "fmax v25.4s, v25.4s, v16.4s\n\t" )
        IF_N_GT(5 , "str  q25   , [x11, #80 ]\n\t")        
        IF_N_GT(6 , "fmax v26.4s, v26.4s, v16.4s\n\t" )
        IF_N_GT(6 , "str  q26   , [x11, #96 ]\n\t")        
        IF_N_GT(7 , "fmax v27.4s, v27.4s, v16.4s\n\t" )
        IF_N_GT(7 , "str  q27   , [x11, #112]\n\t")        
        IF_N_GT(8 , "fmax v28.4s, v28.4s, v16.4s\n\t" )
        IF_N_GT(8 , "str  q28   , [x11, #128]\n\t")        
        IF_N_GT(9 , "fmax v29.4s, v29.4s, v16.4s\n\t" )
        IF_N_GT(9 , "str  q29   , [x11, #144]\n\t")         
        IF_N_GT(10, "fmax v30.4s, v30.4s, v16.4s\n\t"  )
        IF_N_GT(10, "str  q30   , [x11, #160]\n\t")         
        IF_N_GT(11, "fmax v31.4s, v31.4s, v16.4s\n\t"  )
        IF_N_GT(11, "str  q31   , [x11, #176]\n\t")
        
".endif\n\t.endif\n\t.endif\n\t.endif\n\t.endif\n\t.endif\n\t"
    :
    : [Bptr]      "r" ((intptr_t)A), 
      [Aptr]      "r" ((intptr_t)B), 
      [const_ptr] "r" ((intptr_t)constant),
      [Dptr]      "r" ((intptr_t)DX),
      [Cptr]      "r" ((intptr_t)C), 
      [k]         "r" (K), 
      [Bstride]   "i" ((intptr_t)64),        // [Bstride] "i" ((intptr_t)CVL() * 4 * sizeof(float))
      [Astride]   "r" ((intptr_t)ldb * 16)   // [Astride] "r" ((intptr_t)ldb * CVL() * sizeof(float)),
    : "cc", "memory"
#ifndef __INTELLISENSE__
          ,
          "x9", "x10", "x11", "x12", "x13", "x14", "x15",
          
          "v0" , "v1" , "v2" , "v3" , "v4" , "v5" , "v6" , "v7" ,
          "v8" , "v9" , "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", /*"v17", "v18", "v19",*/ "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
#endif
    );
}

