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

#ifndef PPL_ARM_SERVER_KERNEL_FP32_SGEMM_KERNEL_MXNX_FUNC
#define PPL_ARM_SERVER_KERNEL_FP32_SGEMM_KERNEL_MXNX_FUNC(M_BLOCK0, N_BLOCK0) \
    ppl_arm_server_kernel_fp32_sgemm_kernel_mxnx_func<M_BLOCK0, N_BLOCK0>
#endif

#ifndef PPL_ARM_SERVER_KERNEL_PP_STR
#define PPL_ARM_SERVER_KERNEL_PP_STR(X) #X
#endif
#ifndef PPL_ARM_SERVER_KERNEL_STR
#define PPL_ARM_SERVER_KERNEL_STR(X) PPL_ARM_SERVER_KERNEL_PP_STR(X)
#endif

template<>
void PPL_ARM_SERVER_KERNEL_FP32_SGEMM_KERNEL_MXNX_FUNC(M_BLOCK0(), N_BLOCK0()) (
    const float* A,
    const float* B,
    const float* Vconst,
    float* C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t a_m_stride,
    const int64_t a_k_stride,
    const int64_t b_k_stride,
    const int64_t b_n_stride,
    const int64_t ldc,
    const uint32_t load_c,
    const uint32_t fuse_flag)
{
#define IF_M_GT(M0, INSTRUCTION) ".if " PPL_ARM_SERVER_KERNEL_STR(M_BLOCK0()) " > " #M0 "\n\t  " INSTRUCTION ".endif\n\t"

    asm volatile (
        ".align 3\n\t"
        "mov x9,  %[Aptr]\n\t"
        "mov x10, %[Bptr]\n\t"
        "mov x11, %[Cptr]\n\t"
        "add x12, %[Aptr], %[Akstride]\n\t"
        
        "cmp %[load_c], #0\n\t"
    
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 1 \n\t"
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
        "ld1 {v30.4s}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 2 \n\t"
        
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 3 \n\t"

        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 4 \n\t"

        "ld1 {v28.4s}, [x10], %[Bnstride]\n\t"
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"

        "beq 1f\n\t"

    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 1 \n\t"
                    "ld1  {v15.4s}, [x11], %[Cstride]\n\t"
        IF_M_GT(1 , "ld1  {v16.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(2 , "ld1  {v17.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(3 , "ld1  {v18.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(4 , "ld1  {v19.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(5 , "ld1  {v20.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(6 , "ld1  {v21.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(7 , "ld1  {v22.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(8 , "ld1  {v23.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(9 , "ld1  {v24.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(10, "ld1  {v25.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(11, "ld1  {v26.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(12, "ld1  {v27.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(13, "ld1  {v28.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(14, "ld1  {v29.4s}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 2 \n\t"
                    "ld1  {v10.4s, v11.4s}, [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "ld1  {v12.4s, v13.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "ld1  {v14.4s, v15.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "ld1  {v16.4s, v17.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "ld1  {v18.4s, v19.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(5,  "ld1  {v20.4s, v21.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(6,  "ld1  {v22.4s, v23.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(7,  "ld1  {v24.4s, v25.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(8,  "ld1  {v26.4s, v27.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(9,  "ld1  {v28.4s, v29.4s}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 3 \n\t"
                    "ld1  {v7.4s, v8.4s, v9.4s},    [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "ld1  {v10.4s, v11.4s, v12.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "ld1  {v13.4s, v14.4s, v15.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "ld1  {v16.4s, v17.4s, v18.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "ld1  {v19.4s, v20.4s, v21.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(5,  "ld1  {v22.4s, v23.4s, v24.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(6,  "ld1  {v25.4s, v26.4s, v27.4s}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 4 \n\t"
                    "ld1  {v8.4s, v9.4s, v10.4s, v11.4s},   [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "ld1  {v12.4s, v13.4s, v14.4s, v15.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "ld1  {v16.4s, v17.4s, v18.4s, v19.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "ld1  {v20.4s, v21.4s, v22.4s, v23.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "ld1  {v24.4s, v25.4s, v26.4s, v27.4s}, [x11]            \n\t")
    ".endif\n\t"
        "mov x11, %[Cptr]\n\t"

        "b 2f\n\t"
        
    "1:\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 1 \n\t"
                    "ldr  q15, [%[Vcnst_ptr]]\n\t"
        IF_M_GT(1 , "mov  v16.16b, v15.16b\n\t")
        IF_M_GT(2 , "mov  v17.16b, v15.16b\n\t")
        IF_M_GT(3 , "mov  v18.16b, v15.16b\n\t")
        IF_M_GT(4 , "mov  v19.16b, v15.16b\n\t")
        IF_M_GT(5 , "mov  v20.16b, v15.16b\n\t")
        IF_M_GT(6 , "mov  v21.16b, v15.16b\n\t")
        IF_M_GT(7 , "mov  v22.16b, v15.16b\n\t")
        IF_M_GT(8 , "mov  v23.16b, v15.16b\n\t")
        IF_M_GT(9 , "mov  v24.16b, v15.16b\n\t")
        IF_M_GT(10, "mov  v25.16b, v15.16b\n\t")
        IF_M_GT(11, "mov  v26.16b, v15.16b\n\t")
        IF_M_GT(12, "mov  v27.16b, v15.16b\n\t")
        IF_M_GT(13, "mov  v28.16b, v15.16b\n\t")
        IF_M_GT(14, "mov  v29.16b, v15.16b\n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 2 \n\t"
                    "ld1  {v10.4s, v11.4s}, [%[Vcnst_ptr]]\n\t"
        IF_M_GT(1,  "mov  v12.16b, v10.16b\n\t")
        IF_M_GT(1,  "mov  v13.16b, v11.16b\n\t")
        IF_M_GT(2,  "mov  v14.16b, v10.16b\n\t")
        IF_M_GT(2,  "mov  v15.16b, v11.16b\n\t")
        IF_M_GT(3,  "mov  v16.16b, v10.16b\n\t")
        IF_M_GT(3,  "mov  v17.16b, v11.16b\n\t")
        IF_M_GT(4,  "mov  v18.16b, v10.16b\n\t")
        IF_M_GT(4,  "mov  v19.16b, v11.16b\n\t")
        IF_M_GT(5,  "mov  v20.16b, v10.16b\n\t")
        IF_M_GT(5,  "mov  v21.16b, v11.16b\n\t")
        IF_M_GT(6,  "mov  v22.16b, v10.16b\n\t")
        IF_M_GT(6,  "mov  v23.16b, v11.16b\n\t")
        IF_M_GT(7,  "mov  v24.16b, v10.16b\n\t")
        IF_M_GT(7,  "mov  v25.16b, v11.16b\n\t")
        IF_M_GT(8,  "mov  v26.16b, v10.16b\n\t")
        IF_M_GT(8,  "mov  v27.16b, v11.16b\n\t")
        IF_M_GT(9,  "mov  v28.16b, v10.16b\n\t")
        IF_M_GT(9,  "mov  v29.16b, v11.16b\n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 3 \n\t"
                    "ld1  {v7.4s, v8.4s, v9.4s}, [%[Vcnst_ptr]]\n\t"
        IF_M_GT(1,  "mov  v10.16b, v7.16b\n\t")
        IF_M_GT(1,  "mov  v11.16b, v8.16b\n\t")
        IF_M_GT(1,  "mov  v12.16b, v9.16b\n\t")
        IF_M_GT(2,  "mov  v13.16b, v7.16b\n\t")
        IF_M_GT(2,  "mov  v14.16b, v8.16b\n\t")
        IF_M_GT(2,  "mov  v15.16b, v9.16b\n\t")
        IF_M_GT(3,  "mov  v16.16b, v7.16b\n\t")
        IF_M_GT(3,  "mov  v17.16b, v8.16b\n\t")
        IF_M_GT(3,  "mov  v18.16b, v9.16b\n\t")
        IF_M_GT(4,  "mov  v19.16b, v7.16b\n\t")
        IF_M_GT(4,  "mov  v20.16b, v8.16b\n\t")
        IF_M_GT(4,  "mov  v21.16b, v9.16b\n\t")
        IF_M_GT(5,  "mov  v22.16b, v7.16b\n\t")
        IF_M_GT(5,  "mov  v23.16b, v8.16b\n\t")
        IF_M_GT(5,  "mov  v24.16b, v9.16b\n\t")
        IF_M_GT(6,  "mov  v25.16b, v7.16b\n\t")
        IF_M_GT(6,  "mov  v26.16b, v8.16b\n\t")
        IF_M_GT(6,  "mov  v27.16b, v9.16b\n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 4 \n\t"
                    "ld1  {v8.4s, v9.4s, v10.4s, v11.4s}, [%[Vcnst_ptr]]\n\t"
        IF_M_GT(1,  "mov  v12.16b, v8.16b\n\t")
        IF_M_GT(1,  "mov  v13.16b, v9.16b\n\t")
        IF_M_GT(1,  "mov  v14.16b, v10.16b\n\t")
        IF_M_GT(1,  "mov  v15.16b, v11.16b\n\t")
        IF_M_GT(2,  "mov  v16.16b, v8.16b\n\t")
        IF_M_GT(2,  "mov  v17.16b, v9.16b\n\t")
        IF_M_GT(2,  "mov  v18.16b, v10.16b\n\t")
        IF_M_GT(2,  "mov  v19.16b, v11.16b\n\t")
        IF_M_GT(3,  "mov  v20.16b, v8.16b\n\t")
        IF_M_GT(3,  "mov  v21.16b, v9.16b\n\t")
        IF_M_GT(3,  "mov  v22.16b, v10.16b\n\t")
        IF_M_GT(3,  "mov  v23.16b, v11.16b\n\t")
        IF_M_GT(4,  "mov  v24.16b, v8.16b\n\t")
        IF_M_GT(4,  "mov  v25.16b, v9.16b\n\t")
        IF_M_GT(4,  "mov  v26.16b, v10.16b\n\t")
        IF_M_GT(4,  "mov  v27.16b, v11.16b\n\t")
    ".endif\n\t"
        
        ".align 3\n\t"
    "2:\n\t"
        "cmp %[k], #4\n\t"
                    "ld1  {v0.4s},  [x9], %[Amstride]\n\t"
        IF_M_GT(1 , "ld1  {v1.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(2 , "ld1  {v2.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(3 , "ld1  {v3.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(4 , "ld1  {v4.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(5 , "ld1  {v5.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(6 , "ld1  {v6.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(7 , "ld1  {v7.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(8 , "ld1  {v8.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(9 , "ld1  {v9.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(10, "ld1  {v10.4s}, [x9], %[Amstride]\n\t")
        IF_M_GT(11, "ld1  {v11.4s}, [x9], %[Amstride]\n\t")
        IF_M_GT(12, "ld1  {v12.4s}, [x9], %[Amstride]\n\t")
        IF_M_GT(13, "ld1  {v13.4s}, [x9], %[Amstride]\n\t")
        IF_M_GT(14, "ld1  {v14.4s}, [x9]             \n\t")
        "ble 4f\n\t"
    
        ".align 3\n\t"
    "3:\n\t"
        "mov x9,  x12\n\t"
        "add x12, x12, %[Akstride]\n\t"
        "sub %[k], %[k], #4\n\t"  // scheduled earlier
        "cmp %[k], #4\n\t"

    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 1 \n\t"
                    "fmla v15.4s, v31.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v16.4s, v31.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v31.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v31.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v21.4s, v31.4s, v6.s[0]\n\t")
        IF_M_GT(7 , "fmla v22.4s, v31.4s, v7.s[0]\n\t")
        IF_M_GT(8 , "fmla v23.4s, v31.4s, v8.s[0]\n\t")
        IF_M_GT(9 , "fmla v24.4s, v31.4s, v9.s[0]\n\t")
        IF_M_GT(10, "fmla v25.4s, v31.4s, v10.s[0]\n\t")
        IF_M_GT(11, "fmla v26.4s, v31.4s, v11.s[0]\n\t")
        IF_M_GT(12, "fmla v27.4s, v31.4s, v12.s[0]\n\t")
        IF_M_GT(13, "fmla v28.4s, v31.4s, v13.s[0]\n\t")
        IF_M_GT(14, "fmla v29.4s, v31.4s, v14.s[0]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v15.4s, v30.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v16.4s, v30.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v30.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v30.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v30.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v21.4s, v30.4s, v6.s[1]\n\t")
        IF_M_GT(7 , "fmla v22.4s, v30.4s, v7.s[1]\n\t")
        IF_M_GT(8 , "fmla v23.4s, v30.4s, v8.s[1]\n\t")
        IF_M_GT(9 , "fmla v24.4s, v30.4s, v9.s[1]\n\t")
        IF_M_GT(10, "fmla v25.4s, v30.4s, v10.s[1]\n\t")
        IF_M_GT(11, "fmla v26.4s, v30.4s, v11.s[1]\n\t")
        IF_M_GT(12, "fmla v27.4s, v30.4s, v12.s[1]\n\t")
        IF_M_GT(13, "fmla v28.4s, v30.4s, v13.s[1]\n\t")
        IF_M_GT(14, "fmla v29.4s, v30.4s, v14.s[1]\n\t")
        "ld1 {v30.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v15.4s, v31.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v16.4s, v31.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v31.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v31.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v21.4s, v31.4s, v6.s[2]\n\t")
        IF_M_GT(7 , "fmla v22.4s, v31.4s, v7.s[2]\n\t")
        IF_M_GT(8 , "fmla v23.4s, v31.4s, v8.s[2]\n\t")
        IF_M_GT(9 , "fmla v24.4s, v31.4s, v9.s[2]\n\t")
        IF_M_GT(10, "fmla v25.4s, v31.4s, v10.s[2]\n\t")
        IF_M_GT(11, "fmla v26.4s, v31.4s, v11.s[2]\n\t")
        IF_M_GT(12, "fmla v27.4s, v31.4s, v12.s[2]\n\t")
        IF_M_GT(13, "fmla v28.4s, v31.4s, v13.s[2]\n\t")
        IF_M_GT(14, "fmla v29.4s, v31.4s, v14.s[2]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v15.4s, v30.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v16.4s, v30.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v30.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v30.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v30.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v21.4s, v30.4s, v6.s[3]\n\t")
        IF_M_GT(7 , "fmla v22.4s, v30.4s, v7.s[3]\n\t")
        IF_M_GT(8 , "fmla v23.4s, v30.4s, v8.s[3]\n\t")
        IF_M_GT(9 , "fmla v24.4s, v30.4s, v9.s[3]\n\t")
        IF_M_GT(10, "fmla v25.4s, v30.4s, v10.s[3]\n\t")
        IF_M_GT(11, "fmla v26.4s, v30.4s, v11.s[3]\n\t")
        IF_M_GT(12, "fmla v27.4s, v30.4s, v12.s[3]\n\t")
        IF_M_GT(13, "fmla v28.4s, v30.4s, v13.s[3]\n\t")
        IF_M_GT(14, "fmla v29.4s, v30.4s, v14.s[3]\n\t")
        "ld1 {v30.4s}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 2 \n\t"
                    "fmla v10.4s, v30.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v30.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v30.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v18.4s, v30.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v22.4s, v30.4s, v6.s[0]\n\t")
        IF_M_GT(7 , "fmla v24.4s, v30.4s, v7.s[0]\n\t")
        IF_M_GT(8 , "fmla v26.4s, v30.4s, v8.s[0]\n\t")
        IF_M_GT(9 , "fmla v28.4s, v30.4s, v9.s[0]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v31.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v31.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v21.4s, v31.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v23.4s, v31.4s, v6.s[0]\n\t")
        IF_M_GT(7 , "fmla v25.4s, v31.4s, v7.s[0]\n\t")
        IF_M_GT(8 , "fmla v27.4s, v31.4s, v8.s[0]\n\t")
        IF_M_GT(9 , "fmla v29.4s, v31.4s, v9.s[0]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v10.4s, v30.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v30.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v30.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v18.4s, v30.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v22.4s, v30.4s, v6.s[1]\n\t")
        IF_M_GT(7 , "fmla v24.4s, v30.4s, v7.s[1]\n\t")
        IF_M_GT(8 , "fmla v26.4s, v30.4s, v8.s[1]\n\t")
        IF_M_GT(9 , "fmla v28.4s, v30.4s, v9.s[1]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v31.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v31.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v21.4s, v31.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v23.4s, v31.4s, v6.s[1]\n\t")
        IF_M_GT(7 , "fmla v25.4s, v31.4s, v7.s[1]\n\t")
        IF_M_GT(8 , "fmla v27.4s, v31.4s, v8.s[1]\n\t")
        IF_M_GT(9 , "fmla v29.4s, v31.4s, v9.s[1]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v10.4s, v30.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v30.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v30.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v18.4s, v30.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v22.4s, v30.4s, v6.s[2]\n\t")
        IF_M_GT(7 , "fmla v24.4s, v30.4s, v7.s[2]\n\t")
        IF_M_GT(8 , "fmla v26.4s, v30.4s, v8.s[2]\n\t")
        IF_M_GT(9 , "fmla v28.4s, v30.4s, v9.s[2]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v31.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v31.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v21.4s, v31.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v23.4s, v31.4s, v6.s[2]\n\t")
        IF_M_GT(7 , "fmla v25.4s, v31.4s, v7.s[2]\n\t")
        IF_M_GT(8 , "fmla v27.4s, v31.4s, v8.s[2]\n\t")
        IF_M_GT(9 , "fmla v29.4s, v31.4s, v9.s[2]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v10.4s, v30.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v30.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v30.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v18.4s, v30.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v22.4s, v30.4s, v6.s[3]\n\t")
        IF_M_GT(7 , "fmla v24.4s, v30.4s, v7.s[3]\n\t")
        IF_M_GT(8 , "fmla v26.4s, v30.4s, v8.s[3]\n\t")
        IF_M_GT(9 , "fmla v28.4s, v30.4s, v9.s[3]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v31.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v31.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v21.4s, v31.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v23.4s, v31.4s, v6.s[3]\n\t")
        IF_M_GT(7 , "fmla v25.4s, v31.4s, v7.s[3]\n\t")
        IF_M_GT(8 , "fmla v27.4s, v31.4s, v8.s[3]\n\t")
        IF_M_GT(9 , "fmla v29.4s, v31.4s, v9.s[3]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 3 \n\t"
                    "fmla v7.4s,  v29.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v10.4s, v29.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v13.4s, v29.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v29.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v29.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v22.4s, v29.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v25.4s, v29.4s, v6.s[0]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v8.4s,  v30.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v11.4s, v30.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v30.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v20.4s, v30.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v23.4s, v30.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v26.4s, v30.4s, v6.s[0]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v31.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v31.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v21.4s, v31.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v24.4s, v31.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v27.4s, v31.4s, v6.s[0]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v7.4s,  v29.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v10.4s, v29.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v13.4s, v29.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v29.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v29.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v22.4s, v29.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v25.4s, v29.4s, v6.s[1]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v8.4s,  v30.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v11.4s, v30.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v30.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v20.4s, v30.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v23.4s, v30.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v26.4s, v30.4s, v6.s[1]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v31.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v31.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v21.4s, v31.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v24.4s, v31.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v27.4s, v31.4s, v6.s[1]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v7.4s,  v29.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v10.4s, v29.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v13.4s, v29.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v29.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v29.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v22.4s, v29.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v25.4s, v29.4s, v6.s[2]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v8.4s,  v30.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v11.4s, v30.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v30.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v20.4s, v30.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v23.4s, v30.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v26.4s, v30.4s, v6.s[2]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v31.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v31.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v21.4s, v31.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v24.4s, v31.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v27.4s, v31.4s, v6.s[2]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v7.4s,  v29.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v10.4s, v29.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v13.4s, v29.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v29.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v29.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v22.4s, v29.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v25.4s, v29.4s, v6.s[3]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v8.4s,  v30.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v11.4s, v30.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v30.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v20.4s, v30.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v23.4s, v30.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v26.4s, v30.4s, v6.s[3]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v31.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v31.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v21.4s, v31.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v24.4s, v31.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v27.4s, v31.4s, v6.s[3]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 4 \n\t"
                    "fmla v8.4s,  v28.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v28.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v16.4s, v28.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v20.4s, v28.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v24.4s, v28.4s, v4.s[0]\n\t")
        "ld1 {v28.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v29.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v29.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v29.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v21.4s, v29.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v25.4s, v29.4s, v4.s[0]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v10.4s, v30.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v14.4s, v30.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v18.4s, v30.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v22.4s, v30.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v26.4s, v30.4s, v4.s[0]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v15.4s, v31.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v19.4s, v31.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v23.4s, v31.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v27.4s, v31.4s, v4.s[0]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v8.4s,  v28.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v28.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v16.4s, v28.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v20.4s, v28.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v24.4s, v28.4s, v4.s[1]\n\t")
        "ld1 {v28.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v29.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v29.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v29.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v21.4s, v29.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v25.4s, v29.4s, v4.s[1]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v10.4s, v30.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v14.4s, v30.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v18.4s, v30.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v22.4s, v30.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v26.4s, v30.4s, v4.s[1]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v15.4s, v31.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v19.4s, v31.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v23.4s, v31.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v27.4s, v31.4s, v4.s[1]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v8.4s,  v28.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v28.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v16.4s, v28.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v20.4s, v28.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v24.4s, v28.4s, v4.s[2]\n\t")
        "ld1 {v28.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v29.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v29.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v29.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v21.4s, v29.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v25.4s, v29.4s, v4.s[2]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v10.4s, v30.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v14.4s, v30.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v18.4s, v30.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v22.4s, v30.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v26.4s, v30.4s, v4.s[2]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v15.4s, v31.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v19.4s, v31.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v23.4s, v31.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v27.4s, v31.4s, v4.s[2]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

                    "fmla v8.4s,  v28.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v28.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v16.4s, v28.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v20.4s, v28.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v24.4s, v28.4s, v4.s[3]\n\t")
        "ld1 {v28.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v29.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v29.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v29.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v21.4s, v29.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v25.4s, v29.4s, v4.s[3]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v10.4s, v30.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v14.4s, v30.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v18.4s, v30.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v22.4s, v30.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v26.4s, v30.4s, v4.s[3]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v15.4s, v31.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v19.4s, v31.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v23.4s, v31.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v27.4s, v31.4s, v4.s[3]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"

                    "ld1  {v0.4s},  [x9], %[Amstride]\n\t"
        IF_M_GT(1 , "ld1  {v1.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(2 , "ld1  {v2.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(3 , "ld1  {v3.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(4 , "ld1  {v4.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(5 , "ld1  {v5.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(6 , "ld1  {v6.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(7 , "ld1  {v7.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(8 , "ld1  {v8.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(9 , "ld1  {v9.4s},  [x9], %[Amstride]\n\t")
        IF_M_GT(10, "ld1  {v10.4s}, [x9], %[Amstride]\n\t")
        IF_M_GT(11, "ld1  {v11.4s}, [x9], %[Amstride]\n\t")
        IF_M_GT(12, "ld1  {v12.4s}, [x9], %[Amstride]\n\t")
        IF_M_GT(13, "ld1  {v13.4s}, [x9], %[Amstride]\n\t")
        IF_M_GT(14, "ld1  {v14.4s}, [x9]            \n\t")
        "bgt 3b\n\t"
        
    "4:\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 1 \n\t"
        "subs %[k], %[k], #1\n\t"
                    "fmla v15.4s, v31.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v16.4s, v31.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v31.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v31.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v21.4s, v31.4s, v6.s[0]\n\t")
        IF_M_GT(7 , "fmla v22.4s, v31.4s, v7.s[0]\n\t")
        IF_M_GT(8 , "fmla v23.4s, v31.4s, v8.s[0]\n\t")
        IF_M_GT(9 , "fmla v24.4s, v31.4s, v9.s[0]\n\t")
        IF_M_GT(10, "fmla v25.4s, v31.4s, v10.s[0]\n\t")
        IF_M_GT(11, "fmla v26.4s, v31.4s, v11.s[0]\n\t")
        IF_M_GT(12, "fmla v27.4s, v31.4s, v12.s[0]\n\t")
        IF_M_GT(13, "fmla v28.4s, v31.4s, v13.s[0]\n\t")
        IF_M_GT(14, "fmla v29.4s, v31.4s, v14.s[0]\n\t")
        "beq 5f\n\t"
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"

        "subs %[k], %[k], #1\n\t"
                    "fmla v15.4s, v30.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v16.4s, v30.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v30.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v30.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v30.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v21.4s, v30.4s, v6.s[1]\n\t")
        IF_M_GT(7 , "fmla v22.4s, v30.4s, v7.s[1]\n\t")
        IF_M_GT(8 , "fmla v23.4s, v30.4s, v8.s[1]\n\t")
        IF_M_GT(9 , "fmla v24.4s, v30.4s, v9.s[1]\n\t")
        IF_M_GT(10, "fmla v25.4s, v30.4s, v10.s[1]\n\t")
        IF_M_GT(11, "fmla v26.4s, v30.4s, v11.s[1]\n\t")
        IF_M_GT(12, "fmla v27.4s, v30.4s, v12.s[1]\n\t")
        IF_M_GT(13, "fmla v28.4s, v30.4s, v13.s[1]\n\t")
        IF_M_GT(14, "fmla v29.4s, v30.4s, v14.s[1]\n\t")
        "beq 5f\n\t"
        "ld1 {v30.4s}, [x10]\n\t"

        "subs %[k], %[k], #1\n\t"
                    "fmla v15.4s, v31.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v16.4s, v31.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v31.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v31.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v21.4s, v31.4s, v6.s[2]\n\t")
        IF_M_GT(7 , "fmla v22.4s, v31.4s, v7.s[2]\n\t")
        IF_M_GT(8 , "fmla v23.4s, v31.4s, v8.s[2]\n\t")
        IF_M_GT(9 , "fmla v24.4s, v31.4s, v9.s[2]\n\t")
        IF_M_GT(10, "fmla v25.4s, v31.4s, v10.s[2]\n\t")
        IF_M_GT(11, "fmla v26.4s, v31.4s, v11.s[2]\n\t")
        IF_M_GT(12, "fmla v27.4s, v31.4s, v12.s[2]\n\t")
        IF_M_GT(13, "fmla v28.4s, v31.4s, v13.s[2]\n\t")
        IF_M_GT(14, "fmla v29.4s, v31.4s, v14.s[2]\n\t")
        "beq 5f\n\t"

                    "fmla v15.4s, v30.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v16.4s, v30.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v30.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v30.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v30.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v21.4s, v30.4s, v6.s[3]\n\t")
        IF_M_GT(7 , "fmla v22.4s, v30.4s, v7.s[3]\n\t")
        IF_M_GT(8 , "fmla v23.4s, v30.4s, v8.s[3]\n\t")
        IF_M_GT(9 , "fmla v24.4s, v30.4s, v9.s[3]\n\t")
        IF_M_GT(10, "fmla v25.4s, v30.4s, v10.s[3]\n\t")
        IF_M_GT(11, "fmla v26.4s, v30.4s, v11.s[3]\n\t")
        IF_M_GT(12, "fmla v27.4s, v30.4s, v12.s[3]\n\t")
        IF_M_GT(13, "fmla v28.4s, v30.4s, v13.s[3]\n\t")
        IF_M_GT(14, "fmla v29.4s, v30.4s, v14.s[3]\n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 2 \n\t"
        "subs %[k], %[k], #1\n\t"
                    "fmla v10.4s, v30.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v30.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v30.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v18.4s, v30.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v22.4s, v30.4s, v6.s[0]\n\t")
        IF_M_GT(7 , "fmla v24.4s, v30.4s, v7.s[0]\n\t")
        IF_M_GT(8 , "fmla v26.4s, v30.4s, v8.s[0]\n\t")
        IF_M_GT(9 , "fmla v28.4s, v30.4s, v9.s[0]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v31.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v31.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v21.4s, v31.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v23.4s, v31.4s, v6.s[0]\n\t")
        IF_M_GT(7 , "fmla v25.4s, v31.4s, v7.s[0]\n\t")
        IF_M_GT(8 , "fmla v27.4s, v31.4s, v8.s[0]\n\t")
        IF_M_GT(9 , "fmla v29.4s, v31.4s, v9.s[0]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
                    "fmla v10.4s, v30.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v30.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v30.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v18.4s, v30.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v22.4s, v30.4s, v6.s[1]\n\t")
        IF_M_GT(7 , "fmla v24.4s, v30.4s, v7.s[1]\n\t")
        IF_M_GT(8 , "fmla v26.4s, v30.4s, v8.s[1]\n\t")
        IF_M_GT(9 , "fmla v28.4s, v30.4s, v9.s[1]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v31.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v31.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v21.4s, v31.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v23.4s, v31.4s, v6.s[1]\n\t")
        IF_M_GT(7 , "fmla v25.4s, v31.4s, v7.s[1]\n\t")
        IF_M_GT(8 , "fmla v27.4s, v31.4s, v8.s[1]\n\t")
        IF_M_GT(9 , "fmla v29.4s, v31.4s, v9.s[1]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
                    "fmla v10.4s, v30.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v30.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v30.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v18.4s, v30.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v22.4s, v30.4s, v6.s[2]\n\t")
        IF_M_GT(7 , "fmla v24.4s, v30.4s, v7.s[2]\n\t")
        IF_M_GT(8 , "fmla v26.4s, v30.4s, v8.s[2]\n\t")
        IF_M_GT(9 , "fmla v28.4s, v30.4s, v9.s[2]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v31.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v31.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v21.4s, v31.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v23.4s, v31.4s, v6.s[2]\n\t")
        IF_M_GT(7 , "fmla v25.4s, v31.4s, v7.s[2]\n\t")
        IF_M_GT(8 , "fmla v27.4s, v31.4s, v8.s[2]\n\t")
        IF_M_GT(9 , "fmla v29.4s, v31.4s, v9.s[2]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
        "beq 5f\n\t"

                    "fmla v10.4s, v30.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v30.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v30.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v18.4s, v30.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v20.4s, v30.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v22.4s, v30.4s, v6.s[3]\n\t")
        IF_M_GT(7 , "fmla v24.4s, v30.4s, v7.s[3]\n\t")
        IF_M_GT(8 , "fmla v26.4s, v30.4s, v8.s[3]\n\t")
        IF_M_GT(9 , "fmla v28.4s, v30.4s, v9.s[3]\n\t")
                    "fmla v11.4s, v31.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v31.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v31.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v31.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v21.4s, v31.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v23.4s, v31.4s, v6.s[3]\n\t")
        IF_M_GT(7 , "fmla v25.4s, v31.4s, v7.s[3]\n\t")
        IF_M_GT(8 , "fmla v27.4s, v31.4s, v8.s[3]\n\t")
        IF_M_GT(9 , "fmla v29.4s, v31.4s, v9.s[3]\n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 3 \n\t"
        "subs %[k], %[k], #1\n\t"
                    "fmla v7.4s,  v29.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v10.4s, v29.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v13.4s, v29.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v29.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v29.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v22.4s, v29.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v25.4s, v29.4s, v6.s[0]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v8.4s,  v30.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v11.4s, v30.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v30.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v20.4s, v30.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v23.4s, v30.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v26.4s, v30.4s, v6.s[0]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v31.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v31.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v21.4s, v31.4s, v4.s[0]\n\t")
        IF_M_GT(5 , "fmla v24.4s, v31.4s, v5.s[0]\n\t")
        IF_M_GT(6 , "fmla v27.4s, v31.4s, v6.s[0]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
                    "fmla v7.4s,  v29.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v10.4s, v29.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v13.4s, v29.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v29.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v29.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v22.4s, v29.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v25.4s, v29.4s, v6.s[1]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v8.4s,  v30.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v11.4s, v30.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v30.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v20.4s, v30.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v23.4s, v30.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v26.4s, v30.4s, v6.s[1]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v31.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v31.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v21.4s, v31.4s, v4.s[1]\n\t")
        IF_M_GT(5 , "fmla v24.4s, v31.4s, v5.s[1]\n\t")
        IF_M_GT(6 , "fmla v27.4s, v31.4s, v6.s[1]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
                    "fmla v7.4s,  v29.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v10.4s, v29.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v13.4s, v29.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v29.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v29.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v22.4s, v29.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v25.4s, v29.4s, v6.s[2]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v8.4s,  v30.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v11.4s, v30.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v30.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v20.4s, v30.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v23.4s, v30.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v26.4s, v30.4s, v6.s[2]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v31.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v31.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v21.4s, v31.4s, v4.s[2]\n\t")
        IF_M_GT(5 , "fmla v24.4s, v31.4s, v5.s[2]\n\t")
        IF_M_GT(6 , "fmla v27.4s, v31.4s, v6.s[2]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
        "beq 5f\n\t"

                    "fmla v7.4s,  v29.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v10.4s, v29.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v13.4s, v29.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v16.4s, v29.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v19.4s, v29.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v22.4s, v29.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v25.4s, v29.4s, v6.s[3]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v8.4s,  v30.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v11.4s, v30.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v14.4s, v30.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v17.4s, v30.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v20.4s, v30.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v23.4s, v30.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v26.4s, v30.4s, v6.s[3]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v31.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v31.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v15.4s, v31.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v18.4s, v31.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v21.4s, v31.4s, v4.s[3]\n\t")
        IF_M_GT(5 , "fmla v24.4s, v31.4s, v5.s[3]\n\t")
        IF_M_GT(6 , "fmla v27.4s, v31.4s, v6.s[3]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 4 \n\t"
        "subs %[k], %[k], #1\n\t"
                    "fmla v8.4s,  v28.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v28.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v16.4s, v28.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v20.4s, v28.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v24.4s, v28.4s, v4.s[0]\n\t")
        "ld1 {v28.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v29.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v29.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v29.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v21.4s, v29.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v25.4s, v29.4s, v4.s[0]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v10.4s, v30.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v14.4s, v30.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v18.4s, v30.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v22.4s, v30.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v26.4s, v30.4s, v4.s[0]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[0]\n\t"
        IF_M_GT(1 , "fmla v15.4s, v31.4s, v1.s[0]\n\t")
        IF_M_GT(2 , "fmla v19.4s, v31.4s, v2.s[0]\n\t")
        IF_M_GT(3 , "fmla v23.4s, v31.4s, v3.s[0]\n\t")
        IF_M_GT(4 , "fmla v27.4s, v31.4s, v4.s[0]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
                    "fmla v8.4s,  v28.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v28.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v16.4s, v28.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v20.4s, v28.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v24.4s, v28.4s, v4.s[1]\n\t")
        "ld1 {v28.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v29.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v29.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v29.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v21.4s, v29.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v25.4s, v29.4s, v4.s[1]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v10.4s, v30.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v14.4s, v30.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v18.4s, v30.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v22.4s, v30.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v26.4s, v30.4s, v4.s[1]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[1]\n\t"
        IF_M_GT(1 , "fmla v15.4s, v31.4s, v1.s[1]\n\t")
        IF_M_GT(2 , "fmla v19.4s, v31.4s, v2.s[1]\n\t")
        IF_M_GT(3 , "fmla v23.4s, v31.4s, v3.s[1]\n\t")
        IF_M_GT(4 , "fmla v27.4s, v31.4s, v4.s[1]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
                    "fmla v8.4s,  v28.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v28.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v16.4s, v28.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v20.4s, v28.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v24.4s, v28.4s, v4.s[2]\n\t")
        "ld1 {v28.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v29.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v29.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v29.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v21.4s, v29.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v25.4s, v29.4s, v4.s[2]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v10.4s, v30.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v14.4s, v30.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v18.4s, v30.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v22.4s, v30.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v26.4s, v30.4s, v4.s[2]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[2]\n\t"
        IF_M_GT(1 , "fmla v15.4s, v31.4s, v1.s[2]\n\t")
        IF_M_GT(2 , "fmla v19.4s, v31.4s, v2.s[2]\n\t")
        IF_M_GT(3 , "fmla v23.4s, v31.4s, v3.s[2]\n\t")
        IF_M_GT(4 , "fmla v27.4s, v31.4s, v4.s[2]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
        "beq 5f\n\t"

                    "fmla v8.4s,  v28.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v12.4s, v28.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v16.4s, v28.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v20.4s, v28.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v24.4s, v28.4s, v4.s[3]\n\t")
        "ld1 {v28.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v9.4s,  v29.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v13.4s, v29.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v17.4s, v29.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v21.4s, v29.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v25.4s, v29.4s, v4.s[3]\n\t")
        "ld1 {v29.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v10.4s, v30.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v14.4s, v30.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v18.4s, v30.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v22.4s, v30.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v26.4s, v30.4s, v4.s[3]\n\t")
        "ld1 {v30.4s}, [x10], %[Bnstride]\n\t"
                    "fmla v11.4s, v31.4s, v0.s[3]\n\t"
        IF_M_GT(1 , "fmla v15.4s, v31.4s, v1.s[3]\n\t")
        IF_M_GT(2 , "fmla v19.4s, v31.4s, v2.s[3]\n\t")
        IF_M_GT(3 , "fmla v23.4s, v31.4s, v3.s[3]\n\t")
        IF_M_GT(4 , "fmla v27.4s, v31.4s, v4.s[3]\n\t")
        "ld1 {v31.4s}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"

        ".align 3\n\t"
    "5:\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 1 \n\t"
                    "st1  {v15.4s}, [x11], %[Cstride]\n\t"
        IF_M_GT(1 , "st1  {v16.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(2 , "st1  {v17.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(3 , "st1  {v18.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(4 , "st1  {v19.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(5 , "st1  {v20.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(6 , "st1  {v21.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(7 , "st1  {v22.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(8 , "st1  {v23.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(9 , "st1  {v24.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(10, "st1  {v25.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(11, "st1  {v26.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(12, "st1  {v27.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(13, "st1  {v28.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(14, "st1  {v29.4s}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 2 \n\t"
                    "st1  {v10.4s, v11.4s}, [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "st1  {v12.4s, v13.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "st1  {v14.4s, v15.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "st1  {v16.4s, v17.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "st1  {v18.4s, v19.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(5,  "st1  {v20.4s, v21.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(6,  "st1  {v22.4s, v23.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(7,  "st1  {v24.4s, v25.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(8,  "st1  {v26.4s, v27.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(9,  "st1  {v28.4s, v29.4s}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 3 \n\t"
                    "st1  {v7.4s, v8.4s, v9.4s},    [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "st1  {v10.4s, v11.4s, v12.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "st1  {v13.4s, v14.4s, v15.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "st1  {v16.4s, v17.4s, v18.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "st1  {v19.4s, v20.4s, v21.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(5,  "st1  {v22.4s, v23.4s, v24.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(6,  "st1  {v25.4s, v26.4s, v27.4s}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 4 \n\t"
                    "st1  {v8.4s, v9.4s, v10.4s, v11.4s},   [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "st1  {v12.4s, v13.4s, v14.4s, v15.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "st1  {v16.4s, v17.4s, v18.4s, v19.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "st1  {v20.4s, v21.4s, v22.4s, v23.4s}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "st1  {v24.4s, v25.4s, v26.4s, v27.4s}, [x11]            \n\t")
    ".endif\n\t"
    :
    : [Aptr]      "r" ((intptr_t)A), 
      [Bptr]      "r" ((intptr_t)B), 
      [Vcnst_ptr] "r" ((intptr_t)Vconst),
      [Cptr]      "r" ((intptr_t)C), 
      [k]         "r" (K), 
      [Amstride]  "r" ((intptr_t)a_m_stride * 4),
      [Akstride]  "r" ((intptr_t)a_k_stride * 4),
      [Bkstride]  "r" ((intptr_t)(b_k_stride * 4 - (N_BLOCK0()-1) * b_n_stride * 4)),
      [Bnstride]  "r" ((intptr_t)b_n_stride * 4),
      [Cstride]   "r" ((intptr_t)ldc * 4),
      [load_c]    "r" ((intptr_t)load_c)
    : "cc", "memory"
#ifndef __INTELLISENSE__
          ,
          "x9", "x10", "x11", "x12", //"x13", "x14", "x15",
          
          "v0" , "v1" , "v2" , "v3" , "v4" , "v5" , "v6" , "v7" ,
          "v8" , "v9" , "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
#endif
    );
}
