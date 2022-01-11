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

#ifndef PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC
#define PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC(M_BLOCK0, N_BLOCK0) \
    ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<M_BLOCK0, N_BLOCK0>
#endif

#ifndef PPL_ARM_SERVER_KERNEL_PP_STR
#define PPL_ARM_SERVER_KERNEL_PP_STR(X) #X
#endif
#ifndef PPL_ARM_SERVER_KERNEL_STR
#define PPL_ARM_SERVER_KERNEL_STR(X) PPL_ARM_SERVER_KERNEL_PP_STR(X)
#endif

template<>
void PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC(M_BLOCK0(), N_BLOCK0()) (
    const __fp16* A,
    const __fp16* B,
    const __fp16* Vconst,
    __fp16* C,
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
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"
        "ld1 {v30.8h}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 2 \n\t"
        
        "ld1 {v30.8h}, [x10], %[Bnstride]\n\t"
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 3 \n\t"

        "ld1 {v29.8h}, [x10], %[Bnstride]\n\t"
        "ld1 {v30.8h}, [x10], %[Bnstride]\n\t"
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 4 \n\t"

        "ld1 {v28.8h}, [x10], %[Bnstride]\n\t"
        "ld1 {v29.8h}, [x10], %[Bnstride]\n\t"
        "ld1 {v30.8h}, [x10], %[Bnstride]\n\t"
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"

        "beq 1f\n\t"

    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 1 \n\t"
                    "ld1  {v15.8h}, [x11], %[Cstride]\n\t"
        IF_M_GT(1 , "ld1  {v16.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(2 , "ld1  {v17.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(3 , "ld1  {v18.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(4 , "ld1  {v19.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(5 , "ld1  {v20.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(6 , "ld1  {v21.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(7 , "ld1  {v22.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(8 , "ld1  {v23.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(9 , "ld1  {v24.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(10, "ld1  {v25.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(11, "ld1  {v26.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(12, "ld1  {v27.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(13, "ld1  {v28.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(14, "ld1  {v29.8h}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 2 \n\t"
                    "ld1  {v10.8h, v11.8h}, [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "ld1  {v12.8h, v13.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "ld1  {v14.8h, v15.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "ld1  {v16.8h, v17.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "ld1  {v18.8h, v19.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(5,  "ld1  {v20.8h, v21.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(6,  "ld1  {v22.8h, v23.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(7,  "ld1  {v24.8h, v25.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(8,  "ld1  {v26.8h, v27.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(9,  "ld1  {v28.8h, v29.8h}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 3 \n\t"
                    "ld1  {v7.8h, v8.8h, v9.8h},    [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "ld1  {v10.8h, v11.8h, v12.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "ld1  {v13.8h, v14.8h, v15.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "ld1  {v16.8h, v17.8h, v18.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "ld1  {v19.8h, v20.8h, v21.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(5,  "ld1  {v22.8h, v23.8h, v24.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(6,  "ld1  {v25.8h, v26.8h, v27.8h}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 4 \n\t"
                    "ld1  {v8.8h, v9.8h, v10.8h, v11.8h},   [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "ld1  {v12.8h, v13.8h, v14.8h, v15.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "ld1  {v16.8h, v17.8h, v18.8h, v19.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "ld1  {v20.8h, v21.8h, v22.8h, v23.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "ld1  {v24.8h, v25.8h, v26.8h, v27.8h}, [x11]            \n\t")
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
                    "ld1  {v10.8h, v11.8h}, [%[Vcnst_ptr]]\n\t"
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
                    "ld1  {v7.8h, v8.8h, v9.8h}, [%[Vcnst_ptr]]\n\t"
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
                    "ld1  {v8.8h, v9.8h, v10.8h, v11.8h}, [%[Vcnst_ptr]]\n\t"
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
        "cmp %[k], #8\n\t"
                    "ld1  {v0.8h},  [x9], %[Amstride]\n\t"
        IF_M_GT(1 , "ld1  {v1.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(2 , "ld1  {v2.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(3 , "ld1  {v3.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(4 , "ld1  {v4.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(5 , "ld1  {v5.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(6 , "ld1  {v6.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(7 , "ld1  {v7.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(8 , "ld1  {v8.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(9 , "ld1  {v9.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(10, "ld1  {v10.8h}, [x9], %[Amstride]\n\t")
        IF_M_GT(11, "ld1  {v11.8h}, [x9], %[Amstride]\n\t")
        IF_M_GT(12, "ld1  {v12.8h}, [x9], %[Amstride]\n\t")
        IF_M_GT(13, "ld1  {v13.8h}, [x9], %[Amstride]\n\t")
        IF_M_GT(14, "ld1  {v14.8h}, [x9]            \n\t")
        "ble 4f\n\t"
    
        ".align 3\n\t"
    "3:\n\t"
        "mov x9,  x12\n\t"
        "add x12, x12, %[Akstride]\n\t"
        "sub %[k], %[k], #8\n\t"  // scheduled earlier
        "cmp %[k], #8\n\t"

    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 1 \n\t"
#define N1_COMPUTE_BLOCK(k_idx, db_idx) \
                    "fmla v15.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v0.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t" \
        IF_M_GT(1 , "fmla v16.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v1.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(2 , "fmla v17.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v2.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(3 , "fmla v18.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v3.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(4 , "fmla v19.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v4.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(5 , "fmla v20.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v5.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(6 , "fmla v21.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v6.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(7 , "fmla v22.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v7.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(8 , "fmla v23.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v8.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(9 , "fmla v24.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v9.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(10, "fmla v25.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v10.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(11, "fmla v26.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v11.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(12, "fmla v27.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v12.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(13, "fmla v28.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v13.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(14, "fmla v29.8h, v3" PPL_ARM_SERVER_KERNEL_STR(db_idx) ".8h, v14.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t")

        N1_COMPUTE_BLOCK(0, 1)
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"

        N1_COMPUTE_BLOCK(1, 0)
        "ld1 {v30.8h}, [x10], %[Bkstride]\n\t"

        N1_COMPUTE_BLOCK(2, 1)
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"

        N1_COMPUTE_BLOCK(3, 0)
        "ld1 {v30.8h}, [x10], %[Bkstride]\n\t"

        N1_COMPUTE_BLOCK(4, 1)
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"

        N1_COMPUTE_BLOCK(5, 0)
        "ld1 {v30.8h}, [x10], %[Bkstride]\n\t"

        N1_COMPUTE_BLOCK(6, 1)
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"

        N1_COMPUTE_BLOCK(7, 0)
        "ld1 {v30.8h}, [x10], %[Bkstride]\n\t"
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 2 \n\t"
#define N2_COMPUTE_BLOCK(k_idx) \
                    "fmla v10.8h, v30.8h, v0.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t" \
        IF_M_GT(1 , "fmla v12.8h, v30.8h, v1.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(2 , "fmla v14.8h, v30.8h, v2.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(3 , "fmla v16.8h, v30.8h, v3.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(4 , "fmla v18.8h, v30.8h, v4.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(5 , "fmla v20.8h, v30.8h, v5.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(6 , "fmla v22.8h, v30.8h, v6.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(7 , "fmla v24.8h, v30.8h, v7.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(8 , "fmla v26.8h, v30.8h, v8.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(9 , "fmla v28.8h, v30.8h, v9.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        "ld1 {v30.8h}, [x10], %[Bnstride]\n\t" \
                    "fmla v11.8h, v31.8h, v0.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t" \
        IF_M_GT(1 , "fmla v13.8h, v31.8h, v1.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(2 , "fmla v15.8h, v31.8h, v2.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(3 , "fmla v17.8h, v31.8h, v3.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(4 , "fmla v19.8h, v31.8h, v4.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(5 , "fmla v21.8h, v31.8h, v5.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(6 , "fmla v23.8h, v31.8h, v6.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(7 , "fmla v25.8h, v31.8h, v7.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(8 , "fmla v27.8h, v31.8h, v8.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(9 , "fmla v29.8h, v31.8h, v9.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"

        N2_COMPUTE_BLOCK(0)
        N2_COMPUTE_BLOCK(1)
        N2_COMPUTE_BLOCK(2)
        N2_COMPUTE_BLOCK(3)
        N2_COMPUTE_BLOCK(4)
        N2_COMPUTE_BLOCK(5)
        N2_COMPUTE_BLOCK(6)
        N2_COMPUTE_BLOCK(7)

    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 3 \n\t"
#define N3_COMPUTE_BLOCK(k_idx) \
                    "fmla v7.8h,  v29.8h, v0.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t" \
        IF_M_GT(1 , "fmla v10.8h, v29.8h, v1.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(2 , "fmla v13.8h, v29.8h, v2.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(3 , "fmla v16.8h, v29.8h, v3.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(4 , "fmla v19.8h, v29.8h, v4.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(5 , "fmla v22.8h, v29.8h, v5.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(6 , "fmla v25.8h, v29.8h, v6.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        "ld1 {v29.8h}, [x10], %[Bnstride]\n\t" \
                    "fmla v8.8h,  v30.8h, v0.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t" \
        IF_M_GT(1 , "fmla v11.8h, v30.8h, v1.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(2 , "fmla v14.8h, v30.8h, v2.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(3 , "fmla v17.8h, v30.8h, v3.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(4 , "fmla v20.8h, v30.8h, v4.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(5 , "fmla v23.8h, v30.8h, v5.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(6 , "fmla v26.8h, v30.8h, v6.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        "ld1 {v30.8h}, [x10], %[Bnstride]\n\t" \
                    "fmla v9.8h,  v31.8h, v0.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t" \
        IF_M_GT(1 , "fmla v12.8h, v31.8h, v1.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(2 , "fmla v15.8h, v31.8h, v2.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(3 , "fmla v18.8h, v31.8h, v3.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(4 , "fmla v21.8h, v31.8h, v4.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(5 , "fmla v24.8h, v31.8h, v5.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(6 , "fmla v27.8h, v31.8h, v6.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"

        N3_COMPUTE_BLOCK(0)
        N3_COMPUTE_BLOCK(1)
        N3_COMPUTE_BLOCK(2)
        N3_COMPUTE_BLOCK(3)
        N3_COMPUTE_BLOCK(4)
        N3_COMPUTE_BLOCK(5)
        N3_COMPUTE_BLOCK(6)
        N3_COMPUTE_BLOCK(7)

    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 4 \n\t"
#define N4_COMPUTE_BLOCK(k_idx) \
                    "fmla v8.8h,  v28.8h, v0.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t" \
        IF_M_GT(1 , "fmla v12.8h, v28.8h, v1.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(2 , "fmla v16.8h, v28.8h, v2.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(3 , "fmla v20.8h, v28.8h, v3.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(4 , "fmla v24.8h, v28.8h, v4.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        "ld1 {v28.8h}, [x10], %[Bnstride]\n\t" \
                    "fmla v9.8h,  v29.8h, v0.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t" \
        IF_M_GT(1 , "fmla v13.8h, v29.8h, v1.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(2 , "fmla v17.8h, v29.8h, v2.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(3 , "fmla v21.8h, v29.8h, v3.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(4 , "fmla v25.8h, v29.8h, v4.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        "ld1 {v29.8h}, [x10], %[Bnstride]\n\t" \
                    "fmla v10.8h, v30.8h, v0.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t" \
        IF_M_GT(1 , "fmla v14.8h, v30.8h, v1.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(2 , "fmla v18.8h, v30.8h, v2.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(3 , "fmla v22.8h, v30.8h, v3.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(4 , "fmla v26.8h, v30.8h, v4.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        "ld1 {v30.8h}, [x10], %[Bnstride]\n\t" \
                    "fmla v11.8h, v31.8h, v0.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t" \
        IF_M_GT(1 , "fmla v15.8h, v31.8h, v1.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(2 , "fmla v19.8h, v31.8h, v2.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(3 , "fmla v23.8h, v31.8h, v3.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        IF_M_GT(4 , "fmla v27.8h, v31.8h, v4.h[" PPL_ARM_SERVER_KERNEL_STR(k_idx) "]\n\t") \
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"

        N4_COMPUTE_BLOCK(0)
        N4_COMPUTE_BLOCK(1)
        N4_COMPUTE_BLOCK(2)
        N4_COMPUTE_BLOCK(3)
        N4_COMPUTE_BLOCK(4)
        N4_COMPUTE_BLOCK(5)
        N4_COMPUTE_BLOCK(6)
        N4_COMPUTE_BLOCK(7)

    ".endif\n\t"

                    "ld1  {v0.8h},  [x9], %[Amstride]\n\t"
        IF_M_GT(1 , "ld1  {v1.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(2 , "ld1  {v2.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(3 , "ld1  {v3.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(4 , "ld1  {v4.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(5 , "ld1  {v5.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(6 , "ld1  {v6.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(7 , "ld1  {v7.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(8 , "ld1  {v8.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(9 , "ld1  {v9.8h},  [x9], %[Amstride]\n\t")
        IF_M_GT(10, "ld1  {v10.8h}, [x9], %[Amstride]\n\t")
        IF_M_GT(11, "ld1  {v11.8h}, [x9], %[Amstride]\n\t")
        IF_M_GT(12, "ld1  {v12.8h}, [x9], %[Amstride]\n\t")
        IF_M_GT(13, "ld1  {v13.8h}, [x9], %[Amstride]\n\t")
        IF_M_GT(14, "ld1  {v14.8h}, [x9]            \n\t")
        "bgt 3b\n\t"
        
    "4:\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 1 \n\t"
        "subs %[k], %[k], #1\n\t"
        N1_COMPUTE_BLOCK(0, 1)
        "beq 5f\n\t"
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"

        "subs %[k], %[k], #1\n\t"
        N1_COMPUTE_BLOCK(1, 0)
        "beq 5f\n\t"
        "ld1 {v30.8h}, [x10], %[Bkstride]\n\t"

        "subs %[k], %[k], #1\n\t"
        N1_COMPUTE_BLOCK(2, 1)
        "beq 5f\n\t"
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"

        "subs %[k], %[k], #1\n\t"
        N1_COMPUTE_BLOCK(3, 0)
        "beq 5f\n\t"
        "ld1 {v30.8h}, [x10], %[Bkstride]\n\t"

        "subs %[k], %[k], #1\n\t"
        N1_COMPUTE_BLOCK(4, 1)
        "beq 5f\n\t"
        "ld1 {v31.8h}, [x10], %[Bkstride]\n\t"

        "subs %[k], %[k], #1\n\t"
        N1_COMPUTE_BLOCK(5, 0)
        "beq 5f\n\t"
        "ld1 {v30.8h}, [x10]\n\t"

        "subs %[k], %[k], #1\n\t"
        N1_COMPUTE_BLOCK(6, 1)
        "beq 5f\n\t"

        N1_COMPUTE_BLOCK(7, 0)

    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 2 \n\t"
        "subs %[k], %[k], #1\n\t"
        N2_COMPUTE_BLOCK(0)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N2_COMPUTE_BLOCK(1)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N2_COMPUTE_BLOCK(2)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N2_COMPUTE_BLOCK(3)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N2_COMPUTE_BLOCK(4)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N2_COMPUTE_BLOCK(5)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N2_COMPUTE_BLOCK(6)
        "beq 5f\n\t"

        N2_COMPUTE_BLOCK(7)
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 3 \n\t"
        "subs %[k], %[k], #1\n\t"
        N3_COMPUTE_BLOCK(0)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N3_COMPUTE_BLOCK(1)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N3_COMPUTE_BLOCK(2)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N3_COMPUTE_BLOCK(3)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N3_COMPUTE_BLOCK(4)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N3_COMPUTE_BLOCK(5)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N3_COMPUTE_BLOCK(6)
        "beq 5f\n\t"

        N3_COMPUTE_BLOCK(7)
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 4 \n\t"
        "subs %[k], %[k], #1\n\t"
        N4_COMPUTE_BLOCK(0)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N4_COMPUTE_BLOCK(1)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N4_COMPUTE_BLOCK(2)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N4_COMPUTE_BLOCK(3)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N4_COMPUTE_BLOCK(4)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N4_COMPUTE_BLOCK(5)
        "beq 5f\n\t"

        "subs %[k], %[k], #1\n\t"
        N4_COMPUTE_BLOCK(6)
        "beq 5f\n\t"

        N4_COMPUTE_BLOCK(7)
    ".endif\n\t"

        ".align 3\n\t"
    "5:\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 1 \n\t"
                    "st1  {v15.8h}, [x11], %[Cstride]\n\t"
        IF_M_GT(1 , "st1  {v16.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(2 , "st1  {v17.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(3 , "st1  {v18.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(4 , "st1  {v19.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(5 , "st1  {v20.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(6 , "st1  {v21.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(7 , "st1  {v22.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(8 , "st1  {v23.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(9 , "st1  {v24.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(10, "st1  {v25.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(11, "st1  {v26.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(12, "st1  {v27.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(13, "st1  {v28.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(14, "st1  {v29.8h}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 2 \n\t"
                    "st1  {v10.8h, v11.8h}, [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "st1  {v12.8h, v13.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "st1  {v14.8h, v15.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "st1  {v16.8h, v17.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "st1  {v18.8h, v19.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(5,  "st1  {v20.8h, v21.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(6,  "st1  {v22.8h, v23.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(7,  "st1  {v24.8h, v25.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(8,  "st1  {v26.8h, v27.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(9,  "st1  {v28.8h, v29.8h}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 3 \n\t"
                    "st1  {v7.8h, v8.8h, v9.8h},    [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "st1  {v10.8h, v11.8h, v12.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "st1  {v13.8h, v14.8h, v15.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "st1  {v16.8h, v17.8h, v18.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "st1  {v19.8h, v20.8h, v21.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(5,  "st1  {v22.8h, v23.8h, v24.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(6,  "st1  {v25.8h, v26.8h, v27.8h}, [x11]            \n\t")
    ".endif\n\t"
    ".if " PPL_ARM_SERVER_KERNEL_STR(N_BLOCK0()) " == 4 \n\t"
                    "st1  {v8.8h, v9.8h, v10.8h, v11.8h},   [x11], %[Cstride]\n\t"
        IF_M_GT(1,  "st1  {v12.8h, v13.8h, v14.8h, v15.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(2,  "st1  {v16.8h, v17.8h, v18.8h, v19.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(3,  "st1  {v20.8h, v21.8h, v22.8h, v23.8h}, [x11], %[Cstride]\n\t")
        IF_M_GT(4,  "st1  {v24.8h, v25.8h, v26.8h, v27.8h}, [x11]            \n\t")
    ".endif\n\t"
    :
    : [Aptr]      "r" ((intptr_t)A), 
      [Bptr]      "r" ((intptr_t)B), 
      [Vcnst_ptr] "r" ((intptr_t)Vconst),
      [Cptr]      "r" ((intptr_t)C), 
      [k]         "r" (K), 
      [Amstride]  "r" ((intptr_t)a_m_stride * 2),
      [Akstride]  "r" ((intptr_t)a_k_stride * 2),
      [Bkstride]  "r" ((intptr_t)(b_k_stride * 2 - (N_BLOCK0()-1) * b_n_stride * 2)),
      [Bnstride]  "r" ((intptr_t)b_n_stride * 2),
      [Cstride]   "r" ((intptr_t)ldc * 2),
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
#undef IF_M_GT
#undef N1_COMPUTE_BLOCK
#undef N2_COMPUTE_BLOCK
#undef N3_COMPUTE_BLOCK
#undef N4_COMPUTE_BLOCK
}
