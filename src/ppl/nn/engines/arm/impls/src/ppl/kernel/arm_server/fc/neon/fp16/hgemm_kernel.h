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

#ifndef PPL_ARM_SERVER_KERNEL_SRC_FP16_FC_SGEMM_KERNEL_H_
#define PPL_ARM_SERVER_KERNEL_SRC_FP16_FC_SGEMM_KERNEL_H_

#ifdef PPLNN_USE_ARMV8_2_FP16

#include <iostream>
#include <cstdlib>

template<const uint32_t m, const uint32_t n>
void ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func(
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
    const uint32_t fuse_flag);

#define PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(M_BLOCK0, N_BLOCK0) \
    template<> \
    void ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<M_BLOCK0, N_BLOCK0>(  \
        const __fp16* A,  \
        const __fp16* B,  \
        const __fp16* Vconst, \
        __fp16* C,  \
        const int64_t M,  \
        const int64_t N,  \
        const int64_t K,  \
        const int64_t a_m_stride,  \
        const int64_t a_k_stride,  \
        const int64_t b_k_stride,  \
        const int64_t b_n_stride,  \
        const int64_t ldc,  \
        const uint32_t load_c,  \
        const uint32_t fuse_flag)
    
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(15, 1);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(8,  1);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(4,  1);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(2,  1);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(1,  1);

PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(10, 2);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(8,  2);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(4,  2);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(2,  2);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(1,  2);

PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(7, 3);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(4, 3);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(2, 3);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(1, 3);

PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(5, 4);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(4, 4);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(2, 4);
PPL_ARM_SERVER_KERNEL_FP16_SGEMM_KERNEL_MXNX_FUNC_SIGNATURE(1, 4);


typedef void (*ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func_t)(
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
    const uint32_t first_c,
    const uint32_t fuse_flag);

const ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func_t hgemm_kernel_func_table[4][4] = {
    {
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<1, 1>,
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<1, 2>,
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<1, 3>,
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<1, 4>
    },
    {
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<2, 1>,
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<2, 2>,
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<2, 3>,
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<2, 4>
    },
    {
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<4, 1>,
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<4, 2>,
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<4, 3>,
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<4, 4>
    },
    {
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<8, 1>,
        ppl_arm_server_kernel_fp16_sgemm_kernel_mxnx_func<8, 2>,
        nullptr,
        nullptr
    },
};

#endif

#endif
