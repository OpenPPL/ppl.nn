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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_IM2COL_GEMM_FMA_CONV_GEMM_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_IM2COL_GEMM_FMA_CONV_GEMM_KERNEL_FP32_FMA_H_

#include "ppl/kernel/x86/common/internal_include.h"

#define KERNEL_FLAG_LOAD_C() (1 << 1)
#define KERNEL_FLAG_ADD_V() (1 << 8)
#define KERNEL_FLAG_ADD_H() (1 << 9)
#define KERNEL_FLAG_RELU()  (1 << 10)
#define KERNEL_FLAG_RELU6() (1 << 11)

#define PICK_PARAM(T, PARAM, IDX) *(T*)(PARAM + IDX)

#define PRIV_PARAM_LEN() 8
#define A_IDX()          0 // matrix A, const float*
#define B_IDX()          1 // matrix B, const float*
#define V_IDX()          2 // broadcast vector V, const float*
#define H_IDX()          3 // history matrix H (usually pass C here), const float*
#define C_IDX()          4 // matrix C, float*
#define M_IDX()          5 // critical: M % m_len == 0, int64_t

#define SHAR_PARAM_LEN()    6
#define K_IDX()             0 // int64_t
#define A_MBLK_STRIDE_IDX() 1 // int64_t
#define A_KBLK_STRIDE_IDX() 2 // int64_t
#define H_M_STRIDE_IDX()    3 // int64_t
#define C_M_STRIDE_IDX()    4 // int64_t
#define FLAGS_IDX()         5 // uint64_t

#define NT_STORE_OPT() 2

#define N_RF_BLK() 8

#define M6_N_DT_BLK() 16
#define M6_K_DT_BLK() 16
#define M6_N_RF()     2
#define M6_M_RF()     6

namespace ppl { namespace kernel { namespace x86 {

/* conv_gemm_kernel_fp32_fma data layout

--------------------------------------------------------------------------------

A: can be 2 layout, depend on A_MBLK_STRIDE and A_KBLK_STRIDE.
1). A_MBLK_STRIDE = MBLK(1~6) * KBLK(16)
    A_KBLK_STRIDE = M * KBLK(16)
    {
        {
            m_0[k_{0:15}],
            m_1[k_{0:15}],
            ...
            m_M[k_{0:15}],
        },
        {
            m_0[k_{16:31}],
            m_1[k_{16:31}],
            ...
            m_M[k_{16:31}],
        },
        ...
    }
2). A_MBLK_STRIDE = MBLK(1~6) * RND_UP(K, KBLK(16))
    A_KBLK_STRIDE = MBLK(1~6) * KBLK(16)
    {
        {
            {
                m_0[k_{0:15}],
                m_1[k_{0:15}],
                ...
                m_5[k_{0:15}],
            },
            {
                m_0[k_{16:31}],
                m_1[k_{16:31}],
                ...
                m_5[k_{16:31}],
            },
            ...
        },
        {
            {
                m_6[k_{0:15}],
                m_7[k_{0:15}],
                ...
                m_11[k_{0:15}],
            },
            {
                m_6[k_{16:31}],
                m_7[k_{16:31}],
                ...
                m_11[k_{16:31}],
            },
            ...
        },
        ...
    }

--------------------------------------------------------------------------------

B: only one layout
    {
        k_0[n_{0:15}],
        k_1[n_{0:15}],
        ...
        k_K[n_{0:15}],
    }

--------------------------------------------------------------------------------

V: only one layout
    {
        n_{0:7/15},
    }

--------------------------------------------------------------------------------

H: only one layout
    {
        m_0[n_{0:H_M_STRIDE-1}],
        m_1[n_{0:H_M_STRIDE-1}],
        ...
        m_M[n_{0:H_M_STRIDE-1}],
    }

--------------------------------------------------------------------------------

C: only one layout
    {
        m_0[n_{0:C_M_STRIDE-1}],
        m_1[n_{0:C_M_STRIDE-1}],
        ...
        m_M[n_{0:C_M_STRIDE-1}],
    }

*/

typedef void (*conv_gemm_kernel_fp32_fma_func_t)(const int64_t*, const int64_t*);

/*
    suggestion: K >= 128 is better.

        FMA = K*M*(NBLK/N_RF_BLK)
        LD = M*(NBLK/N_RF_BLK)
        ST = M*(NBLK/N_RF_BLK)
        POST_FUNC = 2to4*M*(NBLK/N_RF_BLK)
        IA >= K/(K+2to6)
*/
extern conv_gemm_kernel_fp32_fma_func_t
    conv_gemm_kernel_fp32_fma_table[M6_N_RF()][M6_M_RF()];

}}}; // namespace ppl::kernel::x86

#endif
