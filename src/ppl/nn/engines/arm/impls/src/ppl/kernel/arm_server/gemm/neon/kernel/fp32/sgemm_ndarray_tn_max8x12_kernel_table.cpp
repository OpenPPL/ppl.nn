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

#include "ppl/kernel/arm_server/gemm/neon/kernel/fp32/sgemm_ndarray_kernel.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <int64_t prefetch_a, int64_t prefetch_b, int64_t init_t, int64_t m_block, int64_t n_block>
void sgemm_ndarray_tn_max8x12_kernel_func(
    const float* A, 
    const float* B, 
    const int64_t K, 
    const int64_t lda, 
    const int64_t ldb, 
    const int64_t ldc, 
    float* C);

#define PREFETCH_A()    0   // not prefetch A
    #define PREFETCH_B()    0   // not prefetch B
        #define INIT_T()    0   // init C as 0
            #include "sgemm_ndarray_tn_max8x12_kernel_generate.inc"
        #undef  INIT_T
        #define INIT_T()    1   // init C by load
            #include "sgemm_ndarray_tn_max8x12_kernel_generate.inc"
        #undef  INIT_T
    #undef  PREFETCH_B
    #define PREFETCH_B()    1   // prefetch B
        #define INIT_T()    0   // init C as 0
            #include "sgemm_ndarray_tn_max8x12_kernel_generate.inc"
        #undef  INIT_T
        #define INIT_T()    1   // init C by load
            #include "sgemm_ndarray_tn_max8x12_kernel_generate.inc"
        #undef  INIT_T
    #undef  PREFETCH_B
#undef  PREFETCH_A
#define PREFETCH_A()    1   // prefetch A
        #define PREFETCH_B()    0   // not prefetch B
        #define INIT_T()    0   // init C as 0
            #include "sgemm_ndarray_tn_max8x12_kernel_generate.inc"
        #undef  INIT_T
        #define INIT_T()    1   // init C by load
            #include "sgemm_ndarray_tn_max8x12_kernel_generate.inc"
        #undef  INIT_T
    #undef  PREFETCH_B
    #define PREFETCH_B()    1   // prefetch B
        #define INIT_T()    0   // init C as 0
            #include "sgemm_ndarray_tn_max8x12_kernel_generate.inc"
        #undef  INIT_T
        #define INIT_T()    1   // init C by load
            #include "sgemm_ndarray_tn_max8x12_kernel_generate.inc"
        #undef  INIT_T
    #undef  PREFETCH_B
#undef  PREFETCH_A

#define SGEMM_NDARRAY_TN_MAX8X12_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T) \
    {\
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 1, 1>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 1, 2>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 1, 3>, \
    }, \
    {\
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 2, 1>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 2, 2>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 2, 3>, \
    }, \
    {\
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 3, 1>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 3, 2>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 3, 3>, \
    }, \
    {\
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 4, 1>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 4, 2>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 4, 3>, \
    }, \
    {\
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 5, 1>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 5, 2>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 5, 3>, \
    }, \
    {\
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 6, 1>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 6, 2>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 6, 3>, \
    }, \
    {\
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 7, 1>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 7, 2>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 7, 3>, \
    }, \
    {\
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 8, 1>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 8, 2>, \
        sgemm_ndarray_tn_max8x12_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 8, 3>, \
    }, \

const sgemm_ndarray_kernel_func_t sgemm_ndarray_kernel_tn_max8x12_func_table[2][2][2][8][3] = {
#define PREFETCH_A      0
{
    #define PREFETCH_B      0
    {
        #define INIT_T          0
        {
            SGEMM_NDARRAY_TN_MAX8X12_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
        #define INIT_T          1
        {
            SGEMM_NDARRAY_TN_MAX8X12_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
    }, 
    #undef  PREFETCH_B
    #define PREFETCH_B      1
    {
        #define INIT_T          0
        {
            SGEMM_NDARRAY_TN_MAX8X12_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
        #define INIT_T          1
        {
            SGEMM_NDARRAY_TN_MAX8X12_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
    }, 
    #undef  PREFETCH_B
}, 
#undef  PREFETCH_A
#define PREFETCH_A      1
{
    #define PREFETCH_B      0
    {
        #define INIT_T          0
        {
            SGEMM_NDARRAY_TN_MAX8X12_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
        #define INIT_T          1
        {
            SGEMM_NDARRAY_TN_MAX8X12_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
    }, 
    #undef  PREFETCH_B
    #define PREFETCH_B      1
    {
        #define INIT_T          0
        {
            SGEMM_NDARRAY_TN_MAX8X12_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
        #define INIT_T          1
        {
            SGEMM_NDARRAY_TN_MAX8X12_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
    }, 
    #undef  PREFETCH_B
}, 
#undef  PREFETCH_A
};

}}}}
