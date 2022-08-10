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

#include "ppl/kernel/arm_server/gemm/neon/kernel/fp16/hgemm_ndarray_kernel.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

#ifdef PPLNN_USE_ARMV8_2_FP16

template <int64_t prefetch_a, int64_t prefetch_b, int64_t init_t, int64_t m_block, int64_t n_block>
void hgemm_ndarray_tn_max8x24_kernel_func(
    const __fp16* A, 
    const __fp16* B, 
    const int64_t K, 
    const int64_t lda, 
    const int64_t ldb, 
    const int64_t ldc, 
    __fp16* C);

#define PREFETCH_A()    0   // not prefetch A
    #define PREFETCH_B()    0   // not prefetch B
        #define INIT_T()    0   // init C as 0
            #include "hgemm_ndarray_tn_max8x24_kernel_generate.inc"
        #undef  INIT_T
        #define INIT_T()    1   // init C by load
            #include "hgemm_ndarray_tn_max8x24_kernel_generate.inc"
        #undef  INIT_T
    #undef  PREFETCH_B
    #define PREFETCH_B()    1   // prefetch B
        #define INIT_T()    0   // init C as 0
            #include "hgemm_ndarray_tn_max8x24_kernel_generate.inc"
        #undef  INIT_T
        #define INIT_T()    1   // init C by load
            #include "hgemm_ndarray_tn_max8x24_kernel_generate.inc"
        #undef  INIT_T
    #undef  PREFETCH_B
#undef  PREFETCH_A
#define PREFETCH_A()    1   // prefetch A
        #define PREFETCH_B()    0   // not prefetch B
        #define INIT_T()    0   // init C as 0
            #include "hgemm_ndarray_tn_max8x24_kernel_generate.inc"
        #undef  INIT_T
        #define INIT_T()    1   // init C by load
            #include "hgemm_ndarray_tn_max8x24_kernel_generate.inc"
        #undef  INIT_T
    #undef  PREFETCH_B
    #define PREFETCH_B()    1   // prefetch B
        #define INIT_T()    0   // init C as 0
            #include "hgemm_ndarray_tn_max8x24_kernel_generate.inc"
        #undef  INIT_T
        #define INIT_T()    1   // init C by load
            #include "hgemm_ndarray_tn_max8x24_kernel_generate.inc"
        #undef  INIT_T
    #undef  PREFETCH_B
#undef  PREFETCH_A

#define HGEMM_NDARRAY_TN_MAX8X24_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T) \
    {\
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 1, 1>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 1, 2>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 1, 3>, \
    }, \
    {\
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 2, 1>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 2, 2>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 2, 3>, \
    }, \
    {\
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 3, 1>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 3, 2>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 3, 3>, \
    }, \
    {\
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 4, 1>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 4, 2>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 4, 3>, \
    }, \
    {\
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 5, 1>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 5, 2>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 5, 3>, \
    }, \
    {\
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 6, 1>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 6, 2>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 6, 3>, \
    }, \
    {\
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 7, 1>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 7, 2>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 7, 3>, \
    }, \
    {\
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 8, 1>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 8, 2>, \
        hgemm_ndarray_tn_max8x24_kernel_func<PREFETCH_A, PREFETCH_B, INIT_T, 8, 3>, \
    }, \

const hgemm_ndarray_kernel_func_t hgemm_ndarray_kernel_tn_max8x24_func_table[2][2][2][8][3] = {
#define PREFETCH_A      0
{
    #define PREFETCH_B      0
    {
        #define INIT_T          0
        {
            HGEMM_NDARRAY_TN_MAX8X24_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
        #define INIT_T          1
        {
            HGEMM_NDARRAY_TN_MAX8X24_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
    }, 
    #undef  PREFETCH_B
    #define PREFETCH_B      1
    {
        #define INIT_T          0
        {
            HGEMM_NDARRAY_TN_MAX8X24_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
        #define INIT_T          1
        {
            HGEMM_NDARRAY_TN_MAX8X24_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
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
            HGEMM_NDARRAY_TN_MAX8X24_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
        #define INIT_T          1
        {
            HGEMM_NDARRAY_TN_MAX8X24_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
    }, 
    #undef  PREFETCH_B
    #define PREFETCH_B      1
    {
        #define INIT_T          0
        {
            HGEMM_NDARRAY_TN_MAX8X24_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
        #define INIT_T          1
        {
            HGEMM_NDARRAY_TN_MAX8X24_KERNEL_FUNC_GROUP(PREFETCH_A, PREFETCH_B, INIT_T)
        }, 
        #undef  INIT_T
    }, 
    #undef  PREFETCH_B
}, 
#undef  PREFETCH_A
};

#endif

}}}}
