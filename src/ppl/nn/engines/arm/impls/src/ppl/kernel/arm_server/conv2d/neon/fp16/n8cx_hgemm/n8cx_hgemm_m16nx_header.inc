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

template<const uint32_t n_block, const uint32_t init_type, const uint32_t fuse_type>
void hgemm_n8cx_kernel_m16nx_fp16_func( 
    const __fp16 * A,    
    const __fp16 * B,
    const __fp16 * constant,    
    const __fp16 * DX,   
          __fp16 * C,  
    const int64_t M,    
    const int64_t N,    
    const int64_t K,    
    const int64_t lda,  
    const int64_t ldb,  
    const int64_t lddx, 
    const int64_t ldc);

#ifndef HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE
#define HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0, INIT_T, FUSE_T) \
    template<> \
    void hgemm_n8cx_kernel_m16nx_fp16_func<N_BLOCK0, INIT_T, FUSE_T>( \
        const __fp16 * A,    \
        const __fp16 * B,    \
        const __fp16 * constant,    \
        const __fp16 * DX,   \
              __fp16 * C,    \
        const int64_t M,    \
        const int64_t N,    \
        const int64_t K,    \
        const int64_t lda,  \
        const int64_t ldb,  \
        const int64_t lddx, \
        const int64_t ldc   )
#endif
#ifndef HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE
#define HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0, INIT_T, FUSE_T) \
    HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0, INIT_T, FUSE_T)
#endif

#define FUSE_T() 0
    #define INIT_T() 0
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 1
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 2
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
#undef FUSE_T
#define FUSE_T() 1
    #define INIT_T() 0
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 1
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 2
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
#undef FUSE_T
#define FUSE_T() 2
    #define INIT_T() 0
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 1
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 2
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
#undef FUSE_T
#define FUSE_T() 4
    #define INIT_T() 0
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 1
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 2
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
#undef FUSE_T
#define FUSE_T() 5
    #define INIT_T() 0
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 1
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 2
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
#undef FUSE_T
#define FUSE_T() 6
    #define INIT_T() 0
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 1
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
    #define INIT_T() 2
        #define N_BLOCK0() 10
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 9
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 8
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 7
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 6
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 5
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 4
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 3
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 2
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
        #define N_BLOCK0() 1
            HGEMM_N8CX_KERNEL_M16NX_FP16_FUNC_SIGNATURE(N_BLOCK0(), INIT_T(), FUSE_T());
        #undef N_BLOCK0
    #undef INIT_T
#undef FUSE_T
