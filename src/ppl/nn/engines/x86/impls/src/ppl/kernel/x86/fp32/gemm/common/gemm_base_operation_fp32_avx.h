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

#ifndef __ST_PPL_KERNEL_X86_FP32_GEMM_AVX512_GEMM_M6_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_GEMM_AVX512_GEMM_M6_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/fp32/transpose/avx/transpose_fp32_avx.h"

namespace ppl { namespace kernel { namespace x86 {


template<gemm_m_type_t typeB>
void gemm_pack_b_n8_operation_fp32_avx(
    const float *B,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB)
{
    const int64_t unroll_n = 8;
    const int64_t unroll_k = 8;
    if (typeB == gemm_m_type::NOTRANS) { // B: (K, N) -> (N/8, K, 8)
        for (int64_t k = 0; k < K; k += unroll_k) {
            const int64_t k_eff = min(unroll_k, K - k);
            const float *src = B + k * ldb;
            float *dst = packedB + k * unroll_n;
            int64_t n = N;
            if (k_eff == unroll_k) {
                while (n >= unroll_n) {
                    n -= unroll_n;
                    _mm256_storeu_ps(dst + 0 * unroll_n, _mm256_loadu_ps(src + 0 * ldb));
                    _mm256_storeu_ps(dst + 1 * unroll_n, _mm256_loadu_ps(src + 1 * ldb));
                    _mm256_storeu_ps(dst + 2 * unroll_n, _mm256_loadu_ps(src + 2 * ldb));
                    _mm256_storeu_ps(dst + 3 * unroll_n, _mm256_loadu_ps(src + 3 * ldb));
                    _mm256_storeu_ps(dst + 4 * unroll_n, _mm256_loadu_ps(src + 4 * ldb));
                    _mm256_storeu_ps(dst + 5 * unroll_n, _mm256_loadu_ps(src + 5 * ldb));
                    _mm256_storeu_ps(dst + 6 * unroll_n, _mm256_loadu_ps(src + 6 * ldb));
                    _mm256_storeu_ps(dst + 7 * unroll_n, _mm256_loadu_ps(src + 7 * ldb));
                    src += unroll_n;
                    dst += K * unroll_n;
                }
                if (n & 4) {
                    _mm_storeu_ps(dst + 0 * unroll_n, _mm_loadu_ps(src + 0 * ldb));
                    _mm_storeu_ps(dst + 1 * unroll_n, _mm_loadu_ps(src + 1 * ldb));
                    _mm_storeu_ps(dst + 2 * unroll_n, _mm_loadu_ps(src + 2 * ldb));
                    _mm_storeu_ps(dst + 3 * unroll_n, _mm_loadu_ps(src + 3 * ldb));
                    _mm_storeu_ps(dst + 4 * unroll_n, _mm_loadu_ps(src + 4 * ldb));
                    _mm_storeu_ps(dst + 5 * unroll_n, _mm_loadu_ps(src + 5 * ldb));
                    _mm_storeu_ps(dst + 6 * unroll_n, _mm_loadu_ps(src + 6 * ldb));
                    _mm_storeu_ps(dst + 7 * unroll_n, _mm_loadu_ps(src + 7 * ldb));
                    src += 4;
                    dst += 4;
                }
                if (n & 2) {
                    *(int64_t*)(dst + 0 * unroll_n) = *(int64_t*)(src + 0 * ldb);
                    *(int64_t*)(dst + 1 * unroll_n) = *(int64_t*)(src + 1 * ldb);
                    *(int64_t*)(dst + 2 * unroll_n) = *(int64_t*)(src + 2 * ldb);
                    *(int64_t*)(dst + 3 * unroll_n) = *(int64_t*)(src + 3 * ldb);
                    *(int64_t*)(dst + 4 * unroll_n) = *(int64_t*)(src + 4 * ldb);
                    *(int64_t*)(dst + 5 * unroll_n) = *(int64_t*)(src + 5 * ldb);
                    *(int64_t*)(dst + 6 * unroll_n) = *(int64_t*)(src + 6 * ldb);
                    *(int64_t*)(dst + 7 * unroll_n) = *(int64_t*)(src + 7 * ldb);
                    src += 2;
                    dst += 2;
                }
                if (n & 1) {
                    *(dst + 0 * unroll_n) = *(src + 0 * ldb);
                    *(dst + 1 * unroll_n) = *(src + 1 * ldb);
                    *(dst + 2 * unroll_n) = *(src + 2 * ldb);
                    *(dst + 3 * unroll_n) = *(src + 3 * ldb);
                    *(dst + 4 * unroll_n) = *(src + 4 * ldb);
                    *(dst + 5 * unroll_n) = *(src + 5 * ldb);
                    *(dst + 6 * unroll_n) = *(src + 6 * ldb);
                    *(dst + 7 * unroll_n) = *(src + 7 * ldb);
                }
            } else {
                while (n >= unroll_n) {
                    n -= unroll_n;
                    const float *l_src = src;
                    float *l_dst = dst;
                    if (k_eff & 4) {
                        _mm256_storeu_ps(l_dst + 0 * unroll_n, _mm256_loadu_ps(l_src + 0 * ldb));
                        _mm256_storeu_ps(l_dst + 1 * unroll_n, _mm256_loadu_ps(l_src + 1 * ldb));
                        _mm256_storeu_ps(l_dst + 2 * unroll_n, _mm256_loadu_ps(l_src + 2 * ldb));
                        _mm256_storeu_ps(l_dst + 3 * unroll_n, _mm256_loadu_ps(l_src + 3 * ldb));
                        l_src += 4 * ldb;
                        l_dst += 4 * unroll_n;
                    }
                    if (k_eff & 2) {
                        _mm256_storeu_ps(l_dst + 0 * unroll_n, _mm256_loadu_ps(l_src + 0 * ldb));
                        _mm256_storeu_ps(l_dst + 1 * unroll_n, _mm256_loadu_ps(l_src + 1 * ldb));
                        l_src += 2 * ldb;
                        l_dst += 2 * unroll_n;
                    }
                    if (k_eff & 1) {
                        _mm256_storeu_ps(l_dst + 0 * unroll_n, _mm256_loadu_ps(l_src + 0 * ldb));
                    }
                    src += unroll_n;
                    dst += K * unroll_n;
                }
                if (n & 4) {
                    const float *l_src = src;
                    float *l_dst = dst;
                    if (k_eff & 4) {
                        _mm_storeu_ps(l_dst + 0 * unroll_n, _mm_loadu_ps(l_src + 0 * ldb));
                        _mm_storeu_ps(l_dst + 1 * unroll_n, _mm_loadu_ps(l_src + 1 * ldb));
                        _mm_storeu_ps(l_dst + 2 * unroll_n, _mm_loadu_ps(l_src + 2 * ldb));
                        _mm_storeu_ps(l_dst + 3 * unroll_n, _mm_loadu_ps(l_src + 3 * ldb));
                        l_src += 4 * ldb;
                        l_dst += 4 * unroll_n;
                    }
                    if (k_eff & 2) {
                        _mm_storeu_ps(l_dst + 0 * unroll_n, _mm_loadu_ps(l_src + 0 * ldb));
                        _mm_storeu_ps(l_dst + 1 * unroll_n, _mm_loadu_ps(l_src + 1 * ldb));
                        l_src += 2 * ldb;
                        l_dst += 2 * unroll_n;
                    }
                    if (k_eff & 1) {
                        _mm_storeu_ps(l_dst + 0 * unroll_n, _mm_loadu_ps(l_src + 0 * ldb));
                    }
                    src += 4;
                    dst += 4;
                }
                if (n & 2) {
                    const float *l_src = src;
                    float *l_dst = dst;
                    if (k_eff & 4) {
                        *(int64_t*)(l_dst + 0 * unroll_n) = *(int64_t*)(l_src + 0 * ldb);
                        *(int64_t*)(l_dst + 1 * unroll_n) = *(int64_t*)(l_src + 1 * ldb);
                        *(int64_t*)(l_dst + 2 * unroll_n) = *(int64_t*)(l_src + 2 * ldb);
                        *(int64_t*)(l_dst + 3 * unroll_n) = *(int64_t*)(l_src + 3 * ldb);
                        l_src += 4 * ldb;
                        l_dst += 4 * unroll_n;
                    }
                    if (k_eff & 2) {
                        *(int64_t*)(l_dst + 0 * unroll_n) = *(int64_t*)(l_src + 0 * ldb);
                        *(int64_t*)(l_dst + 1 * unroll_n) = *(int64_t*)(l_src + 1 * ldb);
                        l_src += 2 * ldb;
                        l_dst += 2 * unroll_n;
                    }
                    if (k_eff & 1) {
                        *(int64_t*)(l_dst + 0 * unroll_n) = *(int64_t*)(l_src + 0 * ldb);
                    }
                    src += 2;
                    dst += 2;
                }
                if (n & 1) {
                    const float *l_src = src;
                    float *l_dst = dst;
                    if (k_eff & 4) {
                        *(l_dst + 0 * unroll_n) = *(l_src + 0 * ldb);
                        *(l_dst + 1 * unroll_n) = *(l_src + 1 * ldb);
                        *(l_dst + 2 * unroll_n) = *(l_src + 2 * ldb);
                        *(l_dst + 3 * unroll_n) = *(l_src + 3 * ldb);
                        l_src += 4 * ldb;
                        l_dst += 4 * unroll_n;
                    }
                    if (k_eff & 2) {
                        *(l_dst + 0 * unroll_n) = *(l_src + 0 * ldb);
                        *(l_dst + 1 * unroll_n) = *(l_src + 1 * ldb);
                        l_src += 2 * ldb;
                        l_dst += 2 * unroll_n;
                    }
                    if (k_eff & 1) {
                        *(l_dst + 0 * unroll_n) = *(l_src + 0 * ldb);
                    }
                }
            }
        }
    } else { // B: (N, K) -> (N/8, K, 8)
        for (int64_t n = 0; n < N; n += unroll_n) {
            const int64_t n_eff = min(unroll_n, N - n);
            const float *src = B + n * ldb;
            float *dst = packedB + K * n;
            int64_t k = K;
            if (n_eff == unroll_n) {
// ========================================================================== //
#define K_TRANS_PACK_B_STEP(K) do {\
    dst[0 + K * unroll_n] = src[0 * ldb + K];\
    dst[1 + K * unroll_n] = src[1 * ldb + K];\
    dst[2 + K * unroll_n] = src[2 * ldb + K];\
    dst[3 + K * unroll_n] = src[3 * ldb + K];\
    dst[4 + K * unroll_n] = src[4 * ldb + K];\
    dst[5 + K * unroll_n] = src[5 * ldb + K];\
    dst[6 + K * unroll_n] = src[6 * ldb + K];\
    dst[7 + K * unroll_n] = src[7 * ldb + K];\
} while (0)
// ========================================================================== //
                while (k >= unroll_k) {
                    k -= unroll_k;
                    transpose_8x8_fp32_avx(src, ldb, unroll_n, dst);
                    src += unroll_k;
                    dst += unroll_k * unroll_n;
                }
                if (k & 4) {
                    K_TRANS_PACK_B_STEP(0);
                    K_TRANS_PACK_B_STEP(1);
                    K_TRANS_PACK_B_STEP(2);
                    K_TRANS_PACK_B_STEP(3);
                    src += 4;
                    dst += 4 * unroll_n;
                }
                if (k & 2) {
                    K_TRANS_PACK_B_STEP(0);
                    K_TRANS_PACK_B_STEP(1);
                    src += 2;
                    dst += 2 * unroll_n;
                }
                if (k & 1) {
                    K_TRANS_PACK_B_STEP(0);
                    src += 1;
                    dst += 1 * unroll_n;
                }
#undef K_TRANS_PACK_B_STEP
            } else {
// ========================================================================== //
#define K_TRANS_PACK_B_STEP(K) do {\
    const float *l_src = src;\
    float *l_dst = dst;\
    if (n_eff & 4) {\
        l_dst[0 + K * unroll_n] = l_src[0 * ldb + K];\
        l_dst[1 + K * unroll_n] = l_src[1 * ldb + K];\
        l_dst[2 + K * unroll_n] = l_src[2 * ldb + K];\
        l_dst[3 + K * unroll_n] = l_src[3 * ldb + K];\
        l_dst += 4;\
        l_src += 4 * ldb;\
    }\
    if (n_eff & 2) {\
        l_dst[0 + K * unroll_n] = l_src[0 * ldb + K];\
        l_dst[1 + K * unroll_n] = l_src[1 * ldb + K];\
        l_dst += 2;\
        l_src += 2 * ldb;\
    }\
    if (n_eff & 1) {\
        l_dst[0 + K * unroll_n] = l_src[0 * ldb + K];\
    }\
} while (0)
// ========================================================================== //
                while (k >= unroll_k) {
                    k -= unroll_k;
                    K_TRANS_PACK_B_STEP(0);
                    K_TRANS_PACK_B_STEP(1);
                    K_TRANS_PACK_B_STEP(2);
                    K_TRANS_PACK_B_STEP(3);
                    K_TRANS_PACK_B_STEP(4);
                    K_TRANS_PACK_B_STEP(5);
                    K_TRANS_PACK_B_STEP(6);
                    K_TRANS_PACK_B_STEP(7);
                    src += unroll_k;
                    dst += unroll_k * unroll_n;
                }
                if (k & 4) {
                    K_TRANS_PACK_B_STEP(0);
                    K_TRANS_PACK_B_STEP(1);
                    K_TRANS_PACK_B_STEP(2);
                    K_TRANS_PACK_B_STEP(3);
                    src += 4;
                    dst += 4 * unroll_n;
                }
                if (k & 2) {
                    K_TRANS_PACK_B_STEP(0);
                    K_TRANS_PACK_B_STEP(1);
                    src += 2;
                    dst += 2 * unroll_n;
                }
                if (k & 1) {
                    K_TRANS_PACK_B_STEP(0);
                    src += 1;
                    dst += 1 * unroll_n;
                }
#undef K_TRANS_PACK_B_STEP
            }
        }
    }
}

template<gemm_m_type_t typeB>
void gemm_pack_b_n16_operation_fp32_avx(
    const float *B,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB)
{
    const int64_t VEC_REG_ELTS = 8;
    const int64_t unroll_n = 16;
    if (typeB == gemm_m_type::NOTRANS) { // B: (K, N) -> (N/16, K, 16)
        const int64_t unroll_k = 4;
        for (int64_t k = 0; k < K; k += unroll_k) {
            const int64_t k_eff = min(unroll_k, K - k);
            const float *src = B + k * ldb;
            float *dst = packedB + k * unroll_n;
            int64_t n = N;
            if (k_eff == unroll_k) {
                while (n >= unroll_n) {
                    n -= unroll_n;
                    _mm256_storeu_ps(dst + 0 * unroll_n + 0 * VEC_REG_ELTS, _mm256_loadu_ps(src + 0 * ldb + 0 * VEC_REG_ELTS));
                    _mm256_storeu_ps(dst + 0 * unroll_n + 1 * VEC_REG_ELTS, _mm256_loadu_ps(src + 0 * ldb + 1 * VEC_REG_ELTS));
                    _mm256_storeu_ps(dst + 1 * unroll_n + 0 * VEC_REG_ELTS, _mm256_loadu_ps(src + 1 * ldb + 0 * VEC_REG_ELTS));
                    _mm256_storeu_ps(dst + 1 * unroll_n + 1 * VEC_REG_ELTS, _mm256_loadu_ps(src + 1 * ldb + 1 * VEC_REG_ELTS));
                    _mm256_storeu_ps(dst + 2 * unroll_n + 0 * VEC_REG_ELTS, _mm256_loadu_ps(src + 2 * ldb + 0 * VEC_REG_ELTS));
                    _mm256_storeu_ps(dst + 2 * unroll_n + 1 * VEC_REG_ELTS, _mm256_loadu_ps(src + 2 * ldb + 1 * VEC_REG_ELTS));
                    _mm256_storeu_ps(dst + 3 * unroll_n + 0 * VEC_REG_ELTS, _mm256_loadu_ps(src + 3 * ldb + 0 * VEC_REG_ELTS));
                    _mm256_storeu_ps(dst + 3 * unroll_n + 1 * VEC_REG_ELTS, _mm256_loadu_ps(src + 3 * ldb + 1 * VEC_REG_ELTS));
                    src += unroll_n;
                    dst += K * unroll_n;
                }
                if (n & 8) {
                    _mm256_storeu_ps(dst + 0 * unroll_n, _mm256_loadu_ps(src + 0 * ldb));
                    _mm256_storeu_ps(dst + 1 * unroll_n, _mm256_loadu_ps(src + 1 * ldb));
                    _mm256_storeu_ps(dst + 2 * unroll_n, _mm256_loadu_ps(src + 2 * ldb));
                    _mm256_storeu_ps(dst + 3 * unroll_n, _mm256_loadu_ps(src + 3 * ldb));
                    src += 8;
                    dst += 8;
                }
                if (n & 4) {
                    _mm_storeu_ps(dst + 0 * unroll_n, _mm_loadu_ps(src + 0 * ldb));
                    _mm_storeu_ps(dst + 1 * unroll_n, _mm_loadu_ps(src + 1 * ldb));
                    _mm_storeu_ps(dst + 2 * unroll_n, _mm_loadu_ps(src + 2 * ldb));
                    _mm_storeu_ps(dst + 3 * unroll_n, _mm_loadu_ps(src + 3 * ldb));
                    src += 4;
                    dst += 4;
                }
                if (n & 2) {
                    *(int64_t*)(dst + 0 * unroll_n) = *(int64_t*)(src + 0 * ldb);
                    *(int64_t*)(dst + 1 * unroll_n) = *(int64_t*)(src + 1 * ldb);
                    *(int64_t*)(dst + 2 * unroll_n) = *(int64_t*)(src + 2 * ldb);
                    *(int64_t*)(dst + 3 * unroll_n) = *(int64_t*)(src + 3 * ldb);
                    src += 2;
                    dst += 2;
                }
                if (n & 1) {
                    *(dst + 0 * unroll_n) = *(src + 0 * ldb);
                    *(dst + 1 * unroll_n) = *(src + 1 * ldb);
                    *(dst + 2 * unroll_n) = *(src + 2 * ldb);
                    *(dst + 3 * unroll_n) = *(src + 3 * ldb);
                }
            } else {
                while (n >= unroll_n) {
                    n -= unroll_n;
                    const float *l_src = src;
                    float *l_dst = dst;
                    if (k_eff & 2) {
                        _mm256_storeu_ps(l_dst + 0 * unroll_n + 0 * VEC_REG_ELTS, _mm256_loadu_ps(l_src + 0 * ldb + 0 * VEC_REG_ELTS));
                        _mm256_storeu_ps(l_dst + 0 * unroll_n + 1 * VEC_REG_ELTS, _mm256_loadu_ps(l_src + 0 * ldb + 1 * VEC_REG_ELTS));
                        _mm256_storeu_ps(l_dst + 1 * unroll_n + 0 * VEC_REG_ELTS, _mm256_loadu_ps(l_src + 1 * ldb + 0 * VEC_REG_ELTS));
                        _mm256_storeu_ps(l_dst + 1 * unroll_n + 1 * VEC_REG_ELTS, _mm256_loadu_ps(l_src + 1 * ldb + 1 * VEC_REG_ELTS));
                        l_src += 2 * ldb;
                        l_dst += 2 * unroll_n;
                    }
                    if (k_eff & 1) {
                        _mm256_storeu_ps(l_dst + 0 * unroll_n + 0 * VEC_REG_ELTS, _mm256_loadu_ps(l_src + 0 * ldb + 0 * VEC_REG_ELTS));
                        _mm256_storeu_ps(l_dst + 0 * unroll_n + 1 * VEC_REG_ELTS, _mm256_loadu_ps(l_src + 0 * ldb + 1 * VEC_REG_ELTS));
                    }
                    src += unroll_n;
                    dst += K * unroll_n;
                }
                if (n & 8) {
                    const float *l_src = src;
                    float *l_dst = dst;
                    if (k_eff & 2) {
                        _mm256_storeu_ps(l_dst + 0 * unroll_n, _mm256_loadu_ps(l_src + 0 * ldb));
                        _mm256_storeu_ps(l_dst + 1 * unroll_n, _mm256_loadu_ps(l_src + 1 * ldb));
                        l_src += 2 * ldb;
                        l_dst += 2 * unroll_n;
                    }
                    if (k_eff & 1) {
                        _mm256_storeu_ps(l_dst + 0 * unroll_n, _mm256_loadu_ps(l_src + 0 * ldb));
                    }
                    src += 8;
                    dst += 8;
                }
                if (n & 4) {
                    const float *l_src = src;
                    float *l_dst = dst;
                    if (k_eff & 2) {
                        _mm_storeu_ps(l_dst + 0 * unroll_n, _mm_loadu_ps(l_src + 0 * ldb));
                        _mm_storeu_ps(l_dst + 1 * unroll_n, _mm_loadu_ps(l_src + 1 * ldb));
                        l_src += 2 * ldb;
                        l_dst += 2 * unroll_n;
                    }
                    if (k_eff & 1) {
                        _mm_storeu_ps(l_dst + 0 * unroll_n, _mm_loadu_ps(l_src + 0 * ldb));
                    }
                    src += 4;
                    dst += 4;
                }
                if (n & 2) {
                    const float *l_src = src;
                    float *l_dst = dst;
                    if (k_eff & 2) {
                        *(int64_t*)(l_dst + 0 * unroll_n) = *(int64_t*)(l_src + 0 * ldb);
                        *(int64_t*)(l_dst + 1 * unroll_n) = *(int64_t*)(l_src + 1 * ldb);
                        l_src += 2 * ldb;
                        l_dst += 2 * unroll_n;
                    }
                    if (k_eff & 1) {
                        *(int64_t*)(l_dst + 0 * unroll_n) = *(int64_t*)(l_src + 0 * ldb);
                    }
                    src += 2;
                    dst += 2;
                }
                if (n & 1) {
                    const float *l_src = src;
                    float *l_dst = dst;
                    if (k_eff & 2) {
                        *(l_dst + 0 * unroll_n) = *(l_src + 0 * ldb);
                        *(l_dst + 1 * unroll_n) = *(l_src + 1 * ldb);
                        l_src += 2 * ldb;
                        l_dst += 2 * unroll_n;
                    }
                    if (k_eff & 1) {
                        *(l_dst + 0 * unroll_n) = *(l_src + 0 * ldb);
                    }
                }
            }
        }
    } else { // B: (N, K) -> (N/16, K, 16)
        const int64_t unroll_k = 8;
        for (int64_t n = 0; n < N; n += unroll_n) {
            const int64_t n_eff = min(unroll_n, N - n);
            const float *src = B + n * ldb;
            float *dst = packedB + K * n;
            int64_t k = K;
            if (n_eff == unroll_n) {
                while (k >= unroll_k) {
                    k -= unroll_k;
                    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
                    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
                    ymm0 = _mm256_loadu_ps(src + 0 * ldb); // K:0-7, N:0-7
                    ymm1 = _mm256_loadu_ps(src + 1 * ldb);
                    ymm2 = _mm256_loadu_ps(src + 2 * ldb);
                    ymm3 = _mm256_loadu_ps(src + 3 * ldb);
                    ymm4 = _mm256_loadu_ps(src + 4 * ldb);
                    ymm5 = _mm256_loadu_ps(src + 5 * ldb);
                    ymm6 = _mm256_loadu_ps(src + 6 * ldb);
                    ymm7 = _mm256_loadu_ps(src + 7 * ldb);
                    TRANSPOSE_8X8_FP32_AVX_MACRO();
                    _mm256_storeu_ps(dst + 0 * unroll_n + 0 * VEC_REG_ELTS, ymm8); // K:0-7, N:0-7
                    _mm256_storeu_ps(dst + 1 * unroll_n + 0 * VEC_REG_ELTS, ymm9);
                    _mm256_storeu_ps(dst + 2 * unroll_n + 0 * VEC_REG_ELTS, ymm10);
                    _mm256_storeu_ps(dst + 3 * unroll_n + 0 * VEC_REG_ELTS, ymm11);
                    _mm256_storeu_ps(dst + 4 * unroll_n + 0 * VEC_REG_ELTS, ymm12);
                    _mm256_storeu_ps(dst + 5 * unroll_n + 0 * VEC_REG_ELTS, ymm13);
                    _mm256_storeu_ps(dst + 6 * unroll_n + 0 * VEC_REG_ELTS, ymm14);
                    _mm256_storeu_ps(dst + 7 * unroll_n + 0 * VEC_REG_ELTS, ymm15);

                    ymm0 = _mm256_loadu_ps(src + 8 * ldb); // K:0-7, N:8-15
                    ymm1 = _mm256_loadu_ps(src + 9 * ldb);
                    ymm2 = _mm256_loadu_ps(src + 10 * ldb);
                    ymm3 = _mm256_loadu_ps(src + 11 * ldb);
                    ymm4 = _mm256_loadu_ps(src + 12 * ldb);
                    ymm5 = _mm256_loadu_ps(src + 13 * ldb);
                    ymm6 = _mm256_loadu_ps(src + 14 * ldb);
                    ymm7 = _mm256_loadu_ps(src + 15 * ldb);
                    TRANSPOSE_8X8_FP32_AVX_MACRO();
                    _mm256_storeu_ps(dst + 0 * unroll_n + 1 * VEC_REG_ELTS, ymm8); // K:0-7, N:8-15
                    _mm256_storeu_ps(dst + 1 * unroll_n + 1 * VEC_REG_ELTS, ymm9);
                    _mm256_storeu_ps(dst + 2 * unroll_n + 1 * VEC_REG_ELTS, ymm10);
                    _mm256_storeu_ps(dst + 3 * unroll_n + 1 * VEC_REG_ELTS, ymm11);
                    _mm256_storeu_ps(dst + 4 * unroll_n + 1 * VEC_REG_ELTS, ymm12);
                    _mm256_storeu_ps(dst + 5 * unroll_n + 1 * VEC_REG_ELTS, ymm13);
                    _mm256_storeu_ps(dst + 6 * unroll_n + 1 * VEC_REG_ELTS, ymm14);
                    _mm256_storeu_ps(dst + 7 * unroll_n + 1 * VEC_REG_ELTS, ymm15);

                    src += unroll_k;
                    dst += unroll_k * unroll_n;
                }
// ========================================================================== //
#define K_TRANS_PACK_B_STEP(K) do {\
    dst[0 + K * unroll_n] = src[0 * ldb + K];\
    dst[1 + K * unroll_n] = src[1 * ldb + K];\
    dst[2 + K * unroll_n] = src[2 * ldb + K];\
    dst[3 + K * unroll_n] = src[3 * ldb + K];\
    dst[4 + K * unroll_n] = src[4 * ldb + K];\
    dst[5 + K * unroll_n] = src[5 * ldb + K];\
    dst[6 + K * unroll_n] = src[6 * ldb + K];\
    dst[7 + K * unroll_n] = src[7 * ldb + K];\
    dst[8 + K * unroll_n] = src[8 * ldb + K];\
    dst[9 + K * unroll_n] = src[9 * ldb + K];\
    dst[10 + K * unroll_n] = src[10 * ldb + K];\
    dst[11 + K * unroll_n] = src[11 * ldb + K];\
    dst[12 + K * unroll_n] = src[12 * ldb + K];\
    dst[13 + K * unroll_n] = src[13 * ldb + K];\
    dst[14 + K * unroll_n] = src[14 * ldb + K];\
    dst[15 + K * unroll_n] = src[15 * ldb + K];\
} while (0)
// ========================================================================== //
                if (k & 4) {
                    K_TRANS_PACK_B_STEP(0);
                    K_TRANS_PACK_B_STEP(1);
                    K_TRANS_PACK_B_STEP(2);
                    K_TRANS_PACK_B_STEP(3);
                    src += 4;
                    dst += 4 * unroll_n;
                }
                if (k & 2) {
                    K_TRANS_PACK_B_STEP(0);
                    K_TRANS_PACK_B_STEP(1);
                    src += 2;
                    dst += 2 * unroll_n;
                }
                if (k & 1) {
                    K_TRANS_PACK_B_STEP(0);
                    src += 1;
                    dst += 1 * unroll_n;
                }
#undef K_TRANS_PACK_B_STEP
            } else {
// ========================================================================== //
#define K_TRANS_PACK_B_STEP(K) do {\
    const float *l_src = src;\
    float *l_dst = dst;\
    if (n_eff & 8) {\
        l_dst[0 + K * unroll_n] = l_src[0 * ldb + K];\
        l_dst[1 + K * unroll_n] = l_src[1 * ldb + K];\
        l_dst[2 + K * unroll_n] = l_src[2 * ldb + K];\
        l_dst[3 + K * unroll_n] = l_src[3 * ldb + K];\
        l_dst[4 + K * unroll_n] = l_src[4 * ldb + K];\
        l_dst[5 + K * unroll_n] = l_src[5 * ldb + K];\
        l_dst[6 + K * unroll_n] = l_src[6 * ldb + K];\
        l_dst[7 + K * unroll_n] = l_src[7 * ldb + K];\
        l_dst += 8;\
        l_src += 8 * ldb;\
    }\
    if (n_eff & 4) {\
        l_dst[0 + K * unroll_n] = l_src[0 * ldb + K];\
        l_dst[1 + K * unroll_n] = l_src[1 * ldb + K];\
        l_dst[2 + K * unroll_n] = l_src[2 * ldb + K];\
        l_dst[3 + K * unroll_n] = l_src[3 * ldb + K];\
        l_dst += 4;\
        l_src += 4 * ldb;\
    }\
    if (n_eff & 2) {\
        l_dst[0 + K * unroll_n] = l_src[0 * ldb + K];\
        l_dst[1 + K * unroll_n] = l_src[1 * ldb + K];\
        l_dst += 2;\
        l_src += 2 * ldb;\
    }\
    if (n_eff & 1) {\
        l_dst[0 + K * unroll_n] = l_src[0 * ldb + K];\
    }\
} while (0)
// ========================================================================== //
                while (k >= unroll_k) {
                    k -= unroll_k;
                    K_TRANS_PACK_B_STEP(0);
                    K_TRANS_PACK_B_STEP(1);
                    K_TRANS_PACK_B_STEP(2);
                    K_TRANS_PACK_B_STEP(3);
                    K_TRANS_PACK_B_STEP(4);
                    K_TRANS_PACK_B_STEP(5);
                    K_TRANS_PACK_B_STEP(6);
                    K_TRANS_PACK_B_STEP(7);
                    src += unroll_k;
                    dst += unroll_k * unroll_n;
                }
                if (k & 4) {
                    K_TRANS_PACK_B_STEP(0);
                    K_TRANS_PACK_B_STEP(1);
                    K_TRANS_PACK_B_STEP(2);
                    K_TRANS_PACK_B_STEP(3);
                    src += 4;
                    dst += 4 * unroll_n;
                }
                if (k & 2) {
                    K_TRANS_PACK_B_STEP(0);
                    K_TRANS_PACK_B_STEP(1);
                    src += 2;
                    dst += 2 * unroll_n;
                }
                if (k & 1) {
                    K_TRANS_PACK_B_STEP(0);
                    src += 1;
                    dst += 1 * unroll_n;
                }
#undef K_TRANS_PACK_B_STEP
            }
        }
    }
}

template<gemm_m_type_t typeB, int64_t ldpacked_b, int64_t kN>
void gemm_pack_b_operation_fp32_avx(
    const float *B,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB)
{
    const int64_t N_REG_ELTS = 8;
    const bool is_constant_N = kN != 0;
    const int64_t lN = is_constant_N ? kN : N;
    const bool is_aligned_kN = is_constant_N && (kN % N_REG_ELTS == 0);
    if (typeB == gemm_m_type::TRANS) {
        const int64_t unroll_n = N_REG_ELTS;
        const int64_t unroll_k = 8;
        const float *src = B;
        float *dst = packedB;
        int64_t n = lN;
        while (n >= unroll_n) { // B(N, K) -> (K, N)
            n -= unroll_n;
            const float *l_src = src;
            float *l_dst = dst;
            int64_t k = K;
// ========================================================================== //
#define K_TRANS_B_STEP(K) do {\
    l_dst[K * ldpacked_b + 0] = l_src[K + ldb * 0];\
    l_dst[K * ldpacked_b + 1] = l_src[K + ldb * 1];\
    l_dst[K * ldpacked_b + 2] = l_src[K + ldb * 2];\
    l_dst[K * ldpacked_b + 3] = l_src[K + ldb * 3];\
    l_dst[K * ldpacked_b + 4] = l_src[K + ldb * 4];\
    l_dst[K * ldpacked_b + 5] = l_src[K + ldb * 5];\
    l_dst[K * ldpacked_b + 6] = l_src[K + ldb * 6];\
    l_dst[K * ldpacked_b + 7] = l_src[K + ldb * 7];\
} while (0)
// ========================================================================== //
            while (k >= unroll_k) {
                k -= unroll_k;
                transpose_8x8_fp32_avx(l_src, ldb, ldpacked_b, l_dst);
                l_src += unroll_k;
                l_dst += unroll_k * ldpacked_b;
            }
            if (k & 4) {
                K_TRANS_B_STEP(0);
                K_TRANS_B_STEP(1);
                K_TRANS_B_STEP(2);
                K_TRANS_B_STEP(3);
                l_src += 4;
                l_dst += 4 * ldpacked_b;
            }
            if (k & 2) {
                K_TRANS_B_STEP(0);
                K_TRANS_B_STEP(1);
                l_src += 2;
                l_dst += 2 * ldpacked_b;
            }
            if (k & 1) {
                K_TRANS_B_STEP(0);
            }
#undef K_TRANS_B_STEP
            src += unroll_n * ldb;
            dst += unroll_n;
        }
        if (n & 4) {
            const float *l_src = src;
            float *l_dst = dst;
            int64_t k = K;
// ========================================================================== //
#define K_TRANS_B_STEP(K) do {\
    l_dst[K * ldpacked_b + 0] = l_src[K + ldb * 0];\
    l_dst[K * ldpacked_b + 1] = l_src[K + ldb * 1];\
    l_dst[K * ldpacked_b + 2] = l_src[K + ldb * 2];\
    l_dst[K * ldpacked_b + 3] = l_src[K + ldb * 3];\
} while (0)
// ========================================================================== //
            while (k >= unroll_k) {
                k -= unroll_k;
                K_TRANS_B_STEP(0);
                K_TRANS_B_STEP(1);
                K_TRANS_B_STEP(2);
                K_TRANS_B_STEP(3);
                K_TRANS_B_STEP(4);
                K_TRANS_B_STEP(5);
                K_TRANS_B_STEP(6);
                K_TRANS_B_STEP(7);
                l_src += unroll_k;
                l_dst += unroll_k * ldpacked_b;
            }
            if (k & 4) {
                K_TRANS_B_STEP(0);
                K_TRANS_B_STEP(1);
                K_TRANS_B_STEP(2);
                K_TRANS_B_STEP(3);
                l_src += 4;
                l_dst += 4 * ldpacked_b;
            }
            if (k & 2) {
                K_TRANS_B_STEP(0);
                K_TRANS_B_STEP(1);
                l_src += 2;
                l_dst += 2 * ldpacked_b;
            }
            if (k & 1) {
                K_TRANS_B_STEP(0);
            }
#undef K_TRANS_B_STEP
            src += 4 * ldb;
            dst += 4;
        }
        if (n & 2) {
            const float *l_src = src;
            float *l_dst = dst;
            int64_t k = K;
// ========================================================================== //
#define K_TRANS_B_STEP(K) do {\
    l_dst[K * ldpacked_b + 0] = l_src[K + ldb * 0];\
    l_dst[K * ldpacked_b + 1] = l_src[K + ldb * 1];\
} while (0)
// ========================================================================== //
            while (k >= unroll_k) {
                k -= unroll_k;
                K_TRANS_B_STEP(0);
                K_TRANS_B_STEP(1);
                K_TRANS_B_STEP(2);
                K_TRANS_B_STEP(3);
                K_TRANS_B_STEP(4);
                K_TRANS_B_STEP(5);
                K_TRANS_B_STEP(6);
                K_TRANS_B_STEP(7);
                l_src += unroll_k;
                l_dst += unroll_k * ldpacked_b;
            }
            if (k & 4) {
                K_TRANS_B_STEP(0);
                K_TRANS_B_STEP(1);
                K_TRANS_B_STEP(2);
                K_TRANS_B_STEP(3);
                l_src += 4;
                l_dst += 4 * ldpacked_b;
            }
            if (k & 2) {
                K_TRANS_B_STEP(0);
                K_TRANS_B_STEP(1);
                l_src += 2;
                l_dst += 2 * ldpacked_b;
            }
            if (k & 1) {
                K_TRANS_B_STEP(0);
            }
#undef K_TRANS_B_STEP
            src += 2 * ldb;
            dst += 2;
        }
        if (n & 1) {
            const float *l_src = src;
            float *l_dst = dst;
            int64_t k = K;
// ========================================================================== //
#define K_TRANS_B_STEP(K) do {\
    l_dst[K * ldpacked_b + 0] = l_src[K + ldb * 0];\
} while (0)
// ========================================================================== //
            while (k >= unroll_k) {
                k -= unroll_k;
                K_TRANS_B_STEP(0);
                K_TRANS_B_STEP(1);
                K_TRANS_B_STEP(2);
                K_TRANS_B_STEP(3);
                K_TRANS_B_STEP(4);
                K_TRANS_B_STEP(5);
                K_TRANS_B_STEP(6);
                K_TRANS_B_STEP(7);
                l_src += unroll_k;
                l_dst += unroll_k * ldpacked_b;
            }
            if (k & 4) {
                K_TRANS_B_STEP(0);
                K_TRANS_B_STEP(1);
                K_TRANS_B_STEP(2);
                K_TRANS_B_STEP(3);
                l_src += 4;
                l_dst += 4 * ldpacked_b;
            }
            if (k & 2) {
                K_TRANS_B_STEP(0);
                K_TRANS_B_STEP(1);
                l_src += 2;
                l_dst += 2 * ldpacked_b;
            }
            if (k & 1) {
                K_TRANS_B_STEP(0);
            }
#undef K_TRANS_B_STEP
        }
    } else {
        const int64_t unroll_k = 4;
        const float *src = B;
        float *dst = packedB;
        int64_t k = K;
        if (is_aligned_kN && kN <= 8 * N_REG_ELTS) {
#define K_PACK_B_STEP(K) do{\
    if (kN > 0 * N_REG_ELTS) _mm256_storeu_ps(dst + K * ldpacked_b + 0 * N_REG_ELTS, _mm256_loadu_ps(src + K * ldb + 0 * N_REG_ELTS));\
    if (kN > 1 * N_REG_ELTS) _mm256_storeu_ps(dst + K * ldpacked_b + 1 * N_REG_ELTS, _mm256_loadu_ps(src + K * ldb + 1 * N_REG_ELTS));\
    if (kN > 2 * N_REG_ELTS) _mm256_storeu_ps(dst + K * ldpacked_b + 2 * N_REG_ELTS, _mm256_loadu_ps(src + K * ldb + 2 * N_REG_ELTS));\
    if (kN > 3 * N_REG_ELTS) _mm256_storeu_ps(dst + K * ldpacked_b + 3 * N_REG_ELTS, _mm256_loadu_ps(src + K * ldb + 3 * N_REG_ELTS));\
    if (kN > 4 * N_REG_ELTS) _mm256_storeu_ps(dst + K * ldpacked_b + 4 * N_REG_ELTS, _mm256_loadu_ps(src + K * ldb + 4 * N_REG_ELTS));\
    if (kN > 5 * N_REG_ELTS) _mm256_storeu_ps(dst + K * ldpacked_b + 5 * N_REG_ELTS, _mm256_loadu_ps(src + K * ldb + 5 * N_REG_ELTS));\
    if (kN > 6 * N_REG_ELTS) _mm256_storeu_ps(dst + K * ldpacked_b + 6 * N_REG_ELTS, _mm256_loadu_ps(src + K * ldb + 6 * N_REG_ELTS));\
    if (kN > 7 * N_REG_ELTS) _mm256_storeu_ps(dst + K * ldpacked_b + 7 * N_REG_ELTS, _mm256_loadu_ps(src + K * ldb + 7 * N_REG_ELTS));\
} while (0)
            while (k >= unroll_k) { // B(K, N) -> (K, N)
                k -= unroll_k;
                K_PACK_B_STEP(0);
                K_PACK_B_STEP(1);
                K_PACK_B_STEP(2);
                K_PACK_B_STEP(3);
                src += unroll_k * ldb;
                dst += unroll_k * ldpacked_b;
            }
            if (k & 2) {
                K_PACK_B_STEP(0);
                K_PACK_B_STEP(1);
                src += 2 * ldb;
                dst += 2 * ldpacked_b;
            }
            if (k & 1) {
                K_PACK_B_STEP(0);
            }
#undef K_PACK_B_STEP
        } else {
            while (k >= unroll_k) { // B(K, N) -> (K, N)
                k -= unroll_k;
                memcpy32_avx(dst + 0 * ldpacked_b, src + 0 * ldb, lN);
                memcpy32_avx(dst + 1 * ldpacked_b, src + 1 * ldb, lN);
                memcpy32_avx(dst + 2 * ldpacked_b, src + 2 * ldb, lN);
                memcpy32_avx(dst + 3 * ldpacked_b, src + 3 * ldb, lN);
                src += unroll_k * ldb;
                dst += unroll_k * ldpacked_b;
            }
            while (k > 0) {
                k -= 1;
                memcpy32_avx(dst, src, lN);
                src += ldb;
                dst += ldpacked_b;
            }
        }
    }
}

template<gemm_m_type_t typeA>
void gemm_pack_a_m8_operation_fp32_avx(
    const float *A,
    const int64_t M,
    const int64_t K,
    const int64_t lda,
    float *packedA)
{
    if (typeA == gemm_m_type::TRANS) { // A: (K, M) -> (M/8, K, 8)
        const int64_t unroll_m = 8;
        const int64_t unroll_k = 4;
        int64_t k = K;
        const float *src = A;
        const int64_t m_body = round(M, unroll_m);
        const int64_t m_tail = M - m_body;
        float *dst_body = packedA;
        float *dst_tail = packedA + K * m_body;
        while (k >= unroll_k) {
            k -= unroll_k;
            if (m_body) {
                const float *l_src = src;
                float *l_dst = dst_body;
                int64_t m = m_body;
                while (m >= unroll_m) {
                    m -= unroll_m;
                    _mm256_storeu_ps(l_dst + 0 * unroll_m, _mm256_loadu_ps(l_src + 0 * lda));
                    _mm256_storeu_ps(l_dst + 1 * unroll_m, _mm256_loadu_ps(l_src + 1 * lda));
                    _mm256_storeu_ps(l_dst + 2 * unroll_m, _mm256_loadu_ps(l_src + 2 * lda));
                    _mm256_storeu_ps(l_dst + 3 * unroll_m, _mm256_loadu_ps(l_src + 3 * lda));
                    l_src += unroll_m;
                    l_dst += K * unroll_m;
                }
            }
#define M_TAIL_1(DST, SRC) do { *(DST) = *(SRC); } while (0)
#define M_TAIL_2(DST, SRC) do { *(int64_t*)(DST) = *(const int64_t*)(SRC); } while (0)
#define M_TAIL_3(DST, SRC) do { *(int64_t*)(DST) = *(const int64_t*)(SRC); *(DST + 2) = *(SRC + 2); } while (0)
#define M_TAIL_4(DST, SRC) do { _mm_storeu_ps(DST, _mm_loadu_ps(SRC)); } while (0)
#define M_TAIL_5(DST, SRC) do { _mm_storeu_ps(DST, _mm_loadu_ps(SRC)); *(DST + 4) = *(SRC + 4); } while (0)
#define M_TAIL_6(DST, SRC) do { _mm_storeu_ps(DST, _mm_loadu_ps(SRC)); *(int64_t*)(DST + 4) = *(const int64_t*)(SRC + 4); } while (0)
#define M_TAIL_7(DST, SRC) do { _mm_storeu_ps(DST, _mm_loadu_ps(SRC)); *(int64_t*)(DST + 4) = *(const int64_t*)(SRC + 4); *(DST + 6) = *(SRC + 6); } while (0)
            if (m_tail) {
                const float *l_src = src + m_body;
                float *l_dst = dst_tail;
                switch (m_tail) {
                    case 7: {
                        M_TAIL_7(l_dst + 0 * 7, l_src + 0 * lda);
                        M_TAIL_7(l_dst + 1 * 7, l_src + 1 * lda);
                        M_TAIL_7(l_dst + 2 * 7, l_src + 2 * lda);
                        M_TAIL_7(l_dst + 3 * 7, l_src + 3 * lda);
                        break;
                    }
                    case 6: {
                        M_TAIL_6(l_dst + 0 * 6, l_src + 0 * lda);
                        M_TAIL_6(l_dst + 1 * 6, l_src + 1 * lda);
                        M_TAIL_6(l_dst + 2 * 6, l_src + 2 * lda);
                        M_TAIL_6(l_dst + 3 * 6, l_src + 3 * lda);
                        break;
                    }
                    case 5: {
                        M_TAIL_5(l_dst + 0 * 5, l_src + 0 * lda);
                        M_TAIL_5(l_dst + 1 * 5, l_src + 1 * lda);
                        M_TAIL_5(l_dst + 2 * 5, l_src + 2 * lda);
                        M_TAIL_5(l_dst + 3 * 5, l_src + 3 * lda);
                        break;
                    }
                    case 4: {
                        M_TAIL_4(l_dst + 0 * 4, l_src + 0 * lda);
                        M_TAIL_4(l_dst + 1 * 4, l_src + 1 * lda);
                        M_TAIL_4(l_dst + 2 * 4, l_src + 2 * lda);
                        M_TAIL_4(l_dst + 3 * 4, l_src + 3 * lda);
                        break;
                    }
                    case 3: {
                        M_TAIL_3(l_dst + 0 * 3, l_src + 0 * lda);
                        M_TAIL_3(l_dst + 1 * 3, l_src + 1 * lda);
                        M_TAIL_3(l_dst + 2 * 3, l_src + 2 * lda);
                        M_TAIL_3(l_dst + 3 * 3, l_src + 3 * lda);
                        break;
                    }
                    case 2: {
                        M_TAIL_2(l_dst + 0 * 2, l_src + 0 * lda);
                        M_TAIL_2(l_dst + 1 * 2, l_src + 1 * lda);
                        M_TAIL_2(l_dst + 2 * 2, l_src + 2 * lda);
                        M_TAIL_2(l_dst + 3 * 2, l_src + 3 * lda);
                        break;
                    }
                    case 1: {
                        M_TAIL_1(l_dst + 0 * 1, l_src + 0 * lda);
                        M_TAIL_1(l_dst + 1 * 1, l_src + 1 * lda);
                        M_TAIL_1(l_dst + 2 * 1, l_src + 2 * lda);
                        M_TAIL_1(l_dst + 3 * 1, l_src + 3 * lda);
                        break;
                    }
                    default: break;
                }
            }
#undef M_TAIL_1
#undef M_TAIL_2
#undef M_TAIL_3
#undef M_TAIL_4
#undef M_TAIL_5
#undef M_TAIL_6
#undef M_TAIL_7
            src += unroll_k * lda;
            dst_body += unroll_k * unroll_m;
            dst_tail += unroll_k * m_tail;
        }
        while (k > 0) {
            k -= 1;
            if (m_body) {
                const float *l_src = src;
                float *l_dst = dst_body;
                int64_t m = m_body;
                while (m >= unroll_m * 2) {
                    m -= unroll_m * 2;
                    _mm256_storeu_ps(l_dst + 0 * K * unroll_m, _mm256_loadu_ps(l_src + 0 * unroll_m));
                    _mm256_storeu_ps(l_dst + 1 * K * unroll_m, _mm256_loadu_ps(l_src + 1 * unroll_m));
                    l_src += unroll_m * 2;
                    l_dst += K * unroll_m * 2;
                }
                if (m > 0) {
                    _mm256_storeu_ps(l_dst + 0 * K * unroll_m, _mm256_loadu_ps(l_src + 0 * unroll_m));
                }
            }
            if (m_tail) {
                const float *l_src = src + m_body;
                float *l_dst = dst_tail;
                if (m_tail & 4) {
                    _mm_storeu_ps(l_dst, _mm_loadu_ps(l_src));
                    l_src += 4;
                    l_dst += 4;
                }
                if (m_tail & 2) {
                    *(int64_t*)(l_dst) = *(int64_t*)(l_src);
                    l_src += 2;
                    l_dst += 2;
                }
                if (m_tail & 1) {
                    *l_dst = *l_src;
                }
            }
            src += lda;
            dst_body += unroll_m;
            dst_tail += m_tail;
        }
    } else { // A: (M, K) -> (M/8, K, 8)
        const int64_t unroll_m = 8;
        const int64_t unroll_k = 8;
        int64_t m = M;
        const float *src = A;
        float *dst = packedA;
        while (m >= unroll_m) {
            m -= unroll_m;
            int64_t k = K;
            const float *l_src = src;
            float *l_dst = dst;
            while (k >= unroll_k) {
                k -= unroll_k;
                transpose_8x8_fp32_avx(l_src, lda, unroll_m, l_dst);
                l_src += unroll_k;
                l_dst += unroll_k * unroll_m;
            }
// ========================================================================== //
#define K_TRANS_A_STEP(K) do {\
    l_dst[K * unroll_m + 0] = l_src[K + lda * 0];\
    l_dst[K * unroll_m + 1] = l_src[K + lda * 1];\
    l_dst[K * unroll_m + 2] = l_src[K + lda * 2];\
    l_dst[K * unroll_m + 3] = l_src[K + lda * 3];\
    l_dst[K * unroll_m + 4] = l_src[K + lda * 4];\
    l_dst[K * unroll_m + 5] = l_src[K + lda * 5];\
    l_dst[K * unroll_m + 6] = l_src[K + lda * 6];\
    l_dst[K * unroll_m + 7] = l_src[K + lda * 7];\
} while (0)
// ========================================================================== //
            if (k & 4) {
                K_TRANS_A_STEP(0);
                K_TRANS_A_STEP(1);
                K_TRANS_A_STEP(2);
                K_TRANS_A_STEP(3);
                l_src += 4;
                l_dst += 4 * unroll_m;
            }
            if (k & 2) {
                K_TRANS_A_STEP(0);
                K_TRANS_A_STEP(1);
                l_src += 2;
                l_dst += 2 * unroll_m;
            }
            if (k & 1) {
                K_TRANS_A_STEP(0);
            }
#undef K_TRANS_A_STEP
            src += unroll_m * lda;
            dst += unroll_m * K;
        }
        if (m > 0) {
            int64_t k = K;
// ========================================================================== //
#define K_TRANS_A_STEP(K) do {\
    const float *l_src = src;\
    float *l_dst = dst;\
    if (m & 4) {\
        l_dst[K * m + 0] = l_src[K + lda * 0];\
        l_dst[K * m + 1] = l_src[K + lda * 1];\
        l_dst[K * m + 2] = l_src[K + lda * 2];\
        l_dst[K * m + 3] = l_src[K + lda * 3];\
        l_src += 4 * lda;\
        l_dst += 4;\
    }\
    if (m & 2) {\
        l_dst[K * m + 0] = l_src[K + lda * 0];\
        l_dst[K * m + 1] = l_src[K + lda * 1];\
        l_src += 2 * lda;\
        l_dst += 2;\
    }\
    if (m & 1) {\
        l_dst[K * m + 0] = l_src[K + lda * 0];\
    }\
} while (0)
// ========================================================================== //
            while (k >= unroll_k) {
                k -= unroll_k;
                K_TRANS_A_STEP(0);
                K_TRANS_A_STEP(1);
                K_TRANS_A_STEP(2);
                K_TRANS_A_STEP(3);
                K_TRANS_A_STEP(4);
                K_TRANS_A_STEP(5);
                K_TRANS_A_STEP(6);
                K_TRANS_A_STEP(7);
                src += unroll_k;
                dst += unroll_k * m;
            }
            if (k & 4) {
                K_TRANS_A_STEP(0);
                K_TRANS_A_STEP(1);
                K_TRANS_A_STEP(2);
                K_TRANS_A_STEP(3);
                src += 4;
                dst += 4 * m;
            }
            if (k & 2) {
                K_TRANS_A_STEP(0);
                K_TRANS_A_STEP(1);
                src += 2;
                dst += 2 * m;
            }
            if (k & 1) {
                K_TRANS_A_STEP(0);
            }
#undef K_TRANS_A_STEP
        }
    }
}

template<gemm_m_type_t typeA>
void gemm_pack_a_operation_fp32_avx(
    const float *A,
    const int64_t M,
    const int64_t K,
    const int64_t lda,
    const int64_t ldpacked_a,
    float *packedA)
{
    if (typeA == gemm_m_type::TRANS) {
        const int64_t unroll_m = 8;
        const int64_t unroll_k = 8;
        for (int64_t k = 0; k < K; k += unroll_k) { // A: (K, M) -> (M, K)
            const int64_t k_eff = min(unroll_k, K - k);
            const float *src = A + k * lda;
            float *dst = packedA + k;
            int64_t m = M;
            if (k_eff == unroll_k) {
// ========================================================================== //
#define M_TRANS_A_STEP(M) do {\
    dst[M * ldpacked_a + 0] = src[M + lda * 0];\
    dst[M * ldpacked_a + 1] = src[M + lda * 1];\
    dst[M * ldpacked_a + 2] = src[M + lda * 2];\
    dst[M * ldpacked_a + 3] = src[M + lda * 3];\
    dst[M * ldpacked_a + 4] = src[M + lda * 4];\
    dst[M * ldpacked_a + 5] = src[M + lda * 5];\
    dst[M * ldpacked_a + 6] = src[M + lda * 6];\
    dst[M * ldpacked_a + 7] = src[M + lda * 7];\
} while (0)
// ========================================================================== //
                while (m >= unroll_m) {
                    m -= unroll_m;
                    transpose_8x8_fp32_avx(src, lda, ldpacked_a, dst);
                    src += unroll_m;
                    dst += unroll_m * ldpacked_a;
                }
                if (m & 4) {
                    M_TRANS_A_STEP(0);
                    M_TRANS_A_STEP(1);
                    M_TRANS_A_STEP(2);
                    M_TRANS_A_STEP(3);
                    src += 4;
                    dst += 4 * ldpacked_a;
                }
                if (m & 2) {
                    M_TRANS_A_STEP(0);
                    M_TRANS_A_STEP(1);
                    src += 2;
                    dst += 2 * ldpacked_a;
                }
                if (m & 1) {
                    M_TRANS_A_STEP(0);
                }
#undef M_TRANS_A_STEP
            } else {
// ========================================================================== //
#define M_TRANS_A_STEP(M) do {\
    const float *l_src = src;\
    float *l_dst = dst;\
    if (k_eff & 4) {\
        l_dst[M * ldpacked_a + 0] = l_src[M + lda * 0];\
        l_dst[M * ldpacked_a + 1] = l_src[M + lda * 1];\
        l_dst[M * ldpacked_a + 2] = l_src[M + lda * 2];\
        l_dst[M * ldpacked_a + 3] = l_src[M + lda * 3];\
        l_src += lda * 4;\
        l_dst += 4;\
    }\
    if (k_eff & 2) {\
        l_dst[M * ldpacked_a + 0] = l_src[M + lda * 0];\
        l_dst[M * ldpacked_a + 1] = l_src[M + lda * 1];\
        l_src += lda * 2;\
        l_dst += 2;\
    }\
    if (k_eff & 1) {\
        l_dst[M * ldpacked_a + 0] = l_src[M + lda * 0];\
    }\
} while (0)
// ========================================================================== //
                while (m >= unroll_m) {
                    m -= unroll_m;
                    M_TRANS_A_STEP(0);
                    M_TRANS_A_STEP(1);
                    M_TRANS_A_STEP(2);
                    M_TRANS_A_STEP(3);
                    M_TRANS_A_STEP(4);
                    M_TRANS_A_STEP(5);
                    M_TRANS_A_STEP(6);
                    M_TRANS_A_STEP(7);
                    src += unroll_m;
                    dst += unroll_m * ldpacked_a;
                }
                if (m & 4) {
                    M_TRANS_A_STEP(0);
                    M_TRANS_A_STEP(1);
                    M_TRANS_A_STEP(2);
                    M_TRANS_A_STEP(3);
                    src += 4;
                    dst += 4 * ldpacked_a;
                }
                if (m & 2) {
                    M_TRANS_A_STEP(0);
                    M_TRANS_A_STEP(1);
                    src += 2;
                    dst += 2 * ldpacked_a;
                }
                if (m & 1) {
                    M_TRANS_A_STEP(0);
                }
#undef M_TRANS_A_STEP
            }
        }
    } else {
        for (int64_t m = 0; m < M; ++m) { // A: (M, K) -> (M, K)
            const float *src = A + m * lda;
            float *dst = packedA + m * ldpacked_a;
            memcpy32_avx(dst, src, K);
        }
    }
}

template<gemm_v_type_t typeH, gemm_m_type_t typeV>
void gemm_fp32_apply_beta_avx(
    const float *V,
    const float *H,
    const int64_t M,
    const int64_t N,
    const int64_t ldh,
    const int64_t ldc,
    const float beta,
    float *C)
{
    if (typeV == gemm_v_type::EMPTY && typeH == gemm_m_type::EMPTY)
        return;

    const int64_t VEC_REG_ELTS = 8;
    const int64_t unroll_n = VEC_REG_ELTS * 2;
 
    __m256 ymm_v, ymm_beta;
    if (typeV == gemm_v_type::SCALAR) ymm_v = _mm256_set1_ps(V[0]);
    ymm_beta = _mm256_set1_ps(beta);

    for (int64_t m = 0; m < M; ++m) {
        const float *l_v = nullptr;
        const float *l_h = nullptr;
        if (typeV == gemm_v_type::COL_VEC) ymm_v = _mm256_set1_ps(V[m]);
        if (typeV == gemm_v_type::ROW_VEC) l_v = V;
        if (typeH == gemm_m_type::NOTRANS) l_h = H + m * ldh;
        float *l_c = C + m * ldc;
        int64_t n = N;
        while (n >= unroll_n) {
            __m256 ymm0, ymm1;
            if (typeV != gemm_v_type::EMPTY) {
                if (typeV == gemm_v_type::ROW_VEC) {
                    ymm0 = _mm256_loadu_ps(l_v + 0 * VEC_REG_ELTS);
                    ymm1 = _mm256_loadu_ps(l_v + 1 * VEC_REG_ELTS);
                } else {
                    ymm0 = ymm_v;
                    ymm1 = ymm_v;
                }
            }
            if (typeH == gemm_m_type::NOTRANS) {
                if (typeV == gemm_v_type::EMPTY) {
                    ymm0 = _mm256_loadu_ps(l_h + 0 * VEC_REG_ELTS);
                    ymm1 = _mm256_loadu_ps(l_h + 1 * VEC_REG_ELTS);
                } else {
                    ymm0 = _mm256_add_ps(_mm256_loadu_ps(l_h + 0 * VEC_REG_ELTS), ymm0);
                    ymm1 = _mm256_add_ps(_mm256_loadu_ps(l_h + 1 * VEC_REG_ELTS), ymm1);
                }
                _mm_prefetch((const char*)(l_h + 4 * ldh), _MM_HINT_T0);
            }
            ymm0 = _mm256_mul_ps(ymm0, ymm_beta);
            ymm1 = _mm256_mul_ps(ymm1, ymm_beta);
            _mm256_storeu_ps(l_c + 0 * VEC_REG_ELTS, ymm0);
            _mm256_storeu_ps(l_c + 1 * VEC_REG_ELTS, ymm1);
            _mm_prefetch((const char*)(l_c + 4 * ldc), _MM_HINT_T0);
            if (typeV == gemm_v_type::ROW_VEC) l_v += unroll_n;
            if (typeH == gemm_m_type::NOTRANS) l_h += unroll_n;
            l_c += unroll_n;
            n -= unroll_n;
        }
        while (n > 0) {
            float y = 0.0f;
            if (typeV != gemm_v_type::EMPTY) {
                if (typeV == gemm_v_type::SCALAR) {
                    y = V[0];
                }
                if (typeV == gemm_v_type::ROW_VEC) {
                    y = l_v[0];
                }
                if (typeV == gemm_v_type::COL_VEC) {
                    y = V[m];
                }
            }
            if (typeH == gemm_m_type::NOTRANS) {
                if (typeV == gemm_v_type::EMPTY) {
                    y = l_h[0];
                } else {
                    y += l_h[0];
                }
            }
            l_c[0] = y * beta;
            if (typeV == gemm_v_type::ROW_VEC) l_v += 1;
            if (typeH == gemm_m_type::NOTRANS) l_h += 1;
            l_c += 1;
            n -= 1;
        }
    }
}

inline void gemm_fp32_copy_c_buf_avx(
    const float *c_buf,
    const int64_t M,
    const int64_t N,
    const int64_t ldc_buf,
    const int64_t ldc,
    float *C)
{
    const int64_t unroll_m = 4;
    const float *src = c_buf;
    float *dst = C;
    int64_t m = M;
    while (m >= unroll_m) {
        m -= unroll_m;
        memcpy32_avx(dst + 0 * ldc, src + 0 * ldc_buf, N);
        memcpy32_avx(dst + 1 * ldc, src + 1 * ldc_buf, N);
        memcpy32_avx(dst + 2 * ldc, src + 2 * ldc_buf, N);
        memcpy32_avx(dst + 3 * ldc, src + 3 * ldc_buf, N);
        src += unroll_m * ldc_buf;
        dst += unroll_m * ldc;
    }
    if (m & 2) {
        memcpy32_avx(dst + 0 * ldc, src + 0 * ldc_buf, N);
        memcpy32_avx(dst + 1 * ldc, src + 1 * ldc_buf, N);
        src += 2 * ldc_buf;
        dst += 2 * ldc;
    }
    if (m & 1) {
        memcpy32_avx(dst + 0 * ldc, src + 0 * ldc_buf, N);
    }
}

}}};

#endif
