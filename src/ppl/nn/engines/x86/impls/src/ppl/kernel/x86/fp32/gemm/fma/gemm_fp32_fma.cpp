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

#include <string.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gemm/fma/gemm_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/fp32/transpose/avx/transpose_fp32_avx.h"
#include "ppl/kernel/x86/common/array_param_helper.h"
#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/common/generic_cpu_allocator.h"

namespace ppl { namespace kernel { namespace x86 {

static const int64_t INIT_K_L2_BLK_S = 128;
static const int64_t INIT_K_L2_BLK_M = 192;
static const int64_t INIT_K_L2_BLK_L = 256;
static const int64_t INIT_N_L2_BLK_S = 144;
static const int64_t INIT_N_L2_BLK_L = 192;
static const int64_t INIT_M_L3_BLK_L = 1024;
static const int64_t INIT_M_L3_BLK_S = 512;
static const int64_t M_L2_BLK = 64;

typedef uint64_t opt_flag_t;

class opt_flag {
public:
    static const opt_flag_t c_overflow_opt = (1 << 1);
    static const opt_flag_t chiplet_opt = (1 << 2);
    static const opt_flag_t pack_a_opt = (1 << 3);
};

template<gemm_m_type_t typeB>
void gemm_pack_b_operation_fp32_avx(
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

template<gemm_m_type_t typeA>
void gemm_pack_a_operation_fp32_avx(
    const float *A,
    const int64_t M,
    const int64_t K,
    const int64_t lda,
    float *packedA)
{
    const int64_t ldpacked_a = K;

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

void gemm_fp32_copy_c_buf_avx(
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

    const int64_t N_REG_ELTS = gemm_kernel_fp32_fma::config::N_REG_ELTS;
    const int64_t unroll_n = N_REG_ELTS * 2;
 
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
                    ymm0 = _mm256_loadu_ps(l_v + 0 * N_REG_ELTS);
                    ymm1 = _mm256_loadu_ps(l_v + 1 * N_REG_ELTS);
                } else {
                    ymm0 = ymm_v;
                    ymm1 = ymm_v;
                }
            }
            if (typeH == gemm_m_type::NOTRANS) {
                if (typeV == gemm_v_type::EMPTY) {
                    ymm0 = _mm256_loadu_ps(l_h + 0 * N_REG_ELTS);
                    ymm1 = _mm256_loadu_ps(l_h + 1 * N_REG_ELTS);
                } else {
                    ymm0 = _mm256_add_ps(_mm256_loadu_ps(l_h + 0 * N_REG_ELTS), ymm0);
                    ymm1 = _mm256_add_ps(_mm256_loadu_ps(l_h + 1 * N_REG_ELTS), ymm1);
                }
            }
            ymm0 = _mm256_mul_ps(ymm0, ymm_beta);
            ymm1 = _mm256_mul_ps(ymm1, ymm_beta);
            _mm256_storeu_ps(l_c + 0 * N_REG_ELTS, ymm0);
            _mm256_storeu_ps(l_c + 1 * N_REG_ELTS, ymm1);
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

// Row-major impl, H and C could be the same matrix
ppl::common::RetCode gemm_operation_fp32_fma(
    const float *A,
    const float *B,
    const float *V,
    const float *H,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typeV,
    const gemm_m_type_t typeH,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldh,
    const float alpha,
    const float beta,
    const gemm_post_t post,
    const opt_flag_t flags,
    float *C)
{
    if (typeA == gemm_m_type::PACKED || typeB == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typeH != gemm_m_type::EMPTY && typeH != gemm_m_type::NOTRANS) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t N_REG_ELTS = gemm_kernel_fp32_fma::config::N_REG_ELTS;
    const bool apply_aAB = alpha != 0.0f && typeA != gemm_m_type::EMPTY && typeB != gemm_m_type::EMPTY;
    const bool apply_bVpbH = beta != 0.0f && (typeV != gemm_v_type::EMPTY || typeH != gemm_m_type::EMPTY);

    if (!apply_aAB && !apply_bVpbH) {
        return ppl::common::RC_SUCCESS;
    }

    if (alpha == 0.0f && beta == 0.0f) {
        for (int64_t m = 0; m < M; ++m) {
            memset32_avx(C + m * ldc, 0, N);
        }
        return ppl::common::RC_SUCCESS;
    }

    // blocking
    int64_t l2_size = ppl::common::GetCpuCacheL2();
    if (l2_size == 0) {
        l2_size = 256 * 1024;
    }
    int64_t sel_k_l2_blk;
    int64_t sel_n_l2_blk;
    if (l2_size > 512 * 1024) {
        sel_k_l2_blk = INIT_K_L2_BLK_L;
        sel_n_l2_blk = INIT_N_L2_BLK_L;
    } else if (l2_size > 256 * 1024) {
        sel_k_l2_blk = INIT_K_L2_BLK_M;
        sel_n_l2_blk = INIT_N_L2_BLK_S;
    } else {
        sel_k_l2_blk = INIT_K_L2_BLK_S;
        sel_n_l2_blk = INIT_N_L2_BLK_S;
    }

    bool force_c_buffer = flags & opt_flag::c_overflow_opt;
    if (!(flags & opt_flag::chiplet_opt)) {
        force_c_buffer = force_c_buffer && (K / sel_k_l2_blk) > 4;
    }
    const bool alloc_c_buffer = force_c_buffer || ((N % N_REG_ELTS) && apply_aAB);

    const int64_t max_packed_b_len = sel_n_l2_blk * sel_k_l2_blk;
    const int64_t max_c_buffer_len = INIT_M_L3_BLK_S * sel_n_l2_blk;

    int64_t k_l2_blk = min(sel_k_l2_blk, K);
    int64_t n_l2_blk = round_up(min(max_packed_b_len / k_l2_blk, N), N_REG_ELTS);
    if (typeA == gemm_m_type::NOTRANS && n_l2_blk < 0.75f * sel_n_l2_blk) {
        k_l2_blk = min(max_packed_b_len / n_l2_blk, K);
    }
    int64_t m_l3_blk = alloc_c_buffer ? min(max_c_buffer_len / n_l2_blk, M) : INIT_M_L3_BLK_L;
    const bool sliding_packed_a = n_l2_blk < N;
    const bool do_packed_a = (flags & opt_flag::pack_a_opt) || typeA == gemm_m_type::TRANS;

    ppl::common::GenericCpuAllocator allocator;
    float *packed_b = (float*)allocator.Alloc(k_l2_blk * n_l2_blk * sizeof(float));
    float *packed_a = do_packed_a ? (float*)allocator.Alloc((sliding_packed_a ? m_l3_blk : M_L2_BLK) * k_l2_blk * sizeof(float)) : nullptr;
    float *c_buffer = alloc_c_buffer ? (float*)allocator.Alloc(m_l3_blk * n_l2_blk * sizeof(float)) : nullptr;

    auto apply_beta_func = gemm_fp32_apply_beta_avx<gemm_m_type::EMPTY, gemm_v_type::EMPTY>;
    if (typeH == gemm_m_type::NOTRANS) {
        if (typeV == gemm_v_type::EMPTY) apply_beta_func = gemm_fp32_apply_beta_avx<gemm_m_type::NOTRANS, gemm_v_type::EMPTY>;
        if (typeV == gemm_v_type::SCALAR) apply_beta_func = gemm_fp32_apply_beta_avx<gemm_m_type::NOTRANS, gemm_v_type::SCALAR>;
        if (typeV == gemm_v_type::COL_VEC) apply_beta_func = gemm_fp32_apply_beta_avx<gemm_m_type::NOTRANS, gemm_v_type::COL_VEC>;
        if (typeV == gemm_v_type::ROW_VEC) apply_beta_func = gemm_fp32_apply_beta_avx<gemm_m_type::NOTRANS, gemm_v_type::ROW_VEC>;
    } else {
        if (typeV == gemm_v_type::SCALAR) apply_beta_func = gemm_fp32_apply_beta_avx<gemm_m_type::EMPTY, gemm_v_type::SCALAR>;
        if (typeV == gemm_v_type::COL_VEC) apply_beta_func = gemm_fp32_apply_beta_avx<gemm_m_type::EMPTY, gemm_v_type::COL_VEC>;
        if (typeV == gemm_v_type::ROW_VEC) apply_beta_func = gemm_fp32_apply_beta_avx<gemm_m_type::EMPTY, gemm_v_type::ROW_VEC>;
    }

    auto pack_b_func = gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS>;
    if (typeB == gemm_m_type::TRANS) pack_b_func = gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS>;

    auto pack_a_func = gemm_pack_a_operation_fp32_avx<gemm_m_type::NOTRANS>;
    if (typeA == gemm_m_type::TRANS) pack_a_func = gemm_pack_a_operation_fp32_avx<gemm_m_type::TRANS>;

    int64_t kernel_param[gemm_kernel_fp32_fma::param_def::LENGTH];
    array_param_helper ker_p(kernel_param);
    gemm_kernel_fp32_fma ker(kernel_param);
    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::ALPHA_IDX) = alpha;

    for (int64_t ml3 = 0; ml3 < M; ml3 += m_l3_blk) {
        const int64_t ml3_eff = min(m_l3_blk, M - ml3);
        for (int64_t kl2 = 0; kl2 < K; kl2 += k_l2_blk) {
            const int64_t kl2_eff = min(k_l2_blk, K - kl2);
            const bool is_first_k = kl2 == 0;
            const bool is_last_k = kl2 + kl2_eff == K;
            for (int64_t nl2 = 0; nl2 < N; nl2 += n_l2_blk) {
                const int64_t nl2_eff = min(n_l2_blk, N - nl2);
                const int64_t padded_nl2_eff = round_up(nl2_eff, N_REG_ELTS);

                const int64_t nl2_body = round(nl2_eff, gemm_kernel_fp32_fma::config::MAX_N_BLK);
                const int64_t nl2_treg = div_up(nl2_eff - nl2_body, N_REG_ELTS);
                const int64_t nl2_tail = nl2_treg * N_REG_ELTS;

                const bool use_c_buffer = force_c_buffer || (alloc_c_buffer && (nl2_eff % N_REG_ELTS));

                float *local_c = C + ml3 * ldc + nl2;
                float *local_c_buf = use_c_buffer ? c_buffer : local_c;
                const int64_t ldc_buf = use_c_buffer ? padded_nl2_eff : ldc;
                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = local_c_buf;
                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::LDC_IDX) = ldc_buf;
                ker_p.pick<gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::FLAGS_IDX) = is_first_k ? 0 : gemm_kernel_fp32_fma::flag::LOAD_C;

                if (apply_bVpbH && is_first_k) {
                    const float *l_h = nullptr;
                    const float *l_v = nullptr;
                    if (typeH == gemm_m_type::NOTRANS) l_h = H + ml3 * ldh + nl2;
                    if (typeV == gemm_v_type::SCALAR) l_v = V;
                    if (typeV == gemm_v_type::COL_VEC) l_v = V + ml3;
                    if (typeV == gemm_v_type::ROW_VEC) l_v = V + nl2;
                    apply_beta_func(l_v, l_h, ml3_eff, nl2_eff, ldh, ldc_buf, beta, local_c_buf);
                    ker_p.pick<gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::FLAGS_IDX) |= gemm_kernel_fp32_fma::flag::LOAD_C;
                }

                if (!apply_aAB)
                    continue;

                if (is_last_k) {
                    if (post == gemm_post::RELU6) ker_p.pick<gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::FLAGS_IDX) |= gemm_kernel_fp32_fma::flag::RELU6;
                    if (post == gemm_post::RELU) ker_p.pick<gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::FLAGS_IDX) |= gemm_kernel_fp32_fma::flag::RELU;
                }

                const float *base_b = B + (typeB == gemm_m_type::NOTRANS ? kl2 * ldb + nl2 : nl2 * ldb + kl2);
                const float *base_a = A + (typeA == gemm_m_type::NOTRANS ? ml3 * lda + kl2 : kl2 * lda + ml3);
                float *base_c_buf = local_c_buf;
                pack_b_func(base_b, nl2_eff, kl2_eff, ldb, packed_b);
                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::LDPACKED_B_IDX) = kl2_eff * N_REG_ELTS;
                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::K_IDX) = kl2_eff;

                if (!do_packed_a) {
                    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::LDA_IDX) = lda;
                    int64_t m = ml3_eff;
                    while (m >= gemm_kernel_fp32_fma::config::MAX_M_BLK) {
                        m -= gemm_kernel_fp32_fma::config::MAX_M_BLK;
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_a;
                        if (nl2_body) {
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::PACKED_B_PTR_IDX) = packed_b;
                            ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c_buf;
                            ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::N_IDX) = nl2_body;
                            ker.execute(gemm_kernel_fp32_fma::config::MAX_M_REGS, gemm_kernel_fp32_fma::config::MAX_N_REGS);
                        }
                        if (nl2_tail) {
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::PACKED_B_PTR_IDX) = packed_b + nl2_body * kl2_eff;
                            ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c_buf + nl2_body;
                            ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::N_IDX) = nl2_tail;
                            ker.execute(gemm_kernel_fp32_fma::config::MAX_M_REGS, nl2_treg);
                        }

                        base_c_buf += gemm_kernel_fp32_fma::config::MAX_M_BLK * ldc_buf;
                        base_a += gemm_kernel_fp32_fma::config::MAX_M_BLK * lda;
                    }
                    if (m > 0) {
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_a;
                        if (nl2_body) {
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::PACKED_B_PTR_IDX) = packed_b;
                            ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c_buf;
                            ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::N_IDX) = nl2_body;
                            ker.execute(m, gemm_kernel_fp32_fma::config::MAX_N_REGS);
                        }
                        if (nl2_tail) {
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::PACKED_B_PTR_IDX) = packed_b + nl2_body * kl2_eff;
                            ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c_buf + nl2_body;
                            ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::N_IDX) = nl2_tail;
                            ker.execute(m, nl2_treg);
                        }
                    }
                } else {
                    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::LDA_IDX) = kl2_eff;
                    for (int64_t ml2 = 0; ml2 < ml3_eff; ml2 += M_L2_BLK) {
                        const int64_t ml2_eff = min(M_L2_BLK, ml3_eff - ml2);
                        float *local_packed_a = packed_a + (sliding_packed_a ? ml2 * kl2_eff : 0);
                        if (!sliding_packed_a || (sliding_packed_a && nl2 == 0)) {
                            pack_a_func(base_a + (typeA == gemm_m_type::TRANS ? ml2 : ml2 * lda), ml2_eff, kl2_eff, lda, local_packed_a);
                        }
                        const float *base_packed_a = local_packed_a;
                        const int64_t ldpacked_a = kl2_eff;
                        int64_t m = ml2_eff;
                        while (m >= gemm_kernel_fp32_fma::config::MAX_M_BLK) {
                            m -= gemm_kernel_fp32_fma::config::MAX_M_BLK;
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_packed_a;
                            if (nl2_body) {
                                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::PACKED_B_PTR_IDX) = packed_b;
                                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c_buf;
                                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::N_IDX) = nl2_body;
                                ker.execute(gemm_kernel_fp32_fma::config::MAX_M_REGS, gemm_kernel_fp32_fma::config::MAX_N_REGS);
                            }
                            if (nl2_tail) {
                                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::PACKED_B_PTR_IDX) = packed_b + nl2_body * kl2_eff;
                                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c_buf + nl2_body;
                                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::N_IDX) = nl2_tail;
                                ker.execute(gemm_kernel_fp32_fma::config::MAX_M_REGS, nl2_treg);
                            }

                            base_c_buf += gemm_kernel_fp32_fma::config::MAX_M_BLK * ldc_buf;
                            base_packed_a += gemm_kernel_fp32_fma::config::MAX_M_BLK * ldpacked_a;
                        }
                        if (m > 0) {
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_packed_a;
                            if (nl2_body) {
                                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::PACKED_B_PTR_IDX) = packed_b;
                                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c_buf;
                                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::N_IDX) = nl2_body;
                                ker.execute(m, gemm_kernel_fp32_fma::config::MAX_N_REGS);
                            }
                            if (nl2_tail) {
                                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::PACKED_B_PTR_IDX) = packed_b + nl2_body * kl2_eff;
                                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c_buf + nl2_body;
                                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::N_IDX) = nl2_tail;
                                ker.execute(m, nl2_treg);
                            }
                        }
                    }
                }

                if (use_c_buffer && is_last_k) {
                    gemm_fp32_copy_c_buf_avx(c_buffer, ml3_eff, nl2_eff, ldc_buf, ldc, local_c);
                }
            }
        }
    }

    if (packed_b) allocator.Free(packed_b);
    if (packed_a) allocator.Free(packed_a);
    if (c_buffer) allocator.Free(c_buffer);

    return ppl::common::RC_SUCCESS;
}

// Row-major impl, H and C could be the same matrix
ppl::common::RetCode gemm_fp32_fma(
    const float *A,
    const float *B,
    const float *V,
    const float *H,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typeV,
    const gemm_m_type_t typeH,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldh,
    const float alpha,
    const float beta,
    const gemm_post_t post,
    float *C)
{
    if (typeA == gemm_m_type::PACKED || typeB == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typeH != gemm_m_type::EMPTY && typeH != gemm_m_type::NOTRANS) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if ((typeA == gemm_m_type::NOTRANS || lda == 1) && M == 1) {
        return gemv_fp32_fma(
            A, B, V, H,
            gemm_v_type::ROW_VEC, typeB, typeV, typeH,
            N, K, ldb,
            alpha, beta, post, C);
    }

    if (N == 1 && ((typeB == gemm_m_type::NOTRANS && ldb == 1) || (typeB == gemm_m_type::TRANS && ldb == K)) && ldc == 1) {
        auto l_typeA = typeA == gemm_m_type::NOTRANS ? gemm_m_type::TRANS : gemm_m_type::NOTRANS;
        auto l_typeV = typeV == gemm_v_type::ROW_VEC ? gemm_v_type::COL_VEC : gemm_v_type::ROW_VEC;
        return gemv_fp32_fma(
            B, A, V, H,
            gemm_v_type::ROW_VEC, l_typeA, l_typeV, typeH,
            M, K, lda,
            alpha, beta, post, C);
    }

    const int64_t num_threads = PPL_OMP_MAX_THREADS();
    opt_flag_t flags = 0;

    const bool intel_platform = strstr(ppl::common::GetCpuVendor(), "Intel") != nullptr;
    if (!intel_platform) {
        flags |= opt_flag::chiplet_opt; // assume all other platform are chiplet
    }

    // how to detect AMD chiplet?
    if (M * K >= (intel_platform ? (4096 * 4096) : (2048 * 2048)) && M >= 512 && K >= 512) { // A oversize
        flags |= opt_flag::pack_a_opt;
    }

    if (num_threads == 1) {
        return gemm_operation_fp32_fma(
            A, B, V, H,
            typeA, typeB, typeV, typeH,
            M, N, K, lda, ldb ,ldc ,ldh,
            alpha, beta, post, flags, C);
    }

    int64_t m_task_blk;
    int64_t n_task_blk;
    int64_t m_task;
    int64_t n_task;

    if (N > M) {
        if (intel_platform) {
            flags &= ~opt_flag::pack_a_opt;
        }
        n_task_blk = round_up(div_up(N, num_threads), gemm_kernel_fp32_fma::config::N_REG_ELTS);
        n_task = div_up(N, n_task_blk);
        m_task = max<int64_t>(1, num_threads / n_task);
        m_task_blk = round_up(div_up(M, m_task), INIT_M_L3_BLK_S);
        m_task = div_up(M, m_task_blk);
    } else {
        m_task_blk = round_up(div_up(M, num_threads), INIT_M_L3_BLK_S / 2);
        m_task = div_up(M, m_task_blk);
        n_task = max<int64_t>(1, num_threads / m_task);
        n_task_blk = round_up(div_up(N, n_task), gemm_kernel_fp32_fma::config::N_REG_ELTS);
        n_task = div_up(N, n_task_blk);
    }

    int64_t l2_size = ppl::common::GetCpuCacheL2();
    if (l2_size == 0) {
        l2_size = 256 * 1024;
    }
    const int64_t l2_elts = l2_size / sizeof(float);
    if (M * N > 4 * l2_elts * num_threads) { // C oversize
        flags |= opt_flag::c_overflow_opt;
    }

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t t = 0; t < m_task * n_task; ++t) {
        int64_t mb, nb;
        if (2 * N >= M) {
            mb = (t / n_task) * m_task_blk;
            nb = (t % n_task) * n_task_blk;
        } else {
            nb = (t / m_task) * n_task_blk;
            mb = (t % m_task) * m_task_blk;
        }

        const float *lA = A;
        if (typeA == gemm_m_type::NOTRANS) {
            lA += mb * lda;
        } else {
            lA += mb;
        }

        const float *lB = B;
        if (typeB == gemm_m_type::NOTRANS) {
            lB += nb;
        } else {
            lB += nb * ldb;
        }

        const float *lV = V;
        if (typeV == gemm_v_type::COL_VEC) {
            lV += mb;
        } else if (typeV == gemm_v_type::ROW_VEC) {
            lV += nb;
        }

        const float *lH = H;
        if (typeH == gemm_m_type::NOTRANS) {
            lH += mb * ldh + nb;
        }

        float *lC = C + mb * ldh + nb;

        const int64_t mb_eff = min(m_task_blk, M - mb);
        const int64_t nb_eff = min(n_task_blk, N - nb);

        auto ret = gemm_operation_fp32_fma(
            lA, lB, lV, lH,
            typeA, typeB, typeV, typeH,
            mb_eff, nb_eff, K, lda, ldb ,ldc ,ldh,
            alpha, beta, post, flags, lC);

        (void) ret;
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
