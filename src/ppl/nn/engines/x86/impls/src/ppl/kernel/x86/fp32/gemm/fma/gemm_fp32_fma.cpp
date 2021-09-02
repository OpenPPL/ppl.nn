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

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gemm/fma/gemm_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/fp32/transpose/avx/transpose_fp32_avx.h"
#include "ppl/kernel/x86/common/array_param_helper.h"
#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/common/generic_cpu_allocator.h"

namespace ppl { namespace kernel { namespace x86 {

static const int64_t init_k_l2_blk_s = 128;
static const int64_t init_k_l2_blk_m = 192;
static const int64_t init_k_l2_blk_l = 256;
static const int64_t init_n_l2_blk_s = 144;
static const int64_t init_n_l2_blk_l = 192;
static const int64_t init_m_l3_blk = 512;
static const int64_t m_l2_blk = 64;

void gemm_pack_b_operation_fp32_fma(
    const float *B,
    const gemm_m_type_t typeB,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB)
{
    const int64_t unroll_n = 8;
    const int64_t unroll_k = 8;
    if (typeB == gemm_m_type::notrans) { // B: (K, N) -> (N/8, K, 8)
        for (int64_t n = 0; n < N; n += unroll_n) {
            const int64_t n_eff = min(unroll_n, N - n);
            const float *src = B + n;
            float *dst = packedB + K * n;
            int64_t k = K;
            if (n_eff == unroll_n) {
                while (k >= unroll_k) {
                    k -= unroll_k;
                    memcpy32_avx(dst + 0 * unroll_n, src + 0 * ldb, unroll_n);
                    memcpy32_avx(dst + 1 * unroll_n, src + 1 * ldb, unroll_n);
                    memcpy32_avx(dst + 2 * unroll_n, src + 2 * ldb, unroll_n);
                    memcpy32_avx(dst + 3 * unroll_n, src + 3 * ldb, unroll_n);
                    memcpy32_avx(dst + 4 * unroll_n, src + 4 * ldb, unroll_n);
                    memcpy32_avx(dst + 5 * unroll_n, src + 5 * ldb, unroll_n);
                    memcpy32_avx(dst + 6 * unroll_n, src + 6 * ldb, unroll_n);
                    memcpy32_avx(dst + 7 * unroll_n, src + 7 * ldb, unroll_n);
                    src += unroll_k * ldb;
                    dst += unroll_k * unroll_n;
                }
                if (k & 4) {
                    memcpy32_avx(dst + 0 * unroll_n, src + 0 * ldb, unroll_n);
                    memcpy32_avx(dst + 1 * unroll_n, src + 1 * ldb, unroll_n);
                    memcpy32_avx(dst + 2 * unroll_n, src + 2 * ldb, unroll_n);
                    memcpy32_avx(dst + 3 * unroll_n, src + 3 * ldb, unroll_n);
                    src += 4 * ldb;
                    dst += 4 * unroll_n;
                }
                if (k & 2) {
                    memcpy32_avx(dst + 0 * unroll_n, src + 0 * ldb, unroll_n);
                    memcpy32_avx(dst + 1 * unroll_n, src + 1 * ldb, unroll_n);
                    src += 2 * ldb;
                    dst += 2 * unroll_n;
                }
                if (k & 1) {
                    memcpy32_avx(dst + 0 * unroll_n, src + 0 * ldb, unroll_n);
                    src += 1 * ldb;
                    dst += 1 * unroll_n;
                }
            } else {
// ========================================================================== //
#define K_PACK_B_STEP(K) do {\
    const float *l_src = src;\
    float *l_dst = dst;\
    if (n_eff & 4) {\
        l_dst[0 + K * unroll_n] = l_src[0 + ldb * K];\
        l_dst[1 + K * unroll_n] = l_src[1 + ldb * K];\
        l_dst[2 + K * unroll_n] = l_src[2 + ldb * K];\
        l_dst[3 + K * unroll_n] = l_src[3 + ldb * K];\
        l_dst += 4;\
        l_src += 4;\
    }\
    if (n_eff & 2) {\
        l_dst[0 + K * unroll_n] = l_src[0 + ldb * K];\
        l_dst[1 + K * unroll_n] = l_src[1 + ldb * K];\
        l_dst += 2;\
        l_src += 2;\
    }\
    if (n_eff & 1) {\
        l_dst[0 + K * unroll_n] = l_src[0 + ldb * K];\
    }\
} while (0)
// ========================================================================== //
                while (k >= unroll_k) {
                    k -= unroll_k;
                    K_PACK_B_STEP(0);
                    K_PACK_B_STEP(1);
                    K_PACK_B_STEP(2);
                    K_PACK_B_STEP(3);
                    K_PACK_B_STEP(4);
                    K_PACK_B_STEP(5);
                    K_PACK_B_STEP(6);
                    K_PACK_B_STEP(7);
                    src += unroll_k * ldb;
                    dst += unroll_k * unroll_n;
                }
                if (k & 4) {
                    K_PACK_B_STEP(0);
                    K_PACK_B_STEP(1);
                    K_PACK_B_STEP(2);
                    K_PACK_B_STEP(3);
                    src += 4 * ldb;
                    dst += 4 * unroll_n;
                }
                if (k & 2) {
                    K_PACK_B_STEP(0);
                    K_PACK_B_STEP(1);
                    src += 2 * ldb;
                    dst += 2 * unroll_n;
                }
                if (k & 1) {
                    K_PACK_B_STEP(0);
                    src += 1 * ldb;
                    dst += 1 * unroll_n;
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

void gemm_trans_a_operation_fp32(
    const float *A,
    const int64_t M,
    const int64_t K,
    const int64_t lda,
    float *transA)
{
    const int64_t unroll_m = 8;
    const int64_t unroll_k = 8;
    const int64_t ldtrans_a = K;

    for (int64_t m = 0; m < M; m += unroll_m) { // A: (K, M) -> (M, K)
        const int64_t m_eff = min(unroll_m, M - m);
        const float *src = A + m;
        float *dst = transA + m * ldtrans_a;
        int64_t k = K;
        if (m_eff == unroll_m) {
// ========================================================================== //
#define K_TRANS_A_STEP(K) do {\
    dst[0 * ldtrans_a + K] = src[0 + lda * K];\
    dst[1 * ldtrans_a + K] = src[1 + lda * K];\
    dst[2 * ldtrans_a + K] = src[2 + lda * K];\
    dst[3 * ldtrans_a + K] = src[3 + lda * K];\
    dst[4 * ldtrans_a + K] = src[4 + lda * K];\
    dst[5 * ldtrans_a + K] = src[5 + lda * K];\
    dst[6 * ldtrans_a + K] = src[6 + lda * K];\
    dst[7 * ldtrans_a + K] = src[7 + lda * K];\
} while (0)
// ========================================================================== //
            while (k >= unroll_k) {
                k -= unroll_k;
                transpose_8x8_fp32_avx(src, lda, ldtrans_a, dst);
                src += unroll_k * lda;
                dst += unroll_k;
            }
            if (k & 4) {
                K_TRANS_A_STEP(0);
                K_TRANS_A_STEP(1);
                K_TRANS_A_STEP(2);
                K_TRANS_A_STEP(3);
                src += 4 * lda;
                dst += 4;
            }
            if (k & 2) {
                K_TRANS_A_STEP(0);
                K_TRANS_A_STEP(1);
                src += 2 * lda;
                dst += 2;
            }
            if (k & 1) {
                K_TRANS_A_STEP(0);
                src += 1 * lda;
                dst += 1;
            }
#undef K_TRANS_A_STEP
        } else {
// ========================================================================== //
#define K_TRANS_A_STEP(K) do {\
    const float *l_src = src;\
    float *l_dst = dst;\
    if (m_eff & 4) {\
        l_dst[0 * ldtrans_a + K] = l_src[0 + lda * K];\
        l_dst[1 * ldtrans_a + K] = l_src[1 + lda * K];\
        l_dst[2 * ldtrans_a + K] = l_src[2 + lda * K];\
        l_dst[3 * ldtrans_a + K] = l_src[3 + lda * K];\
        l_dst += 4 * ldtrans_a;\
        l_src += 4;\
    }\
    if (m_eff & 2) {\
        l_dst[0 * ldtrans_a + K] = l_src[0 + lda * K];\
        l_dst[1 * ldtrans_a + K] = l_src[1 + lda * K];\
        l_dst += 2 * ldtrans_a;\
        l_src += 2;\
    }\
    if (m_eff & 1) {\
        l_dst[0 * ldtrans_a + K] = l_src[0 + lda * K];\
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
                src += unroll_k * lda;
                dst += unroll_k;
            }
            if (k & 4) {
                K_TRANS_A_STEP(0);
                K_TRANS_A_STEP(1);
                K_TRANS_A_STEP(2);
                K_TRANS_A_STEP(3);
                src += 4 * lda;
                dst += 4;
            }
            if (k & 2) {
                K_TRANS_A_STEP(0);
                K_TRANS_A_STEP(1);
                src += 2 * lda;
                dst += 2;
            }
            if (k & 1) {
                K_TRANS_A_STEP(0);
                src += 1 * lda;
                dst += 1;
            }
#undef K_TRANS_A_STEP
        }
    }
}

void gemm_fp32_fma_copy_c_buf(
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
void gemm_fp32_fma_apply_beta(
    const float *V,
    const float *H,
    const int64_t M,
    const int64_t N,
    const int64_t ldh,
    const int64_t ldc,
    const float beta,
    float *C)
{
    if (typeV == gemm_v_type::empty && typeH == gemm_m_type::empty)
        return;

    const int64_t n_reg_elts = gemm_kernel_fp32_fma::config::n_reg_elts;
    const int64_t unroll_n = n_reg_elts * 2;
 
    __m256 ymm_v, ymm_beta;
    if (typeV == gemm_v_type::scalar) ymm_v = _mm256_set1_ps(V[0]);
    ymm_beta = _mm256_set1_ps(beta);

    for (int64_t m = 0; m < M; ++m) {
        const float *l_v = nullptr;
        const float *l_h = nullptr;
        if (typeV == gemm_v_type::col_vec) ymm_v = _mm256_set1_ps(V[m]);
        if (typeV == gemm_v_type::row_vec) l_v = V;
        if (typeH == gemm_m_type::notrans) l_h = H + m * ldh;
        float *l_c = C + m * ldc;
        int64_t n = N;
        while (n >= unroll_n) {
            __m256 ymm0, ymm1;
            if (typeV != gemm_v_type::empty) {
                if (typeV == gemm_v_type::row_vec) {
                    ymm0 = _mm256_loadu_ps(l_v + 0 * n_reg_elts);
                    ymm1 = _mm256_loadu_ps(l_v + 1 * n_reg_elts);
                } else {
                    ymm0 = ymm_v;
                    ymm1 = ymm_v;
                }
            }
            if (typeH == gemm_m_type::notrans) {
                if (typeV == gemm_v_type::empty) {
                    ymm0 = _mm256_loadu_ps(l_h + 0 * n_reg_elts);
                    ymm1 = _mm256_loadu_ps(l_h + 1 * n_reg_elts);
                } else {
                    ymm0 = _mm256_add_ps(_mm256_loadu_ps(l_h + 0 * n_reg_elts), ymm0);
                    ymm1 = _mm256_add_ps(_mm256_loadu_ps(l_h + 1 * n_reg_elts), ymm1);
                }
            }
            ymm0 = _mm256_mul_ps(ymm0, ymm_beta);
            ymm1 = _mm256_mul_ps(ymm1, ymm_beta);
            _mm256_storeu_ps(l_c + 0 * n_reg_elts, ymm0);
            _mm256_storeu_ps(l_c + 1 * n_reg_elts, ymm1);
            if (typeV == gemm_v_type::row_vec) l_v += unroll_n;
            if (typeH == gemm_m_type::notrans) l_h += unroll_n;
            l_c += unroll_n;
            n -= unroll_n;
        }
        while (n > 0) {
            float y = 0.0f;
            if (typeV != gemm_v_type::empty) {
                if (typeV == gemm_v_type::scalar) {
                    y = V[0];
                }
                if (typeV == gemm_v_type::row_vec) {
                    y = l_v[0];
                }
                if (typeV == gemm_v_type::col_vec) {
                    y = V[m];
                }
            }
            if (typeH == gemm_m_type::notrans) {
                if (typeV == gemm_v_type::empty) {
                    y = l_h[0];
                } else {
                    y += l_h[0];
                }
            }
            l_c[0] = y * beta;
            if (typeV == gemm_v_type::row_vec) l_v += 1;
            if (typeH == gemm_m_type::notrans) l_h += 1;
            l_c += 1;
            n -= 1;
        }
    }
}

// Row-major impl, H and C could be the same matrix
// Parameter <hint> is for NUMA-Aware schedule
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
    const int32_t hint,
    float *C)
{
    if (typeA == gemm_m_type::packed || typeB == gemm_m_type::packed) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typeH != gemm_m_type::empty && typeH != gemm_m_type::notrans) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t n_reg_elts = gemm_kernel_fp32_fma::config::n_reg_elts;
    const bool apply_aAB = alpha != 0.0f && typeA != gemm_m_type::empty && typeB != gemm_m_type::empty;
    const bool apply_bVpbH = beta != 0.0f && (typeV != gemm_v_type::empty || typeH != gemm_m_type::empty);
    const bool alloc_c_buffer = (hint || (N % n_reg_elts)) && apply_aAB;

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
        sel_k_l2_blk = init_k_l2_blk_l;
        sel_n_l2_blk = init_n_l2_blk_l;
    } else if (l2_size > 256 * 1024) {
        sel_k_l2_blk = init_k_l2_blk_m;
        sel_n_l2_blk = init_n_l2_blk_s;
    } else {
        sel_k_l2_blk = init_k_l2_blk_s;
        sel_n_l2_blk = init_n_l2_blk_s;
    }

    const int64_t max_packed_b_len = sel_n_l2_blk * sel_k_l2_blk;
    const int64_t max_c_buffer_len = init_m_l3_blk * sel_n_l2_blk;

    int64_t k_l2_blk = min(sel_k_l2_blk, K);
    int64_t n_l2_blk = round_up(min(max_packed_b_len / k_l2_blk, N), n_reg_elts);
    if (typeA == gemm_m_type::notrans && n_l2_blk < 0.75f * sel_n_l2_blk) {
        k_l2_blk = min(max_packed_b_len / n_l2_blk, K);
    }
    int64_t m_l3_blk = alloc_c_buffer ? min(max_c_buffer_len / n_l2_blk, M) : M;

    ppl::common::GenericCpuAllocator allocator;
    float *packed_b = (float*)allocator.Alloc(k_l2_blk * n_l2_blk * sizeof(float));
    float *transposed_a = typeA == gemm_m_type::trans ? (float*)allocator.Alloc(m_l2_blk * k_l2_blk * sizeof(float)) : nullptr;
    float *c_buffer = alloc_c_buffer ? (float*)allocator.Alloc(m_l3_blk * n_l2_blk * sizeof(float)) : nullptr;

    auto apply_beta_func = gemm_fp32_fma_apply_beta<gemm_m_type::empty, gemm_v_type::empty>;
    if (typeH == gemm_m_type::notrans) {
        if (typeV == gemm_v_type::empty) apply_beta_func = gemm_fp32_fma_apply_beta<gemm_m_type::notrans, gemm_v_type::empty>;
        if (typeV == gemm_v_type::scalar) apply_beta_func = gemm_fp32_fma_apply_beta<gemm_m_type::notrans, gemm_v_type::scalar>;
        if (typeV == gemm_v_type::col_vec) apply_beta_func = gemm_fp32_fma_apply_beta<gemm_m_type::notrans, gemm_v_type::col_vec>;
        if (typeV == gemm_v_type::row_vec) apply_beta_func = gemm_fp32_fma_apply_beta<gemm_m_type::notrans, gemm_v_type::row_vec>;
    } else {
        if (typeV == gemm_v_type::scalar) apply_beta_func = gemm_fp32_fma_apply_beta<gemm_m_type::empty, gemm_v_type::scalar>;
        if (typeV == gemm_v_type::col_vec) apply_beta_func = gemm_fp32_fma_apply_beta<gemm_m_type::empty, gemm_v_type::col_vec>;
        if (typeV == gemm_v_type::row_vec) apply_beta_func = gemm_fp32_fma_apply_beta<gemm_m_type::empty, gemm_v_type::row_vec>;
    }

    int64_t kernel_param[gemm_kernel_fp32_fma::param_def::length];
    array_param_helper ker_p(kernel_param);
    gemm_kernel_fp32_fma ker(kernel_param);
    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::alpha_idx) = alpha;

    for (int64_t ml3 = 0; ml3 < M; ml3 += m_l3_blk) {
        for (int64_t nl2 = 0; nl2 < N; nl2 += n_l2_blk) {
            const int64_t ml3_eff = min(m_l3_blk, M - ml3);
            const int64_t nl2_eff = min(n_l2_blk, N - nl2);
            const int64_t padded_nl2_eff = round_up(nl2_eff, n_reg_elts);

            const int64_t nl2_body = round(nl2_eff, gemm_kernel_fp32_fma::config::max_n_blk);
            const int64_t nl2_treg = div_up(nl2_eff - nl2_body, n_reg_elts);
            const int64_t nl2_tail = nl2_treg * n_reg_elts;

            const bool use_c_buffer = alloc_c_buffer && (hint || (nl2_eff % n_reg_elts));

            float *local_c = C + ml3 * ldc + nl2;
            float *local_c_buf = use_c_buffer ? c_buffer : local_c;
            const int64_t ldc_buf = use_c_buffer ? padded_nl2_eff : ldc;
            ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::c_ptr_idx) = local_c_buf;
            ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::ldc_idx) = ldc_buf;
            ker_p.pick<gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::flags_idx) = 0;

            if (apply_bVpbH) {
                const float *l_h = nullptr;
                const float *l_v = nullptr;
                if (typeH == gemm_m_type::notrans) l_h = H + ml3 * ldh + nl2;
                if (typeV == gemm_v_type::scalar) l_v = V;
                if (typeV == gemm_v_type::col_vec) l_v = V + ml3;
                if (typeV == gemm_v_type::row_vec) l_v = V + nl2;
                apply_beta_func(l_v, l_h, ml3_eff, nl2_eff, ldh, ldc_buf, beta, local_c_buf);
                ker_p.pick<gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::flags_idx) |= gemm_kernel_fp32_fma::flag::load_c;
            }

            if (!apply_aAB)
                continue;

            for (int64_t kl2 = 0; kl2 < K; kl2 += k_l2_blk) {
                const int64_t kl2_eff = min(k_l2_blk, K - kl2);
                const bool is_last_k = kl2 + kl2_eff == K;
                if (is_last_k) {
                    if (post == gemm_post::relu6) ker_p.pick<gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::flags_idx) |= gemm_kernel_fp32_fma::flag::relu6;
                    if (post == gemm_post::relu) ker_p.pick<gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::flags_idx) |= gemm_kernel_fp32_fma::flag::relu;
                }
                const float *base_b = B + (typeB == gemm_m_type::notrans ? kl2 * ldb + nl2 : nl2 * ldb + kl2);
                const float *base_a = A + (typeA == gemm_m_type::notrans ? ml3 * lda + kl2 : kl2 * lda + ml3);
                float *base_c_buf = local_c_buf;
                gemm_pack_b_operation_fp32_fma(base_b, typeB, nl2_eff, kl2_eff, ldb, packed_b);
                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::ldpacked_b_idx) = kl2_eff * n_reg_elts;
                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::k_idx) = kl2_eff;

                if (typeA == gemm_m_type::notrans) {
                    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::lda_idx) = lda;
                    int64_t m = ml3_eff;
                    while (m >= gemm_kernel_fp32_fma::config::max_m_blk) {
                        m -= gemm_kernel_fp32_fma::config::max_m_blk;
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::a_ptr_idx) = base_a;
                        if (nl2_body) {
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::packed_b_ptr_idx) = packed_b;
                            ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::c_ptr_idx) = base_c_buf;
                            ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::n_idx) = nl2_body;
                            ker.execute(gemm_kernel_fp32_fma::config::max_m_regs, gemm_kernel_fp32_fma::config::max_n_regs);
                        }
                        if (nl2_tail) {
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::packed_b_ptr_idx) = packed_b + nl2_body * kl2_eff;
                            ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::c_ptr_idx) = base_c_buf + nl2_body;
                            ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::n_idx) = nl2_tail;
                            ker.execute(gemm_kernel_fp32_fma::config::max_m_regs, nl2_treg);
                        }

                        base_c_buf += gemm_kernel_fp32_fma::config::max_m_blk * ldc_buf;
                        base_a += gemm_kernel_fp32_fma::config::max_m_blk * lda;
                    }
                    if (m > 0) {
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::a_ptr_idx) = base_a;
                        if (nl2_body) {
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::packed_b_ptr_idx) = packed_b;
                            ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::c_ptr_idx) = base_c_buf;
                            ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::n_idx) = nl2_body;
                            ker.execute(m, gemm_kernel_fp32_fma::config::max_n_regs);
                        }
                        if (nl2_tail) {
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::packed_b_ptr_idx) = packed_b + nl2_body * kl2_eff;
                            ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::c_ptr_idx) = base_c_buf + nl2_body;
                            ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::n_idx) = nl2_tail;
                            ker.execute(m, nl2_treg);
                        }
                    }
                } else {
                    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::lda_idx) = kl2_eff;
                    for (int64_t ml2 = 0; ml2 < ml3_eff; ml2 += m_l2_blk) {
                        const int64_t ml2_eff = min(m_l2_blk, ml3_eff - ml2);
                        gemm_trans_a_operation_fp32(base_a + ml2, ml2_eff, kl2_eff, lda, transposed_a);
                        const float *base_trans_a = transposed_a;
                        const int64_t ldtrans_a = kl2_eff;
                        int64_t m = ml2_eff;
                        while (m >= gemm_kernel_fp32_fma::config::max_m_blk) {
                            m -= gemm_kernel_fp32_fma::config::max_m_blk;
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::a_ptr_idx) = base_trans_a;
                            if (nl2_body) {
                                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::packed_b_ptr_idx) = packed_b;
                                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::c_ptr_idx) = base_c_buf;
                                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::n_idx) = nl2_body;
                                ker.execute(gemm_kernel_fp32_fma::config::max_m_regs, gemm_kernel_fp32_fma::config::max_n_regs);
                            }
                            if (nl2_tail) {
                                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::packed_b_ptr_idx) = packed_b + nl2_body * kl2_eff;
                                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::c_ptr_idx) = base_c_buf + nl2_body;
                                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::n_idx) = nl2_tail;
                                ker.execute(gemm_kernel_fp32_fma::config::max_m_regs, nl2_treg);
                            }

                            base_c_buf += gemm_kernel_fp32_fma::config::max_m_blk * ldc_buf;
                            base_trans_a += gemm_kernel_fp32_fma::config::max_m_blk * ldtrans_a;
                        }
                        if (m > 0) {
                            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::a_ptr_idx) = base_trans_a;
                            if (nl2_body) {
                                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::packed_b_ptr_idx) = packed_b;
                                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::c_ptr_idx) = base_c_buf;
                                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::n_idx) = nl2_body;
                                ker.execute(m, gemm_kernel_fp32_fma::config::max_n_regs);
                            }
                            if (nl2_tail) {
                                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::packed_b_ptr_idx) = packed_b + nl2_body * kl2_eff;
                                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::c_ptr_idx) = base_c_buf + nl2_body;
                                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::n_idx) = nl2_tail;
                                ker.execute(m, nl2_treg);
                            }
                        }
                    }
                }
                
                ker_p.pick<gemm_kernel_fp32_fma::flag_t>(gemm_kernel_fp32_fma::param_def::flags_idx) |= gemm_kernel_fp32_fma::flag::load_c;
            }

            if (use_c_buffer) {
                gemm_fp32_fma_copy_c_buf(c_buffer, ml3_eff, nl2_eff, ldc_buf, ldc, local_c);
            }
        }
    }

    if (packed_b) allocator.Free(packed_b);
    if (transposed_a) allocator.Free(transposed_a);
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
    if (typeA == gemm_m_type::packed || typeB == gemm_m_type::packed) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typeH != gemm_m_type::empty && typeH != gemm_m_type::notrans) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t num_threads = PPL_OMP_MAX_THREADS();

    if (num_threads == 1) {
        return gemm_operation_fp32_fma(
            A, B, V, H,
            typeA, typeB, typeV, typeH,
            M, N, K, lda, ldb ,ldc ,ldh,
            alpha, beta, post, 0, C);
    }

    int64_t m_task_blk;
    int64_t n_task_blk;
    int64_t m_task;
    int64_t n_task;

    if (N > M) {
        n_task_blk = round_up(div_up(N, num_threads), gemm_kernel_fp32_fma::config::n_reg_elts);
        n_task = div_up(N, n_task_blk);
        m_task = max<int64_t>(1, num_threads / n_task);
        m_task_blk = round_up(div_up(M, m_task), init_m_l3_blk);
        m_task = div_up(M, m_task_blk);
    } else {
        m_task_blk = round_up(div_up(M, num_threads), init_m_l3_blk / 2);
        m_task = div_up(M, m_task_blk);
        n_task = max<int64_t>(1, num_threads / m_task);
        n_task_blk = round_up(div_up(N, n_task), gemm_kernel_fp32_fma::config::n_reg_elts);
        n_task = div_up(N, n_task_blk);
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
        if (typeA == gemm_m_type::notrans) {
            lA += mb * lda;
        } else {
            lA += mb;
        }

        const float *lB = B;
        if (typeB == gemm_m_type::notrans) {
            lB += nb;
        } else {
            lB += nb * ldb;
        }

        const float *lV = V;
        if (typeV == gemm_v_type::col_vec) {
            lV += mb;
        } else if (typeV == gemm_v_type::row_vec) {
            lV += nb;
        }

        const float *lH = H;
        if (typeH == gemm_m_type::notrans) {
            lH += mb * ldh + nb;
        }

        float *lC = C + mb * ldh + nb;

        const int64_t mb_eff = min(m_task_blk, M - mb);
        const int64_t nb_eff = min(n_task_blk, N - nb);

        auto ret = gemm_operation_fp32_fma(
            lA, lB, lV, lH,
            typeA, typeB, typeV, typeH,
            mb_eff, nb_eff, K, lda, ldb ,ldc ,ldh,
            alpha, beta, post, 0, lC);

        (void) ret;
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
