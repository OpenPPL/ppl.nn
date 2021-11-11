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

#include <nmmintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gemm_v2/sse/gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse.h"
#include "ppl/kernel/x86/fp32/gemm_v2/sse/kernel/gemm_kernel_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

const int32_t simd_w = 4;

#define TRANSPOSE_4X4()                          \
    do {                                         \
        xmm4 = _mm_unpacklo_ps(xmm0, xmm1);      \
        xmm5 = _mm_unpackhi_ps(xmm0, xmm1);      \
        xmm6 = _mm_unpacklo_ps(xmm2, xmm3);      \
        xmm7 = _mm_unpackhi_ps(xmm2, xmm3);      \
        xmm0 = _mm_shuffle_ps(xmm4, xmm6, 0x44); \
        xmm1 = _mm_shuffle_ps(xmm4, xmm6, 0xee); \
        xmm2 = _mm_shuffle_ps(xmm5, xmm7, 0x44); \
        xmm3 = _mm_shuffle_ps(xmm5, xmm7, 0xee); \
    } while (0);

void gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse::load_a_data(
    const float* src,
    const int32_t m_len,
    const int32_t k_len,
    float* dst)
{
    const int32_t& lda       = param_.lda;
    const int32_t& m_blk_len = blk_partition_.m_blk_len;

    if (param_.trans_A) {
        for (int32_t k = 0; k < k_len; k++) {
            int32_t m = 0;
            for (; m + simd_w <= m_len; m += simd_w) {
                __m128 xmm0 = _mm_loadu_ps(src + k * lda + m);
                _mm_storeu_ps(dst + k * m_blk_len + m, xmm0);
            }
            for (; m < m_len; m++) {
                dst[k * m_blk_len + m] = src[k * lda + m];
            }
        }
    } else {
        __m128 xmm0, xmm1, xmm2, xmm3;
        __m128 xmm4, xmm5, xmm6, xmm7;

        int32_t m = 0;
        for (; m + simd_w <= m_len; m += simd_w) {
            int32_t k = 0;
            for (; k + simd_w <= k_len; k += simd_w) {
                xmm0 = _mm_loadu_ps(src + (m + 0) * lda + k);
                xmm1 = _mm_loadu_ps(src + (m + 1) * lda + k);
                xmm2 = _mm_loadu_ps(src + (m + 2) * lda + k);
                xmm3 = _mm_loadu_ps(src + (m + 3) * lda + k);

                TRANSPOSE_4X4();

                _mm_storeu_ps(dst + (k + 0) * m_blk_len + m, xmm0);
                _mm_storeu_ps(dst + (k + 1) * m_blk_len + m, xmm1);
                _mm_storeu_ps(dst + (k + 2) * m_blk_len + m, xmm2);
                _mm_storeu_ps(dst + (k + 3) * m_blk_len + m, xmm3);
            }
            for (; k < k_len; k++) {
                dst[k * m_blk_len + m + 0] = src[(m + 0) * lda + k];
                dst[k * m_blk_len + m + 1] = src[(m + 1) * lda + k];
                dst[k * m_blk_len + m + 2] = src[(m + 2) * lda + k];
                dst[k * m_blk_len + m + 3] = src[(m + 3) * lda + k];
            }
        }
        for (; m < m_len; m++) {
            for (int32_t k = 0; k < k_len; k++) {
                dst[k * m_blk_len + m] = src[m * lda + k];
            }
        }
    }
}

void gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse::load_b_data(
    const float* src,
    const int32_t n_len,
    const int32_t k_len,
    float* dst)
{
    const int32_t& ldb       = param_.ldb;
    const int32_t& n_blk_len = blk_partition_.n_blk_len;

    if (param_.trans_B) {
        __m128 xmm0, xmm1, xmm2, xmm3;
        __m128 xmm4, xmm5, xmm6, xmm7;

        int32_t n = 0;
        for (; n + simd_w <= n_len; n += simd_w) {
            int32_t k = 0;
            for (; k + simd_w <= k_len; k += simd_w) {
                xmm0 = _mm_loadu_ps(src + (n + 0) * ldb + k);
                xmm1 = _mm_loadu_ps(src + (n + 1) * ldb + k);
                xmm2 = _mm_loadu_ps(src + (n + 2) * ldb + k);
                xmm3 = _mm_loadu_ps(src + (n + 3) * ldb + k);

                TRANSPOSE_4X4();

                _mm_storeu_ps(dst + (k + 0) * n_blk_len + n, xmm0);
                _mm_storeu_ps(dst + (k + 1) * n_blk_len + n, xmm1);
                _mm_storeu_ps(dst + (k + 2) * n_blk_len + n, xmm2);
                _mm_storeu_ps(dst + (k + 3) * n_blk_len + n, xmm3);
            }
            for (; k < k_len; k++) {
                dst[k * n_blk_len + n + 0] = src[(n + 0) * ldb + k];
                dst[k * n_blk_len + n + 1] = src[(n + 1) * ldb + k];
                dst[k * n_blk_len + n + 2] = src[(n + 2) * ldb + k];
                dst[k * n_blk_len + n + 3] = src[(n + 3) * ldb + k];
            }
        }
        for (; n < n_len; n++) {
            for (int32_t k = 0; k < k_len; k++) {
                dst[k * n_blk_len + n] = src[n * ldb + k];
            }
        }
    } else {
        for (int32_t k = 0; k < k_len; k++) {
            int32_t n = 0;
            for (; n + simd_w <= n_len; n += simd_w) {
                __m128 xmm0 = _mm_loadu_ps(src + k * ldb + n);
                _mm_storeu_ps(dst + k * n_blk_len + n, xmm0);
            }
            for (; n < n_len; n++) {
                dst[k * n_blk_len + n] = src[k * ldb + n];
            }
        }
    }
}

template <gemm_v2_C_type_t c_type, gemm_v2_fuse_flag_t fuse_flag, bool alpha_1, bool beta_0>
static void store_dst_data_impl(
    const float* src,
    const int32_t m_len,
    const int32_t n_len,
    const float alpha,
    const float beta,
    const int32_t n_blk_len,
    const float* C,
    const int32_t ldc,
    const int32_t ldy,
    float* dst)
{
    __m128 v_zero  = _mm_set1_ps(0);
    __m128 v_alpha = _mm_set1_ps(alpha);
    __m128 v_beta  = _mm_set1_ps(beta);

    float s_c  = 0;
    __m128 v_c = v_zero;
    if (!beta_0 && c_type == gemm_v2_C_type::SCALAR) {
        s_c = C[0];
        v_c = _mm_set1_ps(s_c);
    }

    for (int32_t m = 0; m < m_len; m++) {
        if (!beta_0 && c_type == gemm_v2_C_type::VECTOR_H) {
            s_c = C[m];
            v_c = _mm_set1_ps(s_c);
        }
        int32_t n = 0;
        for (; n + simd_w <= n_len; n += simd_w) {
            __m128 v_data = _mm_loadu_ps(src + m * n_blk_len + n);
            if (!alpha_1) {
                v_data = _mm_mul_ps(v_data, v_alpha);
            }
            if (!beta_0 && c_type != gemm_v2_C_type::EMPTY) {
                if (c_type == gemm_v2_C_type::MATRIX) {
                    v_c = _mm_loadu_ps(C + m * ldc + n);
                }
                if (c_type == gemm_v2_C_type::VECTOR_W) {
                    v_c = _mm_loadu_ps(C + n);
                }
                v_data = _mm_add_ps(v_data, _mm_mul_ps(v_beta, v_c));
            }
            if (fuse_flag & gemm_v2_fuse_flag::RELU) {
                v_data = _mm_max_ps(v_data, v_zero);
            }
            _mm_storeu_ps(dst + m * ldy + n, v_data);
        }
        for (; n < n_len; n++) {
            float data = src[m * n_blk_len + n];
            if (!alpha_1) {
                data *= alpha;
            }
            if (!beta_0 && c_type != gemm_v2_C_type::EMPTY) {
                if (c_type == gemm_v2_C_type::MATRIX) {
                    s_c = C[m * ldc + n];
                }
                if (c_type == gemm_v2_C_type::VECTOR_W) {
                    s_c = C[n];
                }
                data += beta * s_c;
            }
            if (fuse_flag & gemm_v2_fuse_flag::RELU) {
                data = max(data, 0.0f);
            }
            dst[m * ldy + n] = data;
        }
    }
}

typedef void (*store_dst_data_func_type_t)(const float*, const int32_t, const int32_t, const float, const float, const int32_t, const float*, const int32_t, const int32_t, float*);
static const store_dst_data_func_type_t store_dst_data_func_tab[5][2][2][2] = {
    {
        {
            {
                store_dst_data_impl<gemm_v2_C_type::EMPTY, gemm_v2_fuse_flag::NONE, false, false>,
                store_dst_data_impl<gemm_v2_C_type::EMPTY, gemm_v2_fuse_flag::NONE, false, true>,
            },
            {
                store_dst_data_impl<gemm_v2_C_type::EMPTY, gemm_v2_fuse_flag::NONE, true, false>,
                store_dst_data_impl<gemm_v2_C_type::EMPTY, gemm_v2_fuse_flag::NONE, true, true>,
            },
        },
        {
            {
                store_dst_data_impl<gemm_v2_C_type::EMPTY, gemm_v2_fuse_flag::RELU, false, false>,
                store_dst_data_impl<gemm_v2_C_type::EMPTY, gemm_v2_fuse_flag::RELU, false, true>,
            },
            {
                store_dst_data_impl<gemm_v2_C_type::EMPTY, gemm_v2_fuse_flag::RELU, true, false>,
                store_dst_data_impl<gemm_v2_C_type::EMPTY, gemm_v2_fuse_flag::RELU, true, true>,
            },
        },
    },
    {
        {
            {
                store_dst_data_impl<gemm_v2_C_type::SCALAR, gemm_v2_fuse_flag::NONE, false, false>,
                store_dst_data_impl<gemm_v2_C_type::SCALAR, gemm_v2_fuse_flag::NONE, false, true>,
            },
            {
                store_dst_data_impl<gemm_v2_C_type::SCALAR, gemm_v2_fuse_flag::NONE, true, false>,
                store_dst_data_impl<gemm_v2_C_type::SCALAR, gemm_v2_fuse_flag::NONE, true, true>,
            },
        },
        {
            {
                store_dst_data_impl<gemm_v2_C_type::SCALAR, gemm_v2_fuse_flag::RELU, false, false>,
                store_dst_data_impl<gemm_v2_C_type::SCALAR, gemm_v2_fuse_flag::RELU, false, true>,
            },
            {
                store_dst_data_impl<gemm_v2_C_type::SCALAR, gemm_v2_fuse_flag::RELU, true, false>,
                store_dst_data_impl<gemm_v2_C_type::SCALAR, gemm_v2_fuse_flag::RELU, true, true>,
            },
        },
    },
    {
        {
            {
                store_dst_data_impl<gemm_v2_C_type::VECTOR_H, gemm_v2_fuse_flag::NONE, false, false>,
                store_dst_data_impl<gemm_v2_C_type::VECTOR_H, gemm_v2_fuse_flag::NONE, false, true>,
            },
            {
                store_dst_data_impl<gemm_v2_C_type::VECTOR_H, gemm_v2_fuse_flag::NONE, true, false>,
                store_dst_data_impl<gemm_v2_C_type::VECTOR_H, gemm_v2_fuse_flag::NONE, true, true>,
            },
        },
        {
            {
                store_dst_data_impl<gemm_v2_C_type::VECTOR_H, gemm_v2_fuse_flag::RELU, false, false>,
                store_dst_data_impl<gemm_v2_C_type::VECTOR_H, gemm_v2_fuse_flag::RELU, false, true>,
            },
            {
                store_dst_data_impl<gemm_v2_C_type::VECTOR_H, gemm_v2_fuse_flag::RELU, true, false>,
                store_dst_data_impl<gemm_v2_C_type::VECTOR_H, gemm_v2_fuse_flag::RELU, true, true>,
            },
        },
    },
    {
        {
            {
                store_dst_data_impl<gemm_v2_C_type::VECTOR_W, gemm_v2_fuse_flag::NONE, false, false>,
                store_dst_data_impl<gemm_v2_C_type::VECTOR_W, gemm_v2_fuse_flag::NONE, false, true>,
            },
            {
                store_dst_data_impl<gemm_v2_C_type::VECTOR_W, gemm_v2_fuse_flag::NONE, true, false>,
                store_dst_data_impl<gemm_v2_C_type::VECTOR_W, gemm_v2_fuse_flag::NONE, true, true>,
            },
        },
        {
            {
                store_dst_data_impl<gemm_v2_C_type::VECTOR_W, gemm_v2_fuse_flag::RELU, false, false>,
                store_dst_data_impl<gemm_v2_C_type::VECTOR_W, gemm_v2_fuse_flag::RELU, false, true>,
            },
            {
                store_dst_data_impl<gemm_v2_C_type::VECTOR_W, gemm_v2_fuse_flag::RELU, true, false>,
                store_dst_data_impl<gemm_v2_C_type::VECTOR_W, gemm_v2_fuse_flag::RELU, true, true>,
            },
        },
    },
    {
        {
            {
                store_dst_data_impl<gemm_v2_C_type::MATRIX, gemm_v2_fuse_flag::NONE, false, false>,
                store_dst_data_impl<gemm_v2_C_type::MATRIX, gemm_v2_fuse_flag::NONE, false, true>,
            },
            {
                store_dst_data_impl<gemm_v2_C_type::MATRIX, gemm_v2_fuse_flag::NONE, true, false>,
                store_dst_data_impl<gemm_v2_C_type::MATRIX, gemm_v2_fuse_flag::NONE, true, true>,
            },
        },
        {
            {
                store_dst_data_impl<gemm_v2_C_type::MATRIX, gemm_v2_fuse_flag::RELU, false, false>,
                store_dst_data_impl<gemm_v2_C_type::MATRIX, gemm_v2_fuse_flag::RELU, false, true>,
            },
            {
                store_dst_data_impl<gemm_v2_C_type::MATRIX, gemm_v2_fuse_flag::RELU, true, false>,
                store_dst_data_impl<gemm_v2_C_type::MATRIX, gemm_v2_fuse_flag::RELU, true, true>,
            },
        },
    },
};

void gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse::store_dst_data(
    const float* src,
    const int32_t m_len,
    const int32_t n_len,
    const float* C,
    float* dst)
{
    const float& alpha      = param_.alpha;
    const float& beta       = param_.beta;
    const int32_t c_type    = (int32_t)param_.c_type;
    const int32_t fuse_flag = (int32_t)param_.fuse_flag;

    store_dst_data_func_tab[c_type][fuse_flag][alpha == 1.0f ? 1 : 0][beta == 0 ? 1 : 0](
        src, m_len, n_len, alpha, beta, blk_partition_.n_blk_len, C, param_.ldc, param_.ldy, dst);
}

void gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse::execute_sub_blk(
    const float* A,
    const float* B,
    const int32_t m_len,
    const int32_t n_len,
    const int32_t k_len,
    float* dst)
{
    const int32_t& m_kernel_blk_len = blk_partition_.m_kernel_blk_len;
    const int32_t& n_kernel_blk_len = blk_partition_.n_kernel_blk_len;

    const int32_t& lda = blk_partition_.m_blk_len;
    const int32_t& ldb = blk_partition_.n_blk_len;
    const int32_t& ldc = blk_partition_.n_blk_len;

    if (m_len % m_kernel_blk_len == 0 && n_len % n_kernel_blk_len == 0) {
        for (int32_t n = 0; n < n_len; n += n_kernel_blk_len) {
            for (int32_t m = 0; m < m_len; m += m_kernel_blk_len) {
                gemm_kernel_4x12_fp32_sse(A + m, B + n, k_len, lda, ldb, ldc, dst + m * ldc + n);
            }
        }
    } else {
        for (int32_t n = 0; n < n_len; n += n_kernel_blk_len) {
            for (int32_t m = 0; m < m_len; m += m_kernel_blk_len) {
                const int32_t m_kernel_blk_eff = min(m_kernel_blk_len, m_len - m);
                const int32_t n_kernel_blk_eff = min(n_kernel_blk_len, n_len - n);
                gemm_kernel_max4x12_fp32_sse_func_tab[m_kernel_blk_eff][div_up(n_kernel_blk_eff, simd_w)](
                    A + m, B + n, k_len, lda, ldb, ldc, dst + m * ldc + n);
            }
        }
    }
}

common::RetCode gemm_v2_mnk_sub_kmn_kernel_nm_atbn_executor_fp32_sse::execute(void)
{
    const int32_t& M               = param_.M;
    const int32_t& N               = param_.N;
    const int32_t& K               = param_.K;
    const int32_t& lda             = param_.lda;
    const int32_t& ldb             = param_.ldb;
    const int32_t& ldc             = param_.ldc;
    const int32_t& ldy             = param_.ldy;
    const float* A                 = param_.src_A;
    const float* B                 = param_.src_B;
    const float* C                 = param_.src_C;
    float* dst                     = param_.dst_Y;
    const int32_t& trans_A         = param_.trans_A;
    const int32_t& trans_B         = param_.trans_B;
    const gemm_v2_C_type_t& c_type = param_.c_type;

    const int32_t& m_blk_len     = blk_partition_.m_blk_len;
    const int32_t& n_blk_len     = blk_partition_.n_blk_len;
    const int32_t& k_blk_len     = blk_partition_.k_blk_len;
    const int32_t& m_sub_blk_len = blk_partition_.m_sub_blk_len;
    const int32_t& n_sub_blk_len = blk_partition_.n_sub_blk_len;
    const int32_t& k_sub_blk_len = blk_partition_.k_sub_blk_len;

    float* temp_buffer = (float*)temp_buffer_;

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
    PRAGMA_OMP_PARALLEL_FOR()
#endif
    for (int32_t m = 0; m < M; m += m_blk_len) {
        for (int32_t n = 0; n < N; n += n_blk_len) {
            float* l_temp   = temp_buffer + PPL_OMP_THREAD_ID() * get_buffer_len_per_thread();
            float* temp_a   = l_temp;
            float* temp_b   = temp_a + get_a_buffer_len();
            float* temp_dst = temp_b + get_b_buffer_len();

            memset(temp_dst, 0, get_dst_buffer_len() * sizeof(float));

            const int32_t m_blk_eff = min(m_blk_len, M - m);
            const int32_t n_blk_eff = min(n_blk_len, N - n);

            for (int32_t k = 0; k < K; k += k_blk_len) {
                const int32_t k_blk_eff = min(k_blk_len, K - k);
                // load data into L2
                const float* l_src_a    = nullptr;
                const float* l_src_b    = nullptr;
                if (trans_A) {
                    l_src_a = A + k * lda + m;
                } else {
                    l_src_a = A + m * lda + k;
                }
                if (trans_B) {
                    l_src_b = B + n * ldb + k;
                } else {
                    l_src_b = B + k * ldb + n;
                }
                load_a_data(l_src_a, m_blk_eff, k_blk_eff, temp_a);
                load_b_data(l_src_b, n_blk_eff, k_blk_eff, temp_b);

                for (int32_t kk = 0; kk < k_blk_eff; kk += k_sub_blk_len) {
                    for (int32_t mm = 0; mm < m_blk_eff; mm += m_sub_blk_len) {
                        for (int32_t nn = 0; nn < n_blk_eff; nn += n_sub_blk_len) {
                            const int32_t m_sub_blk_eff = min(m_sub_blk_len, m_blk_eff - mm);
                            const int32_t n_sub_blk_eff = min(n_sub_blk_len, n_blk_eff - nn);
                            const int32_t k_sub_blk_eff = min(k_sub_blk_len, k_blk_eff - kk);

                            execute_sub_blk(
                                temp_a + kk * m_blk_len + mm,
                                temp_b + kk * n_blk_len + nn,
                                m_sub_blk_eff,
                                n_sub_blk_eff,
                                k_sub_blk_eff,
                                temp_dst + mm * n_blk_len + nn);
                        }
                    }
                }
            }

            const float* l_src_c = nullptr;
            if (c_type == gemm_v2_C_type::EMPTY || C == nullptr) {
                l_src_c = nullptr;
            } else if (c_type == gemm_v2_C_type::SCALAR) {
                l_src_c = C;
            } else if (c_type == gemm_v2_C_type::VECTOR_H) {
                l_src_c = C + m;
            } else if (c_type == gemm_v2_C_type::VECTOR_W) {
                l_src_c = C + n;
            } else if (c_type == gemm_v2_C_type::MATRIX) {
                l_src_c = C + m * ldc + n;
            }
            store_dst_data(temp_dst, m_blk_eff, n_blk_eff, l_src_c, dst + m * ldy + n);
        }
    }

    return common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86