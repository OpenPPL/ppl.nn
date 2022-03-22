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
#include "ppl/kernel/x86/common/array_param_helper.h"
#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/kernel/x86/fp32/gemm/common/gemm_base_operation_fp32_avx.h"

namespace ppl { namespace kernel { namespace x86 {

static const int64_t INIT_K_L2_BLK_S = 128;
static const int64_t INIT_K_L2_BLK_M = 192;
static const int64_t INIT_K_L2_BLK_L = 256;
static const int64_t INIT_N_L2_BLK_S = 144;
static const int64_t INIT_N_L2_BLK_L = 192;
static const int64_t INIT_M_L3_BLK_S = 512;
static const int64_t INIT_M_L3_BLK_L = 1024;
static const int64_t M_L2_BLK = 64;

typedef uint64_t opt_flag_t;

class opt_flag {
public:
    static const opt_flag_t c_overflow_opt = (1 << 1);
    static const opt_flag_t chiplet_opt = (1 << 2);
    static const opt_flag_t pack_a_opt = (1 << 3);
};

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

    const int64_t max_packed_b_len = sel_n_l2_blk * sel_k_l2_blk;
    const int64_t max_c_buffer_len = INIT_M_L3_BLK_S * sel_n_l2_blk;

    int64_t k_l2_blk = min(sel_k_l2_blk, K);
    int64_t n_l2_blk = round_up(min(max_packed_b_len / k_l2_blk, N), N_REG_ELTS);
    if (typeA == gemm_m_type::NOTRANS && n_l2_blk < 0.75f * sel_n_l2_blk) {
        k_l2_blk = min(max_packed_b_len / n_l2_blk, K);
    }

    bool force_c_buffer = (flags & opt_flag::c_overflow_opt) && N <= n_l2_blk;
    if (!(flags & opt_flag::chiplet_opt)) {
        force_c_buffer = force_c_buffer && (K / sel_k_l2_blk) > 4;
    }
    const bool alloc_c_buffer = force_c_buffer || ((N % N_REG_ELTS) && apply_aAB);

    int64_t m_l3_blk = alloc_c_buffer ? min(max_c_buffer_len / n_l2_blk, M) : INIT_M_L3_BLK_L;
    const bool sliding_packed_a = N <= n_l2_blk;
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

    auto pack_b_func = gemm_pack_b_n8_operation_fp32_avx<gemm_m_type::NOTRANS>;
    if (typeB == gemm_m_type::TRANS) pack_b_func = gemm_pack_b_n8_operation_fp32_avx<gemm_m_type::TRANS>;

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
                            pack_a_func(base_a + (typeA == gemm_m_type::TRANS ? ml2 : ml2 * lda), ml2_eff, kl2_eff, lda, kl2_eff, local_packed_a);
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
