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
#include "ppl/kernel/x86/common/threading_tools.h"

namespace ppl { namespace kernel { namespace x86 {

static const int64_t K_L2_BLK_MAX_LARGE = 256;
static const int64_t K_L2_BLK_MAX_SMALL = 192;
static const int64_t K_L1_BLK_MAX_SMALL_M = 2048;
static const int64_t N_L3_BLK_MAX = 10000;
static const int64_t N_THR_BLK_MIN = 384;
static const int64_t M_L3_BLK_MAX = 384;

typedef uint64_t opt_flag_t;

class opt_flag {
public:
    static const opt_flag_t large_c = (1 << 0);
    static const opt_flag_t large_l2 = (1 << 1);
    static const opt_flag_t multi_thread = (1 << 2);
};

typedef decltype(gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, 0, 0>)(*gemm_fp32_fma_pack_b_func_t);

ppl::common::RetCode gemm_packed_b_operation_fp32_fma(
    const float *A,
    const float *packedB,
    const float *bias,
    const float *sum,
    const gemm_m_type_t typeA,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldc,
    const int64_t ldsum,
    const float alpha,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    const gemm_post_t post,
    const opt_flag_t flags,
    float *C)
{
    if (typeA == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typesum != gemm_m_type::EMPTY && typesum != gemm_m_type::NOTRANS) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const bool is_trans_a = typeA == gemm_m_type::TRANS;

    // blocking
    int64_t k_blk = K;
    int64_t k_blk_max = (flags & opt_flag::large_l2) ? K_L2_BLK_MAX_LARGE : K_L2_BLK_MAX_SMALL;
    if (M <= gemm_kernel_fp32_fma::config::MAX_M_BLK) k_blk_max = K_L1_BLK_MAX_SMALL_M;
    if ((flags & opt_flag::large_c) && (flags & opt_flag::multi_thread)) k_blk_max *= 2; // avoid write c too many times
    if (k_blk >= 2 * k_blk_max) k_blk = k_blk_max;
    else if (k_blk >= 1.5 * k_blk_max) k_blk = div_up(k_blk, 2);

    const int64_t n_blk = round_up(min(max(gemm_kernel_fp32_fma::config::MAX_N_BLK, N), N_L3_BLK_MAX), gemm_kernel_fp32_fma::config::MAX_N_BLK);
    const int64_t m_blk = round_up(min(max(gemm_kernel_fp32_fma::config::MAX_M_BLK, M), M_L3_BLK_MAX), gemm_kernel_fp32_fma::config::MAX_M_BLK);

    const int64_t packed_a_bytes = k_blk * m_blk * sizeof(float) + PPL_X86_PAGE_BYTES();
    uint8_t *temp_buffer = (uint8_t*)ppl::common::AlignedAlloc(packed_a_bytes, PPL_X86_CACHELINE_BYTES());
    float *packed_a = (float*)round_up((uintptr_t)(temp_buffer), PPL_X86_PAGE_BYTES());

    const auto pack_a_func = is_trans_a ? gemm_pack_a_m4_operation_fp32_avx<gemm_m_type::TRANS> : gemm_pack_a_m4_operation_fp32_avx<gemm_m_type::NOTRANS>;

    int64_t kernel_param[gemm_kernel_fp32_fma::param_def::LENGTH];
    array_param_helper ker_p(kernel_param);
    gemm_kernel_fp32_fma ker(kernel_param);
    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::ALPHA_IDX) = alpha;
    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_BIAS_IDX) = beta_bias;
    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_SUM_IDX) = beta_sum;
    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::LDC_IDX) = ldc;
    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::LDSUM_IDX) = ldsum;
    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::PRF_C_LDK_IDX) =
        (flags & opt_flag::large_c) ?
        gemm_kernel_fp32_fma::config::PRF_C_LDK_MEM :
        gemm_kernel_fp32_fma::config::PRF_C_LDK_L3;

    for (int64_t nb = 0; nb < N; nb += n_blk) {
        const int64_t nb_eff = min(n_blk, N - nb);
        const int64_t nb_body = round(nb_eff, gemm_kernel_fp32_fma::config::MAX_N_BLK);
        const int64_t nb_tail = nb_eff - nb_body;
        const int64_t nb_body_reg = gemm_kernel_fp32_fma::config::MAX_N_REGS;
        const int64_t nb_tail_reg = div_up(nb_tail, gemm_kernel_fp32_fma::config::N_REG_ELTS);
        const int64_t nb_tail_mask = nb_tail % gemm_kernel_fp32_fma::config::N_REG_ELTS;
        const int64_t nb_tail_need_mask = nb_tail_mask ? 1 : 0;
        const int64_t padded_nb_tail = nb_tail_reg * gemm_kernel_fp32_fma::config::N_REG_ELTS;
        if (nb_tail_need_mask) ker.gen_mask(nb_tail_mask);
        for (int64_t kb = 0; kb < K; kb += k_blk) {
            const int64_t kb_eff = min(k_blk, K - kb);
            const bool is_first_k = kb == 0;
            const bool is_last_k = kb + kb_eff == K;
            ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::K_IDX) = kb_eff;
            for (int64_t mb = 0; mb < M; mb += m_blk) {
                const int64_t mb_eff = min(m_blk, M - mb);
                const int64_t mb_body = round(mb_eff, gemm_kernel_fp32_fma::config::MAX_M_BLK);
                const int64_t mb_tail = mb_eff - mb_body;
                const int64_t mb_body_reg = gemm_kernel_fp32_fma::config::MAX_M_REGS;
                const int64_t mb_tail_reg = div_up(mb_tail, gemm_kernel_fp32_fma::config::M_REG_ELTS);

                int64_t ker_flags = 0;
                if (is_first_k) {
                    if (beta != 0.0f) {
                        ker_flags |= gemm_kernel_fp32_fma::flag::LOAD_C;
                        ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_IDX) = beta;
                    }
                    if (typebias == gemm_v_type::SCALAR) ker_flags |= gemm_kernel_fp32_fma::flag::SCA_BIAS;
                    if (typebias == gemm_v_type::COL_VEC) ker_flags |= gemm_kernel_fp32_fma::flag::COL_BIAS;
                    if (typebias == gemm_v_type::ROW_VEC) ker_flags |= gemm_kernel_fp32_fma::flag::ROW_BIAS;
                    if (typesum == gemm_m_type::NOTRANS) ker_flags |= gemm_kernel_fp32_fma::flag::WITH_SUM;
                } else {
                    ker_flags |= gemm_kernel_fp32_fma::flag::LOAD_C;
                    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_IDX) = 1.0f;
                }
                if (is_last_k) {
                    if (post == gemm_post::RELU6) ker_flags |= gemm_kernel_fp32_fma::flag::RELU6;
                    if (post == gemm_post::RELU) ker_flags |= gemm_kernel_fp32_fma::flag::RELU;
                }
                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::FLAGS_IDX) = ker_flags;

                const float *base_a = A + (!is_trans_a ? mb * lda + kb : kb * lda + mb);
                float *base_c = C + mb * ldc + nb;
                const float *base_p = packedB + nb * K;
                float *base_q = packed_a;
                const float *base_next_p = base_p + K * gemm_kernel_fp32_fma::config::MAX_N_BLK;

                const float *base_sum = sum + mb * ldsum + nb;
                const float *base_bias = bias;
                if (typebias == gemm_v_type::COL_VEC) base_bias += mb;
                if (typebias == gemm_v_type::ROW_VEC) base_bias += nb;

                pack_a_func(base_a, mb_eff, kb_eff, lda, base_q);

                int64_t n = nb_body;
                while (n > 0) {
                    n -= gemm_kernel_fp32_fma::config::MAX_N_BLK;
                    const float *l_p = base_p + kb * gemm_kernel_fp32_fma::config::MAX_N_BLK;
                    const float *l_next_p = base_next_p + (n == 0 ? kb * padded_nb_tail : kb * gemm_kernel_fp32_fma::config::MAX_N_BLK);
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::BIAS_PTR_IDX) = base_bias;
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::SUM_PTR_IDX) = base_sum;
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::B_PTR_IDX) = l_p;
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::NEXT_B_PTR_IDX) = l_next_p;

                    if (mb_body) {
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q;
                        ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c;
                        ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_body;
                        ker.execute(0, mb_body_reg, nb_body_reg);
                    }

                    if (mb_tail) {
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q + mb_body * kb_eff;
                        ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c + mb_body * ldc;
                        ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_tail;
                        ker.execute(0, mb_tail_reg, nb_body_reg);
                    }

                    base_p = base_next_p, base_next_p += K * gemm_kernel_fp32_fma::config::MAX_N_BLK;
                    base_c += gemm_kernel_fp32_fma::config::MAX_N_BLK;
                    base_sum += gemm_kernel_fp32_fma::config::MAX_N_BLK;
                    if (typebias == gemm_v_type::ROW_VEC) base_bias += gemm_kernel_fp32_fma::config::MAX_N_BLK;
                }
                if (nb_tail) {
                    const float *l_p = base_p + kb * padded_nb_tail;
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::BIAS_PTR_IDX) = base_bias;
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::SUM_PTR_IDX) = base_sum;
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::B_PTR_IDX) = l_p;

                    if (mb_body) {
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q;
                        ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c;
                        ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_body;
                        ker.execute(nb_tail_need_mask, mb_body_reg, nb_tail_reg);
                    }

                    if (mb_tail) {
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q + mb_body * kb_eff;
                        ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c + mb_body * ldc;
                        ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_tail;
                        ker.execute(nb_tail_need_mask, mb_tail_reg, nb_tail_reg);
                    }
                }
            }
        }
    }

    ppl::common::AlignedFree(temp_buffer);
    return ppl::common::RC_SUCCESS;
}

// Row-major impl
ppl::common::RetCode gemm_operation_fp32_fma(
    const float *A,
    const float *B,
    const float *bias,
    const float *sum,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldsum,
    const float alpha,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    const gemm_post_t post,
    const opt_flag_t flags,
    float *C)
{
    if (typesum != gemm_m_type::EMPTY && typesum != gemm_m_type::NOTRANS) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const bool apply_alpha = alpha != 0.0f && typeA != gemm_m_type::EMPTY && typeB != gemm_m_type::EMPTY;
    const bool apply_betas = beta != 0.0f || (beta_bias != 0.0f && typebias != gemm_v_type::EMPTY) || (beta_sum != 0.0f && typesum != gemm_m_type::EMPTY);

    if (!apply_alpha && !apply_betas) {
        for (int64_t m = 0; m < M; ++m) {
            memset32_avx(C + m * ldc, 0, N);
        }
        return ppl::common::RC_SUCCESS;
    }

    if (K == 0) {
        auto apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::EMPTY, gemm_v_type::EMPTY>;
        if (typesum == gemm_m_type::NOTRANS) {
            if (typebias == gemm_v_type::EMPTY) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::NOTRANS, gemm_v_type::EMPTY>;
            if (typebias == gemm_v_type::SCALAR) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::NOTRANS, gemm_v_type::SCALAR>;
            if (typebias == gemm_v_type::COL_VEC) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::NOTRANS, gemm_v_type::COL_VEC>;
            if (typebias == gemm_v_type::ROW_VEC) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::NOTRANS, gemm_v_type::ROW_VEC>;
        } else {
            if (typebias == gemm_v_type::SCALAR) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::EMPTY, gemm_v_type::SCALAR>;
            if (typebias == gemm_v_type::COL_VEC) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::EMPTY, gemm_v_type::COL_VEC>;
            if (typebias == gemm_v_type::ROW_VEC) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::EMPTY, gemm_v_type::ROW_VEC>;
        }
        apply_betas_func(bias, sum, M, N, ldc, ldsum, beta, beta_bias, beta_sum, C);
    }

    if (typeA == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typeB == gemm_m_type::PACKED) {
        return gemm_packed_b_operation_fp32_fma(
            A, B, bias, sum,
            typeA, typebias, typesum,
            M, N, K, lda, ldc, ldsum,
            alpha, beta, beta_bias, beta_sum,
            post, flags, C);
    }

    const bool is_trans_a = typeA == gemm_m_type::TRANS;
    const bool is_trans_b = typeB == gemm_m_type::TRANS;

    // blocking
    int64_t k_blk = K;
    int64_t k_blk_max = (flags & opt_flag::large_l2) ? K_L2_BLK_MAX_LARGE : K_L2_BLK_MAX_SMALL;
    if (k_blk >= 2 * k_blk_max) k_blk = k_blk_max;
    else if (k_blk >= 1.5 * k_blk_max) k_blk = div_up(k_blk, 2);

    const int64_t n_blk = round_up(min(max(gemm_kernel_fp32_fma::config::MAX_N_BLK, N), N_L3_BLK_MAX), gemm_kernel_fp32_fma::config::MAX_N_BLK);
    const int64_t m_blk = round_up(min(max(gemm_kernel_fp32_fma::config::MAX_M_BLK, M), M_L3_BLK_MAX), gemm_kernel_fp32_fma::config::MAX_M_BLK);
    const bool use_sliding_packed_b = m_blk < M; // no need to save packed_b if only one m_blk
    const int64_t n_packed_b_blk = use_sliding_packed_b ? n_blk : gemm_kernel_fp32_fma::config::MAX_N_BLK;

    const int64_t packed_a_bytes = k_blk * m_blk * sizeof(float) + PPL_X86_PAGE_BYTES();
    const int64_t packed_b_bytes = (k_blk * n_packed_b_blk + gemm_kernel_fp32_fma::config::MAX_N_BLK) * sizeof(float) + PPL_X86_PAGE_BYTES();
    uint8_t *temp_buffer = (uint8_t*)ppl::common::AlignedAlloc(packed_a_bytes + packed_b_bytes, PPL_X86_CACHELINE_BYTES());
    float *packed_b = (float*)round_up((uintptr_t)temp_buffer, PPL_X86_PAGE_BYTES());
    float *packed_a = (float*)round_up((uintptr_t)(temp_buffer + packed_b_bytes), PPL_X86_PAGE_BYTES());

    static const gemm_fp32_fma_pack_b_func_t pack_b_body_func[2] = {
        gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::MAX_N_BLK, gemm_kernel_fp32_fma::config::MAX_N_BLK>,
        gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::MAX_N_BLK, gemm_kernel_fp32_fma::config::MAX_N_BLK>,
    };

    static const gemm_fp32_fma_pack_b_func_t pack_b_tail_func[2][4] = {
        {
            nullptr,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 1, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 2, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 3, 0>,
        },
        {
            nullptr,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 1, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 2, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 3, 0>,
        },
    };

    const auto pack_a_func = is_trans_a ? gemm_pack_a_m4_operation_fp32_avx<gemm_m_type::TRANS> : gemm_pack_a_m4_operation_fp32_avx<gemm_m_type::NOTRANS>;

    int64_t kernel_param[gemm_kernel_fp32_fma::param_def::LENGTH];
    array_param_helper ker_p(kernel_param);
    gemm_kernel_fp32_fma ker(kernel_param);
    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::ALPHA_IDX) = alpha;
    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_BIAS_IDX) = beta_bias;
    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_SUM_IDX) = beta_sum;
    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::LDC_IDX) = ldc;
    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::LDSUM_IDX) = ldsum;
    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::PRF_C_LDK_IDX) =
        (flags & opt_flag::large_c) ?
        gemm_kernel_fp32_fma::config::PRF_C_LDK_MEM :
        gemm_kernel_fp32_fma::config::PRF_C_LDK_L3;

    for (int64_t nb = 0; nb < N; nb += n_blk) {
        const int64_t nb_eff = min(n_blk, N - nb);
        const int64_t nb_body = round(nb_eff, gemm_kernel_fp32_fma::config::MAX_N_BLK);
        const int64_t nb_tail = nb_eff - nb_body;
        const int64_t nb_body_reg = gemm_kernel_fp32_fma::config::MAX_N_REGS;
        const int64_t nb_tail_reg = div_up(nb_tail, gemm_kernel_fp32_fma::config::N_REG_ELTS);
        const int64_t nb_tail_mask = nb_tail % gemm_kernel_fp32_fma::config::N_REG_ELTS;
        const int64_t nb_tail_need_mask = nb_tail_mask ? 1 : 0;
        if (nb_tail_need_mask) ker.gen_mask(nb_tail_mask);
        for (int64_t kb = 0; kb < K; kb += k_blk) {
            const int64_t kb_eff = min(k_blk, K - kb);
            const bool is_first_k = kb == 0;
            const bool is_last_k = kb + kb_eff == K;
            ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::K_IDX) = kb_eff;
            for (int64_t mb = 0; mb < M; mb += m_blk) {
                const int64_t mb_eff = min(m_blk, M - mb);
                const int64_t mb_body = round(mb_eff, gemm_kernel_fp32_fma::config::MAX_M_BLK);
                const int64_t mb_tail = mb_eff - mb_body;
                const int64_t mb_body_reg = gemm_kernel_fp32_fma::config::MAX_M_REGS;
                const int64_t mb_tail_reg = div_up(mb_tail, gemm_kernel_fp32_fma::config::M_REG_ELTS);

                int64_t ker_flags = 0;
                if (is_first_k) {
                    if (beta != 0.0f) {
                        ker_flags |= gemm_kernel_fp32_fma::flag::LOAD_C;
                        ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_IDX) = beta;
                    }
                    if (typebias == gemm_v_type::SCALAR) ker_flags |= gemm_kernel_fp32_fma::flag::SCA_BIAS;
                    if (typebias == gemm_v_type::COL_VEC) ker_flags |= gemm_kernel_fp32_fma::flag::COL_BIAS;
                    if (typebias == gemm_v_type::ROW_VEC) ker_flags |= gemm_kernel_fp32_fma::flag::ROW_BIAS;
                    if (typesum == gemm_m_type::NOTRANS) ker_flags |= gemm_kernel_fp32_fma::flag::WITH_SUM;
                } else {
                    ker_flags |= gemm_kernel_fp32_fma::flag::LOAD_C;
                    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_IDX) = 1.0f;
                }
                if (is_last_k) {
                    if (post == gemm_post::RELU6) ker_flags |= gemm_kernel_fp32_fma::flag::RELU6;
                    if (post == gemm_post::RELU) ker_flags |= gemm_kernel_fp32_fma::flag::RELU;
                }
                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::FLAGS_IDX) = ker_flags;

                const float *base_b = B + (!is_trans_b ? kb * ldb + nb : nb * ldb + kb);
                const float *base_a = A + (!is_trans_a ? mb * lda + kb : kb * lda + mb);
                float *base_c = C + mb * ldc + nb;
                float *base_p = packed_b;
                float *base_q = packed_a;
                float *base_next_p = base_p + kb_eff * gemm_kernel_fp32_fma::config::MAX_N_BLK;

                const float *base_sum = sum + mb * ldsum + nb;
                const float *base_bias = bias;
                if (typebias == gemm_v_type::COL_VEC) base_bias += mb;
                if (typebias == gemm_v_type::ROW_VEC) base_bias += nb;

                pack_a_func(base_a, mb_eff, kb_eff, lda, base_q);

                int64_t n = nb_body;
                while (n > 0) {
                    n -= gemm_kernel_fp32_fma::config::MAX_N_BLK;
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::BIAS_PTR_IDX) = base_bias;
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::SUM_PTR_IDX) = base_sum;

                    if (mb == 0) pack_b_body_func[is_trans_b](base_b, gemm_kernel_fp32_fma::config::MAX_N_BLK, kb_eff, ldb, base_p);
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::B_PTR_IDX) = base_p;
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::NEXT_B_PTR_IDX) = base_next_p;

                    if (mb_body) {
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q;
                        ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c;
                        ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_body;
                        ker.execute(0, mb_body_reg, nb_body_reg);
                    }

                    if (mb_tail) {
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q + mb_body * kb_eff;
                        ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c + mb_body * ldc;
                        ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_tail;
                        ker.execute(0, mb_tail_reg, nb_body_reg);
                    }

                    if (is_trans_b) base_b += gemm_kernel_fp32_fma::config::MAX_N_BLK * ldb;
                    else base_b += gemm_kernel_fp32_fma::config::MAX_N_BLK;
                    if (use_sliding_packed_b) base_p = base_next_p, base_next_p += kb_eff * gemm_kernel_fp32_fma::config::MAX_N_BLK;
                    base_c += gemm_kernel_fp32_fma::config::MAX_N_BLK;
                    base_sum += gemm_kernel_fp32_fma::config::MAX_N_BLK;
                    if (typebias == gemm_v_type::ROW_VEC) base_bias += gemm_kernel_fp32_fma::config::MAX_N_BLK;
                }
                if (nb_tail) {
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::BIAS_PTR_IDX) = base_bias;
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::SUM_PTR_IDX) = base_sum;

                    if (mb == 0) pack_b_tail_func[is_trans_b][nb_tail_reg](base_b, nb_tail, kb_eff, ldb, base_p);
                    ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::B_PTR_IDX) = base_p;

                    if (mb_body) {
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q;
                        ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c;
                        ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_body;
                        ker.execute(nb_tail_need_mask, mb_body_reg, nb_tail_reg);
                    }

                    if (mb_tail) {
                        ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q + mb_body * kb_eff;
                        ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c + mb_body * ldc;
                        ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_tail;
                        ker.execute(nb_tail_need_mask, mb_tail_reg, nb_tail_reg);
                    }
                }
            }
        }
    }

    ppl::common::AlignedFree(temp_buffer);
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode gemm_shared_packed_b_operation_fp32_fma(
    const float *A,
    const float *packedB,
    const float *bias,
    const float *sum,
    const gemm_m_type_t typeA,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldc,
    const int64_t ldsum,
    const float alpha,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    const gemm_post_t post,
    const opt_flag_t flags,
    float *C)
{
    const bool is_trans_a = typeA == gemm_m_type::TRANS;
    const int64_t m_blk = round_up(min(max(gemm_kernel_fp32_fma::config::MAX_M_BLK, M), M_L3_BLK_MAX), gemm_kernel_fp32_fma::config::MAX_M_BLK);

    const int64_t packed_a_bytes = K * m_blk * sizeof(float) + PPL_X86_PAGE_BYTES();
    uint8_t *temp_buffer = (uint8_t*)ppl::common::AlignedAlloc(packed_a_bytes, PPL_X86_CACHELINE_BYTES());
    float *packed_a = (float*)round_up((uintptr_t)temp_buffer, PPL_X86_PAGE_BYTES());

    const auto pack_a_func = is_trans_a ? gemm_pack_a_m4_operation_fp32_avx<gemm_m_type::TRANS> : gemm_pack_a_m4_operation_fp32_avx<gemm_m_type::NOTRANS>;

    int64_t kernel_param[gemm_kernel_fp32_fma::param_def::LENGTH];
    array_param_helper ker_p(kernel_param);
    gemm_kernel_fp32_fma ker(kernel_param);
    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::ALPHA_IDX) = alpha;
    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_BIAS_IDX) = beta_bias;
    ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_SUM_IDX) = beta_sum;
    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::LDC_IDX) = ldc;
    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::LDSUM_IDX) = ldsum;
    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::PRF_C_LDK_IDX) =
        (flags & opt_flag::large_c) ?
        gemm_kernel_fp32_fma::config::PRF_C_LDK_MEM :
        gemm_kernel_fp32_fma::config::PRF_C_LDK_L3;

    const int64_t n_body = round(N, gemm_kernel_fp32_fma::config::MAX_N_BLK);
    const int64_t n_tail = N - n_body;
    const int64_t n_body_reg = gemm_kernel_fp32_fma::config::MAX_N_REGS;
    const int64_t n_tail_reg = div_up(n_tail, gemm_kernel_fp32_fma::config::N_REG_ELTS);
    const int64_t n_tail_mask = n_tail % gemm_kernel_fp32_fma::config::N_REG_ELTS;
    const int64_t n_tail_need_mask = n_tail_mask ? 1 : 0;
    if (n_tail_need_mask) ker.gen_mask(n_tail_mask);
    ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::K_IDX) = K;
    for (int64_t mb = 0; mb < M; mb += m_blk) {
        const int64_t mb_eff = min(m_blk, M - mb);
        const int64_t mb_body = round(mb_eff, gemm_kernel_fp32_fma::config::MAX_M_BLK);
        const int64_t mb_tail = mb_eff - mb_body;
        const int64_t mb_body_reg = gemm_kernel_fp32_fma::config::MAX_M_REGS;
        const int64_t mb_tail_reg = div_up(mb_tail, gemm_kernel_fp32_fma::config::M_REG_ELTS);

        int64_t ker_flags = 0; // control beta outside
        if (beta != 0.0f) {
            ker_flags |= gemm_kernel_fp32_fma::flag::LOAD_C;
            ker_p.pick<float>(gemm_kernel_fp32_fma::param_def::BETA_IDX) = beta;
        }
        if (typebias == gemm_v_type::SCALAR) ker_flags |= gemm_kernel_fp32_fma::flag::SCA_BIAS;
        if (typebias == gemm_v_type::COL_VEC) ker_flags |= gemm_kernel_fp32_fma::flag::COL_BIAS;
        if (typebias == gemm_v_type::ROW_VEC) ker_flags |= gemm_kernel_fp32_fma::flag::ROW_BIAS;
        if (typesum == gemm_m_type::NOTRANS) ker_flags |= gemm_kernel_fp32_fma::flag::WITH_SUM;
        if (post == gemm_post::RELU6) ker_flags |= gemm_kernel_fp32_fma::flag::RELU6;
        if (post == gemm_post::RELU) ker_flags |= gemm_kernel_fp32_fma::flag::RELU;
        ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::FLAGS_IDX) = ker_flags;

        const float *base_a = A + (!is_trans_a ? mb * lda : mb);
        float *base_c = C + mb * ldc;
        const float *base_p = packedB;
        float *base_q = packed_a;
        const float *base_next_p = base_p + K * gemm_kernel_fp32_fma::config::MAX_N_BLK;

        const float *base_sum = sum + mb * ldsum;
        const float *base_bias = bias;
        if (typebias == gemm_v_type::COL_VEC) base_bias += mb;

        pack_a_func(base_a, mb_eff, K, lda, base_q);

        int64_t n = n_body;
        while (n > 0) {
            n -= gemm_kernel_fp32_fma::config::MAX_N_BLK;
            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::BIAS_PTR_IDX) = base_bias;
            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::SUM_PTR_IDX) = base_sum;
            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::B_PTR_IDX) = base_p;
            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::NEXT_B_PTR_IDX) = base_next_p;

            if (mb_body) {
                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q;
                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c;
                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_body;
                ker.execute(0, mb_body_reg, n_body_reg);
            }

            if (mb_tail) {
                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q + mb_body * K;
                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c + mb_body * ldc;
                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_tail;
                ker.execute(0, mb_tail_reg, n_body_reg);
            }

            base_c += gemm_kernel_fp32_fma::config::MAX_N_BLK;
            base_p = base_next_p, base_next_p += K * gemm_kernel_fp32_fma::config::MAX_N_BLK;
            base_sum += gemm_kernel_fp32_fma::config::MAX_N_BLK;
            if (typebias == gemm_v_type::ROW_VEC) base_bias += gemm_kernel_fp32_fma::config::MAX_N_BLK;
        }
        if (n_tail) {
            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::BIAS_PTR_IDX) = base_bias;
            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::SUM_PTR_IDX) = base_sum;
            ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::B_PTR_IDX) = base_p;

            if (mb_body) {
                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q;
                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c;
                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_body;
                ker.execute(n_tail_need_mask, mb_body_reg, n_tail_reg);
            }

            if (mb_tail) {
                ker_p.pick<const float*>(gemm_kernel_fp32_fma::param_def::A_PTR_IDX) = base_q + mb_body * K;
                ker_p.pick<float*>(gemm_kernel_fp32_fma::param_def::C_PTR_IDX) = base_c + mb_body * ldc;
                ker_p.pick<int64_t>(gemm_kernel_fp32_fma::param_def::M_IDX) = mb_tail;
                ker.execute(n_tail_need_mask, mb_tail_reg, n_tail_reg);
            }
        }
    }

    ppl::common::AlignedFree(temp_buffer);
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode gemm_shared_pack_b_threaded_operation_fp32_fma(
    const float *A,
    const float *B,
    const float *bias,
    const float *sum,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldsum,
    const float alpha,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    const gemm_post_t post,
    const opt_flag_t flags,
    const int64_t thread_id,
    const int64_t num_threads,
    uint8_t **shared_packed_b,
    float *C)
{
    if (thread_id >= num_threads || thread_id < 0) {
        return ppl::common::RC_INVALID_VALUE;
    }

    if (typesum != gemm_m_type::EMPTY && typesum != gemm_m_type::NOTRANS) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const bool apply_alpha = alpha != 0.0f && typeA != gemm_m_type::EMPTY && typeB != gemm_m_type::EMPTY;
    const bool apply_betas = beta != 0.0f || (beta_bias != 0.0f && typebias != gemm_v_type::EMPTY) || (beta_sum != 0.0f && typesum != gemm_m_type::EMPTY);

    if (!apply_alpha && !apply_betas) {
        for (int64_t m = 0; m < M; ++m) {
            memset32_avx(C + m * ldc, 0, N);
        }
        return ppl::common::RC_SUCCESS;
    }

    if (K == 0) {
        auto apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::EMPTY, gemm_v_type::EMPTY>;
        if (typesum == gemm_m_type::NOTRANS) {
            if (typebias == gemm_v_type::EMPTY) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::NOTRANS, gemm_v_type::EMPTY>;
            if (typebias == gemm_v_type::SCALAR) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::NOTRANS, gemm_v_type::SCALAR>;
            if (typebias == gemm_v_type::COL_VEC) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::NOTRANS, gemm_v_type::COL_VEC>;
            if (typebias == gemm_v_type::ROW_VEC) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::NOTRANS, gemm_v_type::ROW_VEC>;
        } else {
            if (typebias == gemm_v_type::SCALAR) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::EMPTY, gemm_v_type::SCALAR>;
            if (typebias == gemm_v_type::COL_VEC) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::EMPTY, gemm_v_type::COL_VEC>;
            if (typebias == gemm_v_type::ROW_VEC) apply_betas_func = gemm_fp32_apply_betas_avx<gemm_m_type::EMPTY, gemm_v_type::ROW_VEC>;
        }
        apply_betas_func(bias, sum, M, N, ldc, ldsum, beta, beta_bias, beta_sum, C);
    }

    if (typeA == gemm_m_type::PACKED || typeB == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const bool is_trans_a = typeA == gemm_m_type::TRANS;
    const bool is_trans_b = typeB == gemm_m_type::TRANS;

    // blocking
    int64_t k_blk = K;
    int64_t k_blk_max = (flags & opt_flag::large_l2) ? K_L2_BLK_MAX_LARGE : K_L2_BLK_MAX_SMALL;
    if (flags & opt_flag::large_c) k_blk_max *= 2; // avoid write c too many times
    if (k_blk >= 2 * k_blk_max) k_blk = k_blk_max;
    else if (k_blk >= 1.5 * k_blk_max) k_blk = div_up(k_blk, 2);

    int64_t n_blk = round_up(min(max(gemm_kernel_fp32_fma::config::MAX_N_BLK, N), N_L3_BLK_MAX), gemm_kernel_fp32_fma::config::MAX_N_BLK);
    n_blk = round_up(min(n_blk * num_threads, N), gemm_kernel_fp32_fma::config::MAX_N_BLK);

    const int64_t packed_b_bytes = (k_blk * n_blk + gemm_kernel_fp32_fma::config::MAX_N_BLK) * sizeof(float) + PPL_X86_PAGE_BYTES();

    if (thread_id == 0) { // main thread malloc shared buffer
        *shared_packed_b = (uint8_t*)ppl::common::AlignedAlloc(packed_b_bytes, PPL_X86_CACHELINE_BYTES());
    }
    PRAGMA_OMP_BARRIER() // wait for malloc

    float *packed_b = (float*)round_up((uintptr_t)*shared_packed_b, PPL_X86_PAGE_BYTES());


    static const gemm_fp32_fma_pack_b_func_t pack_b_body_func[2] = {
        gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::MAX_N_BLK, gemm_kernel_fp32_fma::config::MAX_N_BLK>,
        gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::MAX_N_BLK, gemm_kernel_fp32_fma::config::MAX_N_BLK>,
    };

    static const gemm_fp32_fma_pack_b_func_t pack_b_tail_func[2][4] = {
        {
            nullptr,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 1, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 2, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 3, 0>,
        },
        {
            nullptr,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 1, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 2, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 3, 0>,
        },
    };

    ppl::common::RetCode ret = ppl::common::RC_SUCCESS;

    for (int64_t kb = 0; kb < K; kb += k_blk) {
        const int64_t kb_eff = min(k_blk, K - kb);
        const bool is_first_k = kb == 0;
        const bool is_last_k = kb + kb_eff == K;
        for (int64_t nb = 0; nb < N; nb += n_blk) {
            const int64_t nb_eff = min(n_blk, N - nb);
            const gemm_v_type_t l_typebias = is_first_k ? typebias : gemm_v_type::EMPTY;
            const gemm_m_type_t l_typesum = is_first_k ? typesum : gemm_m_type::EMPTY;
            const float l_beta = is_first_k ? beta : 1.0f;
            const gemm_post_t l_post = is_last_k ? post : gemm_post::NONE;

            const float *base_b = B + (!is_trans_b ? kb * ldb + nb : nb * ldb + kb);
            const float *base_a = A + (!is_trans_a ? kb : kb * lda);
            float *base_c = C + nb;
            float *base_p = packed_b;

            const float *base_sum = sum + nb;
            const float *base_bias = bias;
            if (typebias == gemm_v_type::ROW_VEC) base_bias += nb;

            // parallel packing
            int64_t nb_thr_pack_b, nb_thr_pack_b_eff;
            parallel_task_distribution_1d(
                thread_id, num_threads,
                div_up(nb_eff, gemm_kernel_fp32_fma::config::MAX_N_BLK),
                &nb_thr_pack_b, &nb_thr_pack_b_eff);
            nb_thr_pack_b *= gemm_kernel_fp32_fma::config::MAX_N_BLK;
            nb_thr_pack_b_eff = max<int64_t>(min(nb_thr_pack_b_eff * gemm_kernel_fp32_fma::config::MAX_N_BLK, nb_eff - nb_thr_pack_b), 0);
            float *thr_packed_b = packed_b + nb_thr_pack_b * kb_eff;
            const float *thr_base_b = base_b + (!is_trans_b ? nb_thr_pack_b : nb_thr_pack_b * ldb);
            const int64_t nb_thr_pack_b_body = round(nb_thr_pack_b_eff, gemm_kernel_fp32_fma::config::MAX_N_BLK);
            const int64_t nb_thr_pack_b_tail = nb_thr_pack_b_eff - nb_thr_pack_b_body;
            const int64_t nb_thr_pack_b_treg = div_up(nb_thr_pack_b_tail, gemm_kernel_fp32_fma::config::N_REG_ELTS);
            int64_t n = nb_thr_pack_b_body;
            while (n > 0) {
                n -= gemm_kernel_fp32_fma::config::MAX_N_BLK;
                pack_b_body_func[is_trans_b](thr_base_b, gemm_kernel_fp32_fma::config::MAX_N_BLK, kb_eff, ldb, thr_packed_b);

                if (is_trans_b) thr_base_b += gemm_kernel_fp32_fma::config::MAX_N_BLK * ldb;
                else thr_base_b += gemm_kernel_fp32_fma::config::MAX_N_BLK;
                thr_packed_b += gemm_kernel_fp32_fma::config::MAX_N_BLK * kb_eff;
            }
            if (nb_thr_pack_b_tail) {
                pack_b_tail_func[is_trans_b][nb_thr_pack_b_treg](thr_base_b, nb_thr_pack_b_tail, kb_eff, ldb, thr_packed_b);
            }
            PRAGMA_OMP_BARRIER() // wait for pack b sync

            auto l_ret = gemm_shared_packed_b_operation_fp32_fma(
                base_a, base_p, base_bias, base_sum, typeA, l_typebias, l_typesum,
                M, nb_eff, kb_eff, lda, ldc, ldsum, alpha,
                l_beta, beta_bias, beta_sum, l_post, flags, base_c);
            if (l_ret != ppl::common::RC_SUCCESS) ret = l_ret;
            PRAGMA_OMP_BARRIER() // wait for compute sync
        }
    }

    if (thread_id == 0) ppl::common::AlignedFree(*shared_packed_b);
    return ret;
}

uint64_t gemm_fp32_fma_get_packed_b_bytes(
    const int64_t N,
    const int64_t K)
{
    return sizeof(float) * K * round_up(N, gemm_kernel_fp32_fma::config::N_REG_ELTS);
}

ppl::common::RetCode gemm_fp32_fma_pack_b(
    const float *B,
    const gemm_m_type_t typeB,
    const int64_t N,
    const int64_t K,
    const int64_t ldb,
    float *packedB)
{
    if (N <= 0 || K <= 0) {
        return ppl::common::RC_SUCCESS;
    }

    const bool is_trans_b = typeB == gemm_m_type::TRANS;

    // blocking
    const int64_t k_blk = K_L2_BLK_MAX_LARGE / 2;
    const int64_t n_blk = gemm_kernel_fp32_fma::config::MAX_N_BLK;

    const int64_t k_task = div_up(K, k_blk);
    const int64_t n_task = div_up(N, n_blk);

    static const gemm_fp32_fma_pack_b_func_t pack_b_body_func[2] = {
        gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::MAX_N_BLK, gemm_kernel_fp32_fma::config::MAX_N_BLK>,
        gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::MAX_N_BLK, gemm_kernel_fp32_fma::config::MAX_N_BLK>,
    };

    static const gemm_fp32_fma_pack_b_func_t pack_b_tail_func[2][4] = {
        {
            nullptr,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 1, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 2, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::NOTRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 3, 0>,
        },
        {
            nullptr,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 1, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 2, 0>,
            gemm_pack_b_operation_fp32_avx<gemm_m_type::TRANS, gemm_kernel_fp32_fma::config::N_REG_ELTS * 3, 0>,
        },
    };

    // packedB: (N/n_blk, K, n_blk)
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t t = 0; t < k_task * n_task; ++t) {
        const int64_t nt = t / k_task;
        const int64_t kt = t % k_task;

        const int64_t nb = nt * n_blk;
        const int64_t kb = kt * k_blk;

        const int64_t nb_eff = min(N - nb, n_blk);
        const int64_t kb_eff = min(K - kb, k_blk);

        const int64_t n_regs = div_up(nb_eff, gemm_kernel_fp32_fma::config::N_REG_ELTS);
        const int64_t padded_nb_eff = n_regs * gemm_kernel_fp32_fma::config::N_REG_ELTS;

        const float *base_b = B + (is_trans_b ? nb * ldb + kb : kb * ldb + nb);
        float *base_p = packedB + kb * padded_nb_eff + nb * K;

        if (n_regs == gemm_kernel_fp32_fma::config::MAX_N_REGS) {
            pack_b_body_func[is_trans_b](base_b, nb_eff, kb_eff, ldb, base_p);
        } else {
            pack_b_tail_func[is_trans_b][n_regs](base_b, nb_eff, kb_eff, ldb, base_p);
        }
    }

    return ppl::common::RC_SUCCESS;
}

// Row-major impl
ppl::common::RetCode gemm_fp32_fma(
    const float *A,
    const float *B,
    const float *bias,
    const float *sum,
    const gemm_m_type_t typeA,
    const gemm_m_type_t typeB,
    const gemm_v_type_t typebias,
    const gemm_m_type_t typesum,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldsum,
    const float alpha,
    const float beta,
    const float beta_bias,
    const float beta_sum,
    const gemm_post_t post,
    float *C)
{
    if (M <= 0 || N <= 0 || K < 0) {
        return ppl::common::RC_SUCCESS;
    }

    if (typeA == gemm_m_type::PACKED) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (typesum != gemm_m_type::EMPTY && typesum != gemm_m_type::NOTRANS) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const bool is_packed_b = typeB == gemm_m_type::PACKED;
    const int64_t n_div = is_packed_b ? gemm_kernel_fp32_fma::config::MAX_N_BLK : gemm_kernel_fp32_fma::config::N_REG_ELTS;
    
    if (!is_packed_b) {
        if ((typeA == gemm_m_type::NOTRANS || lda == 1) && M == 1) {
            return gemv_fp32_fma(
                A, B, bias, sum,
                gemm_v_type::ROW_VEC, typeB, typebias, typesum,
                N, K, ldb,
                alpha, beta, beta_bias, beta_sum, post, C);
        }

        if (N == 1 && ((typeB == gemm_m_type::NOTRANS && ldb == 1) || (typeB == gemm_m_type::TRANS && ldb == K)) && ldc == 1) {
            auto l_typeA = typeA == gemm_m_type::NOTRANS ? gemm_m_type::TRANS : gemm_m_type::NOTRANS;
            auto l_typebias = typebias;
            if (typebias == gemm_v_type::ROW_VEC) l_typebias = gemm_v_type::COL_VEC;
            if (typebias == gemm_v_type::COL_VEC) l_typebias = gemm_v_type::ROW_VEC;
            return gemv_fp32_fma(
                B, A, bias, sum,
                gemm_v_type::ROW_VEC, l_typeA, l_typebias, typesum,
                M, K, lda,
                alpha, beta, beta_bias, beta_sum, post, C);
        }
    }

    const int64_t num_threads = PPL_OMP_MAX_THREADS();
    const uint64_t l3_size = ppl::common::GetCpuCacheL3() == 0 ? (num_threads * 2048 * 1024) : ppl::common::GetCpuCacheL3();
    const uint64_t l2_size = ppl::common::GetCpuCacheL2() == 0 ? (256 * 1024) : ppl::common::GetCpuCacheL2();
    opt_flag_t flags = 0;
    if (M * N * sizeof(float) > l3_size * 2) flags |= opt_flag::large_c;
    if (l2_size >= 512 * 1024) flags |= opt_flag::large_l2;

    if (num_threads == 1) {
        return gemm_operation_fp32_fma(
            A, B, bias, sum,
            typeA, typeB, typebias, typesum,
            M, N, K, lda, ldb ,ldc ,ldsum,
            alpha, beta, beta_bias, beta_sum,
            post, flags, C);
    } else {
        flags |= opt_flag::multi_thread;
    }

    int64_t m_threads = 0;
    int64_t n_threads = 0;
    bool use_shared_packed_b = false;

    // blocking
    if (N >= N_THR_BLK_MIN * 2) {
        if (M >= num_threads * M_L3_BLK_MAX / 4) {
            use_shared_packed_b = is_packed_b ? false : true;
            m_threads = min(div_up(M, gemm_kernel_fp32_fma::config::MAX_M_BLK), num_threads);
            n_threads = 1;
        } else {
            n_threads = min<int64_t>(div_up(N, N_THR_BLK_MIN * 2), 4);
            m_threads = num_threads / n_threads;
            if (M < m_threads * M_L3_BLK_MAX / 2) { // small M
                n_threads = min<int64_t>(max<int64_t>(N / N_THR_BLK_MIN, 1), 4);
            }
            if (M <= M_L3_BLK_MAX && N >= num_threads * N_THR_BLK_MIN && N < num_threads * N_L3_BLK_MAX) { // very small M
                n_threads = min<int64_t>(max<int64_t>(N / N_THR_BLK_MIN, 1), num_threads);
            }
            if (M <= M_L3_BLK_MAX / 4 && N >= num_threads * N_THR_BLK_MIN) { // just stick!
                n_threads = min<int64_t>(max<int64_t>(N / N_THR_BLK_MIN, 1), num_threads);
            }
            if (M <= num_threads * gemm_kernel_fp32_fma::config::MAX_M_BLK) {
                n_threads = min<int64_t>(div_up(N, n_div), num_threads);
            }
            m_threads = num_threads / n_threads;
            m_threads = min(div_up(M, gemm_kernel_fp32_fma::config::MAX_M_BLK), m_threads);
        }
    } else {
        if (N > M && // M too small and N is enough
            (M < num_threads * gemm_kernel_fp32_fma::config::MAX_M_BLK ||
            N >= num_threads * n_div)) {
            m_threads = 1;
            n_threads = min<int64_t>(div_up(N, n_div), num_threads);
        } else {
            m_threads = min(div_up(M, gemm_kernel_fp32_fma::config::MAX_M_BLK), num_threads);
            n_threads = 1;
        }
    }

    std::vector<ppl::common::RetCode> thread_ret(m_threads * n_threads, ppl::common::RC_SUCCESS);
    uint8_t *shared_packed_b = nullptr;
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t t = 0; t < m_threads * n_threads; ++t) {
        const int64_t mt = t % m_threads;
        const int64_t nt = t / m_threads;

        int64_t mb, nb, mb_eff, nb_eff;
        parallel_task_distribution_1d(mt, m_threads, M, &mb, &mb_eff);
        parallel_task_distribution_1d(nt, n_threads, div_up(N, n_div), &nb, &nb_eff);
        nb *= n_div;
        nb_eff = max<int64_t>(min(nb_eff * n_div, N - nb), 0);

        const float *lA = A;
        if (typeA == gemm_m_type::NOTRANS) {
            lA += mb * lda;
        } else {
            lA += mb;
        }

        const float *lB = B;
        if (typeB == gemm_m_type::PACKED) {
            lB += nb * K;
        } else if (typeB == gemm_m_type::NOTRANS) {
            lB += nb;
        } else {
            lB += nb * ldb;
        }

        const float *lbias = bias;
        if (typebias == gemm_v_type::COL_VEC) {
            lbias += mb;
        } else if (typebias == gemm_v_type::ROW_VEC) {
            lbias += nb;
        }

        const float *lsum = sum;
        if (typesum == gemm_m_type::NOTRANS) {
            lsum += mb * ldsum + nb;
        }

        float *lC = C + mb * ldc + nb;

        if (use_shared_packed_b) {
            thread_ret[t] = gemm_shared_pack_b_threaded_operation_fp32_fma(
                lA, lB, lbias, lsum,
                typeA, typeB, typebias, typesum,
                mb_eff, nb_eff, K, lda, ldb ,ldc, ldsum,
                alpha, beta, beta_bias, beta_sum,
                post, flags, mt, m_threads, &shared_packed_b, lC);
        } else {
            thread_ret[t] = gemm_operation_fp32_fma(
                lA, lB, lbias, lsum,
                typeA, typeB, typebias, typesum,
                mb_eff, nb_eff, K, lda, ldb ,ldc, ldsum,
                alpha, beta, beta_bias, beta_sum,
                post, flags, lC);
        }
    }
    for (int64_t t = 0; t < m_threads * n_threads; ++t) {
        if (thread_ret[t] != ppl::common::RC_SUCCESS) return thread_ret[t];
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
