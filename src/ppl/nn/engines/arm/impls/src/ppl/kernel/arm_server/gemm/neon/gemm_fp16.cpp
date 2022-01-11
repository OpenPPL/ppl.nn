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

#ifdef PPL_USE_ARM_SERVER_FP16

#include <arm_neon.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

#include "ppl/common/arm/sysinfo.h"
#include "ppl/kernel/arm_server/common/internal_include.h"
#include "../../fc/neon/fp16/hgemm_kernel.h"

#define VBLOCK() 8
#define PACK_VBLOCK(val) ((val + 7) & (~7))

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

size_t ppl_arm_server_kernel_fp16_gemm_get_buffer_size(
    const int64_t sgemm_m1,
    const int64_t sgemm_n1)
{
    return PPL_OMP_MAX_THREADS() * (sgemm_m1 * sgemm_n1 + 128) * sizeof(__fp16);
}

static void ppl_arm_server_kernel_fp16_gemm_alphax(
    const __fp16 *a,
    const __fp16 *b,
    const __fp16 *c,
    __fp16 *y,
    __fp16 *tmp_buffer,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldy,
    const __fp16 alpha,
    const __fp16 beta,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_m3,
    const int64_t sgemm_k3)
{
PRAGMA_OMP_PARALLEL()
{
    const int64_t k_sgemm_m0 = 15;
    const int64_t k_sgemm_n0 = VBLOCK();

    __fp16 *local_y_buffer = tmp_buffer + PPL_OMP_THREAD_ID() * (sgemm_m1 * sgemm_n1 + 128);

    for (int64_t i_l3 = 0; i_l3 < M; i_l3 += sgemm_m3) {
        for (int64_t p_l3 = 0; p_l3 < K; p_l3 += K) {
            const int64_t m_l3 = std::min((M - i_l3), sgemm_m3);
            const int64_t k_l3 = std::min((K - p_l3), sgemm_k3);


            PRAGMA_OMP_FOR_COLLAPSE(2)
            for (int64_t i_l1 = 0; i_l1 < m_l3; i_l1 += sgemm_m1) {
                for (int64_t j_l1 = 0; j_l1 < N; j_l1 += sgemm_n1) {
                    const int64_t m_l1 = std::min((m_l3-i_l1), sgemm_m1);
                    const int64_t n_l1 = std::min((N-j_l1), sgemm_n1);
                    const int64_t n_l1_pack = PACK_VBLOCK(n_l1);
                    
                    int64_t opt_sgemm_m0 = k_sgemm_m0;
                    int64_t opt_sgemm_n0 = k_sgemm_n0;
                    if (m_l1 < 15 && m_l1 >= 10) {
                        opt_sgemm_m0 = 10;
                        opt_sgemm_n0 = VBLOCK() * 2;
                    }
                    else if (m_l1 < 10 && m_l1 >= 7) {
                        opt_sgemm_m0 = 7;
                        opt_sgemm_n0 = VBLOCK() * 3;
                    }
                    else if (m_l1 < 7) {
                        opt_sgemm_m0 = 5;
                        opt_sgemm_n0 = VBLOCK() * 4;
                    }
                    const int64_t i_local = i_l3 + i_l1;
                    const int64_t ldy_local = n_l1_pack;
                    __fp16 * y_base_local = local_y_buffer;
                    memset(y_base_local, 0, m_l1 * n_l1_pack * sizeof(__fp16));
                    
                    const int64_t m_l1_align = (m_l1 / opt_sgemm_m0) * opt_sgemm_m0;

                    for (int64_t p_l1 = 0; p_l1 < k_l3; p_l1 += sgemm_k1) {
                        const int64_t p = p_l3 + p_l1;
                        const int64_t k_l1 = std::min((k_l3-p_l1), sgemm_k1);
                        for (int64_t i_l0 = 0; i_l0 < m_l1_align; i_l0 += opt_sgemm_m0) {
                            const int64_t i = i_local + i_l0;
                            for (int64_t j_l0 = 0; j_l0 < n_l1_pack; j_l0 += opt_sgemm_n0) {
                                const int64_t j = j_l0 + j_l1;
                                const int64_t n_l0 = std::min((n_l1_pack-j_l0), opt_sgemm_n0);

                                hgemm_kernel_func_table[4][n_l0/VBLOCK()-1](
                                    a + i * lda + p,
                                    b + p * ldb + j,
                                    nullptr,
                                    y_base_local + i_l0 * ldy_local + j_l0,
                                    opt_sgemm_m0,
                                    n_l0,
                                    k_l1,
                                    lda,
                                    VBLOCK(),
                                    ldb,
                                    VBLOCK(),
                                    ldy_local,
                                    true,
                                    0
                                );
                            }
                        }
                        const int64_t m_l1_tail = m_l1 - m_l1_align;
                        int64_t i_l0 = m_l1_align;
                        if (m_l1_tail) {
                            if (m_l1_tail & 8) {
                                for (int64_t j_l0 = 0; j_l0 < n_l1_pack; j_l0 += 8) {
                                    const int64_t j = j_l0 + j_l1;
                                    const int64_t n_l0 = std::min((n_l1_pack-j_l0), (int64_t)8);
                                    hgemm_kernel_func_table[3][n_l0/VBLOCK()-1](
                                        a + (i_local+i_l0) * lda + p,
                                        b + p * ldb + j,
                                        nullptr,
                                        y_base_local + i_l0 * ldy_local + j_l0,
                                        8,
                                        n_l0,
                                        k_l1,
                                        lda,
                                        VBLOCK(),
                                        ldb,
                                        VBLOCK(),
                                        ldy_local,
                                        true,
                                        0
                                    );
                                }
                                i_l0 += 8;
                            }
                            if (m_l1_tail & 4) {
                                for (int64_t j_l0 = 0; j_l0 < n_l1_pack; j_l0 += 16) {
                                    const int64_t j = j_l0 + j_l1;
                                    const int64_t n_l0 = std::min((n_l1_pack-j_l0), (int64_t)16);
                                    hgemm_kernel_func_table[2][n_l0/VBLOCK()-1](
                                        a + (i_local+i_l0) * lda + p,
                                        b + p * ldb + j,
                                        nullptr,
                                        y_base_local + i_l0 * ldy_local + j_l0,
                                        4,
                                        n_l0,
                                        k_l1,
                                        lda,
                                        VBLOCK(),
                                        ldb,
                                        VBLOCK(),
                                        ldy_local,
                                        true,
                                        0
                                    );
                                }
                                i_l0 += 4;
                            }
                            if (m_l1_tail & 2) {
                                for (int64_t j_l0 = 0; j_l0 < n_l1_pack; j_l0 += 16) {
                                    const int64_t j = j_l0 + j_l1;
                                    const int64_t n_l0 = std::min((n_l1_pack-j_l0), (int64_t)16);
                                    hgemm_kernel_func_table[1][n_l0/VBLOCK()-1](
                                        a + (i_local+i_l0) * lda + p,
                                        b + p * ldb + j,
                                        nullptr,
                                        y_base_local + i_l0 * ldy_local + j_l0,
                                        2,
                                        n_l0,
                                        k_l1,
                                        lda,
                                        VBLOCK(),
                                        ldb,
                                        VBLOCK(),
                                        ldy_local,
                                        true,
                                        0
                                    );
                                }
                                i_l0 += 2;
                            }
                            if (m_l1_tail & 1) {
                                for (int64_t j_l0 = 0; j_l0 < n_l1_pack; j_l0 += 16) {
                                    const int64_t j = j_l0 + j_l1;
                                    const int64_t n_l0 = std::min((n_l1_pack-j_l0), (int64_t)16);
                                    hgemm_kernel_func_table[0][n_l0/VBLOCK()-1](
                                        a + (i_local+i_l0) * lda + p,
                                        b + p * ldb + j,
                                        nullptr,
                                        y_base_local + i_l0 * ldy_local + j_l0,
                                        1,
                                        n_l0,
                                        k_l1,
                                        lda,
                                        VBLOCK(),
                                        ldb,
                                        VBLOCK(),
                                        ldy_local,
                                        true,
                                        0
                                    );
                                }
                            }
                        }
                    }  // K block L1

                    if ((p_l3 + sgemm_k3 >= K)) {

                        if (alpha == 1) {
                            if (beta == 0.0f) {
                                const int64_t n_l1_align4 = (n_l1 & (~(VBLOCK()-1)));
                                for (int64_t m_idx = 0; m_idx < m_l1; m_idx++) {
                                    const __fp16 * ybuf_base = y_base_local + m_idx * ldy_local;
                                    __fp16 * y_base = y + (i_local + m_idx) * ldy + j_l1;
                                    for (int64_t n_idx = 0; n_idx < n_l1_align4; n_idx+=VBLOCK()) {
                                        float16x8_t vdata = vld1q_f16(ybuf_base + n_idx);
                                        vst1q_f16(y_base + n_idx, vdata);
                                    }
                                    for (int64_t n_idx = n_l1_align4; n_idx < n_l1; n_idx++) {
                                        y_base[n_idx] = ybuf_base[n_idx];
                                    }
                                }
                            }
                            else if (beta == 1.0f) {
                                const int64_t n_l1_align4 = (n_l1 & (~(VBLOCK()-1)));
                                for (int64_t m_idx = 0; m_idx < m_l1; m_idx++) {
                                    const __fp16 * ybuf_base = y_base_local + m_idx * ldy_local;
                                    __fp16 * y_base = y + (i_local + m_idx) * ldy + j_l1;
                                    const __fp16 * c_base = c + (i_local + m_idx) * ldc + j_l1;
                                    for (int64_t n_idx = 0; n_idx < n_l1_align4; n_idx+=VBLOCK()) {
                                        float16x8_t vdata_y = vld1q_f16(ybuf_base + n_idx);
                                        float16x8_t vdata_c = vld1q_f16(c_base + n_idx);
                                        vdata_y = vaddq_f16(vdata_y, vdata_c);

                                        vst1q_f16(y_base + n_idx, vdata_y);
                                    }
                                    for (int64_t n_idx = n_l1_align4; n_idx < n_l1; n_idx++) {
                                        y_base[n_idx] = ybuf_base[n_idx] + c_base[n_idx];
                                    }
                                }
                            }
                            else if (beta != 1.0f) {
                                float16x8_t vbeta = vdupq_n_f16(beta);
                                const int64_t n_l1_align4 = (n_l1 & (~(VBLOCK()-1)));
                                for (int64_t m_idx = 0; m_idx < m_l1; m_idx++) {
                                    const __fp16 * ybuf_base = y_base_local + m_idx * ldy_local;
                                    __fp16 * y_base = y + (i_local + m_idx) * ldy + j_l1;
                                    const __fp16 * c_base = c + (i_local + m_idx) * ldc + j_l1;
                                    for (int64_t n_idx = 0; n_idx < n_l1_align4; n_idx+=VBLOCK()) {
                                        float16x8_t vdata_y = vld1q_f16(ybuf_base + n_idx);
                                        float16x8_t vdata_c = vld1q_f16(c_base + n_idx);
                                        vdata_y = vfmaq_f16(vdata_y, vdata_c, vbeta);

                                        vst1q_f16(y_base + n_idx, vdata_y);
                                    }
                                    for (int64_t n_idx = n_l1_align4; n_idx < n_l1; n_idx++) {
                                        y_base[n_idx] = ybuf_base[n_idx] + c_base[n_idx] * beta;
                                    }
                                }
                            }
                        }
                        else if (alpha != 1) {
                            if (beta == 0.0f) {
                                float16x8_t valpha = vdupq_n_f16(alpha);
                                const int64_t n_l1_align4 = (n_l1 & (~(VBLOCK()-1)));
                                for (int64_t m_idx = 0; m_idx < m_l1; m_idx++) {
                                    const __fp16 * ybuf_base = y_base_local + m_idx * ldy_local;
                                    __fp16 * y_base = y + (i_local + m_idx) * ldy + j_l1;
                                    for (int64_t n_idx = 0; n_idx < n_l1_align4; n_idx+=VBLOCK()) {
                                        float16x8_t vdata = vld1q_f16(ybuf_base + n_idx);
                                        vdata = vmulq_f16(vdata, valpha);
                                        vst1q_f16(y_base + n_idx, vdata);
                                    }
                                    for (int64_t n_idx = n_l1_align4; n_idx < n_l1; n_idx++) {
                                        y_base[n_idx] = ybuf_base[n_idx] * alpha;
                                    }
                                }
                            }
                            else if (beta == 1.0f) {
                                float16x8_t valpha = vdupq_n_f16(alpha);
                                const int64_t n_l1_align4 = (n_l1 & (~(VBLOCK()-1)));
                                for (int64_t m_idx = 0; m_idx < m_l1; m_idx++) {
                                    const __fp16 * ybuf_base = y_base_local + m_idx * ldy_local;
                                    __fp16 * y_base = y + (i_local + m_idx) * ldy + j_l1;
                                    const __fp16 * c_base = c + (i_local + m_idx) * ldc + j_l1;
                                    for (int64_t n_idx = 0; n_idx < n_l1_align4; n_idx+=VBLOCK()) {
                                        float16x8_t vdata_y = vld1q_f16(ybuf_base + n_idx);
                                        float16x8_t vdata_c = vld1q_f16(c_base + n_idx);
                                        vdata_y = vfmaq_f16(vdata_c, vdata_y, valpha);

                                        vst1q_f16(y_base + n_idx, vdata_y);
                                    }
                                    for (int64_t n_idx = n_l1_align4; n_idx < n_l1; n_idx++) {
                                        y_base[n_idx] = ybuf_base[n_idx] * alpha + c_base[n_idx];
                                    }
                                }
                            }
                            else if (beta != 1.0f) {
                                float16x8_t valpha = vdupq_n_f16(alpha);
                                float16x8_t vbeta = vdupq_n_f16(beta);
                                const int64_t n_l1_align4 = (n_l1 & (~(VBLOCK()-1)));
                                for (int64_t m_idx = 0; m_idx < m_l1; m_idx++) {
                                    const __fp16 * ybuf_base = y_base_local + m_idx * ldy_local;
                                    __fp16 * y_base = y + (i_local + m_idx) * ldy + j_l1;
                                    const __fp16 * c_base = c + (i_local + m_idx) * ldc + j_l1;
                                    for (int64_t n_idx = 0; n_idx < n_l1_align4; n_idx+=VBLOCK()) {
                                        float16x8_t vdata_y = vld1q_f16(ybuf_base + n_idx);
                                        float16x8_t vdata_c = vld1q_f16(c_base + n_idx);
                                        vdata_y = vmulq_f16(vdata_y, valpha);
                                        vdata_y = vfmaq_f16(vdata_y, vdata_c, vbeta);

                                        vst1q_f16(y_base + n_idx, vdata_y);
                                    }
                                    for (int64_t n_idx = n_l1_align4; n_idx < n_l1; n_idx++) {
                                        y_base[n_idx] = ybuf_base[n_idx] * alpha + c_base[n_idx] * beta;
                                    }
                                }
                            }
                        }
                    }

                }  // N block L1
            }  // M block L1
        }  // K block L3
    }  // M block L3

}  // omp parallel region
}

static void ppl_arm_server_kernel_fp16_gemm_alpha0(
    const __fp16 *a,
    const __fp16 *b,
    const __fp16 *c,
    __fp16 *y,
    __fp16 *tmp_buffer,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldy,
    const __fp16 alpha,
    const __fp16 beta,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_m3,
    const int64_t sgemm_k3)
{
PRAGMA_OMP_PARALLEL()
{
    if (beta == 0) {
        const int64_t N_align = PACK_VBLOCK(N);
        float16x8_t vzero = vdupq_n_f16(0.0f);

        PRAGMA_OMP_FOR()
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N_align; j+=VBLOCK()) {
                vst1q_f16(y + i * ldy + j, vzero);
            }
            for (int64_t j = N_align; j < N; j++) {
                y[i*ldy+j] = 0.0f;
            }
        }
    }
    else if (beta == 1) {
        const int64_t N_align = PACK_VBLOCK(N);

        PRAGMA_OMP_FOR()
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N_align; j+=VBLOCK()) {
                float16x8_t vdata_c = vld1q_f16(c + i * ldc + j);
                vst1q_f16(y + i * ldy + j, vdata_c);
            }
            for (int64_t j = N_align; j < N; j++) {
                y[i*ldy+j] = c[i*ldc+j];
            }
        }
    }
    else if (beta != 1) {
        float16x8_t vbeta = vdupq_n_f16(beta);
        const int64_t N_align = PACK_VBLOCK(N);

        PRAGMA_OMP_FOR()
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N_align; j+=VBLOCK()) {
                float16x8_t vdata_c = vld1q_f16(c + i * ldc + j);
                vdata_c = vmulq_f16(vdata_c, vbeta);
                vst1q_f16(y + i * ldy + j, vdata_c);
            }
            for (int64_t j = N_align; j < N; j++) {
                y[i*ldy+j] = c[i*ldc+j]*beta;
            }
        }
    }
}
}

ppl::common::RetCode gemm_fp16(
    const __fp16 *a,
    const __fp16 *b,
    const __fp16 *c,
    __fp16 *y,
    __fp16 *tmp_buffer,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldy,
    const __fp16 alpha,
    const __fp16 beta,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_m3,
    const int64_t sgemm_k3)
{
    if (alpha == 0.0f) {
        ppl_arm_server_kernel_fp16_gemm_alpha0(a, b, c, y, tmp_buffer,
            M, N, K, lda, ldb, ldc, ldy, alpha, beta, sgemm_m1, sgemm_n1, sgemm_k1, sgemm_m3, sgemm_k3);
    }
    else {
        ppl_arm_server_kernel_fp16_gemm_alphax(a, b, c, y, tmp_buffer,
            M, N, K, lda, ldb, ldc, ldy, alpha, beta, sgemm_m1, sgemm_n1, sgemm_k1, sgemm_m3, sgemm_k3);
    }
    return ppl::common::RC_SUCCESS;
}

}}}}  // namespace ppl::kernel::arm_server::neon

#endif
