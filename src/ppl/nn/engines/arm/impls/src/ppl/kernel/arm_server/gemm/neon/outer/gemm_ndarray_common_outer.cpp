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

#include <arm_neon.h>
#include <string.h>

#include "ppl/kernel/arm_server/gemm/neon/gemm.h"
#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"
#include "ppl/kernel/arm_server/common/simd_tools.h"
#include "ppl/kernel/arm_server/gemm/neon/kernel/fp16/hgemm_ndarray_kernel.h"
#include "ppl/kernel/arm_server/gemm/neon/kernel/fp32/sgemm_ndarray_kernel.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

struct gemm_blk_param {
    int64_t m_blk;
    int64_t n_blk;
    int64_t k_blk;
};

#define SGEMM_M_KERNEL 8
#define SGEMM_N_KERNEL 12

#define HGEMM_M_KERNEL 8
#define HGEMM_N_KERNEL 24

template <typename eT>
inline int64_t M_KERNEL();
template <typename eT>
inline int64_t N_KERNEL();

template <>
inline int64_t M_KERNEL<float>()
{
    return SGEMM_M_KERNEL;
}
template <>
inline int64_t N_KERNEL<float>()
{
    return SGEMM_N_KERNEL;
}

#ifdef PPLNN_USE_ARMV8_2_FP16
template <>
inline int64_t M_KERNEL<__fp16>()
{
    return HGEMM_M_KERNEL;
}
template <>
inline int64_t N_KERNEL<__fp16>()
{
    return HGEMM_N_KERNEL;
}
#endif

inline uint64_t temp_buffer_a_elemsize(const gemm_blk_param& blk_param, const int64_t K)
{
    return blk_param.m_blk * round_up(K, blk_param.k_blk);
}

inline uint64_t temp_buffer_b_elemsize(const gemm_blk_param& blk_param)
{
    return blk_param.k_blk * blk_param.n_blk;
}

inline uint64_t temp_buffer_dst_elemsize(const gemm_blk_param& blk_param)
{
    return blk_param.m_blk * blk_param.n_blk;
}

inline uint64_t temp_buffer_per_thread(const gemm_blk_param& blk_param, const int64_t K)
{
    return temp_buffer_a_elemsize(blk_param, K) + temp_buffer_b_elemsize(blk_param) + temp_buffer_dst_elemsize(blk_param) + 128;
}

inline void* align_ptr(const void* ptr, uint64_t align)
{
    return (void*)(((uint64_t)ptr + align - 1) / align * align);
}

inline void transpose_4x4_32bit(
    float32x4_t& v0,
    float32x4_t& v1,
    float32x4_t& v2,
    float32x4_t& v3)
{
    const float32x4_t v4 = vtrn1q_f32(v0, v1);
    const float32x4_t v5 = vtrn2q_f32(v0, v1);
    const float32x4_t v6 = vtrn1q_f32(v2, v3);
    const float32x4_t v7 = vtrn2q_f32(v2, v3);

    v0 = (float32x4_t)vtrn1q_f64((float64x2_t)v4, (float64x2_t)v6);
    v1 = (float32x4_t)vtrn1q_f64((float64x2_t)v5, (float64x2_t)v7);
    v2 = (float32x4_t)vtrn2q_f64((float64x2_t)v4, (float64x2_t)v6);
    v3 = (float32x4_t)vtrn2q_f64((float64x2_t)v5, (float64x2_t)v7);
}

#ifdef PPLNN_USE_ARMV8_2_FP16
inline void transpose_8x8_16bit(
    float16x8_t& v0,
    float16x8_t& v1,
    float16x8_t& v2,
    float16x8_t& v3,
    float16x8_t& v4,
    float16x8_t& v5,
    float16x8_t& v6,
    float16x8_t& v7)
{
    float16x8_t v8  = vtrn1q_f16(v0, v1);
    float16x8_t v9  = vtrn2q_f16(v0, v1);
    float16x8_t v10 = vtrn1q_f16(v2, v3);
    float16x8_t v11 = vtrn2q_f16(v2, v3);
    float16x8_t v12 = vtrn1q_f16(v4, v5);
    float16x8_t v13 = vtrn2q_f16(v4, v5);
    float16x8_t v14 = vtrn1q_f16(v6, v7);
    float16x8_t v15 = vtrn2q_f16(v6, v7);

    float32x4_t v16 = vtrn1q_f32((float32x4_t)v8, (float32x4_t)v10);
    float32x4_t v18 = vtrn2q_f32((float32x4_t)v8, (float32x4_t)v10);
    float32x4_t v17 = vtrn1q_f32((float32x4_t)v9, (float32x4_t)v11);
    float32x4_t v19 = vtrn2q_f32((float32x4_t)v9, (float32x4_t)v11);
    float32x4_t v20 = vtrn1q_f32((float32x4_t)v12, (float32x4_t)v14);
    float32x4_t v22 = vtrn2q_f32((float32x4_t)v12, (float32x4_t)v14);
    float32x4_t v21 = vtrn1q_f32((float32x4_t)v13, (float32x4_t)v15);
    float32x4_t v23 = vtrn2q_f32((float32x4_t)v13, (float32x4_t)v15);

    v0 = (float16x8_t)vtrn1q_f64((float64x2_t)v16, (float64x2_t)v20);
    v4 = (float16x8_t)vtrn2q_f64((float64x2_t)v16, (float64x2_t)v20);
    v1 = (float16x8_t)vtrn1q_f64((float64x2_t)v17, (float64x2_t)v21);
    v5 = (float16x8_t)vtrn2q_f64((float64x2_t)v17, (float64x2_t)v21);
    v2 = (float16x8_t)vtrn1q_f64((float64x2_t)v18, (float64x2_t)v22);
    v6 = (float16x8_t)vtrn2q_f64((float64x2_t)v18, (float64x2_t)v22);
    v3 = (float16x8_t)vtrn1q_f64((float64x2_t)v19, (float64x2_t)v23);
    v7 = (float16x8_t)vtrn2q_f64((float64x2_t)v19, (float64x2_t)v23);
}
#endif

inline void prefetch_l1(const void* addr)
{
    asm volatile(
        "prfm   pldl1keep,  [%[addr],      #0]         \n"
        :
        : [addr] "r"(addr)
        :);
}

template <typename eT>
inline void gemm_ndarray_common_outer_pack_an2at(
    const eT* A,
    const int64_t M,
    const int64_t K,
    const int64_t lda,
    const gemm_blk_param& blk_param,
    eT* dst);

template <>
inline void gemm_ndarray_common_outer_pack_an2at<float>(
    const float* A,
    const int64_t M,
    const int64_t K,
    const int64_t lda,
    const gemm_blk_param& blk_param,
    float* dst)
{
    const int64_t simd_w = 4;
    const int64_t ld_dst = M_KERNEL<float>();

    for (int64_t k_base = 0; k_base < K; k_base += blk_param.k_blk) {
        const int64_t k_eff = min(K - k_base, blk_param.k_blk);
        for (int64_t m_base = 0; m_base < M; m_base += M_KERNEL<float>()) {
            const int64_t m_eff = min(M - m_base, M_KERNEL<float>());
            const float* p_src  = A + m_base * lda + k_base;
            float* p_dst        = dst + k_base * blk_param.m_blk + m_base * blk_param.k_blk;

            int64_t m = 0;
            for (; m + simd_w < m_eff; m += simd_w) {
                const float* p_src_0 = p_src + (m + 0) * lda;
                const float* p_src_1 = p_src + (m + 1) * lda;
                const float* p_src_2 = p_src + (m + 2) * lda;
                const float* p_src_3 = p_src + (m + 3) * lda;

                const float* p_src_4 = p_src + (m + 4) * lda;
                const float* p_src_5 = p_src + (m + 5) * lda;
                const float* p_src_6 = p_src + (m + 6) * lda;
                const float* p_src_7 = p_src + (m + 7) * lda;

                int64_t k = 0;
                for (; k + simd_w <= k_eff; k += simd_w) {
                    prefetch_l1(p_src_4 + k);
                    prefetch_l1(p_src_5 + k);
                    prefetch_l1(p_src_6 + k);
                    prefetch_l1(p_src_7 + k);

                    float32x4_t v0 = vld1q_f32(p_src_0 + k);
                    float32x4_t v1 = vld1q_f32(p_src_1 + k);
                    float32x4_t v2 = vld1q_f32(p_src_2 + k);
                    float32x4_t v3 = vld1q_f32(p_src_3 + k);

                    transpose_4x4_32bit(v0, v1, v2, v3);

                    vst1q_f32(p_dst + (k + 0) * ld_dst + m, v0);
                    vst1q_f32(p_dst + (k + 1) * ld_dst + m, v1);
                    vst1q_f32(p_dst + (k + 2) * ld_dst + m, v2);
                    vst1q_f32(p_dst + (k + 3) * ld_dst + m, v3);
                }
                for (; k < k_eff; k++) {
                    p_dst[k * ld_dst + m + 0] = p_src_0[k];
                    p_dst[k * ld_dst + m + 1] = p_src_1[k];
                    p_dst[k * ld_dst + m + 2] = p_src_2[k];
                    p_dst[k * ld_dst + m + 3] = p_src_3[k];
                }
            }
            for (; m < m_eff; m++) {
                for (int64_t k = 0; k < k_eff; k++) {
                    p_dst[k * ld_dst + m] = p_src[m * lda + k];
                }
            }
        }
    }
}

#ifdef PPLNN_USE_ARMV8_2_FP16
template <>
inline void gemm_ndarray_common_outer_pack_an2at<__fp16>(
    const __fp16* A,
    const int64_t M,
    const int64_t K,
    const int64_t lda,
    const gemm_blk_param& blk_param,
    __fp16* dst)
{
    const int64_t simd_w = 8;
    const int64_t ld_dst = M_KERNEL<__fp16>();

    for (int64_t k_base = 0; k_base < K; k_base += blk_param.k_blk) {
        const int64_t k_eff = min(K - k_base, blk_param.k_blk);
        for (int64_t m_base = 0; m_base < M; m_base += M_KERNEL<__fp16>()) {
            const int64_t m_eff = min(M - m_base, M_KERNEL<__fp16>());
            const __fp16* p_src = A + m_base * lda + k_base;
            __fp16* p_dst       = dst + k_base * blk_param.m_blk + m_base * blk_param.k_blk;

            int64_t m = 0;
            for (; m + simd_w < m_eff; m += simd_w) {
                const __fp16* p_src_0 = p_src + (m + 0) * lda;
                const __fp16* p_src_1 = p_src + (m + 1) * lda;
                const __fp16* p_src_2 = p_src + (m + 2) * lda;
                const __fp16* p_src_3 = p_src + (m + 3) * lda;
                const __fp16* p_src_4 = p_src + (m + 4) * lda;
                const __fp16* p_src_5 = p_src + (m + 5) * lda;
                const __fp16* p_src_6 = p_src + (m + 6) * lda;
                const __fp16* p_src_7 = p_src + (m + 7) * lda;

                const __fp16* p_src_8  = p_src + (m + 8) * lda;
                const __fp16* p_src_9  = p_src + (m + 9) * lda;
                const __fp16* p_src_10 = p_src + (m + 10) * lda;
                const __fp16* p_src_11 = p_src + (m + 11) * lda;
                const __fp16* p_src_12 = p_src + (m + 12) * lda;
                const __fp16* p_src_13 = p_src + (m + 13) * lda;
                const __fp16* p_src_14 = p_src + (m + 14) * lda;
                const __fp16* p_src_15 = p_src + (m + 15) * lda;

                int64_t k = 0;
                for (; k + simd_w <= k_eff; k += simd_w) {
                    prefetch_l1(p_src_8 + k);
                    prefetch_l1(p_src_9 + k);
                    prefetch_l1(p_src_10 + k);
                    prefetch_l1(p_src_11 + k);
                    prefetch_l1(p_src_12 + k);
                    prefetch_l1(p_src_13 + k);
                    prefetch_l1(p_src_14 + k);
                    prefetch_l1(p_src_15 + k);

                    float16x8_t v0 = vld1q_f16(p_src_0 + k);
                    float16x8_t v1 = vld1q_f16(p_src_1 + k);
                    float16x8_t v2 = vld1q_f16(p_src_2 + k);
                    float16x8_t v3 = vld1q_f16(p_src_3 + k);
                    float16x8_t v4 = vld1q_f16(p_src_4 + k);
                    float16x8_t v5 = vld1q_f16(p_src_5 + k);
                    float16x8_t v6 = vld1q_f16(p_src_6 + k);
                    float16x8_t v7 = vld1q_f16(p_src_7 + k);

                    transpose_8x8_16bit(v0, v1, v2, v3, v4, v5, v6, v7);

                    vst1q_f16(p_dst + (k + 0) * ld_dst + m, v0);
                    vst1q_f16(p_dst + (k + 1) * ld_dst + m, v1);
                    vst1q_f16(p_dst + (k + 2) * ld_dst + m, v2);
                    vst1q_f16(p_dst + (k + 3) * ld_dst + m, v3);
                    vst1q_f16(p_dst + (k + 4) * ld_dst + m, v4);
                    vst1q_f16(p_dst + (k + 5) * ld_dst + m, v5);
                    vst1q_f16(p_dst + (k + 6) * ld_dst + m, v6);
                    vst1q_f16(p_dst + (k + 7) * ld_dst + m, v7);
                }
                for (; k < k_eff; k++) {
                    p_dst[k * ld_dst + m + 0] = p_src_0[k];
                    p_dst[k * ld_dst + m + 1] = p_src_1[k];
                    p_dst[k * ld_dst + m + 2] = p_src_2[k];
                    p_dst[k * ld_dst + m + 3] = p_src_3[k];
                    p_dst[k * ld_dst + m + 4] = p_src_4[k];
                    p_dst[k * ld_dst + m + 5] = p_src_5[k];
                    p_dst[k * ld_dst + m + 6] = p_src_6[k];
                    p_dst[k * ld_dst + m + 7] = p_src_7[k];
                }
            }
            for (; m < m_eff; m++) {
                for (int64_t k = 0; k < k_eff; k++) {
                    p_dst[k * ld_dst + m] = p_src[m * lda + k];
                }
            }
        }
    }
}
#endif

template <typename eT>
inline void gemm_ndarray_common_outer_pack_at2at(
    const eT* A,
    const int64_t M,
    const int64_t K,
    const int64_t lda,
    const gemm_blk_param& blk_param,
    eT* dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w = sizeof(vecType) / sizeof(eT);
    const int64_t ld_dst = M_KERNEL<eT>();

    for (int64_t k_base = 0; k_base < K; k_base += blk_param.k_blk) {
        const int64_t k_eff = min(K - k_base, blk_param.k_blk);
        for (int64_t m_base = 0; m_base < M; m_base += M_KERNEL<eT>()) {
            const int64_t m_eff = min(M - m_base, M_KERNEL<eT>());
            const eT* p_src     = A + k_base * lda + m_base;
            eT* p_dst           = dst + k_base * blk_param.m_blk + m_base * blk_param.k_blk;

            const int64_t prefetch_line = 1;
            const eT* p_src_next        = p_src + prefetch_line * lda;

            if (m_eff == M_KERNEL<eT>()) {
                for (int64_t k = 0; k < k_eff; k++) {
                    prefetch_l1(p_src_next);
                    vst<eT, eN>(p_dst + k * ld_dst + simd_w * 0, vld<eT, eN>(p_src + k * lda + simd_w * 0));
                    vst<eT, eN>(p_dst + k * ld_dst + simd_w * 1, vld<eT, eN>(p_src + k * lda + simd_w * 1));
                    vst<eT, eN>(p_dst + k * ld_dst + simd_w * 2, vld<eT, eN>(p_src + k * lda + simd_w * 2));
                }
            } else {
                for (int64_t k = 0; k < k_eff; k++) {
                    prefetch_l1(p_src_next);
                    int64_t m = 0;
                    for (; m + simd_w <= m_eff; m += simd_w) {
                        vst<eT, eN>(p_dst + k * ld_dst + m, vld<eT, eN>(p_src + k * lda + m));
                    }
                    for (; m < m_eff; m++) {
                        p_dst[k * ld_dst + m] = p_src[k * lda + m];
                    }
                }
            }
        }
    }
}

template <typename eT>
inline void gemm_ndarray_common_outer_pack_bn2bn(
    const eT* B,
    const int64_t K,
    const int64_t N,
    const int64_t ldb,
    const gemm_blk_param& blk_param,
    eT* dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w = sizeof(vecType) / sizeof(eT);
    const int64_t ld_dst = N_KERNEL<eT>();

    for (int64_t n_base = 0; n_base < N; n_base += N_KERNEL<eT>()) {
        const int64_t n_eff = min(N - n_base, N_KERNEL<eT>());
        const eT* p_src     = B + n_base;
        eT* p_dst           = dst + n_base * blk_param.k_blk;

        const int64_t prefetch_line = 1;
        const eT* p_src_next        = p_src + prefetch_line * ldb;

        if (n_eff == N_KERNEL<eT>()) {
            for (int64_t k = 0; k < K; k++) {
                prefetch_l1(p_src_next + k * ld_dst);
                vst<eT, eN>(p_dst + k * ld_dst + simd_w * 0, vld<eT, eN>(p_src + k * ldb + simd_w * 0));
                vst<eT, eN>(p_dst + k * ld_dst + simd_w * 1, vld<eT, eN>(p_src + k * ldb + simd_w * 1));
                vst<eT, eN>(p_dst + k * ld_dst + simd_w * 2, vld<eT, eN>(p_src + k * ldb + simd_w * 2));
            }
        } else {
            for (int64_t k = 0; k < K; k++) {
                prefetch_l1(p_src_next + k * ld_dst);
                int64_t n = 0;
                for (; n + simd_w <= n_eff; n += simd_w) {
                    vst<eT, eN>(p_dst + k * ld_dst + n, vld<eT, eN>(p_src + k * ldb + n));
                }
                for (; n < n_eff; n++) {
                    p_dst[k * ld_dst + n] = p_src[k * ldb + n];
                }
            }
        }
    }
}

template <typename eT>
inline void gemm_ndarray_common_outer_pack_bt2bn(
    const eT* B,
    const int64_t K,
    const int64_t N,
    const int64_t ldb,
    const gemm_blk_param& blk_param,
    eT* dst)
{
    const int64_t ld_dst = N_KERNEL<eT>();

    for (int64_t n_base = 0; n_base < N; n_base += N_KERNEL<eT>()) {
        const int64_t n_eff = min(N - n_base, N_KERNEL<eT>());
        const eT* p_src     = B + n_base * ldb;
        eT* p_dst           = dst + n_base * blk_param.k_blk;

        for (int64_t n = 0; n < n_eff; n++) {
            for (int64_t k = 0; k < K; k++) {
                p_dst[k * ld_dst + n] = p_src[n * ldb + k];
            }
        }
    }
}

template <>
inline void gemm_ndarray_common_outer_pack_bt2bn<float>(
    const float* B,
    const int64_t K,
    const int64_t N,
    const int64_t ldb,
    const gemm_blk_param& blk_param,
    float* dst)
{
    const int64_t simd_w = 4;
    const int64_t ld_dst = N_KERNEL<float>();

    for (int64_t n_base = 0; n_base < N; n_base += N_KERNEL<float>()) {
        const int64_t n_eff = min(N - n_base, N_KERNEL<float>());
        const float* p_src  = B + n_base * ldb;
        float* p_dst        = dst + n_base * blk_param.k_blk;

        int64_t n = 0;
        for (; n + simd_w <= n_eff; n += simd_w) {
            const float* p_src_0 = p_src + (n + 0) * ldb;
            const float* p_src_1 = p_src + (n + 1) * ldb;
            const float* p_src_2 = p_src + (n + 2) * ldb;
            const float* p_src_3 = p_src + (n + 3) * ldb;

            const float* p_src_4 = p_src + (n + 4) * ldb;
            const float* p_src_5 = p_src + (n + 5) * ldb;
            const float* p_src_6 = p_src + (n + 6) * ldb;
            const float* p_src_7 = p_src + (n + 7) * ldb;

            int64_t k = 0;
            for (; k + simd_w <= K; k += simd_w) {
                prefetch_l1(p_src_4 + k);
                prefetch_l1(p_src_5 + k);
                prefetch_l1(p_src_6 + k);
                prefetch_l1(p_src_7 + k);

                float32x4_t v0 = vld1q_f32(p_src_0 + k);
                float32x4_t v1 = vld1q_f32(p_src_1 + k);
                float32x4_t v2 = vld1q_f32(p_src_2 + k);
                float32x4_t v3 = vld1q_f32(p_src_3 + k);

                transpose_4x4_32bit(v0, v1, v2, v3);

                vst1q_f32(p_dst + (k + 0) * ld_dst + n, v0);
                vst1q_f32(p_dst + (k + 1) * ld_dst + n, v1);
                vst1q_f32(p_dst + (k + 2) * ld_dst + n, v2);
                vst1q_f32(p_dst + (k + 3) * ld_dst + n, v3);
            }
            for (; k < K; k++) {
                p_dst[k * ld_dst + n + 0] = p_src_0[k];
                p_dst[k * ld_dst + n + 1] = p_src_1[k];
                p_dst[k * ld_dst + n + 2] = p_src_2[k];
                p_dst[k * ld_dst + n + 3] = p_src_3[k];
            }
        }
        for (; n < n_eff; n++) {
            for (int64_t k = 0; k < K; k++) {
                p_dst[k * ld_dst + n] = p_src[n * ldb + k];
            }
        }
    }
}

template <typename eT>
inline void gemm_ndarray_common_outer_store_dst(
    const eT* src,
    const eT* C,
    const int64_t M,
    const int64_t N,
    const int64_t m_offset,
    const int64_t n_offset,
    const float alpha_fp32,
    const float beta_fp32,
    const int64_t ldc,
    const int64_t ld_dst,
    const gemm_C_type_t c_type,
    const gemm_blk_param& blk_param,
    eT* dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w = sizeof(vecType) / sizeof(eT);
    const int64_t ld_src = blk_param.n_blk;

    const eT alpha = (const eT)alpha_fp32;
    const eT beta  = (const eT)beta_fp32;

    const vecType v_alpha = vdup_n<eT, eN>(alpha);

    if (c_type == gemm_C_type::EMPTY || c_type == gemm_C_type::SCALAR) {
        const eT bias        = (c_type == gemm_C_type::SCALAR ? C[0] : 0) * beta;
        const vecType v_bias = vdup_n<eT, eN>(bias);
        for (int64_t m = 0; m < M; m++) {
            int64_t n = 0;
            for (; n + simd_w <= N; n += simd_w) {
                vecType v_src = vld<eT, eN>(src + m * ld_src + n);
                vst<eT, eN>(dst + m * ld_dst + n, v_src * v_alpha + v_bias);
            }
            for (; n < N; n++) {
                dst[m * ld_dst + n] = src[m * ld_src + n] * alpha + bias;
            }
        }
    } else if (c_type == gemm_C_type::VECTOR_H) {
        for (int64_t m = 0; m < M; m++) {
            const eT bias        = C[m + m_offset] * beta;
            const vecType v_bias = vdup_n<eT, eN>(bias);
            int64_t n            = 0;
            for (; n + simd_w <= N; n += simd_w) {
                vecType v_src = vld<eT, eN>(src + m * ld_src + n);
                vst<eT, eN>(dst + m * ld_dst + n, v_src * v_alpha + v_bias);
            }
            for (; n < N; n++) {
                dst[m * ld_dst + n] = src[m * ld_src + n] * alpha + bias;
            }
        }
    } else if (c_type == gemm_C_type::VECTOR_W) {
        const eT* ptr_c      = C + n_offset;
        const vecType v_beta = vdup_n<eT, eN>(beta);
        for (int64_t m = 0; m < M; m++) {
            int64_t n = 0;
            for (; n + simd_w <= N; n += simd_w) {
                vecType v_src = vld<eT, eN>(src + m * ld_src + n);
                vecType v_c   = vld<eT, eN>(ptr_c + n);
                vst<eT, eN>(dst + m * ld_dst + n, v_src * v_alpha + v_c * v_beta);
            }
            for (; n < N; n++) {
                dst[m * ld_dst + n] = src[m * ld_src + n] * alpha + ptr_c[n] * beta;
            }
        }
    } else if (c_type == gemm_C_type::MATRIX) {
        const eT* ptr_c      = C + m_offset * ldc + n_offset;
        const vecType v_beta = vdup_n<eT, eN>(beta);
        for (int64_t m = 0; m < M; m++) {
            int64_t n = 0;
            for (; n + simd_w <= N; n += simd_w) {
                vecType v_src = vld<eT, eN>(src + m * ld_src + n);
                vecType v_c   = vld<eT, eN>(ptr_c + m * ldc + n);
                vst<eT, eN>(dst + m * ld_dst + n, v_src * v_alpha + v_c * v_beta);
            }
            for (; n < N; n++) {
                dst[m * ld_dst + n] = src[m * ld_src + n] * alpha + ptr_c[m * ldc + n] * beta;
            }
        }
    }
}

template <typename eT>
inline gemm_blk_param gemm_ndarray_common_outer_generate_blk_param(
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t num_threads)
{
    gemm_blk_param blk_param;
    // basic blocking param
    if (std::is_same<eT, float>::value) {
        blk_param.m_blk = 512;
        blk_param.n_blk = 96;
        blk_param.k_blk = 128;
    }
#ifdef PPLNN_USE_ARMV8_2_FP16
    if (std::is_same<eT, __fp16>::value) {
        blk_param.m_blk = 512;
        blk_param.n_blk = 96;
        blk_param.k_blk = 256;
    }
#endif

    blk_param.k_blk = min(blk_param.k_blk, K);
    blk_param.m_blk = min(blk_param.m_blk, round_up(M, M_KERNEL<eT>()));
    blk_param.n_blk = min(blk_param.n_blk, round_up(N, N_KERNEL<eT>()));

    if (num_threads == 1) { // for single thread & small M/N case, just limit block size to M/N
        return blk_param;
    }

    blk_param.m_blk = min(blk_param.m_blk, (int64_t)256); // 512 is too large for multi-thread

    // TODO: not a best task divide strategy
    int64_t m_task_num = div_up(M, blk_param.m_blk);
    int64_t n_task_num = div_up(N, blk_param.n_blk);
    while (m_task_num * n_task_num < num_threads * 3) {
        const bool m_can_divide = blk_param.m_blk >= 2 * M_KERNEL<eT>();
        const bool n_can_divide = blk_param.n_blk >= 2 * N_KERNEL<eT>();
        if (m_task_num <= n_task_num && m_can_divide) {
            blk_param.m_blk /= 2;
        } else if (n_task_num <= m_task_num && n_can_divide) {
            blk_param.n_blk /= 2;
        } else if (m_can_divide) {
            blk_param.m_blk /= 2;
        } else if (n_can_divide) {
            blk_param.n_blk /= 2;
        } else {
            break;
        }
        m_task_num = div_up(M, blk_param.m_blk);
        n_task_num = div_up(N, blk_param.n_blk);
    }

    blk_param.m_blk = round_up(blk_param.m_blk, M_KERNEL<eT>());
    blk_param.n_blk = round_up(blk_param.n_blk, N_KERNEL<eT>());

    return blk_param;
}

template <typename eT>
ppl::common::RetCode gemm_ndarray_common_outer(
    const eT* A,
    const eT* B,
    const eT* C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t transA,
    const int64_t transB,
    const float alpha,
    const float beta,
    const int64_t ldy,
    const gemm_C_type_t c_type,
    eT* Y)
{
    const int64_t simd_w      = 128 / 8 / sizeof(eT);
    const int64_t num_threads = PPL_OMP_MAX_THREADS();

    gemm_blk_param blk_param = gemm_ndarray_common_outer_generate_blk_param<eT>(M, N, K, num_threads);

    ppl::common::GenericCpuAllocator allocator;
    void* temp = allocator.Alloc(temp_buffer_per_thread(blk_param, K) * num_threads * sizeof(eT));
    std::vector<const eT*> last_pack_a_ptr(num_threads, nullptr);

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t m = 0; m < M; m += blk_param.m_blk) {
        for (int64_t n = 0; n < N; n += blk_param.n_blk) {
            const int64_t thread_id = PPL_OMP_THREAD_ID();
            eT* temp_buffer         = (eT*)align_ptr((eT*)temp + temp_buffer_per_thread(blk_param, K) * thread_id, 64);
            eT* temp_buffer_a       = temp_buffer;
            eT* temp_buffer_b       = temp_buffer_a + temp_buffer_a_elemsize(blk_param, K);
            eT* temp_buffer_dst     = temp_buffer_b + temp_buffer_b_elemsize(blk_param);

            const int64_t m_eff = min(M - m, blk_param.m_blk);
            const int64_t n_eff = min(N - n, blk_param.n_blk);
            const eT* p_src_a   = transA ? A + m : A + m * lda;
            if (last_pack_a_ptr[thread_id] != p_src_a) {
                if (!transA) {
                    gemm_ndarray_common_outer_pack_an2at<eT>(p_src_a, m_eff, K, lda, blk_param, temp_buffer_a);
                } else {
                    gemm_ndarray_common_outer_pack_at2at<eT>(p_src_a, m_eff, K, lda, blk_param, temp_buffer_a);
                }
                last_pack_a_ptr[thread_id] = p_src_a;
            }

            for (int64_t kk = 0; kk < K; kk += blk_param.k_blk) {
                const int64_t kk_eff = min(K - kk, blk_param.k_blk);
                const eT* p_src_b    = transB ? B + n * ldb + kk : B + kk * ldb + n;
                if (!transB) {
                    gemm_ndarray_common_outer_pack_bn2bn<eT>(p_src_b, kk_eff, n_eff, ldb, blk_param, temp_buffer_b);
                } else {
                    gemm_ndarray_common_outer_pack_bt2bn<eT>(p_src_b, kk_eff, n_eff, ldb, blk_param, temp_buffer_b);
                }

                const int64_t init_t = kk == 0 ? 0 : 1;

                for (int64_t m_kernel = 0; m_kernel < m_eff; m_kernel += M_KERNEL<eT>()) {
                    for (int64_t n_kernel = 0; n_kernel < n_eff; n_kernel += N_KERNEL<eT>()) {
                        const int64_t m_kernel_len = min(m_eff - m_kernel, M_KERNEL<eT>());
                        const int64_t n_kernel_len = min(n_eff - n_kernel, N_KERNEL<eT>());
                        const int64_t n_kernel_blk = div_up(n_kernel_len, simd_w);

                        const int64_t prefetch_a = 1;
                        const int64_t prefetch_b = 1;

                        const int64_t m_kernel_idx = m_kernel_len - 1;
                        const int64_t n_kernel_idx = n_kernel_blk - 1;
                        if (std::is_same<eT, float>::value) {
                            auto gemm_kernel_func = sgemm_ndarray_kernel_tn_max8x12_func_table[prefetch_a][prefetch_b][init_t][m_kernel_idx][n_kernel_idx];
                            gemm_kernel_func(
                                (const float*)temp_buffer_a + kk * blk_param.m_blk + m_kernel * blk_param.k_blk,
                                (const float*)temp_buffer_b + n_kernel * blk_param.k_blk,
                                kk_eff,
                                M_KERNEL<eT>(),
                                N_KERNEL<eT>(),
                                blk_param.n_blk,
                                (float*)temp_buffer_dst + m_kernel * blk_param.n_blk + n_kernel);
                        }
#ifdef PPLNN_USE_ARMV8_2_FP16
                        if (std::is_same<eT, __fp16>::value) {
                            auto gemm_kernel_func = hgemm_ndarray_kernel_tn_max8x24_func_table[prefetch_a][prefetch_b][init_t][m_kernel_idx][n_kernel_idx];
                            gemm_kernel_func(
                                (const __fp16*)temp_buffer_a + kk * blk_param.m_blk + m_kernel * blk_param.k_blk,
                                (const __fp16*)temp_buffer_b + n_kernel * blk_param.k_blk,
                                kk_eff,
                                M_KERNEL<eT>(),
                                N_KERNEL<eT>(),
                                blk_param.n_blk,
                                (__fp16*)temp_buffer_dst + m_kernel * blk_param.n_blk + n_kernel);
                        }
#endif
                    }
                }
            }

            gemm_ndarray_common_outer_store_dst<eT>(temp_buffer_dst, C, m_eff, n_eff, m, n, alpha, beta, ldc, ldy, c_type, blk_param, Y + m * ldy + n);
        }
    }

    allocator.Free(temp);

    return ppl::common::RC_SUCCESS;
}

template ppl::common::RetCode gemm_ndarray_common_outer<float>(
    const float* A,
    const float* B,
    const float* C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t transA,
    const int64_t transB,
    const float alpha,
    const float beta,
    const int64_t ldy,
    const gemm_C_type_t c_type,
    float* Y);

#ifdef PPLNN_USE_ARMV8_2_FP16
template ppl::common::RetCode gemm_ndarray_common_outer<__fp16>(
    const __fp16* A,
    const __fp16* B,
    const __fp16* C,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t transA,
    const int64_t transB,
    const float alpha,
    const float beta,
    const int64_t ldy,
    const gemm_C_type_t c_type,
    __fp16* Y);
#endif

}}}} // namespace ppl::kernel::arm_server::neon
