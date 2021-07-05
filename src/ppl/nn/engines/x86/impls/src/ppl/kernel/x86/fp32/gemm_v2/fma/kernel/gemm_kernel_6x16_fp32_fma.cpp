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

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/gemm_v2/fma/kernel/gemm_kernel_fp32_fma.h"

namespace ppl { namespace kernel { namespace x86 {

const int32_t simd_w = 8;

template <int32_t m_len, int32_t n_len>
void gemm_kernel_max6x16_fp32_fma(
    const float* A,
    const float* B,
    const int32_t k_len,
    const int32_t lda,
    const int32_t ldb,
    const int32_t ldc,
    float* C)
{
    if (k_len <= 0) {
        return;
    }

    __m256 va;
    __m256 vb0, vb1;
    __m256 vc00, vc01;
    __m256 vc10, vc11;
    __m256 vc20, vc21;
    __m256 vc30, vc31;
    __m256 vc40, vc41;
    __m256 vc50, vc51;

    // load C
    if (m_len >= 1) {
        if (n_len > 0 * simd_w)
            vc00 = _mm256_loadu_ps(C + 0 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc01 = _mm256_loadu_ps(C + 0 * ldc + 1 * simd_w);
    }
    if (m_len >= 2) {
        if (n_len > 0 * simd_w)
            vc10 = _mm256_loadu_ps(C + 1 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc11 = _mm256_loadu_ps(C + 1 * ldc + 1 * simd_w);
    }
    if (m_len >= 3) {
        if (n_len > 0 * simd_w)
            vc20 = _mm256_loadu_ps(C + 2 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc21 = _mm256_loadu_ps(C + 2 * ldc + 1 * simd_w);
    }
    if (m_len >= 4) {
        if (n_len > 0 * simd_w)
            vc30 = _mm256_loadu_ps(C + 3 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc31 = _mm256_loadu_ps(C + 3 * ldc + 1 * simd_w);
    }
    if (m_len >= 5) {
        if (n_len > 0 * simd_w)
            vc40 = _mm256_loadu_ps(C + 4 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc41 = _mm256_loadu_ps(C + 4 * ldc + 1 * simd_w);
    }
    if (m_len >= 6) {
        if (n_len > 0 * simd_w)
            vc50 = _mm256_loadu_ps(C + 5 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc51 = _mm256_loadu_ps(C + 5 * ldc + 1 * simd_w);
    }

    const float* a_ptr = A;
    const float* b_ptr = B;

    for (int32_t k = 0; k < k_len; ++k) {
        if (n_len > 0 * simd_w)
            vb0 = _mm256_loadu_ps(b_ptr);
        if (n_len > 1 * simd_w)
            vb1 = _mm256_loadu_ps(b_ptr + simd_w);
        b_ptr += ldb;

        if (m_len >= 1) {
            va = _mm256_set1_ps(a_ptr[0]);
            if (n_len > 0 * simd_w)
                vc00 = _mm256_fmadd_ps(va, vb0, vc00);
            if (n_len > 1 * simd_w)
                vc01 = _mm256_fmadd_ps(va, vb1, vc01);
        }
        if (m_len >= 2) {
            va = _mm256_set1_ps(a_ptr[1]);
            if (n_len > 0 * simd_w)
                vc10 = _mm256_fmadd_ps(va, vb0, vc10);
            if (n_len > 1 * simd_w)
                vc11 = _mm256_fmadd_ps(va, vb1, vc11);
        }
        if (m_len >= 3) {
            va = _mm256_set1_ps(a_ptr[2]);
            if (n_len > 0 * simd_w)
                vc20 = _mm256_fmadd_ps(va, vb0, vc20);
            if (n_len > 1 * simd_w)
                vc21 = _mm256_fmadd_ps(va, vb1, vc21);
        }
        if (m_len >= 4) {
            va = _mm256_set1_ps(a_ptr[3]);
            if (n_len > 0 * simd_w)
                vc30 = _mm256_fmadd_ps(va, vb0, vc30);
            if (n_len > 1 * simd_w)
                vc31 = _mm256_fmadd_ps(va, vb1, vc31);
        }
        if (m_len >= 5) {
            va = _mm256_set1_ps(a_ptr[4]);
            if (n_len > 0 * simd_w)
                vc40 = _mm256_fmadd_ps(va, vb0, vc40);
            if (n_len > 1 * simd_w)
                vc41 = _mm256_fmadd_ps(va, vb1, vc41);
        }
        if (m_len >= 6) {
            va = _mm256_set1_ps(a_ptr[5]);
            if (n_len > 0 * simd_w)
                vc50 = _mm256_fmadd_ps(va, vb0, vc50);
            if (n_len > 1 * simd_w)
                vc51 = _mm256_fmadd_ps(va, vb1, vc51);
        }

        a_ptr += lda;
    }

    // store C
    if (m_len >= 1) {
        if (n_len > 0 * simd_w)
            _mm256_storeu_ps(C + 0 * ldc + 0 * simd_w, vc00);
        if (n_len > 1 * simd_w)
            _mm256_storeu_ps(C + 0 * ldc + 1 * simd_w, vc01);
    }
    if (m_len >= 2) {
        if (n_len > 0 * simd_w)
            _mm256_storeu_ps(C + 1 * ldc + 0 * simd_w, vc10);
        if (n_len > 1 * simd_w)
            _mm256_storeu_ps(C + 1 * ldc + 1 * simd_w, vc11);
    }
    if (m_len >= 3) {
        if (n_len > 0 * simd_w)
            _mm256_storeu_ps(C + 2 * ldc + 0 * simd_w, vc20);
        if (n_len > 1 * simd_w)
            _mm256_storeu_ps(C + 2 * ldc + 1 * simd_w, vc21);
    }
    if (m_len >= 4) {
        if (n_len > 0 * simd_w)
            _mm256_storeu_ps(C + 3 * ldc + 0 * simd_w, vc30);
        if (n_len > 1 * simd_w)
            _mm256_storeu_ps(C + 3 * ldc + 1 * simd_w, vc31);
    }
    if (m_len >= 5) {
        if (n_len > 0 * simd_w)
            _mm256_storeu_ps(C + 4 * ldc + 0 * simd_w, vc40);
        if (n_len > 1 * simd_w)
            _mm256_storeu_ps(C + 4 * ldc + 1 * simd_w, vc41);
    }
    if (m_len >= 6) {
        if (n_len > 0 * simd_w)
            _mm256_storeu_ps(C + 5 * ldc + 0 * simd_w, vc50);
        if (n_len > 1 * simd_w)
            _mm256_storeu_ps(C + 5 * ldc + 1 * simd_w, vc51);
    }
}

void gemm_kernel_6x16_fp32_fma(
    const float* A,
    const float* B,
    const int32_t k_len,
    const int32_t lda,
    const int32_t ldb,
    const int32_t ldc,
    float* C)
{
    if (k_len <= 0) {
        return;
    }

    __m256 va;
    __m256 vb0, vb1;
    __m256 vc00, vc01;
    __m256 vc10, vc11;
    __m256 vc20, vc21;
    __m256 vc30, vc31;
    __m256 vc40, vc41;
    __m256 vc50, vc51;

    // load C
    vc00 = _mm256_loadu_ps(C + 0 * ldc + 0 * simd_w);
    vc01 = _mm256_loadu_ps(C + 0 * ldc + 1 * simd_w);
    vc10 = _mm256_loadu_ps(C + 1 * ldc + 0 * simd_w);
    vc11 = _mm256_loadu_ps(C + 1 * ldc + 1 * simd_w);
    vc20 = _mm256_loadu_ps(C + 2 * ldc + 0 * simd_w);
    vc21 = _mm256_loadu_ps(C + 2 * ldc + 1 * simd_w);
    vc30 = _mm256_loadu_ps(C + 3 * ldc + 0 * simd_w);
    vc31 = _mm256_loadu_ps(C + 3 * ldc + 1 * simd_w);
    vc40 = _mm256_loadu_ps(C + 4 * ldc + 0 * simd_w);
    vc41 = _mm256_loadu_ps(C + 4 * ldc + 1 * simd_w);
    vc50 = _mm256_loadu_ps(C + 5 * ldc + 0 * simd_w);
    vc51 = _mm256_loadu_ps(C + 5 * ldc + 1 * simd_w);

    const float* a_ptr = A;
    const float* b_ptr = B;

    // TODO: asm code here

    for (int32_t k = 0; k < k_len; ++k) {
        vb0 = _mm256_loadu_ps(b_ptr);
        vb1 = _mm256_loadu_ps(b_ptr + simd_w);
        b_ptr += ldb;

        va   = _mm256_set1_ps(a_ptr[0]);
        vc00 = _mm256_fmadd_ps(va, vb0, vc00);
        vc01 = _mm256_fmadd_ps(va, vb1, vc01);

        va   = _mm256_set1_ps(a_ptr[1]);
        vc10 = _mm256_fmadd_ps(va, vb0, vc10);
        vc11 = _mm256_fmadd_ps(va, vb1, vc11);

        va   = _mm256_set1_ps(a_ptr[2]);
        vc20 = _mm256_fmadd_ps(va, vb0, vc20);
        vc21 = _mm256_fmadd_ps(va, vb1, vc21);

        va   = _mm256_set1_ps(a_ptr[3]);
        vc30 = _mm256_fmadd_ps(va, vb0, vc30);
        vc31 = _mm256_fmadd_ps(va, vb1, vc31);

        va   = _mm256_set1_ps(a_ptr[4]);
        vc40 = _mm256_fmadd_ps(va, vb0, vc40);
        vc41 = _mm256_fmadd_ps(va, vb1, vc41);

        va   = _mm256_set1_ps(a_ptr[5]);
        vc50 = _mm256_fmadd_ps(va, vb0, vc50);
        vc51 = _mm256_fmadd_ps(va, vb1, vc51);

        a_ptr += lda;
    }

    // store C
    _mm256_storeu_ps(C + 0 * ldc + 0 * simd_w, vc00);
    _mm256_storeu_ps(C + 0 * ldc + 1 * simd_w, vc01);
    _mm256_storeu_ps(C + 1 * ldc + 0 * simd_w, vc10);
    _mm256_storeu_ps(C + 1 * ldc + 1 * simd_w, vc11);
    _mm256_storeu_ps(C + 2 * ldc + 0 * simd_w, vc20);
    _mm256_storeu_ps(C + 2 * ldc + 1 * simd_w, vc21);
    _mm256_storeu_ps(C + 3 * ldc + 0 * simd_w, vc30);
    _mm256_storeu_ps(C + 3 * ldc + 1 * simd_w, vc31);
    _mm256_storeu_ps(C + 4 * ldc + 0 * simd_w, vc40);
    _mm256_storeu_ps(C + 4 * ldc + 1 * simd_w, vc41);
    _mm256_storeu_ps(C + 5 * ldc + 0 * simd_w, vc50);
    _mm256_storeu_ps(C + 5 * ldc + 1 * simd_w, vc51);
}

const gemm_kernel_fp32_fma_func_type_t gemm_kernel_max6x16_fp32_fma_func_tab[7][3] = {
    {
        gemm_kernel_max6x16_fp32_fma<0, 0>,
        gemm_kernel_max6x16_fp32_fma<0, 8>,
        gemm_kernel_max6x16_fp32_fma<0, 16>,
    },
    {
        gemm_kernel_max6x16_fp32_fma<1, 0>,
        gemm_kernel_max6x16_fp32_fma<1, 8>,
        gemm_kernel_max6x16_fp32_fma<1, 16>,
    },
    {
        gemm_kernel_max6x16_fp32_fma<2, 0>,
        gemm_kernel_max6x16_fp32_fma<2, 8>,
        gemm_kernel_max6x16_fp32_fma<2, 16>,
    },
    {
        gemm_kernel_max6x16_fp32_fma<3, 0>,
        gemm_kernel_max6x16_fp32_fma<3, 8>,
        gemm_kernel_max6x16_fp32_fma<3, 16>,
    },
    {
        gemm_kernel_max6x16_fp32_fma<4, 0>,
        gemm_kernel_max6x16_fp32_fma<4, 8>,
        gemm_kernel_max6x16_fp32_fma<4, 16>,
    },
    {
        gemm_kernel_max6x16_fp32_fma<5, 0>,
        gemm_kernel_max6x16_fp32_fma<5, 8>,
        gemm_kernel_max6x16_fp32_fma<5, 16>,
    },
    {
        gemm_kernel_max6x16_fp32_fma<6, 0>,
        gemm_kernel_max6x16_fp32_fma<6, 8>,
        gemm_kernel_6x16_fp32_fma,
    },
};

}}} // namespace ppl::kernel::x86