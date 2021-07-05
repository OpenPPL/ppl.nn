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

#include "ppl/kernel/x86/fp32/gemm_v2/avx512/kernel/gemm_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

const int32_t simd_w = 16;

template <int32_t m_len, int32_t n_len>
void gemm_kernel_max14x32_fp32_avx512(
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

    __m512 va;
    __m512 vb0, vb1;
    __m512 vc00, vc01;
    __m512 vc10, vc11;
    __m512 vc20, vc21;
    __m512 vc30, vc31;
    __m512 vc40, vc41;
    __m512 vc50, vc51;
    __m512 vc60, vc61;
    __m512 vc70, vc71;
    __m512 vc80, vc81;
    __m512 vc90, vc91;
    __m512 vc100, vc101;
    __m512 vc110, vc111;
    __m512 vc120, vc121;
    __m512 vc130, vc131;

    // load C
    if (m_len >= 1) {
        if (n_len > 0 * simd_w)
            vc00 = _mm512_loadu_ps(C + 0 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc01 = _mm512_loadu_ps(C + 0 * ldc + 1 * simd_w);
    }
    if (m_len >= 2) {
        if (n_len > 0 * simd_w)
            vc10 = _mm512_loadu_ps(C + 1 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc11 = _mm512_loadu_ps(C + 1 * ldc + 1 * simd_w);
    }
    if (m_len >= 3) {
        if (n_len > 0 * simd_w)
            vc20 = _mm512_loadu_ps(C + 2 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc21 = _mm512_loadu_ps(C + 2 * ldc + 1 * simd_w);
    }
    if (m_len >= 4) {
        if (n_len > 0 * simd_w)
            vc30 = _mm512_loadu_ps(C + 3 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc31 = _mm512_loadu_ps(C + 3 * ldc + 1 * simd_w);
    }
    if (m_len >= 5) {
        if (n_len > 0 * simd_w)
            vc40 = _mm512_loadu_ps(C + 4 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc41 = _mm512_loadu_ps(C + 4 * ldc + 1 * simd_w);
    }
    if (m_len >= 6) {
        if (n_len > 0 * simd_w)
            vc50 = _mm512_loadu_ps(C + 5 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc51 = _mm512_loadu_ps(C + 5 * ldc + 1 * simd_w);
    }
    if (m_len >= 7) {
        if (n_len > 0 * simd_w)
            vc60 = _mm512_loadu_ps(C + 6 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc61 = _mm512_loadu_ps(C + 6 * ldc + 1 * simd_w);
    }
    if (m_len >= 8) {
        if (n_len > 0 * simd_w)
            vc70 = _mm512_loadu_ps(C + 7 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc71 = _mm512_loadu_ps(C + 7 * ldc + 1 * simd_w);
    }
    if (m_len >= 9) {
        if (n_len > 0 * simd_w)
            vc80 = _mm512_loadu_ps(C + 8 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc81 = _mm512_loadu_ps(C + 8 * ldc + 1 * simd_w);
    }
    if (m_len >= 10) {
        if (n_len > 0 * simd_w)
            vc90 = _mm512_loadu_ps(C + 9 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc91 = _mm512_loadu_ps(C + 9 * ldc + 1 * simd_w);
    }
    if (m_len >= 11) {
        if (n_len > 0 * simd_w)
            vc100 = _mm512_loadu_ps(C + 10 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc101 = _mm512_loadu_ps(C + 10 * ldc + 1 * simd_w);
    }
    if (m_len >= 12) {
        if (n_len > 0 * simd_w)
            vc110 = _mm512_loadu_ps(C + 11 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc111 = _mm512_loadu_ps(C + 11 * ldc + 1 * simd_w);
    }
    if (m_len >= 13) {
        if (n_len > 0 * simd_w)
            vc120 = _mm512_loadu_ps(C + 12 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc121 = _mm512_loadu_ps(C + 12 * ldc + 1 * simd_w);
    }
    if (m_len >= 14) {
        if (n_len > 0 * simd_w)
            vc130 = _mm512_loadu_ps(C + 13 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc131 = _mm512_loadu_ps(C + 13 * ldc + 1 * simd_w);
    }

    const float* a_ptr = A;
    const float* b_ptr = B;

    for (int32_t k = 0; k < k_len; ++k) {
        if (n_len > 0 * simd_w)
            vb0 = _mm512_loadu_ps(b_ptr);
        if (n_len > 1 * simd_w)
            vb1 = _mm512_loadu_ps(b_ptr + simd_w);
        b_ptr += ldb;

        if (m_len >= 1) {
            va = _mm512_set1_ps(a_ptr[0]);
            if (n_len > 0 * simd_w)
                vc00 = _mm512_fmadd_ps(va, vb0, vc00);
            if (n_len > 1 * simd_w)
                vc01 = _mm512_fmadd_ps(va, vb1, vc01);
        }
        if (m_len >= 2) {
            va = _mm512_set1_ps(a_ptr[1]);
            if (n_len > 0 * simd_w)
                vc10 = _mm512_fmadd_ps(va, vb0, vc10);
            if (n_len > 1 * simd_w)
                vc11 = _mm512_fmadd_ps(va, vb1, vc11);
        }
        if (m_len >= 3) {
            va = _mm512_set1_ps(a_ptr[2]);
            if (n_len > 0 * simd_w)
                vc20 = _mm512_fmadd_ps(va, vb0, vc20);
            if (n_len > 1 * simd_w)
                vc21 = _mm512_fmadd_ps(va, vb1, vc21);
        }
        if (m_len >= 4) {
            va = _mm512_set1_ps(a_ptr[3]);
            if (n_len > 0 * simd_w)
                vc30 = _mm512_fmadd_ps(va, vb0, vc30);
            if (n_len > 1 * simd_w)
                vc31 = _mm512_fmadd_ps(va, vb1, vc31);
        }
        if (m_len >= 5) {
            va = _mm512_set1_ps(a_ptr[4]);
            if (n_len > 0 * simd_w)
                vc40 = _mm512_fmadd_ps(va, vb0, vc40);
            if (n_len > 1 * simd_w)
                vc41 = _mm512_fmadd_ps(va, vb1, vc41);
        }
        if (m_len >= 6) {
            va = _mm512_set1_ps(a_ptr[5]);
            if (n_len > 0 * simd_w)
                vc50 = _mm512_fmadd_ps(va, vb0, vc50);
            if (n_len > 1 * simd_w)
                vc51 = _mm512_fmadd_ps(va, vb1, vc51);
        }
        if (m_len >= 7) {
            va = _mm512_set1_ps(a_ptr[6]);
            if (n_len > 0 * simd_w)
                vc60 = _mm512_fmadd_ps(va, vb0, vc60);
            if (n_len > 1 * simd_w)
                vc61 = _mm512_fmadd_ps(va, vb1, vc61);
        }
        if (m_len >= 8) {
            va = _mm512_set1_ps(a_ptr[7]);
            if (n_len > 0 * simd_w)
                vc70 = _mm512_fmadd_ps(va, vb0, vc70);
            if (n_len > 1 * simd_w)
                vc71 = _mm512_fmadd_ps(va, vb1, vc71);
        }
        if (m_len >= 9) {
            va = _mm512_set1_ps(a_ptr[8]);
            if (n_len > 0 * simd_w)
                vc80 = _mm512_fmadd_ps(va, vb0, vc80);
            if (n_len > 1 * simd_w)
                vc81 = _mm512_fmadd_ps(va, vb1, vc81);
        }
        if (m_len >= 10) {
            va = _mm512_set1_ps(a_ptr[9]);
            if (n_len > 0 * simd_w)
                vc90 = _mm512_fmadd_ps(va, vb0, vc90);
            if (n_len > 1 * simd_w)
                vc91 = _mm512_fmadd_ps(va, vb1, vc91);
        }
        if (m_len >= 11) {
            va = _mm512_set1_ps(a_ptr[10]);
            if (n_len > 0 * simd_w)
                vc100 = _mm512_fmadd_ps(va, vb0, vc100);
            if (n_len > 1 * simd_w)
                vc101 = _mm512_fmadd_ps(va, vb1, vc101);
        }
        if (m_len >= 12) {
            va = _mm512_set1_ps(a_ptr[11]);
            if (n_len > 0 * simd_w)
                vc110 = _mm512_fmadd_ps(va, vb0, vc110);
            if (n_len > 1 * simd_w)
                vc111 = _mm512_fmadd_ps(va, vb1, vc111);
        }
        if (m_len >= 13) {
            va = _mm512_set1_ps(a_ptr[12]);
            if (n_len > 0 * simd_w)
                vc120 = _mm512_fmadd_ps(va, vb0, vc120);
            if (n_len > 1 * simd_w)
                vc121 = _mm512_fmadd_ps(va, vb1, vc121);
        }
        if (m_len >= 14) {
            va = _mm512_set1_ps(a_ptr[13]);
            if (n_len > 0 * simd_w)
                vc130 = _mm512_fmadd_ps(va, vb0, vc130);
            if (n_len > 1 * simd_w)
                vc131 = _mm512_fmadd_ps(va, vb1, vc131);
        }

        a_ptr += lda;
    }

    // store C
    if (m_len >= 1) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 0 * ldc + 0 * simd_w, vc00);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 0 * ldc + 1 * simd_w, vc01);
    }
    if (m_len >= 2) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 1 * ldc + 0 * simd_w, vc10);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 1 * ldc + 1 * simd_w, vc11);
    }
    if (m_len >= 3) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 2 * ldc + 0 * simd_w, vc20);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 2 * ldc + 1 * simd_w, vc21);
    }
    if (m_len >= 4) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 3 * ldc + 0 * simd_w, vc30);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 3 * ldc + 1 * simd_w, vc31);
    }
    if (m_len >= 5) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 4 * ldc + 0 * simd_w, vc40);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 4 * ldc + 1 * simd_w, vc41);
    }
    if (m_len >= 6) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 5 * ldc + 0 * simd_w, vc50);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 5 * ldc + 1 * simd_w, vc51);
    }
    if (m_len >= 7) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 6 * ldc + 0 * simd_w, vc60);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 6 * ldc + 1 * simd_w, vc61);
    }
    if (m_len >= 8) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 7 * ldc + 0 * simd_w, vc70);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 7 * ldc + 1 * simd_w, vc71);
    }
    if (m_len >= 9) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 8 * ldc + 0 * simd_w, vc80);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 8 * ldc + 1 * simd_w, vc81);
    }
    if (m_len >= 10) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 9 * ldc + 0 * simd_w, vc90);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 9 * ldc + 1 * simd_w, vc91);
    }
    if (m_len >= 11) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 10 * ldc + 0 * simd_w, vc100);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 10 * ldc + 1 * simd_w, vc101);
    }
    if (m_len >= 12) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 11 * ldc + 0 * simd_w, vc110);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 11 * ldc + 1 * simd_w, vc111);
    }
    if (m_len >= 13) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 12 * ldc + 0 * simd_w, vc120);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 12 * ldc + 1 * simd_w, vc121);
    }
    if (m_len >= 14) {
        if (n_len > 0 * simd_w)
            _mm512_storeu_ps(C + 13 * ldc + 0 * simd_w, vc130);
        if (n_len > 1 * simd_w)
            _mm512_storeu_ps(C + 13 * ldc + 1 * simd_w, vc131);
    }
}

void gemm_kernel_14x32_fp32_avx512(
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

    __m512 va;
    __m512 vb0, vb1;
    __m512 vc00, vc01;
    __m512 vc10, vc11;
    __m512 vc20, vc21;
    __m512 vc30, vc31;
    __m512 vc40, vc41;
    __m512 vc50, vc51;
    __m512 vc60, vc61;
    __m512 vc70, vc71;
    __m512 vc80, vc81;
    __m512 vc90, vc91;
    __m512 vc100, vc101;
    __m512 vc110, vc111;
    __m512 vc120, vc121;
    __m512 vc130, vc131;

    // load C
    vc00  = _mm512_loadu_ps(C + 0 * ldc + 0 * simd_w);
    vc01  = _mm512_loadu_ps(C + 0 * ldc + 1 * simd_w);
    vc10  = _mm512_loadu_ps(C + 1 * ldc + 0 * simd_w);
    vc11  = _mm512_loadu_ps(C + 1 * ldc + 1 * simd_w);
    vc20  = _mm512_loadu_ps(C + 2 * ldc + 0 * simd_w);
    vc21  = _mm512_loadu_ps(C + 2 * ldc + 1 * simd_w);
    vc30  = _mm512_loadu_ps(C + 3 * ldc + 0 * simd_w);
    vc31  = _mm512_loadu_ps(C + 3 * ldc + 1 * simd_w);
    vc40  = _mm512_loadu_ps(C + 4 * ldc + 0 * simd_w);
    vc41  = _mm512_loadu_ps(C + 4 * ldc + 1 * simd_w);
    vc50  = _mm512_loadu_ps(C + 5 * ldc + 0 * simd_w);
    vc51  = _mm512_loadu_ps(C + 5 * ldc + 1 * simd_w);
    vc60  = _mm512_loadu_ps(C + 6 * ldc + 0 * simd_w);
    vc61  = _mm512_loadu_ps(C + 6 * ldc + 1 * simd_w);
    vc70  = _mm512_loadu_ps(C + 7 * ldc + 0 * simd_w);
    vc71  = _mm512_loadu_ps(C + 7 * ldc + 1 * simd_w);
    vc80  = _mm512_loadu_ps(C + 8 * ldc + 0 * simd_w);
    vc81  = _mm512_loadu_ps(C + 8 * ldc + 1 * simd_w);
    vc90  = _mm512_loadu_ps(C + 9 * ldc + 0 * simd_w);
    vc91  = _mm512_loadu_ps(C + 9 * ldc + 1 * simd_w);
    vc100 = _mm512_loadu_ps(C + 10 * ldc + 0 * simd_w);
    vc101 = _mm512_loadu_ps(C + 10 * ldc + 1 * simd_w);
    vc110 = _mm512_loadu_ps(C + 11 * ldc + 0 * simd_w);
    vc111 = _mm512_loadu_ps(C + 11 * ldc + 1 * simd_w);
    vc120 = _mm512_loadu_ps(C + 12 * ldc + 0 * simd_w);
    vc121 = _mm512_loadu_ps(C + 12 * ldc + 1 * simd_w);
    vc130 = _mm512_loadu_ps(C + 13 * ldc + 0 * simd_w);
    vc131 = _mm512_loadu_ps(C + 13 * ldc + 1 * simd_w);

    const float* a_ptr = A;
    const float* b_ptr = B;

    // TODO: asm code here

    for (int32_t k = 0; k < k_len; ++k) {
        vb0 = _mm512_loadu_ps(b_ptr);
        vb1 = _mm512_loadu_ps(b_ptr + simd_w);
        b_ptr += ldb;

        va   = _mm512_set1_ps(a_ptr[0]);
        vc00 = _mm512_fmadd_ps(va, vb0, vc00);
        vc01 = _mm512_fmadd_ps(va, vb1, vc01);

        va   = _mm512_set1_ps(a_ptr[1]);
        vc10 = _mm512_fmadd_ps(va, vb0, vc10);
        vc11 = _mm512_fmadd_ps(va, vb1, vc11);

        va   = _mm512_set1_ps(a_ptr[2]);
        vc20 = _mm512_fmadd_ps(va, vb0, vc20);
        vc21 = _mm512_fmadd_ps(va, vb1, vc21);

        va   = _mm512_set1_ps(a_ptr[3]);
        vc30 = _mm512_fmadd_ps(va, vb0, vc30);
        vc31 = _mm512_fmadd_ps(va, vb1, vc31);

        va   = _mm512_set1_ps(a_ptr[4]);
        vc40 = _mm512_fmadd_ps(va, vb0, vc40);
        vc41 = _mm512_fmadd_ps(va, vb1, vc41);

        va   = _mm512_set1_ps(a_ptr[5]);
        vc50 = _mm512_fmadd_ps(va, vb0, vc50);
        vc51 = _mm512_fmadd_ps(va, vb1, vc51);

        va   = _mm512_set1_ps(a_ptr[6]);
        vc60 = _mm512_fmadd_ps(va, vb0, vc60);
        vc61 = _mm512_fmadd_ps(va, vb1, vc61);

        va   = _mm512_set1_ps(a_ptr[7]);
        vc70 = _mm512_fmadd_ps(va, vb0, vc70);
        vc71 = _mm512_fmadd_ps(va, vb1, vc71);

        va   = _mm512_set1_ps(a_ptr[8]);
        vc80 = _mm512_fmadd_ps(va, vb0, vc80);
        vc81 = _mm512_fmadd_ps(va, vb1, vc81);

        va   = _mm512_set1_ps(a_ptr[9]);
        vc90 = _mm512_fmadd_ps(va, vb0, vc90);
        vc91 = _mm512_fmadd_ps(va, vb1, vc91);

        va    = _mm512_set1_ps(a_ptr[10]);
        vc100 = _mm512_fmadd_ps(va, vb0, vc100);
        vc101 = _mm512_fmadd_ps(va, vb1, vc101);

        va    = _mm512_set1_ps(a_ptr[11]);
        vc110 = _mm512_fmadd_ps(va, vb0, vc110);
        vc111 = _mm512_fmadd_ps(va, vb1, vc111);

        va    = _mm512_set1_ps(a_ptr[12]);
        vc120 = _mm512_fmadd_ps(va, vb0, vc120);
        vc121 = _mm512_fmadd_ps(va, vb1, vc121);

        va    = _mm512_set1_ps(a_ptr[13]);
        vc130 = _mm512_fmadd_ps(va, vb0, vc130);
        vc131 = _mm512_fmadd_ps(va, vb1, vc131);

        a_ptr += lda;
    }

    // store C
    _mm512_storeu_ps(C + 0 * ldc + 0 * simd_w, vc00);
    _mm512_storeu_ps(C + 0 * ldc + 1 * simd_w, vc01);
    _mm512_storeu_ps(C + 1 * ldc + 0 * simd_w, vc10);
    _mm512_storeu_ps(C + 1 * ldc + 1 * simd_w, vc11);
    _mm512_storeu_ps(C + 2 * ldc + 0 * simd_w, vc20);
    _mm512_storeu_ps(C + 2 * ldc + 1 * simd_w, vc21);
    _mm512_storeu_ps(C + 3 * ldc + 0 * simd_w, vc30);
    _mm512_storeu_ps(C + 3 * ldc + 1 * simd_w, vc31);
    _mm512_storeu_ps(C + 4 * ldc + 0 * simd_w, vc40);
    _mm512_storeu_ps(C + 4 * ldc + 1 * simd_w, vc41);
    _mm512_storeu_ps(C + 5 * ldc + 0 * simd_w, vc50);
    _mm512_storeu_ps(C + 5 * ldc + 1 * simd_w, vc51);
    _mm512_storeu_ps(C + 6 * ldc + 0 * simd_w, vc60);
    _mm512_storeu_ps(C + 6 * ldc + 1 * simd_w, vc61);
    _mm512_storeu_ps(C + 7 * ldc + 0 * simd_w, vc70);
    _mm512_storeu_ps(C + 7 * ldc + 1 * simd_w, vc71);
    _mm512_storeu_ps(C + 8 * ldc + 0 * simd_w, vc80);
    _mm512_storeu_ps(C + 8 * ldc + 1 * simd_w, vc81);
    _mm512_storeu_ps(C + 9 * ldc + 0 * simd_w, vc90);
    _mm512_storeu_ps(C + 9 * ldc + 1 * simd_w, vc91);
    _mm512_storeu_ps(C + 10 * ldc + 0 * simd_w, vc100);
    _mm512_storeu_ps(C + 10 * ldc + 1 * simd_w, vc101);
    _mm512_storeu_ps(C + 11 * ldc + 0 * simd_w, vc110);
    _mm512_storeu_ps(C + 11 * ldc + 1 * simd_w, vc111);
    _mm512_storeu_ps(C + 12 * ldc + 0 * simd_w, vc120);
    _mm512_storeu_ps(C + 12 * ldc + 1 * simd_w, vc121);
    _mm512_storeu_ps(C + 13 * ldc + 0 * simd_w, vc130);
    _mm512_storeu_ps(C + 13 * ldc + 1 * simd_w, vc131);
}

const gemm_kernel_fp32_avx512_func_type_t gemm_kernel_max14x32_fp32_avx512_func_tab[15][3] = {
    {
        gemm_kernel_max14x32_fp32_avx512<0, 0>,
        gemm_kernel_max14x32_fp32_avx512<0, 16>,
        gemm_kernel_max14x32_fp32_avx512<0, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<1, 0>,
        gemm_kernel_max14x32_fp32_avx512<1, 16>,
        gemm_kernel_max14x32_fp32_avx512<1, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<2, 0>,
        gemm_kernel_max14x32_fp32_avx512<2, 16>,
        gemm_kernel_max14x32_fp32_avx512<2, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<3, 0>,
        gemm_kernel_max14x32_fp32_avx512<3, 16>,
        gemm_kernel_max14x32_fp32_avx512<3, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<4, 0>,
        gemm_kernel_max14x32_fp32_avx512<4, 16>,
        gemm_kernel_max14x32_fp32_avx512<4, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<5, 0>,
        gemm_kernel_max14x32_fp32_avx512<5, 16>,
        gemm_kernel_max14x32_fp32_avx512<5, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<6, 0>,
        gemm_kernel_max14x32_fp32_avx512<6, 16>,
        gemm_kernel_max14x32_fp32_avx512<6, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<7, 0>,
        gemm_kernel_max14x32_fp32_avx512<7, 16>,
        gemm_kernel_max14x32_fp32_avx512<7, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<8, 0>,
        gemm_kernel_max14x32_fp32_avx512<8, 16>,
        gemm_kernel_max14x32_fp32_avx512<8, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<9, 0>,
        gemm_kernel_max14x32_fp32_avx512<9, 16>,
        gemm_kernel_max14x32_fp32_avx512<9, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<10, 0>,
        gemm_kernel_max14x32_fp32_avx512<10, 16>,
        gemm_kernel_max14x32_fp32_avx512<10, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<11, 0>,
        gemm_kernel_max14x32_fp32_avx512<11, 16>,
        gemm_kernel_max14x32_fp32_avx512<11, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<12, 0>,
        gemm_kernel_max14x32_fp32_avx512<12, 16>,
        gemm_kernel_max14x32_fp32_avx512<12, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<13, 0>,
        gemm_kernel_max14x32_fp32_avx512<13, 16>,
        gemm_kernel_max14x32_fp32_avx512<13, 32>,
    },
    {
        gemm_kernel_max14x32_fp32_avx512<14, 0>,
        gemm_kernel_max14x32_fp32_avx512<14, 16>,
        gemm_kernel_14x32_fp32_avx512,
    },
};

}}} // namespace ppl::kernel::x86