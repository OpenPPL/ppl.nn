#include <nmmintrin.h>

#include "ppl/kernel/x86/fp32/gemm_v2/sse/kernel/gemm_kernel_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

const int32_t simd_w = 4;

template <int32_t m_len, int32_t n_len>
void gemm_kernel_max4x12_fp32_sse(
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

    __m128 va;
    __m128 vb0, vb1, vb2;
    __m128 vc00, vc01, vc02;
    __m128 vc10, vc11, vc12;
    __m128 vc20, vc21, vc22;
    __m128 vc30, vc31, vc32;

    // load C
    if (m_len >= 1) {
        if (n_len > 0 * simd_w)
            vc00 = _mm_loadu_ps(C + 0 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc01 = _mm_loadu_ps(C + 0 * ldc + 1 * simd_w);
        if (n_len > 2 * simd_w)
            vc02 = _mm_loadu_ps(C + 0 * ldc + 2 * simd_w);
    }
    if (m_len >= 2) {
        if (n_len > 0 * simd_w)
            vc10 = _mm_loadu_ps(C + 1 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc11 = _mm_loadu_ps(C + 1 * ldc + 1 * simd_w);
        if (n_len > 2 * simd_w)
            vc12 = _mm_loadu_ps(C + 1 * ldc + 2 * simd_w);
    }
    if (m_len >= 3) {
        if (n_len > 0 * simd_w)
            vc20 = _mm_loadu_ps(C + 2 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc21 = _mm_loadu_ps(C + 2 * ldc + 1 * simd_w);
        if (n_len > 2 * simd_w)
            vc22 = _mm_loadu_ps(C + 2 * ldc + 2 * simd_w);
    }
    if (m_len >= 4) {
        if (n_len > 0 * simd_w)
            vc30 = _mm_loadu_ps(C + 3 * ldc + 0 * simd_w);
        if (n_len > 1 * simd_w)
            vc31 = _mm_loadu_ps(C + 3 * ldc + 1 * simd_w);
        if (n_len > 2 * simd_w)
            vc32 = _mm_loadu_ps(C + 3 * ldc + 2 * simd_w);
    }

    const float* a_ptr = A;
    const float* b_ptr = B;

    for (int32_t k = 0; k < k_len; ++k) {
        if (n_len > 0 * simd_w)
            vb0 = _mm_loadu_ps(b_ptr);
        if (n_len > 1 * simd_w)
            vb1 = _mm_loadu_ps(b_ptr + simd_w);
        if (n_len > 2 * simd_w)
            vb2 = _mm_loadu_ps(b_ptr + 2 * simd_w);
        b_ptr += ldb;

        if (m_len >= 1) {
            va = _mm_set1_ps(a_ptr[0]);
            if (n_len > 0 * simd_w)
                vc00 = _mm_add_ps(vc00, _mm_mul_ps(va, vb0));
            if (n_len > 1 * simd_w)
                vc01 = _mm_add_ps(vc01, _mm_mul_ps(va, vb1));
            if (n_len > 2 * simd_w)
                vc02 = _mm_add_ps(vc02, _mm_mul_ps(va, vb2));
        }
        if (m_len >= 2) {
            va = _mm_set1_ps(a_ptr[1]);
            if (n_len > 0 * simd_w)
                vc10 = _mm_add_ps(vc10, _mm_mul_ps(va, vb0));
            if (n_len > 1 * simd_w)
                vc11 = _mm_add_ps(vc11, _mm_mul_ps(va, vb1));
            if (n_len > 2 * simd_w)
                vc12 = _mm_add_ps(vc12, _mm_mul_ps(va, vb2));
        }
        if (m_len >= 3) {
            va = _mm_set1_ps(a_ptr[2]);
            if (n_len > 0 * simd_w)
                vc20 = _mm_add_ps(vc20, _mm_mul_ps(va, vb0));
            if (n_len > 1 * simd_w)
                vc21 = _mm_add_ps(vc21, _mm_mul_ps(va, vb1));
            if (n_len > 2 * simd_w)
                vc22 = _mm_add_ps(vc22, _mm_mul_ps(va, vb2));
        }
        if (m_len >= 4) {
            va = _mm_set1_ps(a_ptr[3]);
            if (n_len > 0 * simd_w)
                vc30 = _mm_add_ps(vc30, _mm_mul_ps(va, vb0));
            if (n_len > 1 * simd_w)
                vc31 = _mm_add_ps(vc31, _mm_mul_ps(va, vb1));
            if (n_len > 2 * simd_w)
                vc32 = _mm_add_ps(vc32, _mm_mul_ps(va, vb2));
        }

        a_ptr += lda;
    }

    // store C
    if (m_len >= 1) {
        if (n_len > 0 * simd_w)
            _mm_storeu_ps(C + 0 * ldc + 0 * simd_w, vc00);
        if (n_len > 1 * simd_w)
            _mm_storeu_ps(C + 0 * ldc + 1 * simd_w, vc01);
        if (n_len > 2 * simd_w)
            _mm_storeu_ps(C + 0 * ldc + 2 * simd_w, vc02);
    }
    if (m_len >= 2) {
        if (n_len > 0 * simd_w)
            _mm_storeu_ps(C + 1 * ldc + 0 * simd_w, vc10);
        if (n_len > 1 * simd_w)
            _mm_storeu_ps(C + 1 * ldc + 1 * simd_w, vc11);
        if (n_len > 2 * simd_w)
            _mm_storeu_ps(C + 1 * ldc + 2 * simd_w, vc12);
    }
    if (m_len >= 3) {
        if (n_len > 0 * simd_w)
            _mm_storeu_ps(C + 2 * ldc + 0 * simd_w, vc20);
        if (n_len > 1 * simd_w)
            _mm_storeu_ps(C + 2 * ldc + 1 * simd_w, vc21);
        if (n_len > 2 * simd_w)
            _mm_storeu_ps(C + 2 * ldc + 2 * simd_w, vc22);
    }
    if (m_len >= 4) {
        if (n_len > 0 * simd_w)
            _mm_storeu_ps(C + 3 * ldc + 0 * simd_w, vc30);
        if (n_len > 1 * simd_w)
            _mm_storeu_ps(C + 3 * ldc + 1 * simd_w, vc31);
        if (n_len > 2 * simd_w)
            _mm_storeu_ps(C + 3 * ldc + 2 * simd_w, vc32);
    }
}

void gemm_kernel_4x12_fp32_sse(
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

    __m128 va;
    __m128 vb0, vb1, vb2;
    __m128 vc00, vc01, vc02;
    __m128 vc10, vc11, vc12;
    __m128 vc20, vc21, vc22;
    __m128 vc30, vc31, vc32;

    // load C
    vc00 = _mm_loadu_ps(C + 0 * ldc + 0 * simd_w);
    vc01 = _mm_loadu_ps(C + 0 * ldc + 1 * simd_w);
    vc02 = _mm_loadu_ps(C + 0 * ldc + 2 * simd_w);
    vc10 = _mm_loadu_ps(C + 1 * ldc + 0 * simd_w);
    vc11 = _mm_loadu_ps(C + 1 * ldc + 1 * simd_w);
    vc12 = _mm_loadu_ps(C + 1 * ldc + 2 * simd_w);
    vc20 = _mm_loadu_ps(C + 2 * ldc + 0 * simd_w);
    vc21 = _mm_loadu_ps(C + 2 * ldc + 1 * simd_w);
    vc22 = _mm_loadu_ps(C + 2 * ldc + 2 * simd_w);
    vc30 = _mm_loadu_ps(C + 3 * ldc + 0 * simd_w);
    vc31 = _mm_loadu_ps(C + 3 * ldc + 1 * simd_w);
    vc32 = _mm_loadu_ps(C + 3 * ldc + 2 * simd_w);

    const float* a_ptr = A;
    const float* b_ptr = B;

    // TODO: asm code here

    for (int32_t k = 0; k < k_len; ++k) {
        vb0 = _mm_loadu_ps(b_ptr);
        vb1 = _mm_loadu_ps(b_ptr + simd_w);
        vb2 = _mm_loadu_ps(b_ptr + 2 * simd_w);
        b_ptr += ldb;

        va   = _mm_set1_ps(a_ptr[0]);
        vc00 = _mm_add_ps(vc00, _mm_mul_ps(va, vb0));
        vc01 = _mm_add_ps(vc01, _mm_mul_ps(va, vb1));
        vc02 = _mm_add_ps(vc02, _mm_mul_ps(va, vb2));

        va   = _mm_set1_ps(a_ptr[1]);
        vc10 = _mm_add_ps(vc10, _mm_mul_ps(va, vb0));
        vc11 = _mm_add_ps(vc11, _mm_mul_ps(va, vb1));
        vc12 = _mm_add_ps(vc12, _mm_mul_ps(va, vb2));

        va   = _mm_set1_ps(a_ptr[2]);
        vc20 = _mm_add_ps(vc20, _mm_mul_ps(va, vb0));
        vc21 = _mm_add_ps(vc21, _mm_mul_ps(va, vb1));
        vc22 = _mm_add_ps(vc22, _mm_mul_ps(va, vb2));

        va   = _mm_set1_ps(a_ptr[3]);
        vc30 = _mm_add_ps(vc30, _mm_mul_ps(va, vb0));
        vc31 = _mm_add_ps(vc31, _mm_mul_ps(va, vb1));
        vc32 = _mm_add_ps(vc32, _mm_mul_ps(va, vb2));

        a_ptr += lda;
    }

    // store C
    _mm_storeu_ps(C + 0 * ldc + 0 * simd_w, vc00);
    _mm_storeu_ps(C + 0 * ldc + 1 * simd_w, vc01);
    _mm_storeu_ps(C + 0 * ldc + 2 * simd_w, vc02);
    _mm_storeu_ps(C + 1 * ldc + 0 * simd_w, vc10);
    _mm_storeu_ps(C + 1 * ldc + 1 * simd_w, vc11);
    _mm_storeu_ps(C + 1 * ldc + 2 * simd_w, vc12);
    _mm_storeu_ps(C + 2 * ldc + 0 * simd_w, vc20);
    _mm_storeu_ps(C + 2 * ldc + 1 * simd_w, vc21);
    _mm_storeu_ps(C + 2 * ldc + 2 * simd_w, vc22);
    _mm_storeu_ps(C + 3 * ldc + 0 * simd_w, vc30);
    _mm_storeu_ps(C + 3 * ldc + 1 * simd_w, vc31);
    _mm_storeu_ps(C + 3 * ldc + 2 * simd_w, vc32);
}

const gemm_kernel_fp32_sse_func_type_t gemm_kernel_max4x12_fp32_sse_func_tab[5][4] = {
    {
        gemm_kernel_max4x12_fp32_sse<0, 0>,
        gemm_kernel_max4x12_fp32_sse<0, 4>,
        gemm_kernel_max4x12_fp32_sse<0, 8>,
        gemm_kernel_max4x12_fp32_sse<0, 12>,
    },
    {
        gemm_kernel_max4x12_fp32_sse<1, 0>,
        gemm_kernel_max4x12_fp32_sse<1, 4>,
        gemm_kernel_max4x12_fp32_sse<1, 8>,
        gemm_kernel_max4x12_fp32_sse<1, 12>,
    },
    {
        gemm_kernel_max4x12_fp32_sse<2, 0>,
        gemm_kernel_max4x12_fp32_sse<2, 4>,
        gemm_kernel_max4x12_fp32_sse<2, 8>,
        gemm_kernel_max4x12_fp32_sse<2, 12>,
    },
    {
        gemm_kernel_max4x12_fp32_sse<3, 0>,
        gemm_kernel_max4x12_fp32_sse<3, 4>,
        gemm_kernel_max4x12_fp32_sse<3, 8>,
        gemm_kernel_max4x12_fp32_sse<3, 12>,
    },
    {
        gemm_kernel_max4x12_fp32_sse<4, 0>,
        gemm_kernel_max4x12_fp32_sse<4, 4>,
        gemm_kernel_max4x12_fp32_sse<4, 8>,
        gemm_kernel_4x12_fp32_sse,
    },
};

}}} // namespace ppl::kernel::x86