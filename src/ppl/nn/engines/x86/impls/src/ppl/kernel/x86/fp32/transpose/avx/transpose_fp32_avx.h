#ifndef __ST_PPL_KERNEL_X86_FP32_TRANSPOSE_AVX_TRANSPOSE_FP32_AVX_H_
#define __ST_PPL_KERNEL_X86_FP32_TRANSPOSE_AVX_TRANSPOSE_FP32_AVX_H_

#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

#define TRANSPOSE_8X8_FP32_AVX_MACRO()                                    \
    do {                                                                  \
        ymm8  = _mm256_unpacklo_ps(ymm0, ymm1);                           \
        ymm9  = _mm256_unpackhi_ps(ymm0, ymm1);                           \
        ymm10 = _mm256_unpacklo_ps(ymm2, ymm3);                           \
        ymm11 = _mm256_unpackhi_ps(ymm2, ymm3);                           \
        ymm12 = _mm256_unpacklo_ps(ymm4, ymm5);                           \
        ymm13 = _mm256_unpackhi_ps(ymm4, ymm5);                           \
        ymm14 = _mm256_unpacklo_ps(ymm6, ymm7);                           \
        ymm15 = _mm256_unpackhi_ps(ymm6, ymm7);                           \
        ymm0  = _mm256_shuffle_ps(ymm8, ymm10, _MM_SHUFFLE(1, 0, 1, 0));  \
        ymm1  = _mm256_shuffle_ps(ymm8, ymm10, _MM_SHUFFLE(3, 2, 3, 2));  \
        ymm2  = _mm256_shuffle_ps(ymm9, ymm11, _MM_SHUFFLE(1, 0, 1, 0));  \
        ymm3  = _mm256_shuffle_ps(ymm9, ymm11, _MM_SHUFFLE(3, 2, 3, 2));  \
        ymm4  = _mm256_shuffle_ps(ymm12, ymm14, _MM_SHUFFLE(1, 0, 1, 0)); \
        ymm5  = _mm256_shuffle_ps(ymm12, ymm14, _MM_SHUFFLE(3, 2, 3, 2)); \
        ymm6  = _mm256_shuffle_ps(ymm13, ymm15, _MM_SHUFFLE(1, 0, 1, 0)); \
        ymm7  = _mm256_shuffle_ps(ymm13, ymm15, _MM_SHUFFLE(3, 2, 3, 2)); \
        ymm8  = _mm256_permute2f128_ps(ymm0, ymm4, 0x20);                 \
        ymm9  = _mm256_permute2f128_ps(ymm1, ymm5, 0x20);                 \
        ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 0x20);                 \
        ymm11 = _mm256_permute2f128_ps(ymm3, ymm7, 0x20);                 \
        ymm12 = _mm256_permute2f128_ps(ymm0, ymm4, 0x31);                 \
        ymm13 = _mm256_permute2f128_ps(ymm1, ymm5, 0x31);                 \
        ymm14 = _mm256_permute2f128_ps(ymm2, ymm6, 0x31);                 \
        ymm15 = _mm256_permute2f128_ps(ymm3, ymm7, 0x31);                 \
    } while (false)

inline void transpose_8x8_fp32_avx(
    const float *src,
    const int64_t src_stride,
    const int64_t dst_stride,
    float *dst)
{
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
    ymm0 = _mm256_loadu_ps(src + 0 * src_stride);
    ymm1 = _mm256_loadu_ps(src + 1 * src_stride);
    ymm2 = _mm256_loadu_ps(src + 2 * src_stride);
    ymm3 = _mm256_loadu_ps(src + 3 * src_stride);
    ymm4 = _mm256_loadu_ps(src + 4 * src_stride);
    ymm5 = _mm256_loadu_ps(src + 5 * src_stride);
    ymm6 = _mm256_loadu_ps(src + 6 * src_stride);
    ymm7 = _mm256_loadu_ps(src + 7 * src_stride);

    TRANSPOSE_8X8_FP32_AVX_MACRO();

    _mm256_storeu_ps(dst + 0 * dst_stride, ymm8);
    _mm256_storeu_ps(dst + 1 * dst_stride, ymm9);
    _mm256_storeu_ps(dst + 2 * dst_stride, ymm10);
    _mm256_storeu_ps(dst + 3 * dst_stride, ymm11);
    _mm256_storeu_ps(dst + 4 * dst_stride, ymm12);
    _mm256_storeu_ps(dst + 5 * dst_stride, ymm13);
    _mm256_storeu_ps(dst + 6 * dst_stride, ymm14);
    _mm256_storeu_ps(dst + 7 * dst_stride, ymm15);
}

}}}; // namespace ppl::kernel::x86

#endif
