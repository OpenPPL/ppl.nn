#ifndef __ST_PPL_KERNEL_X86_INT64_TRANSPOSE_AVX_TRANSPOSE_INT64_AVX_H_
#define __ST_PPL_KERNEL_X86_INT64_TRANSPOSE_AVX_TRANSPOSE_INT64_AVX_H_

#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

#define TRANSPOSE_4X4_INT64_AVX_MACRO()                  \
    do {                                                 \
        ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);           \
        ymm5 = _mm256_unpackhi_pd(ymm0, ymm1);           \
        ymm6 = _mm256_unpacklo_pd(ymm2, ymm3);           \
        ymm7 = _mm256_unpackhi_pd(ymm2, ymm3);           \
        ymm0 = _mm256_permute2f128_pd(ymm4, ymm6, 0x20); \
        ymm1 = _mm256_permute2f128_pd(ymm4, ymm6, 0x31); \
        ymm2 = _mm256_permute2f128_pd(ymm5, ymm7, 0x20); \
        ymm3 = _mm256_permute2f128_pd(ymm5, ymm7, 0x31); \
    } while (false)

inline void transpose_4x4_int64_avx(
    const int64_t *src,
    const int64_t src_stride,
    const int64_t dst_stride,
    int64_t *dst)
{
    __m256d ymm0, ymm1, ymm2, ymm3;
    __m256d ymm4, ymm5, ymm6, ymm7;
    ymm0 = _mm256_loadu_pd((const double *)src + 0 * src_stride);
    ymm1 = _mm256_loadu_pd((const double *)src + 1 * src_stride);
    ymm2 = _mm256_loadu_pd((const double *)src + 2 * src_stride);
    ymm3 = _mm256_loadu_pd((const double *)src + 3 * src_stride);

    TRANSPOSE_4X4_INT64_AVX_MACRO();

    _mm256_storeu_pd((double *)dst + 0 * dst_stride, ymm0);
    _mm256_storeu_pd((double *)dst + 1 * dst_stride, ymm1);
    _mm256_storeu_pd((double *)dst + 2 * dst_stride, ymm2);
    _mm256_storeu_pd((double *)dst + 3 * dst_stride, ymm3);
}

}}}; // namespace ppl::kernel::x86

#endif
