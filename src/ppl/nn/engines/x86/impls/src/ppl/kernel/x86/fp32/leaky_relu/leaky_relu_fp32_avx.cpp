#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode leaky_relu_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst)
{
    const uint64_t simd_w      = 8;
    const uint64_t unroll_len  = simd_w * 4;
    const uint64_t unroll_body = round(src_shape->GetElementsIncludingPadding(), unroll_len);
    const __m256 v_alpha       = _mm256_set1_ps(alpha);
    const __m256 v_zero        = _mm256_setzero_ps();
    PRAGMA_OMP_PARALLEL_FOR()
    for (uint64_t i = 0; i < unroll_body; i += unroll_len) {
        __m256 v_src0 = _mm256_loadu_ps(src + i + 0 * simd_w);
        __m256 v_src1 = _mm256_loadu_ps(src + i + 1 * simd_w);
        __m256 v_src2 = _mm256_loadu_ps(src + i + 2 * simd_w);
        __m256 v_src3 = _mm256_loadu_ps(src + i + 3 * simd_w);

        __m256 v_ge0 = _mm256_max_ps(v_src0, v_zero);
        __m256 v_ge1 = _mm256_max_ps(v_src1, v_zero);
        __m256 v_ge2 = _mm256_max_ps(v_src2, v_zero);
        __m256 v_ge3 = _mm256_max_ps(v_src3, v_zero);

        __m256 v_le0 = _mm256_mul_ps(_mm256_min_ps(v_src0, v_zero), v_alpha);
        __m256 v_le1 = _mm256_mul_ps(_mm256_min_ps(v_src1, v_zero), v_alpha);
        __m256 v_le2 = _mm256_mul_ps(_mm256_min_ps(v_src2, v_zero), v_alpha);
        __m256 v_le3 = _mm256_mul_ps(_mm256_min_ps(v_src3, v_zero), v_alpha);

        _mm256_storeu_ps(dst + i + 0 * simd_w, _mm256_add_ps(v_ge0, v_le0));
        _mm256_storeu_ps(dst + i + 1 * simd_w, _mm256_add_ps(v_ge1, v_le1));
        _mm256_storeu_ps(dst + i + 2 * simd_w, _mm256_add_ps(v_ge2, v_le2));
        _mm256_storeu_ps(dst + i + 3 * simd_w, _mm256_add_ps(v_ge3, v_le3));
    }
    for (uint64_t i = unroll_body; i < src_shape->GetElementsIncludingPadding(); i++) {
        dst[i] = src[i] >= 0 ? src[i] : alpha * src[i];
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86