#include <nmmintrin.h>
#include <math.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

static inline __m128 _sse_exp_ps(__m128 x)
{
    __m128 tmp = _mm_setzero_ps(), fx;
    __m128i imm0;
    __m128 one = _mm_set1_ps(1.0f);

    x = _mm_min_ps(x, _mm_set1_ps(88.3762626647949f));
    x = _mm_max_ps(x, _mm_set1_ps(-88.3762626647949f));

    fx = _mm_mul_ps(x, _mm_set1_ps(1.44269504088896341));
    fx = _mm_add_ps(fx, _mm_set1_ps(0.5f));

    tmp = _mm_floor_ps(fx);

    __m128 mask = _mm_cmpgt_ps(tmp, fx);
    mask        = _mm_and_ps(mask, one);
    fx          = _mm_sub_ps(tmp, mask);

    tmp      = _mm_mul_ps(fx, _mm_set1_ps(0.693359375));
    __m128 z = _mm_mul_ps(fx, _mm_set1_ps(-2.12194440e-4));
    x        = _mm_sub_ps(x, tmp);
    x        = _mm_sub_ps(x, z);
    z        = _mm_mul_ps(x, x);

    __m128 y = _mm_set1_ps(1.9875691500E-4);
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(1.3981999507E-3));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(8.3334519073E-3));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(4.1665795894E-2));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(1.6666665459E-1));
    y        = _mm_mul_ps(y, x);
    y        = _mm_add_ps(y, _mm_set1_ps(5.0000001201E-1));
    y        = _mm_mul_ps(y, z);
    y        = _mm_add_ps(y, x);
    y        = _mm_add_ps(y, one);

    imm0         = _mm_cvttps_epi32(fx);
    imm0         = _mm_add_epi32(imm0, _mm_set1_epi32(0x7f));
    imm0         = _mm_slli_epi32(imm0, 23);
    __m128 pow2n = _mm_castsi128_ps(imm0);
    y            = _mm_mul_ps(y, pow2n);
    return y;
}

ppl::common::RetCode exp_fp32_sse(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    float *y)
{
    const int64_t n_elem      = x_shape->GetElementsIncludingPadding();
    const int64_t unroll_n    = 16;
    const int64_t unroll_body = round(n_elem, unroll_n);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        __m128 src0 = _mm_loadu_ps(x + i + 0);
        __m128 src1 = _mm_loadu_ps(x + i + 4);
        __m128 src2 = _mm_loadu_ps(x + i + 8);
        __m128 src3 = _mm_loadu_ps(x + i + 12);
        _mm_storeu_ps(y + i + 0, _sse_exp_ps(src0));
        _mm_storeu_ps(y + i + 4, _sse_exp_ps(src1));
        _mm_storeu_ps(y + i + 8, _sse_exp_ps(src2));
        _mm_storeu_ps(y + i + 12, _sse_exp_ps(src3));
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        y[i] = expf(x[i]);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86