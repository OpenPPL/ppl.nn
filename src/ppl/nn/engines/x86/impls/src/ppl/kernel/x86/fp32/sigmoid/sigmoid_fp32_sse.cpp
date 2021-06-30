#include <nmmintrin.h>
#include <math.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

static inline __m128 _sse_sigmoid_ps(__m128 value)
{
    value = _mm_max_ps(_mm_set1_ps(-18.0f), value);
    value = _mm_min_ps(_mm_set1_ps(18.0f), value);

    __m128 value_squared = _mm_mul_ps(value, value);

    __m128 p;
    p = _mm_mul_ps(value_squared, _mm_set1_ps(4.37031012579801e-11f));
    p = _mm_add_ps(p, _mm_set1_ps(1.15627324459942e-07f));
    p = _mm_mul_ps(p, value_squared);
    p = _mm_add_ps(p, _mm_set1_ps(6.08574864600143e-05f));
    p = _mm_mul_ps(p, value_squared);
    p = _mm_add_ps(p, _mm_set1_ps(8.51377133304701e-03f));
    p = _mm_mul_ps(p, value_squared);
    p = _mm_add_ps(p, _mm_set1_ps(2.48287947061529e-01f));
    p = _mm_mul_ps(p, value);

    __m128 q;
    q = _mm_mul_ps(value_squared, _mm_set1_ps(6.10247389755681e-13f));
    q = _mm_add_ps(q, _mm_set1_ps(5.76102136993427e-09f));
    q = _mm_mul_ps(q, value_squared);
    q = _mm_add_ps(q, _mm_set1_ps(6.29106785017040e-06f));
    q = _mm_mul_ps(q, value_squared);
    q = _mm_add_ps(q, _mm_set1_ps(1.70198817374094e-03f));
    q = _mm_mul_ps(q, value_squared);
    q = _mm_add_ps(q, _mm_set1_ps(1.16817656904453e-01f));
    q = _mm_mul_ps(q, value_squared);
    q = _mm_add_ps(q, _mm_set1_ps(9.93151921023180e-01f));

    __m128 dst = _mm_add_ps(_mm_div_ps(p, q), _mm_set1_ps(0.5f));
    return dst;
}

ppl::common::RetCode sigmoid_fp32_sse(
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
        _mm_storeu_ps(y + i + 0, _sse_sigmoid_ps(src0));
        _mm_storeu_ps(y + i + 4, _sse_sigmoid_ps(src1));
        _mm_storeu_ps(y + i + 8, _sse_sigmoid_ps(src2));
        _mm_storeu_ps(y + i + 12, _sse_sigmoid_ps(src3));
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        y[i] = 1.0f / (expf(-x[i]) + 1.0f);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
