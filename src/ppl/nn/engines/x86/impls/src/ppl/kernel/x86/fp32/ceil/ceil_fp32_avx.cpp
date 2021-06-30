#include <immintrin.h>
#include <math.h>
#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode ceil_fp32_avx(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    float *y)
{
    const int64_t n_elem      = x_shape->GetElementsIncludingPadding();
    const int64_t unroll_n    = 8;
    const int64_t unroll_body = round(n_elem, unroll_n);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_n) {
        __m256 src0 = _mm256_loadu_ps(x + i);
        __m256 dst0 = _mm256_ceil_ps(src0);
        _mm256_storeu_ps(y + i, dst0);
    }
    for (int64_t i = unroll_body; i < n_elem; ++i) {
        y[i] = ceil(x[i]);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86