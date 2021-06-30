#include <nmmintrin.h>
#include <math.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode sqrt_fp32_sse(
    const ppl::nn::TensorShape *in_shape,
    const float *in,
    float *out)
{
    const int64_t length     = in_shape->GetElementsIncludingPadding();
    const int64_t simd_w     = 4;
    const int64_t unroll_len = round(length, simd_w);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_len; i += simd_w) {
        _mm_storeu_ps(out + i, _mm_sqrt_ps(_mm_loadu_ps(in + i)));
    }
    for (int64_t i = unroll_len; i < length; i++) {
        out[i] = sqrtf(in[i]);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
