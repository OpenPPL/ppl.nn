#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reorder_n16cx_nxc_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    float *dst)
{
    if (src_shape->GetDataFormat() != ppl::common::DATAFORMAT_N16CX ||
        src_shape->GetDimCount() < 3) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t X        = src_shape->GetElementsExcludingPadding() / batch / channels;

    const int64_t simd_w  = 8;
    const int64_t c_blk   = 16;
    const int64_t pad_c   = round_up(channels, c_blk);
    const int64_t round_c = round(channels, c_blk);

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t n = 0; n < batch; n++) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t x = 0; x < X; x++) {
            float *ldst       = dst + n * X * channels + x * channels;
            const float *lsrc = src + n * pad_c * X + x * c_blk;

            int64_t c = 0;
            for (; c + c_blk <= channels; c += c_blk) {
                __m256 v_src_0 = _mm256_loadu_ps(lsrc + c * X + 0 * simd_w);
                __m256 v_src_1 = _mm256_loadu_ps(lsrc + c * X + 1 * simd_w);
                _mm256_storeu_ps(ldst + c + 0 * simd_w, v_src_0);
                _mm256_storeu_ps(ldst + c + 1 * simd_w, v_src_1);
            }
            for (; c < channels; c++) {
                ldst[c] = lsrc[round_c * X + c - round_c];
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86