#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/int64/transpose/avx/transpose_int64_avx.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reorder_n16cx_ndarray_int64_avx(
    const ppl::nn::TensorShape *src_shape,
    const int64_t *src,
    int64_t *dst)
{
    if (src_shape->GetDataFormat() != ppl::common::DATAFORMAT_N16CX ||
        src_shape->GetDimCount() < 3) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t X        = src_shape->GetElementsExcludingPadding() / batch / channels;

    const int64_t simd_w   = 4;
    const int64_t c_blk    = 16;
    const int64_t padded_c = round_up(channels, c_blk);

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t b = 0; b < batch; ++b) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t c = 0; c < channels; c += c_blk) {
            for (int64_t x = 0; x < X; x += simd_w) {
                const int64_t c_eff = min<int64_t>(channels - c, c_blk);
                const int64_t x_eff = min<int64_t>(X - x, simd_w);
                for (int64_t mc = 0; mc < c_eff; mc += simd_w) {
                    const int64_t mc_eff = min<int64_t>(c_eff - mc, simd_w);
                    int64_t *ldst        = dst + b * channels * X + (c + mc) * X + x;
                    const int64_t *lsrc  = src + b * padded_c * X + c * X + x * c_blk + mc;
                    if (x_eff == simd_w && c_eff == simd_w) {
                        transpose_4x4_int64_avx(lsrc, c_blk, X, ldst);
                    } else {
                        for (int64_t xx = 0; xx < x_eff; ++xx) {
                            for (int64_t cc = 0; cc < mc_eff; ++cc) {
                                ldst[cc * X + xx] = lsrc[xx * c_blk + cc];
                            }
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
