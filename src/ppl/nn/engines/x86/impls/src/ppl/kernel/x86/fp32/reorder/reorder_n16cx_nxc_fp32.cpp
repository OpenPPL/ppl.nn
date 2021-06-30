#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reorder_n16cx_nxc_fp32(
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
                ldst[c + 0]  = lsrc[c * X + 0];
                ldst[c + 1]  = lsrc[c * X + 1];
                ldst[c + 2]  = lsrc[c * X + 2];
                ldst[c + 3]  = lsrc[c * X + 3];
                ldst[c + 4]  = lsrc[c * X + 4];
                ldst[c + 5]  = lsrc[c * X + 5];
                ldst[c + 6]  = lsrc[c * X + 6];
                ldst[c + 7]  = lsrc[c * X + 7];
                ldst[c + 8]  = lsrc[c * X + 8];
                ldst[c + 9]  = lsrc[c * X + 9];
                ldst[c + 10] = lsrc[c * X + 10];
                ldst[c + 11] = lsrc[c * X + 11];
                ldst[c + 12] = lsrc[c * X + 12];
                ldst[c + 13] = lsrc[c * X + 13];
                ldst[c + 14] = lsrc[c * X + 14];
                ldst[c + 15] = lsrc[c * X + 15];
            }
            for (; c < channels; c++) {
                ldst[c] = lsrc[round_c * X + c - round_c];
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86