#include <string.h> // for memcpy

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode max_unpool_nchw_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *indices,
    float *dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t dst_h    = dst_shape->GetDim(2);
    const int64_t dst_w    = dst_shape->GetDim(3);

    memset(dst, 0, batch * channels * dst_h * dst_w * sizeof(float));

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int n = 0; n < batch; n++) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int c = 0; c < channels; c++) {
            const float *p_src       = src + (n * channels + c) * src_h * src_w;
            const int64_t *p_indices = indices + (n * channels + c) * src_h * src_w;
            float *p_dst             = dst + (n * channels + c) * dst_h * dst_w;

            const int64_t indices_offset = c * dst_h * dst_w;
            for (int ih = 0; ih < src_h; ih++) {
                for (int iw = 0; iw < src_w; iw++) {
                    const int64_t src_index = ih * src_w + iw;
                    const float data        = p_src[src_index];
                    const int64_t dst_index = p_indices[src_index] - indices_offset;
                    p_dst[dst_index]        = data;
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
