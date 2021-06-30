#include <float.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode maxpool2d_nchw_with_indices_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    float *dst,
    int64_t *indices)
{
    const int32_t batch    = src_shape->GetDim(0);
    const int32_t channels = src_shape->GetDim(1);
    const int32_t src_h    = src_shape->GetDim(2);
    const int32_t src_w    = src_shape->GetDim(3);
    const int32_t dst_h    = dst_shape->GetDim(2);
    const int32_t dst_w    = dst_shape->GetDim(3);
#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
#else
    PRAGMA_OMP_PARALLEL_FOR()
#endif
    for (int64_t bc = 0; bc < batch * channels; ++bc) {
        for (int64_t oh = 0; oh < dst_h; ++oh) {
            for (int64_t ow = 0; ow < dst_w; ++ow) {
                const float *p_src         = src + bc * src_h * src_w;
                float *p_dst               = dst + bc * dst_h * dst_w;
                int64_t *p_indices         = indices + bc * dst_h * dst_w;
                const int64_t index_offset = bc * src_h * src_w;

                const int64_t pre_ihstart = oh * stride_h - pad_h;
                const int64_t pre_iwstart = ow * stride_w - pad_w;
                const int64_t ihend       = min<int64_t>(pre_ihstart + kernel_h, src_h);
                const int64_t iwend       = min<int64_t>(pre_iwstart + kernel_w, src_w);
                const int64_t ihstart     = max<int64_t>(pre_ihstart, 0);
                const int64_t iwstart     = max<int64_t>(pre_iwstart, 0);

                if (ihstart >= ihend || iwstart >= iwend) {
                    p_dst[oh * dst_w + ow]     = 0.0f;
                    p_indices[oh * dst_w + ow] = 0;
                } else {
                    float max_val     = -FLT_MAX;
                    int64_t max_index = 0;
                    for (int ih = ihstart; ih < ihend; ih++) {
                        for (int iw = iwstart; iw < iwend; iw++) {
                            const int64_t src_index = ih * src_w + iw;
                            const float src_val     = p_src[src_index];
                            if (src_val > max_val) {
                                max_val   = src_val;
                                max_index = src_index;
                            }
                        }
                    }
                    p_dst[oh * dst_w + ow]     = max_val;
                    p_indices[oh * dst_w + ow] = max_index + index_offset;
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
