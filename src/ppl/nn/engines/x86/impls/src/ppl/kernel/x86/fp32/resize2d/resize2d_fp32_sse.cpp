#include <nmmintrin.h>
#include <string.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reisze2d_ndarray_asymmetric_nearest_floor_2times_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst)
{
    const int64_t num_imgs = src_shape->GetDim(0) * src_shape->GetDim(1);
    const int64_t src_h = src_shape->GetDim(2);
    const int64_t src_w = src_shape->GetDim(3);
    const int64_t dst_h = dst_shape->GetDim(2);
    const int64_t dst_w = dst_shape->GetDim(3);
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t ni = 0; ni < num_imgs; ++ni) {
        const float *l_src = src + ni * src_h * src_w;
        float *l_dst       = dst + ni * dst_h * dst_w;
        for (int64_t oh = 0; oh < src_h; ++oh) {
            float *t_dst0      = l_dst + (oh * 2 + 0) * dst_w;
            float *t_dst1      = l_dst + (oh * 2 + 1) * dst_w;
            const float *t_src = l_src + oh * src_w;
            int64_t ow         = 0;
            for (; ow < src_w - 4; ow += 4) {
                __m128 src0 = _mm_loadu_ps(t_src + ow);
                __m128 src1 = src0;
                src1        = _mm_unpacklo_ps(src0, src1);
                src0        = _mm_unpackhi_ps(src0, src0);
                _mm_storeu_ps(t_dst0 + ow * 2 + 0, src1);
                _mm_storeu_ps(t_dst0 + ow * 2 + 4, src0);
            }
            for (; ow < src_w; ow++) {
                t_dst0[ow * 2 + 0] = t_src[ow];
                t_dst0[ow * 2 + 1] = t_src[ow];
                t_dst1[ow * 2 + 0] = t_src[ow];
                t_dst1[ow * 2 + 1] = t_src[ow];
            }
            memcpy(t_dst1, t_dst0, dst_w * sizeof(float));
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86
