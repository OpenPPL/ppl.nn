#include <immintrin.h>
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/transpose.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode channel_shuffle_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int32_t group,
    float *dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t mid_c1   = group;
    const int64_t mid_c2   = channels / group;
    int64_t mid_dims[5]   = {batch, mid_c1, mid_c2, src_h, src_w};

    ppl::nn::TensorShape *mid_shape = new ppl::nn::TensorShape();
    mid_shape->Reshape(mid_dims, 5);

    return transpose_ndarray_continous2d_fp32(mid_shape, src, 1, 2, dst);
}

}}} // namespace ppl::kernel::x86
