#include "ppl/kernel/x86/common/transpose/transpose_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode transpose_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *perm,
    float *dst)
{
    return transpose_ndarray<float>(src_shape, dst_shape, perm, src, dst);
}

ppl::common::RetCode transpose_ndarray_continous2d_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const uint32_t axis0,
    const uint32_t axis1,
    float *dst)
{
    return transpose_ndarray_continous2d<float>(src_shape, axis0, axis1, src, dst);
}

}}}; // namespace ppl::kernel::x86
