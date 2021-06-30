#include "ppl/kernel/x86/common/argmax/argmax_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode argmax_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    int64_t *dst)
{
    return argmax_ndarray<float>(src_shape, src, axis, dst);
}

}}}; // namespace ppl::kernel::x86
