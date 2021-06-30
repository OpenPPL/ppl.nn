#include "ppl/kernel/x86/common/tile/tile_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode tile_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *repeats,
    float *dst)
{
    return tile_ndarray<float>(src_shape, dst_shape, src, repeats, dst);
}

}}}; // namespace ppl::kernel::x86
