#include "ppl/kernel/x86/common/tile/tile_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode tile_ndarray_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int64_t *repeats,
    int64_t *dst)
{
    return tile_ndarray<int64_t>(src_shape, dst_shape, src, repeats, dst);
}

}}}; // namespace ppl::kernel::x86
