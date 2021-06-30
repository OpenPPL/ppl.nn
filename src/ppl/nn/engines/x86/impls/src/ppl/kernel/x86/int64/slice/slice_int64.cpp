#include "ppl/kernel/x86/common/slice/slice_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode slice_ndarray_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int64_t *starts,
    const int64_t *steps,
    const int64_t *axes,
    const int64_t axes_num,
    int64_t *dst)
{
    return slice_ndarray_common<int64_t>(src_shape, dst_shape, src, starts, steps, axes, axes_num, dst);
}

}}} // namespace ppl::kernel::x86