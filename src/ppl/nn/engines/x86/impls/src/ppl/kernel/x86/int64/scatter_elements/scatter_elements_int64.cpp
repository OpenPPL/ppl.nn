#include "ppl/kernel/x86/common/scatter_elements/scatter_elements_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode scatter_elements_ndarray_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *indices_shape,
    const int64_t *src,
    const int64_t *indices,
    const int64_t *updates,
    const int64_t axis,
    int64_t *dst) 
{
    return scatter_elements_ndarray_common<int64_t>(src_shape, indices_shape, src, indices, updates, axis, dst);
}

}}} // namespace ppl::kernel::x86