#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/concat/concat_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode concat_ndarray_int64(
    const ppl::nn::TensorShape **src_shape_list,
    const int64_t **src_list,
    const int32_t num_src,
    const int32_t axis,
    int64_t *dst)
{
    return concat_ndarray<int64_t>(src_shape_list, src_list, num_src, axis, dst);
}

}}}; // namespace ppl::kernel::x86
