#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/split/split_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode split_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape **dst_shape_list,
    const float *src,
    const int32_t slice_axis,
    const int32_t num_dst,
    float **dst_list)
{
    return split_ndarray<float>(src_shape, dst_shape_list, src, slice_axis, num_dst, dst_list);
}

}}} // namespace ppl::kernel::x86
