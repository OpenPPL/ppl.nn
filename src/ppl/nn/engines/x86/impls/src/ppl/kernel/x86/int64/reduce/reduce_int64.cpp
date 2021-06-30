#include <algorithm>

#include "ppl/kernel/x86/int64/reduce/reduce_ndarray_int64.h"
#include "ppl/kernel/x86/int64/reduce/reduce_n16cx_int64.h"

namespace ppl { namespace kernel { namespace x86 {

template <reduce_op_type_t _op>
ppl::common::RetCode reduce_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst)
{
    if (src_shape->GetDimCount() > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    if (src_shape->GetElementsExcludingPadding() ==
        dst_shape->GetElementsExcludingPadding()) { // no actual reduce happened, just copy
        memcpy(dst, src, src_shape->GetBytesIncludingPadding());
        return ppl::common::RC_SUCCESS;
    }

    int32_t real_axes[PPL_X86_TENSOR_MAX_DIMS()] = {0}; // change negative axes to positive &
    // sort axes
    for (int64_t i = 0; i < num_axes; i++) {
        real_axes[i] = axes[i] >= 0 ? axes[i] : axes[i] + src_shape->GetDimCount();
    }
    std::sort(real_axes, real_axes + num_axes);

    if (src_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        return reduce_ndarray_int64<_op>(src_shape, dst_shape, src, real_axes, num_axes, dst);
    } else if (src_shape->GetDataFormat() == ppl::common::DATAFORMAT_N16CX) {
        return reduce_n16cx_int64<_op>(src, src_shape, dst_shape, real_axes, num_axes, 1, dst);
    }

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode reduce_max_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst)
{
    return reduce_int64<REDUCE_MAX>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_min_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst)
{
    return reduce_int64<REDUCE_MIN>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_mean_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst)
{
    return reduce_int64<REDUCE_MEAN>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_sum_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst)
{
    return reduce_int64<REDUCE_SUM>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_prod_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst)
{
    return reduce_int64<REDUCE_PROD>(src_shape, dst_shape, src, axes, num_axes, dst);
}

}}}; // namespace ppl::kernel::x86
