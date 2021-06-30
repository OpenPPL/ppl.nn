#include <algorithm>

#include "ppl/kernel/x86/fp32/reduce/avx/reduce_ndarray_fp32_avx.h"
#include "ppl/kernel/x86/fp32/reduce/avx/reduce_n16cx_fp32_avx.h"
#include "ppl/kernel/x86/fp32/reduce/avx/reduce_single_axis_ndarray_fp32_avx.h"

namespace ppl { namespace kernel { namespace x86 {

template <reduce_op_type_t _op>
ppl::common::RetCode reduce_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst)
{
    if (src_shape->GetElementsExcludingPadding() == dst_shape->GetElementsExcludingPadding()) { // no actual reduce happened, just copy
        memcpy(dst, src, src_shape->GetBytesIncludingPadding());
        return ppl::common::RC_SUCCESS;
    }
    if (src_shape->GetDimCount() > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    int32_t real_axes[PPL_X86_TENSOR_MAX_DIMS()] = {0}; // change negative axes to positive &
    // sort axes
    for (int64_t i = 0; i < num_axes; i++) {
        real_axes[i] = axes[i] >= 0 ? axes[i] : axes[i] + src_shape->GetDimCount();
    }
    std::sort(real_axes, real_axes + num_axes);

    bool continous_reduce_axis = true;
    for (int64_t i = 0; i < num_axes - 1; i++) {
        if (real_axes[i + 1] - real_axes[i] != 1) {
            continous_reduce_axis = false;
            break;
        }
    }

    if (src_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        if (continous_reduce_axis) { // continous_reduce_axis, use special optimized code
            return reduce_single_axis_ndarray_fp32_avx<_op>(src_shape, dst_shape, src, real_axes, num_axes, dst);
        } else {
            return reduce_ndarray_fp32_avx<_op>(src_shape, dst_shape, src, real_axes, num_axes, dst);
        }
    } else if (src_shape->GetDataFormat() == ppl::common::DATAFORMAT_N16CX) {
        return reduce_n16cx_fp32_avx<_op>(src_shape, dst_shape, src, real_axes, num_axes, 1, dst);
    }

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode reduce_max_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst)
{
    return reduce_fp32_avx<REDUCE_MAX>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_min_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst)
{
    return reduce_fp32_avx<REDUCE_MIN>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_mean_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst)
{
    return reduce_fp32_avx<REDUCE_MEAN>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_sum_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst)
{
    return reduce_fp32_avx<REDUCE_SUM>(src_shape, dst_shape, src, axes, num_axes, dst);
}

}}}; // namespace ppl::kernel::x86
