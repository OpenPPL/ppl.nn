#include "ppl/kernel/x86/common/where/where_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode where_eltwise_int64(
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *cond,
    const int64_t *src_x,
    const int64_t *src_y,
    int64_t *dst)
{
    return where_eltwise_common<int64_t>(dst_shape, cond, src_x, src_y, dst);
}

ppl::common::RetCode where_ndarray_int64(
    const ppl::nn::TensorShape *cond_shape,
    const ppl::nn::TensorShape *src_x_shape,
    const ppl::nn::TensorShape *src_y_shape,
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *cond,
    const int64_t *src_x,
    const int64_t *src_y,
    int64_t *dst)
{
    return where_ndarray_common<int64_t>(cond_shape, src_x_shape, src_y_shape, dst_shape, cond, src_x, src_y, dst);
}

}}}; // namespace ppl::kernel::x86
