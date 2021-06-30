#include "relation_int64_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode equal_eltwise_int64(
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst)
{
    return relation_eltwise_binary_op_int64<RELATION_LESS>(dst_shape, src0, src1, dst);
}

ppl::common::RetCode equal_ndarray_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst)
{
    return relation_ndarray_binary_op_int64<RELATION_EQUAL>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
}

}}}; // namespace ppl::kernel::x86