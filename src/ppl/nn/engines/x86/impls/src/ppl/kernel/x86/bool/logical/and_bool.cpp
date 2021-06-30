#include "logical_bool_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode and_eltwise_bool(
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *src0,
    const uint8_t *src1,
    uint8_t *dst)
{
    return logical_eltwise_binary_op_bool<LOGICAL_AND>(dst_shape, src0, src1, dst);
}

ppl::common::RetCode and_ndarray_bool(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *src0,
    const uint8_t *src1,
    uint8_t *dst)
{
    return logical_ndarray_binary_op_bool<LOGICAL_AND>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
}

}}}; // namespace ppl::kernel::x86