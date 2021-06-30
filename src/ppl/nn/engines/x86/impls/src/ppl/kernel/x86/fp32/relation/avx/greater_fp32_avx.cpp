#include "relation_fp32_avx_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode greater_eltwise_fp32_avx(
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst)
{
    return relation_eltwise_binary_op_fp32_avx<RELATION_GREATER>(dst_shape, src0, src1, dst);
}

ppl::common::RetCode greater_ndarray_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t* dst)
{
    return relation_ndarray_binary_op_fp32_avx<RELATION_GREATER>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
}

}}}; // namespace ppl::kernel::x86