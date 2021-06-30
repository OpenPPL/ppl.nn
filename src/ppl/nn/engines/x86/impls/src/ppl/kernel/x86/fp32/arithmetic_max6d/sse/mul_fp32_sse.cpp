#include "arithmetic_fp32_sse_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode mul_ndarray_max6d_fp32_sse(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst)
{
    return arithmetic_binary_op_ndarray_fp32_sse<ARITHMETIC_MUL>(lhs_shape, rhs_shape, lhs, rhs, dst);
}

}}}; // namespace ppl::kernel::x86
