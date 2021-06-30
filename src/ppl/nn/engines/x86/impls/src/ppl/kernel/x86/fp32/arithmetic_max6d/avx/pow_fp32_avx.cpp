#include "arithmetic_fp32_avx_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode pow_ndarray_max6d_fp32_avx(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst)
{
    return arithmetic_binary_op_ndarray_fp32_avx<ARITHMETIC_POW>(lhs_shape, rhs_shape, lhs, rhs, dst);
}

}}}; // namespace ppl::kernel::x86
