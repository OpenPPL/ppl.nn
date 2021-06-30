#ifndef __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_MAX6D_H_
#define __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_MAX6D_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode add_ndarray_max6d_fp32_sse(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst);

ppl::common::RetCode sub_ndarray_max6d_fp32_sse(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst);

ppl::common::RetCode mul_ndarray_max6d_fp32_sse(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst);

ppl::common::RetCode div_ndarray_max6d_fp32_sse(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst);

ppl::common::RetCode pow_ndarray_max6d_fp32_sse(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst);

ppl::common::RetCode add_ndarray_max6d_fp32_avx(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst);

ppl::common::RetCode sub_ndarray_max6d_fp32_avx(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst);

ppl::common::RetCode mul_ndarray_max6d_fp32_avx(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst);

ppl::common::RetCode div_ndarray_max6d_fp32_avx(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst);

ppl::common::RetCode pow_ndarray_max6d_fp32_avx(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
