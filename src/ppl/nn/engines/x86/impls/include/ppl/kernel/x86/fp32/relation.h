#ifndef __ST_PPL_KERNEL_X86_FP32_RELATION_H_
#define __ST_PPL_KERNEL_X86_FP32_RELATION_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode greater_eltwise_fp32_avx(
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

ppl::common::RetCode greater_ndarray_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

ppl::common::RetCode greater_eltwise_fp32_sse(
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

ppl::common::RetCode greater_ndarray_fp32_sse(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

ppl::common::RetCode equal_eltwise_fp32_avx(
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

ppl::common::RetCode equal_ndarray_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

ppl::common::RetCode equal_eltwise_fp32_sse(
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

ppl::common::RetCode equal_ndarray_fp32_sse(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

ppl::common::RetCode less_eltwise_fp32_avx(
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

ppl::common::RetCode less_ndarray_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

ppl::common::RetCode less_eltwise_fp32_sse(
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

ppl::common::RetCode less_ndarray_fp32_sse(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    uint8_t *dst);

}}}; // namespace ppl::kernel::x86

#endif
