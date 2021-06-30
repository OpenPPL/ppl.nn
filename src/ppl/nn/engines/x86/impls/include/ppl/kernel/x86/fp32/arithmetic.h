#ifndef __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_H_
#define __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode add_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst);

ppl::common::RetCode sub_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst);

ppl::common::RetCode mul_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst);

ppl::common::RetCode div_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst);

ppl::common::RetCode add_fp32_sse(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst);

ppl::common::RetCode sub_fp32_sse(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst);

ppl::common::RetCode mul_fp32_sse(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst);

ppl::common::RetCode div_fp32_sse(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
