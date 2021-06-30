#ifndef __ST_PPL_KERNEL_X86_FP32_REDUCE_H_
#define __ST_PPL_KERNEL_X86_FP32_REDUCE_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reduce_max_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst);

ppl::common::RetCode reduce_min_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst);

ppl::common::RetCode reduce_mean_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst);

ppl::common::RetCode reduce_sum_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst);

ppl::common::RetCode reduce_max_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst);

ppl::common::RetCode reduce_min_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst);

ppl::common::RetCode reduce_mean_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst);

ppl::common::RetCode reduce_sum_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
