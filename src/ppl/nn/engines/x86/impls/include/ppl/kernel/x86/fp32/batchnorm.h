#ifndef __ST_PPL_KERNEL_X86_FP32_BATCHNORM_H_
#define __ST_PPL_KERNEL_X86_FP32_BATCHNORM_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool fuse_relu>
ppl::common::RetCode batchnorm_ndarray_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float *mean,
    const float *variance,
    const float *scale,
    const float *shift,
    const float var_eps,
    float *dst);

template <bool fuse_relu>
ppl::common::RetCode batchnorm_n16cx_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float *mean,
    const float *variance,
    const float *scale,
    const float *shift,
    const float var_eps,
    float *dst);

template <bool fuse_relu>
ppl::common::RetCode batchnorm_ndarray_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float *mean,
    const float *variance,
    const float *scale,
    const float *shift,
    const float var_eps,
    float *dst);

template <bool fuse_relu>
ppl::common::RetCode batchnorm_n16cx_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float *mean,
    const float *variance,
    const float *scale,
    const float *shift,
    const float var_eps,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
