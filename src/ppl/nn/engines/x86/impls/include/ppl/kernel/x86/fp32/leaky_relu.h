#ifndef __ST_PPL_KERNEL_X86_FP32_LEAKY_RELU_H_
#define __ST_PPL_KERNEL_X86_FP32_LEAKY_RELU_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode leaky_relu_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst);

ppl::common::RetCode leaky_relu_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
