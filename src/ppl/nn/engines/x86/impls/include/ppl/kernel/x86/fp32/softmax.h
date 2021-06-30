#ifndef __ST_PPL_KERNEL_X86_FP32_SOFTMAX_H_
#define __ST_PPL_KERNEL_X86_FP32_SOFTMAX_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode softmax_ndarray_fp32_fma(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    float *dst);

ppl::common::RetCode softmax_ndarray_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    float *dst);

ppl::common::RetCode softmax_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int64_t axis,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
