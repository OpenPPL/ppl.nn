#ifndef __ST_PPL_KERNEL_X86_FP32_CLIP_H_
#define __ST_PPL_KERNEL_X86_FP32_CLIP_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode clip_fp32_avx(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    const float clip_min,
    const float clip_max,
    float *y);

ppl::common::RetCode clip_fp32_sse(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    const float clip_min,
    const float clip_max,
    float *y);

}}}; // namespace ppl::kernel::x86

#endif
