#ifndef __ST_PPL_KERNEL_X86_FP32_SQRT_H_
#define __ST_PPL_KERNEL_X86_FP32_SQRT_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode sqrt_fp32_sse(
    const ppl::nn::TensorShape *in_shape,
    const float *in,
    float *out);

}}}; // namespace ppl::kernel::x86

#endif
