#ifndef __ST_PPL_KERNEL_X86_FP32_LOG_H_
#define __ST_PPL_KERNEL_X86_FP32_LOG_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode log_fp32(
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    float *y);

}}}; // namespace ppl::kernel::x86

#endif
