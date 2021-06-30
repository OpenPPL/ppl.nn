#ifndef __ST_PPL_KERNEL_X86_FP32_MAX_UNPOOL_H_
#define __ST_PPL_KERNEL_X86_FP32_MAX_UNPOOL_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode max_unpool_nchw_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *indices,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif