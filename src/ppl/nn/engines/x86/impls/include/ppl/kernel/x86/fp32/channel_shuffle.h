#ifndef __ST_PPL_KERNEL_X86_FP32_RESIZE_H_
#define __ST_PPL_KERNEL_X86_FP32_RESIZE_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode channel_shuffle_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int32_t group,
    float *dst);

ppl::common::RetCode channel_shuffle_n16cx_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int32_t group,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
