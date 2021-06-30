#ifndef __ST_PPL_KERNEL_X86_FP32_WHERE_H_
#define __ST_PPL_KERNEL_X86_FP32_WHERE_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode where_eltwise_fp32(
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *cond,
    const float *src_x,
    const float *src_y,
    float *dst);

ppl::common::RetCode where_ndarray_fp32(
    const ppl::nn::TensorShape *cond_shape,
    const ppl::nn::TensorShape *src_x_shape,
    const ppl::nn::TensorShape *src_y_shape,
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *cond,
    const float *src_x,
    const float *src_y,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
