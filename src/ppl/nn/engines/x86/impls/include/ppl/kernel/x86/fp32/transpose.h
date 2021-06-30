#ifndef __ST_PPL_KERNEL_X86_FP32_TRANSPOSE_H_
#define __ST_PPL_KERNEL_X86_FP32_TRANSPOSE_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode transpose_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *perm,
    float *dst);

ppl::common::RetCode transpose_ndarray_continous2d_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const uint32_t axis0,
    const uint32_t axis1,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
