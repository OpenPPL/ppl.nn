#ifndef __ST_PPL_KERNEL_X86_FP32_SCATTER_ELEMENTS_H_
#define __ST_PPL_KERNEL_X86_FP32_SCATTER_ELEMENTS_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode scatter_elements_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *indices_shape,
    const float *src,
    const int64_t *indices,
    const float *updates,
    const int64_t axis,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif