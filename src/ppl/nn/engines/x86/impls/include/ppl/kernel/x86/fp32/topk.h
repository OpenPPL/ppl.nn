#ifndef __ST_PPL_KERNEL_X86_FP32_TOPK_H_
#define __ST_PPL_KERNEL_X86_FP32_TOPK_H_

#include <omp.h>
#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t topk_ndarray_fp32_get_buffer_bytes(
    const ppl::nn::TensorShape *src_shape,
    const int32_t axis);

ppl::common::RetCode topk_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *value_shape,
    const ppl::nn::TensorShape *indices_shape,
    const float *src,
    const int64_t k,
    const int32_t axis,
    const int32_t largest,
    const int32_t sorted,
    void *temp_buffer,
    float *values,
    int64_t *indices);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_TOPK_H_
