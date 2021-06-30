#ifndef __ST_PPL_KERNEL_X86_FP32_MATMUL_H_
#define __ST_PPL_KERNEL_X86_FP32_MATMUL_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t matmul_ndarray_fp32_get_buffer_bytes(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const common::isa_t isa_flag);

common::RetCode matmul_ndarray_fp32(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const ppl::common::isa_t isa_flag,
    void *temp_buffer,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_MATMUL_H_
