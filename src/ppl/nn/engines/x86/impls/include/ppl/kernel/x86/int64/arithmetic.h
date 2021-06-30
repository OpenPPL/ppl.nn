#ifndef __ST_PPL_KERNEL_X86_INT64_ARITHMETIC_H_
#define __ST_PPL_KERNEL_X86_INT64_ARITHMETIC_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode add_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    int64_t *dst);

ppl::common::RetCode sub_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    int64_t *dst);

ppl::common::RetCode mul_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    int64_t *dst);

ppl::common::RetCode div_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    int64_t *dst);

}}}; // namespace ppl::kernel::x86

#endif
