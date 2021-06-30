#ifndef __ST_PPL_KERNEL_X86_BOOL_LOGICAL_H_
#define __ST_PPL_KERNEL_X86_BOOL_LOGICAL_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode and_eltwise_bool(
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *src0,
    const uint8_t *src1,
    uint8_t *dst);

ppl::common::RetCode and_ndarray_bool(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *src0,
    const uint8_t *src1,
    uint8_t *dst);

}}}; // namespace ppl::kernel::x86

#endif
