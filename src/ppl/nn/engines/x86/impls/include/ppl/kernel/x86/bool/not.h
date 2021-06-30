#ifndef __ST_PPL_KERNEL_X86_BOOL_NOT_H_
#define __ST_PPL_KERNEL_X86_BOOL_NOT_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode not_bool(
    const ppl::nn::TensorShape *x_shape,
    const uint8_t *x,
    uint8_t *y);

ppl::common::RetCode not_bool_sse(
    const ppl::nn::TensorShape *x_shape,
    const uint8_t *x,
    uint8_t *y);

ppl::common::RetCode not_bool_avx(
    const ppl::nn::TensorShape *x_shape,
    const uint8_t *x,
    uint8_t *y);

}}}; // namespace ppl::kernel::x86

#endif
