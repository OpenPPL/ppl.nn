#ifndef __ST_PPL_KERNEL_X86_COMMON_CAST_H_
#define __ST_PPL_KERNEL_X86_COMMON_CAST_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode cast(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const void *src,
    void *dst);

}}}; // namespace ppl::kernel::x86

#endif
