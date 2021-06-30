#ifndef __ST_PPL_KERNEL_X86_INT64_EXPAND_H_
#define __ST_PPL_KERNEL_X86_INT64_EXPAND_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode expand_ndarray_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    int64_t *dst);

}}}; // namespace ppl::kernel::x86

#endif
