#ifndef __ST_PPL_KERNEL_X86_INT64_SCATTER_ELEMENTS_H_
#define __ST_PPL_KERNEL_X86_INT64_SCATTER_ELEMENTS_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode scatter_elements_ndarray_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *indices_shape,
    const int64_t *src,
    const int64_t *indices,
    const int64_t *updates,
    const int64_t axis,
    int64_t *dst);

}}}; // namespace ppl::kernel::x86

#endif