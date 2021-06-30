#ifndef __ST_PPL_KERNEL_X86_INT64_GATHER_H_
#define __ST_PPL_KERNEL_X86_INT64_GATHER_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode gather_ndarray_int64(
    const int64_t *src,
    const int64_t *indices,
    const int64_t outer_dim,
    const int64_t gather_dim,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim,
    int64_t *dst);

}}}; // namespace ppl::kernel::x86

#endif
