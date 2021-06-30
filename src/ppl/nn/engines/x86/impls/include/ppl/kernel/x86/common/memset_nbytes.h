#ifndef __ST_PPL_KERNEL_X86_COMMON_MEMSET_NBYTES_H_
#define __ST_PPL_KERNEL_X86_COMMON_MEMSET_NBYTES_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode memset_nbytes(
    const void *src,
    const uint64_t bytes_per_element,
    const uint64_t num_elements,
    void* dst);

}}}; // namespace ppl::kernel::x86

#endif
