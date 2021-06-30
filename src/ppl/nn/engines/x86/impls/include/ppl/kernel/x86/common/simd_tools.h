#ifndef __ST_PPL_KERNEL_X86_COMMON_SIMD_TOOLS_H_
#define __ST_PPL_KERNEL_X86_COMMON_SIMD_TOOLS_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

void set_denormals_zero(const int32_t on);

}}}; // namespace ppl::kernel::x86

#endif
