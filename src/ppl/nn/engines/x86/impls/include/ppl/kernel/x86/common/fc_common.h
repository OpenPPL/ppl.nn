#ifndef __ST_PPL_KERNEL_X86_FP32_FC_COMMON_H_
#define __ST_PPL_KERNEL_X86_FP32_FC_COMMON_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

typedef uint64_t fc_fuse_flag_t;

class fc_fuse_flag {
public:
    enum {
        none  = 0,
        relu  = 1 << 0,
    };
};

}}}; // namespace ppl::kernel::x86

#endif
