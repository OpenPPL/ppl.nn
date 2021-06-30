#ifndef __ST_PPL_KERNEL_X86_FP32_CONV_COMMON_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV_COMMON_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

typedef uint64_t conv_fuse_flag_t;

class conv_fuse_flag {
public:
    enum {
        none  = 0,
        relu  = 1 << 0,
        relu6 = 1 << 1,
        sum   = 1 << 16,
    };
};

}}}; // namespace ppl::kernel::x86

#endif
