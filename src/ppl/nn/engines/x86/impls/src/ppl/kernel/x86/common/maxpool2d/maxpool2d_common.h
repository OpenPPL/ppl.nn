#ifndef __ST_PPL_KERNEL_X86_COMMON_MAXPOOL2D_MAXPOOL2D_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_MAXPOOL2D_MAXPOOL2D_COMMON_H_

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

struct maxpool2d_param {
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;

    int32_t batch;
    int32_t channels;
    int32_t src_h;
    int32_t src_w;
    int32_t dst_h;
    int32_t dst_w;
};

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_MAXPOOL2D_MAXPOOL2D_COMMON_H_
