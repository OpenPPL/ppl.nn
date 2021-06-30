#ifndef __ST_PPL_KERNEL_X86_COMMON_REDUCE_REDUCE_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_REDUCE_REDUCE_COMMON_H_

namespace ppl { namespace kernel { namespace x86 {

enum reduce_op_type_t {
    REDUCE_MAX  = 0,
    REDUCE_MIN  = 1,
    REDUCE_SUM  = 2,
    REDUCE_MEAN = 3,
    REDUCE_PROD = 4,
};

}}}; // namespace ppl::kernel::x86

#endif
