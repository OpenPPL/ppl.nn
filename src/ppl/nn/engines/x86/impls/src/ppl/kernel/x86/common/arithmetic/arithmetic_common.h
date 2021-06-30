#ifndef __ST_PPL_KERNEL_X86_COMMON_ARITHMETIC_ARITHMETIC_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_ARITHMETIC_ARITHMETIC_COMMON_H_

namespace ppl { namespace kernel { namespace x86 {

enum arithmetic_op_type_t {
    ARITHMETIC_ADD = 0,
    ARITHMETIC_SUB = 1,
    ARITHMETIC_MUL = 2,
    ARITHMETIC_DIV = 3,
    ARITHMETIC_POW = 4
};

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_ARITHMETIC_ARITHMETIC_COMMON_H_
