#ifndef __ST_PPL_KERNEL_X86_COMMON_RELATION_RELATION_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_RELATION_RELATION_COMMON_H_

namespace ppl { namespace kernel { namespace x86 {

enum relation_op_type_t {
    RELATION_GREATER          = 0,
    RELATION_GREATER_OR_EQUAL = 1,
    RELATION_LESS             = 2,
    RELATION_LESS_OR_EQUAL    = 3,
    RELATION_EQUAL            = 4,
    RELATION_NOT_EQUAL        = 5
};

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_RELATION_RELATION_COMMON_H_
