#ifndef __ST_PPL_KERNEL_X86_BOOL_NON_ZERO_H_
#define __ST_PPL_KERNEL_X86_BOOL_NON_ZERO_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

inline uint64_t non_zero_ndarray_bool_get_buffer_bytes(
    const ppl::nn::TensorShape *src_shape)
{
    const uint64_t input_dim_count = src_shape->GetDimCount();
    const uint64_t max_output_num  = src_shape->GetElementsExcludingPadding();
    return input_dim_count * max_output_num * sizeof(int64_t);
}

ppl::common::RetCode non_zero_ndarray_bool(
    const ppl::nn::TensorShape *src_shape,
    const uint8_t *src,
    void *temp_buffer,
    int64_t *non_zero_num,
    int64_t *dst);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_BOOL_NON_ZERO_H_
