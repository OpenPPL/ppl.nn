#ifndef __ST_PPL_KERNEL_X86_INT64_CONCAT_H_
#define __ST_PPL_KERNEL_X86_INT64_CONCAT_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode concat_n16cx_int64(
    const ppl::nn::TensorShape **src_shape_list,
    const int64_t **src_list,
    const int32_t num_src,
    const int32_t axis,
    int64_t *dst);

ppl::common::RetCode concat_ndarray_int64(
    const ppl::nn::TensorShape **src_shape_list,
    const int64_t **src_list,
    const int32_t num_src,
    const int32_t axis,
    int64_t *dst);

}}}; // namespace ppl::kernel::x86

#endif
