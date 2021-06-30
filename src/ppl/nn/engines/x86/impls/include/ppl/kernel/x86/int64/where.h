#ifndef __ST_PPL_KERNEL_X86_INT64_WHERE_H_
#define __ST_PPL_KERNEL_X86_INT64_WHERE_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode where_eltwise_int64(
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *cond,
    const int64_t *src_x,
    const int64_t *src_y,
    int64_t *dst);

ppl::common::RetCode where_ndarray_int64(
    const ppl::nn::TensorShape *cond_shape,
    const ppl::nn::TensorShape *src_x_shape,
    const ppl::nn::TensorShape *src_y_shape,
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *cond,
    const int64_t *src_x,
    const int64_t *src_y,
    int64_t *dst);

}}}; // namespace ppl::kernel::x86

#endif
