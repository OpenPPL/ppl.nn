#ifndef __ST_PPL_KERNEL_X86_INT64_REORDER_H_
#define __ST_PPL_KERNEL_X86_INT64_REORDER_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reorder_ndarray_n16cx_int64_avx(
    const ppl::nn::TensorShape *src_shape,
    const int64_t *src,
    int64_t *dst);

ppl::common::RetCode reorder_ndarray_n16cx_int64(
    const ppl::nn::TensorShape *src_shape,
    const int64_t *src,
    int64_t *dst);

ppl::common::RetCode reorder_n16cx_ndarray_int64_avx(
    const ppl::nn::TensorShape *src_shape,
    const int64_t *src,
    int64_t *dst);

ppl::common::RetCode reorder_n16cx_ndarray_int64(
    const ppl::nn::TensorShape *src_shape,
    const int64_t *src,
    int64_t *dst);

}}}; // namespace ppl::kernel::x86

#endif
