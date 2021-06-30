#ifndef __ST_PPL_KERNEL_X86_INT64_TRANSPOSE_H_
#define __ST_PPL_KERNEL_X86_INT64_TRANSPOSE_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode transpose_ndarray_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *perm,
    int64_t *dst);

ppl::common::RetCode transpose_ndarray_continous2d_int64(
    const ppl::nn::TensorShape *src_shape,
    const int64_t *src,
    const uint32_t axis0,
    const uint32_t axis1,
    int64_t *dst);

}}}; // namespace ppl::kernel::x86

#endif
