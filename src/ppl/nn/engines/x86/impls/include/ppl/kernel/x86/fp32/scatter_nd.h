#ifndef __ST_PPL_KERNEL_X86_FP32_SCATTER_ND_H_
#define __ST_PPL_KERNEL_X86_FP32_SCATTER_ND_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode scatter_nd_ndarray_fp32(
    const float *src,
    const float *updates,
    const int64_t *indices,
    const int32_t *strides,
    const int64_t src_length,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
