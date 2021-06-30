#ifndef __ST_PPL_KERNEL_X86_INT64_ARITHMETIC_ARITHMETIC_ELTWISE_INT64_H_
#define __ST_PPL_KERNEL_X86_INT64_ARITHMETIC_ARITHMETIC_ELTWISE_INT64_H_

#include "arithmetic_kernel_int64.h"

namespace ppl { namespace kernel { namespace x86 {

template <arithmetic_op_type_t _op>
static ppl::common::RetCode arithmetic_eltwise_int64(
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    int64_t *dst)
{
    const int64_t unroll_len  = 8;
    const int64_t length      = dst_shape->GetElementsIncludingPadding();
    const int64_t unroll_body = round(length, unroll_len);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        dst[i + 0] = arithmetic_scalar_kernel_int64<_op>(src0[i + 0], src1[i + 0]);
        dst[i + 1] = arithmetic_scalar_kernel_int64<_op>(src0[i + 1], src1[i + 1]);
        dst[i + 2] = arithmetic_scalar_kernel_int64<_op>(src0[i + 2], src1[i + 2]);
        dst[i + 3] = arithmetic_scalar_kernel_int64<_op>(src0[i + 3], src1[i + 3]);
        dst[i + 4] = arithmetic_scalar_kernel_int64<_op>(src0[i + 4], src1[i + 4]);
        dst[i + 5] = arithmetic_scalar_kernel_int64<_op>(src0[i + 5], src1[i + 5]);
        dst[i + 6] = arithmetic_scalar_kernel_int64<_op>(src0[i + 6], src1[i + 6]);
        dst[i + 7] = arithmetic_scalar_kernel_int64<_op>(src0[i + 7], src1[i + 7]);
    }
    for (int64_t i = unroll_body; i < length; i++) {
        dst[i] = arithmetic_scalar_kernel_int64<_op>(src0[i], src1[i]);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_INT64_ARITHMETIC_ARITHMETIC_ELTWISE_INT64_H_