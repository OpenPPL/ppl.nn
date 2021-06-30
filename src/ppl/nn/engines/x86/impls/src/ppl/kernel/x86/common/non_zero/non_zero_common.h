#ifndef __ST_PPL_KERNEL_X86_COMMON_NON_ZERO_NON_ZERO_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_NON_ZERO_NON_ZERO_COMMON_H_

#include "ppl/kernel/x86/common/internal_include.h"
#include <string.h>

namespace ppl { namespace kernel { namespace x86 {

inline void calc_idx(
    const uint64_t *strides,
    const uint64_t global_idx,
    const uint64_t dim_count,
    uint64_t *idx)
{
    uint64_t global_idx_remain = global_idx;
    for (uint64_t i = 0; i < dim_count; i++) {
        idx[i] = global_idx_remain / strides[i];
        global_idx_remain %= strides[i];
    }
}

template <typename eT>
ppl::common::RetCode non_zero_ndarray_common(
    const ppl::nn::TensorShape *src_shape,
    const eT *src,
    void *temp_buffer,
    int64_t *non_zero_num,
    int64_t *dst)
{
    const int64_t dim_count  = src_shape->GetDimCount();
    const int64_t stride_out = src_shape->GetElementsExcludingPadding();
    uint64_t idx[dim_count];
    uint64_t strides[dim_count];
    int64_t *temp_output = (int64_t*)temp_buffer;

    strides[dim_count - 1] = 1;
    for (int64_t i = dim_count - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * src_shape->GetDim(i + 1);
    }

    *non_zero_num = 0;
    for (uint64_t i = 0; i < src_shape->GetElementsExcludingPadding(); i++) {
        if (src[i] != 0) {
            calc_idx(strides, i, dim_count, idx);
            for (int64_t j = 0; j < dim_count; j++) {
                temp_output[j * stride_out + *non_zero_num] = idx[j];
            }
            (*non_zero_num)++;
        }
    }

    for (int64_t i = 0; i < dim_count; i++) {
        memcpy(dst + i * *non_zero_num, temp_output + i * stride_out, *non_zero_num * sizeof(int64_t));
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_NON_ZERO_NON_ZERO_COMMON_H_
