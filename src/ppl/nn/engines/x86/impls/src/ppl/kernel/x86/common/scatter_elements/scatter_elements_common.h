#ifndef __ST_PPL_KERNEL_X86_COMMON_SCATTER_ELEMENTS_SCATTER_ELEMENTS_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_SCATTER_ELEMENTS_SCATTER_ELEMENTS_COMMON_H_

#include "ppl/kernel/x86/common/internal_include.h"
#include <string.h> // memcpy

namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
ppl::common::RetCode scatter_elements_ndarray_common(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *indices_shape,
    const eT *src,
    const int64_t *indices,
    const eT *updates,
    const int64_t axis,
    eT *dst)
{
    memcpy(dst, src, src_shape->GetBytesExcludingPadding());

    const int64_t dim_count    = src_shape->GetDimCount();
    const int64_t scatter_axis = axis < 0 ? axis + dim_count : axis;

    int64_t src_outer_dims     = 1;
    int64_t indices_outer_dims = 1;
    for (int64_t i = 0; i < scatter_axis; i++) {
        src_outer_dims *= src_shape->GetDim(i);
        indices_outer_dims *= indices_shape->GetDim(i);
    }

    const int64_t src_scatter_dims     = src_shape->GetDim(scatter_axis);
    const int64_t indices_scatter_dims = indices_shape->GetDim(scatter_axis);

    int64_t src_inner_dims     = 1;
    int64_t indices_inner_dims = 1;
    for (int64_t i = scatter_axis + 1; i < dim_count; i++) {
        src_inner_dims *= src_shape->GetDim(i);
        indices_inner_dims *= indices_shape->GetDim(i);
    }

    for (int64_t i = 0; i < indices_outer_dims; i++) {
        for (int64_t j = 0; j < indices_scatter_dims; j++) {
            for (int64_t k = 0; k < indices_inner_dims; k++) {
                const int64_t index = indices[i * indices_scatter_dims * indices_inner_dims + j * indices_inner_dims + k];
                const eT update_val  = updates[i * indices_scatter_dims * indices_inner_dims + j * indices_inner_dims + k];
                dst[i * src_scatter_dims * src_inner_dims + index * src_inner_dims + k] = update_val;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_SCATTER_ELEMENTS_SCATTER_ELEMENTS_COMMON_H_