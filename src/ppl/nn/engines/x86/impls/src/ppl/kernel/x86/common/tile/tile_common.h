#ifndef __ST_PPL_KERNEL_X86_COMMON_TILE_TILE_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_TILE_TILE_COMMON_H_

#include <string.h> // for memcpy

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
static ppl::common::RetCode tile_ndarray_recursive(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const eT *src,
    const int64_t *repeats,
    const int64_t dim_idx,
    const bool has_paralleled,
    eT *dst)
{
    int64_t src_stride = 1;
    int64_t dst_stride = 1;

    for (int64_t i = dim_idx + 1; i < src_shape->GetDimCount(); ++i) {
        src_stride *= src_shape->GetDim(i);
        dst_stride *= dst_shape->GetDim(i);
    }

    if (dim_idx + 1 == src_shape->GetDimCount()) {
        if (!has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < repeats[dim_idx]; ++i) {
                memcpy(dst + i * src_shape->GetDim(dim_idx), src, src_shape->GetDim(dim_idx) * sizeof(eT));
            }
        } else {
            for (int64_t i = 0; i < repeats[dim_idx]; ++i) {
                memcpy(dst + i * src_shape->GetDim(dim_idx), src, src_shape->GetDim(dim_idx) * sizeof(eT));
            }
        }
    } else {
        if (src_shape->GetDim(dim_idx) > 1 && !has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < src_shape->GetDim(dim_idx); ++i) {
                tile_ndarray_recursive<eT>(src_shape, dst_shape, src + i * src_stride, repeats, dim_idx + 1, true, dst + i * dst_stride);
            }
        } else {
            for (int64_t i = 0; i < src_shape->GetDim(dim_idx); ++i) {
                tile_ndarray_recursive<eT>(src_shape, dst_shape, src + i * src_stride, repeats, dim_idx + 1, has_paralleled, dst + i * dst_stride);
            }
        }

        if (!has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 1; i < repeats[dim_idx]; ++i) {
                memcpy(dst + i * src_shape->GetDim(dim_idx) * dst_stride, dst, src_shape->GetDim(dim_idx) * dst_stride * sizeof(eT));
            }
        } else {
            for (int64_t i = 1; i < repeats[dim_idx]; ++i) {
                memcpy(dst + i * src_shape->GetDim(dim_idx) * dst_stride, dst, src_shape->GetDim(dim_idx) * dst_stride * sizeof(eT));
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT>
static ppl::common::RetCode tile_ndarray(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const eT *src,
    const int64_t *repeats,
    eT *dst)
{
    return tile_ndarray_recursive<eT>(src_shape, dst_shape, src, repeats, 0, false, dst);
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_COMMON_TILE_TILE_COMMON_H_
