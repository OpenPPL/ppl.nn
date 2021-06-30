#ifndef __ST_PPL_KERNEL_X86_COMMON_ARGMAX_ARGMAX_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_ARGMAX_ARGMAX_COMMON_H_

#include <limits>
#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
ppl::common::RetCode argmax_ndarray(
    const ppl::nn::TensorShape *src_shape,
    const eT *src,
    const int64_t axis,
    int64_t *dst)
{
    eT numeric_min = std::numeric_limits<eT>().min();
    if (std::is_same<eT, float>().value || std::is_same<eT, double>().value || std::is_same<eT, long double>().value) {
        numeric_min = -std::numeric_limits<eT>().max();
    }
    const int64_t real_axis = axis < 0 ? axis + src_shape->GetDimCount() : axis;

    const int64_t argmax_dim = src_shape->GetDim(real_axis);
    int64_t outer_dim        = 1;
    int64_t inner_dim        = 1;
    for (uint32_t i = 0; i < real_axis; i++) {
        outer_dim *= src_shape->GetDim(i);
    }
    for (uint32_t i = real_axis + 1; i < src_shape->GetDimCount(); i++) {
        inner_dim *= src_shape->GetDim(i);
    }

#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t i = 0; i < outer_dim; ++i) {
        for (int64_t j = 0; j < inner_dim; ++j) {
            eT max_value = numeric_min;
            int64_t idx = 0;
            for (int64_t k = 0; k < argmax_dim; ++k) {
                if (src[(i * argmax_dim + k) * inner_dim + j] > max_value) {
                    max_value = src[(i * argmax_dim + k) * inner_dim + j];
                    idx       = k;
                }
            }
            dst[i * inner_dim + j] = idx;
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_ARGMAX_ARGMAX_COMMON_H_
