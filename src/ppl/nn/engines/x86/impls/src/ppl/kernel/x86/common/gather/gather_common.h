#ifndef __ST_PPL_KERNEL_X86_COMMON_GATHER_GATHER_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_GATHER_GATHER_COMMON_H_

#include "ppl/kernel/x86/common/internal_include.h"
#include <string.h> 

namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
ppl::common::RetCode gather_ndarray_common(
    const eT *src,
    const int64_t *indices,
    const int64_t outer_dim,
    const int64_t gather_dim,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim,
    eT *dst)
{
    if (inner_dim >= 4) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                for (int64_t i = 0; i < indices_dim; ++i) {
                    eT *l_dst = dst + o * num_indices * indices_dim * inner_dim +
                               k * indices_dim * inner_dim + i * inner_dim;
                    int64_t index  = indices[k * indices_dim + i];
                    const eT *l_src = src + o * gather_dim * inner_dim + index * inner_dim;
                    memcpy(l_dst, l_src, inner_dim * sizeof(eT));
                }
            }
        }
    } else if (inner_dim >= 2) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                eT *l_dst =
                    dst + o * num_indices * indices_dim * inner_dim + k * indices_dim * inner_dim;
                const int64_t *l_indices = indices + k * indices_dim;
                const eT *l_src           = src + o * gather_dim * inner_dim;
                if (inner_dim == 2) {
                    for (int64_t i = 0; i < indices_dim; ++i) {
                        l_dst[0] = l_src[l_indices[0] + 0];
                        l_dst[1] = l_src[l_indices[0] + 1];
                        l_dst += inner_dim;
                        ++l_indices;
                    }
                } else {
                    for (int64_t i = 0; i < indices_dim; ++i) {
                        l_dst[0] = l_src[l_indices[0] + 0];
                        l_dst[1] = l_src[l_indices[0] + 1];
                        l_dst[2] = l_src[l_indices[0] + 2];
                        l_dst += inner_dim;
                        ++l_indices;
                    }
                }
            }
        }
    } else {
#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t o = 0; o < outer_dim; ++o) {
            for (int64_t k = 0; k < num_indices; ++k) {
                eT *l_dst                 = dst + o * num_indices * indices_dim + k * indices_dim;
                const int64_t *l_indices = indices + k * indices_dim;
                const eT *l_src           = src + o * gather_dim;
                for (int64_t i = 0; i < indices_dim; ++i) {
                    l_dst[0] = l_src[l_indices[0]];
                    ++l_dst;
                    ++l_indices;
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_GATHER_GATHER_COMMON_H_