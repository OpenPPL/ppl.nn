#include "ppl/kernel/x86/common/internal_include.h"
#include <string.h>

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode gather_nd_ndarray_fp32(
    const float *src,
    const int64_t *indices,
    const int32_t *strides,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim,
    float *dst)
{
    if (inner_dim > 1) {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t k = 0; k < num_indices; ++k) {
            int64_t offset = 0;
            const int64_t *l_indices = indices + k * indices_dim;
            float *l_dst = dst + k * inner_dim;
            for (int64_t i = 0; i < indices_dim; ++i) {
                offset += l_indices[i] * strides[i];
            }
            memcpy(l_dst, src + offset, inner_dim * sizeof(float));
        }
    } else {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t k = 0; k < num_indices; ++k) {
            int64_t offset = 0;
            const int64_t *l_indices = indices + k * indices_dim;
            for (int64_t i = 0; i < indices_dim; ++i) {
                offset += l_indices[i] * strides[i];
            }
            dst[k] = src[offset];
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86
