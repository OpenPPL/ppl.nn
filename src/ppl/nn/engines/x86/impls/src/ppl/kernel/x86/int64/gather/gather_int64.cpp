#include "ppl/kernel/x86/common/gather/gather_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode gather_ndarray_int64(
    const int64_t *src,
    const int64_t *indices,
    const int64_t outer_dim,
    const int64_t gather_dim,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim,
    int64_t *dst)
{
    return gather_ndarray_common<int64_t>(src, indices, outer_dim, gather_dim, inner_dim, num_indices, indices_dim, dst);
}

}}} // namespace ppl::kernel::x86
