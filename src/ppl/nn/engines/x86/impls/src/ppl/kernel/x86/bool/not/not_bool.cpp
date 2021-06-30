#include <math.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode not_bool(
    const ppl::nn::TensorShape *x_shape,
    const uint8_t *x,
    uint8_t *y)
{
    const int64_t n_elem = x_shape->GetElementsIncludingPadding();

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < n_elem; ++i) {
        y[i] = x[i] ^ 0x01;
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86