#include "ppl/kernel/x86/common/non_zero/non_zero_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode non_zero_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    void *temp_buffer,
    int64_t *non_zero_num,
    int64_t *dst)
{
    return non_zero_ndarray_common<float>(src_shape, src, temp_buffer, non_zero_num, dst);
}

}}}; // namespace ppl::kernel::x86
