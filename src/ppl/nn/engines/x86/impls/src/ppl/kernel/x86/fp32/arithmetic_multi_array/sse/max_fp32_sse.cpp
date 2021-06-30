#include "ppl/kernel/x86/fp32/arithmetic_multi_array/sse/arithmetic_multi_array_fp32_sse.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t max_fp32_sse_get_temp_buffer_bytes(
    const uint32_t input_num)
{
    return arithmetic_multi_array_fp32_get_temp_buffer_bytes(input_num);
}

ppl::common::RetCode max_eltwise_fp32_sse(
    const ppl::nn::TensorShape *dst_shape,
    const float **src_list,
    const uint32_t num_src,
    float *dst)
{
    if (num_src == 2) {
        return arithmetic_multi_array_eltwise_fp32_sse<ARRAY_MAX, true>(dst_shape, src_list, num_src, dst);
    } else {
        return arithmetic_multi_array_eltwise_fp32_sse<ARRAY_MAX, false>(dst_shape, src_list, num_src, dst);
    }
}

ppl::common::RetCode max_ndarray_fp32_sse(
    const ppl::nn::TensorShape **src_shape_list,
    const ppl::nn::TensorShape *dst_shape,
    const float **src_list,
    const uint32_t num_src,
    void *temp_buffer,
    float *dst)
{
    if (num_src == 2) {
        return arithmetic_multi_array_ndarray_fp32_sse<ARRAY_MAX, true>(src_shape_list, dst_shape, src_list, num_src, temp_buffer, dst);
    } else {
        return arithmetic_multi_array_ndarray_fp32_sse<ARRAY_MAX, false>(src_shape_list, dst_shape, src_list, num_src, temp_buffer, dst);
    }
}

}}}; // namespace ppl::kernel::x86
