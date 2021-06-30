#include "ppl/kernel/x86/fp32/arithmetic_multi_array/avx/arithmetic_multi_array_fp32_avx.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t min_fp32_avx_get_temp_buffer_bytes(
    const uint32_t input_num)
{
    return arithmetic_multi_array_fp32_get_temp_buffer_bytes(input_num);
}

ppl::common::RetCode min_eltwise_fp32_avx(
    const ppl::nn::TensorShape *dst_shape,
    const float **src_list,
    const uint32_t num_src,
    float *dst)
{
    if (num_src == 2) {
        return arithmetic_multi_array_eltwise_fp32_avx<ARRAY_MIN, true>(dst_shape, src_list, num_src, dst);
    } else {
        return arithmetic_multi_array_eltwise_fp32_avx<ARRAY_MIN, false>(dst_shape, src_list, num_src, dst);
    }
}

ppl::common::RetCode min_ndarray_fp32_avx(
    const ppl::nn::TensorShape **src_shape_list,
    const ppl::nn::TensorShape *dst_shape,
    const float **src_list,
    const uint32_t num_src,
    void *temp_buffer,
    float *dst)
{
    if (num_src == 2) {
        return arithmetic_multi_array_ndarray_fp32_avx<ARRAY_MIN, true>(src_shape_list, dst_shape, src_list, num_src, temp_buffer, dst);
    } else {
        return arithmetic_multi_array_ndarray_fp32_avx<ARRAY_MIN, false>(src_shape_list, dst_shape, src_list, num_src, temp_buffer, dst);
    }
}

}}}; // namespace ppl::kernel::x86
