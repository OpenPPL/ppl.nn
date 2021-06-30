#ifndef __ST_PPL_KERNEL_X86_FP32_MIN_H_
#define __ST_PPL_KERNEL_X86_FP32_MIN_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t min_fp32_avx_get_temp_buffer_bytes(
    const uint32_t num_src);

uint64_t min_fp32_sse_get_temp_buffer_bytes(
    const uint32_t num_src);

ppl::common::RetCode min_eltwise_fp32_avx(
    const ppl::nn::TensorShape *dst_shape,
    const float **src_list,
    const uint32_t num_src,
    float *dst);

ppl::common::RetCode min_ndarray_fp32_avx(
    const ppl::nn::TensorShape **src_shape_list,
    const ppl::nn::TensorShape *dst_shape,
    const float **src_list,
    const uint32_t num_src,
    void *temp_buffer,
    float *dst);

ppl::common::RetCode min_eltwise_fp32_sse(
    const ppl::nn::TensorShape *dst_shape,
    const float **src_list,
    const uint32_t num_src,
    float *dst);

ppl::common::RetCode min_ndarray_fp32_sse(
    const ppl::nn::TensorShape **src_shape_list,
    const ppl::nn::TensorShape *dst_shape,
    const float **src_list,
    const uint32_t num_src,
    void *temp_buffer,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_FP32_MIN_H_
