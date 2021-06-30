#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DYNAMIC_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DYNAMIC_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t conv2d_dynamic_ndarray_fp32_avx512_get_buffer_bytes(
    const int32_t batch,
    const int32_t num_output,
    const int32_t group,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t channels,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w);

uint64_t conv2d_dynamic_ndarray_fp32_fma_get_buffer_bytes(
    const int32_t batch,
    const int32_t num_output,
    const int32_t group,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t channels,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w);

uint64_t conv2d_dynamic_ndarray_fp32_sse_get_buffer_bytes(
    const int32_t batch,
    const int32_t num_output,
    const int32_t group,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t channels,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w);

ppl::common::RetCode conv2d_dynamic_ndarray_fp32_avx512(
    const float *input,
    const float *filter,
    const float *bias,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t batch,
    const int32_t group,
    const int32_t channels,
    const int32_t num_output,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t hole_h,
    const int32_t hole_w,
    float *tmp_buffer,
    float *output);

ppl::common::RetCode conv2d_dynamic_ndarray_fp32_fma(
    const float *input,
    const float *filter,
    const float *bias,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t batch,
    const int32_t group,
    const int32_t channels,
    const int32_t num_output,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t hole_h,
    const int32_t hole_w,
    float *tmp_buffer,
    float *output);

ppl::common::RetCode conv2d_dynamic_ndarray_fp32_sse(
    const float *input,
    const float *filter,
    const float *bias,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t batch,
    const int32_t group,
    const int32_t channels,
    const int32_t num_output,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t hole_h,
    const int32_t hole_w,
    float *tmp_buffer,
    float *output);

}}}; // namespace ppl::kernel::x86

#endif
