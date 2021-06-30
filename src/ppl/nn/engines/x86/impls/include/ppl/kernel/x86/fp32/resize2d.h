#ifndef __ST_PPL_KERNEL_X86_FP32_RESIZE_H_
#define __ST_PPL_KERNEL_X86_FP32_RESIZE_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reisze2d_ndarray_pytorch_linear_floor_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst);

ppl::common::RetCode reisze2d_ndarray_pytorch_cubic_floor_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    const float cubic_coeff_a,
    float *dst);

ppl::common::RetCode reisze2d_ndarray_asymmetric_nearest_floor_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst);

ppl::common::RetCode reisze2d_ndarray_asymmetric_nearest_floor_2times_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst);

ppl::common::RetCode reisze2d_n16cx_asymmetric_nearest_floor_fp32_avx512(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst);

ppl::common::RetCode reisze2d_n16cx_asymmetric_nearest_floor_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst);

ppl::common::RetCode resize2d_n16chw_pytorch_2linear_floor_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst);

ppl::common::RetCode resize2d_n16cx_pytorch_2linear_floor_fp32_avx512(
    const ppl::nn::TensorShape* src_shape,
    const ppl::nn::TensorShape* dst_shape,
    const float *src,
    const float scale_h,
    const float scale_w,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
