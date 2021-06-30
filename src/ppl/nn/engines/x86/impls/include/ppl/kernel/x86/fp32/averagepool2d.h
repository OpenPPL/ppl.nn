#ifndef __ST_PPL_KERNEL_X86_FP32_AVERAGEPOOL2D_H_
#define __ST_PPL_KERNEL_X86_FP32_AVERAGEPOOL2D_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

// averagepool2d n16chw blk

ppl::common::RetCode averagepool2d_n16chw_blk1x16_fp32_avx512(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,
    const int32_t ceil_mode,
    float *dst);

ppl::common::RetCode averagepool2d_n16chw_blk1x8_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,
    const int32_t ceil_mode,
    float *dst);

ppl::common::RetCode averagepool2d_n16chw_blk1x4_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,
    const int32_t ceil_mode,
    float *dst);

// averagepool2d nchw normal

ppl::common::RetCode averagepool2d_nchw_normal_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,
    const int32_t ceil_mode,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_AVERAGEPOOL_H_
