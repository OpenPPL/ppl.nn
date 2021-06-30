#ifndef __ST_PPL_KERNEL_X86_FP32_PAD_H_
#define __ST_PPL_KERNEL_X86_FP32_PAD_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode pad_ndarray_constant_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst);

ppl::common::RetCode pad_ndarray_reflect_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float *dst);

ppl::common::RetCode pad_ndarray_edge_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float *dst);

ppl::common::RetCode pad_n16cx_constant_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst);

ppl::common::RetCode pad_n16cx_reflect_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float *dst);

ppl::common::RetCode pad_n16cx_edge_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_PAD_H_
