#ifndef __ST_PPL_KERNEL_X86_FP32_REORDER_H_
#define __ST_PPL_KERNEL_X86_FP32_REORDER_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reorder_ndarray_n16cx_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    float *dst);

ppl::common::RetCode reorder_ndarray_n16cx_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    float *dst);

ppl::common::RetCode reorder_n16cx_ndarray_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    float *dst);

ppl::common::RetCode reorder_n16cx_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    float *dst);

ppl::common::RetCode reorder_n16cx_nxc_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    float *dst);

ppl::common::RetCode reorder_n16cx_nxc_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    float *dst);

uint64_t reorder_goidhw_gIOdhwB16i16o_fp32_get_dst_size(
    const int32_t group,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_d,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t channels_blk);

ppl::common::RetCode reorder_goidhw_gIOdhwB16i16o_fp32(
    const float *src,
    const int32_t group,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_d,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t channels_blk,
    float *dst);

uint64_t reorder_goidhw_gIOBidhw16i16o_fp32_get_dst_size(
    const int32_t group,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_d,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t channels_blk);

ppl::common::RetCode reorder_goidhw_gIOBidhw16i16o_fp32(
    const float *src,
    const int32_t group,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_d,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t channels_blk,
    float *dst);

uint64_t reorder_goidhw_gOdhwi16o_fp32_get_dst_size(
    const int32_t group,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_d,
    const int32_t kernel_h,
    const int32_t kernel_w);

ppl::common::RetCode reorder_goidhw_gOdhwi16o_fp32(
    const float *src,
    const int32_t group,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_d,
    const int32_t kernel_h,
    const int32_t kernel_w,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
