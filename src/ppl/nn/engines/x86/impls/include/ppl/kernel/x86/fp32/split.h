#ifndef __ST_PPL_KERNEL_X86_FP32_SPLIT_H_
#define __ST_PPL_KERNEL_X86_FP32_SPLIT_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode split_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape **dst_shape_list,
    const float *src,
    const int32_t slice_axis,
    const int32_t num_dst,
    float **dst_list);

ppl::common::RetCode split_n16cx_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape **dst_shape_list,
    const float *src,
    const int32_t slice_axis,
    const int32_t num_dst,
    float **dst_list);

ppl::common::RetCode split_n16cx_interleave_channels_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape **dst_shape_list,
    const float *src,
    const int32_t slice_axis,
    const int32_t num_dst,
    const int32_t c_dim_idx,
    float **dst_list);

}}}; // namespace ppl::kernel::x86

#endif
