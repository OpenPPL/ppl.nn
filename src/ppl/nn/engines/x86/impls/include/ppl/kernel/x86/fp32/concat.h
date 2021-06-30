#ifndef __ST_PPL_KERNEL_X86_FP32_CONCAT_H_
#define __ST_PPL_KERNEL_X86_FP32_CONCAT_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode concat_n16cx_fp32(
    const ppl::nn::TensorShape **src_shape_list,
    const float **src_list,
    const int32_t num_src,
    const int32_t axis,
    float *dst);

ppl::common::RetCode concat_ndarray_fp32(
    const ppl::nn::TensorShape **src_shape_list,
    const float **src_list,
    const int32_t num_src,
    const int32_t axis,
    float *dst);

ppl::common::RetCode concat_n16cx_interleave_channels_fp32_avx(
    const ppl::nn::TensorShape **src_shape_list,
    const float **src_list,
    const int32_t num_src,
    const int32_t axis,
    const int32_t c_dim_idx,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif
