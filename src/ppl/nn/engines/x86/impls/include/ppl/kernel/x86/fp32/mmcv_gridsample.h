#ifndef __ST_PPL_KERNEL_X86_FP32_MMCV_GRID_SAMPLE_H_
#define __ST_PPL_KERNEL_X86_FP32_MMCV_GRID_SAMPLE_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode mmcv_gridsample_linear_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *grid_shape,
    const float *src,
    const float *grid,
    const bool align_corners,
    const int64_t padding_mode,
    float *dst);

ppl::common::RetCode mmcv_gridsample_linear_ndarray_fp32_avx512(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *grid_shape,
    const float *src,
    const float *grid,
    const bool align_corners,
    const int64_t padding_mode,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_MMCV_GRID_SAMPLE_H_
