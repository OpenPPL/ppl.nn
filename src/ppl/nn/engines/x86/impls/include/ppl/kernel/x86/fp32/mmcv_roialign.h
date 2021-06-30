#ifndef __ST_PPL_KERNEL_X86_FP32_MMCV_ROIALIGN_H_
#define __ST_PPL_KERNEL_X86_FP32_MMCV_ROIALIGN_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode mmcv_roialign_ndarray_fp32(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *output_shape,
    const float *input,
    const float *rois,
    const int64_t aligned,
    const int64_t sampling_ratio,
    const float spatial_scale,
    const int32_t pool_mode, // 0: max, 1: avg
    float *output);

ppl::common::RetCode mmcv_roialign_n16cx_fp32(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *output_shape,
    const float *input,
    const float *rois,
    const int64_t aligned,
    const int64_t sampling_ratio,
    const float spatial_scale,
    const int32_t pool_mode, // 0: max, 1: avg
    float *output);

ppl::common::RetCode mmcv_roialign_n16cx_fp32_avx(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *output_shape,
    const float *input,
    const float *rois,
    const int64_t aligned,
    const int64_t sampling_ratio,
    const float spatial_scale,
    const int32_t pool_mode, // 0: max, 1: avg
    float *output);

ppl::common::RetCode mmcv_roialign_n16cx_fp32_avx512(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *rois_shape,
    const ppl::nn::TensorShape *output_shape,
    const float *input,
    const float *rois,
    const int64_t aligned,
    const int64_t sampling_ratio,
    const float spatial_scale,
    const int32_t pool_mode, // 0: max, 1: avg
    float *output);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_MMCV_ROIALIGN_H_
