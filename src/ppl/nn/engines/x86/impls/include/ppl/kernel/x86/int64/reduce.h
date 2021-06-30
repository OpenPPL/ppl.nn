#ifndef __ST_PPL_KERNEL_X86_INT64_REDUCE_H_
#define __ST_PPL_KERNEL_X86_INT64_REDUCE_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reduce_max_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst);

ppl::common::RetCode reduce_min_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst);

ppl::common::RetCode reduce_mean_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst);

ppl::common::RetCode reduce_sum_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst);

ppl::common::RetCode reduce_prod_int64(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src,
    const int32_t *axes,
    const int32_t num_axes,
    int64_t *dst);

}}}; // namespace ppl::kernel::x86

#endif
