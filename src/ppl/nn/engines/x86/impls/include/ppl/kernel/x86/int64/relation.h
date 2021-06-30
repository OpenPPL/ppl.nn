#ifndef __ST_PPL_KERNEL_X86_INT64_RELATION_H_
#define __ST_PPL_KERNEL_X86_INT64_RELATION_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode greater_eltwise_int64(
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst);

ppl::common::RetCode greater_ndarray_int64(
    const ppl::nn::TensorShape *src_shape0,
    const ppl::nn::TensorShape *src_shape1,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst);

ppl::common::RetCode equal_eltwise_int64(
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst);

ppl::common::RetCode equal_ndarray_int64(
    const ppl::nn::TensorShape *src_shape0,
    const ppl::nn::TensorShape *src_shape1,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst);

ppl::common::RetCode less_eltwise_int64(
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst);

ppl::common::RetCode less_ndarray_int64(
    const ppl::nn::TensorShape *src_shape0,
    const ppl::nn::TensorShape *src_shape1,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst);

}}}; // namespace ppl::kernel::x86

#endif
