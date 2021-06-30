#include "arithmetic_eltwise_fp32_avx.h"
#include "arithmetic_broadcast_ndarray_fp32_avx.h"
#include "arithmetic_broadcast_n16cx_fp32_avx.h"

namespace ppl { namespace kernel { namespace x86 {

template <arithmetic_op_type_t _op, bool fuse_relu>
static ppl::common::RetCode arithmetic_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    float *dst)
{
    bool is_eltwise =
        src0_shape->GetElementsExcludingPadding() == dst_shape->GetElementsExcludingPadding() &&
        src1_shape->GetElementsExcludingPadding() == dst_shape->GetElementsExcludingPadding();
    if (is_eltwise) {
        return arithmetic_eltwise_fp32_avx<_op, fuse_relu>(dst_shape, src0, src1, dst);
    } else if (dst_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
        return arithmetic_broadcast_ndarray_fp32_avx<_op, fuse_relu>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    } else if (dst_shape->GetDataFormat() == ppl::common::DATAFORMAT_N16CX) {
        return arithmetic_broadcast_n16cx_fp32_avx<_op, fuse_relu>(src0_shape, src1_shape, dst_shape, src0, src1, 1, dst);
    }

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode add_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst)
{
    if (fuse_relu) {
        return arithmetic_fp32_avx<ARITHMETIC_ADD, true>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    }
    else {
        return arithmetic_fp32_avx<ARITHMETIC_ADD, false>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    }
}

ppl::common::RetCode sub_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst)
{
    if (fuse_relu) {
        return arithmetic_fp32_avx<ARITHMETIC_SUB, true>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    } else {
        return arithmetic_fp32_avx<ARITHMETIC_SUB, false>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    }
}

ppl::common::RetCode mul_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst)
{
    if (fuse_relu) {
        return arithmetic_fp32_avx<ARITHMETIC_MUL, true>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    } else {
        return arithmetic_fp32_avx<ARITHMETIC_MUL, false>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    }
}

ppl::common::RetCode div_fp32_avx(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const bool fuse_relu,
    float *dst)
{
    if (fuse_relu) {
        return arithmetic_fp32_avx<ARITHMETIC_DIV, true>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    } else {
        return arithmetic_fp32_avx<ARITHMETIC_DIV, false>(src0_shape, src1_shape, dst_shape, src0, src1, dst);
    }
}

}}}; // namespace ppl::kernel::x86