#ifndef __ST_PPL_KERNEL_X86_INT64_RELATION_RELATION_INT64_COMMON_H_
#define __ST_PPL_KERNEL_X86_INT64_RELATION_RELATION_INT64_COMMON_H_

#include <immintrin.h>
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/relation/relation_common.h"

namespace ppl { namespace kernel { namespace x86 {

template <relation_op_type_t _op>
inline uint8_t relation_scalar_kernel_int64(int64_t a, int64_t b);

template <>
inline uint8_t relation_scalar_kernel_int64<RELATION_GREATER>(int64_t a, int64_t b)
{
    return a > b ? 1 : 0;
}
template <>
inline uint8_t relation_scalar_kernel_int64<RELATION_GREATER_OR_EQUAL>(int64_t a, int64_t b)
{
    return a >= b ? 1 : 0;
}
template <>
inline uint8_t relation_scalar_kernel_int64<RELATION_LESS>(int64_t a, int64_t b)
{
    return a < b ? 1 : 0;
}
template <>
inline uint8_t relation_scalar_kernel_int64<RELATION_LESS_OR_EQUAL>(int64_t a, int64_t b)
{
    return a <= b ? 1 : 0;
}
template <>
inline uint8_t relation_scalar_kernel_int64<RELATION_EQUAL>(int64_t a, int64_t b)
{
    return a == b ? 1 : 0;
}
template <>
inline uint8_t relation_scalar_kernel_int64<RELATION_NOT_EQUAL>(int64_t a, int64_t b)
{
    return a != b ? 1 : 0;
}

template <relation_op_type_t _op>
ppl::common::RetCode relation_eltwise_binary_op_int64(
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst)
{
    const uint64_t length = dst_shape->GetElementsIncludingPadding();
    PRAGMA_OMP_PARALLEL_FOR()
    for (uint64_t i = 0; i < length; i++) {
        dst[i] = relation_scalar_kernel_int64<_op>(src0[i], src1[i]);
    }
    return ppl::common::RC_SUCCESS;
}

template <relation_op_type_t _op>
ppl::common::RetCode relation_ndarray_binary_op_recursive_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    const int64_t *inc0,
    const int64_t *inc1,
    const int64_t *inc_out,
    const int64_t  dim,
    const bool has_paralleled,
    uint8_t *dst)
{
    const int64_t length = dst_shape->GetDim(dim);
    if (dim == dst_shape->GetDimCount() - 1) { // last dim
        if (dst_shape->GetDim(dim) > 1 && !has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                dst[i] = relation_scalar_kernel_int64<_op>(src0[i * inc0[dim]], src1[i * inc1[dim]]);
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                dst[i] = relation_scalar_kernel_int64<_op>(src0[i * inc0[dim]], src1[i * inc1[dim]]);
            }
        }
    } else {
        if (dst_shape->GetDim(dim) > 1 && !has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < dst_shape->GetDim(dim); i++) {
                const int64_t* p_src0 = src0 + i * inc0[dim];
                const int64_t* p_src1 = src1 + i * inc1[dim];
                uint8_t* p_dst        = dst + i * inc_out[dim];
                relation_ndarray_binary_op_recursive_int64<_op>(
                    src0_shape, src1_shape, dst_shape, p_src0, p_src1, inc0, inc1, inc_out, dim + 1, true, p_dst);
            }
        } else {
            for (int64_t i = 0; i < dst_shape->GetDim(dim); i++) {
                const int64_t* p_src0 = src0 + i * inc0[dim];
                const int64_t* p_src1 = src1 + i * inc1[dim];
                uint8_t* p_dst        = dst + i * inc_out[dim];
                relation_ndarray_binary_op_recursive_int64<_op>(
                    src0_shape, src1_shape, dst_shape, p_src0, p_src1, inc0, inc1, inc_out, dim + 1, has_paralleled, p_dst);
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

inline ppl::nn::TensorShape pad_shape(
    const ppl::nn::TensorShape *shape,
    const int64_t padded_dim_count)
{
    ppl::nn::TensorShape padded_shape(*shape);
    padded_shape.SetDimCount(padded_dim_count);
    if (shape->IsScalar()) {
        for (int64_t i = 0; i < padded_dim_count; i++) {
            padded_shape.SetDim(i, 1);
        }
    } else {
        const int64_t dim_diff = padded_dim_count - shape->GetDimCount();
        for (int64_t i = 0; i < dim_diff; i++) {
            padded_shape.SetDim(i, 1);
        }
        for (int64_t i = dim_diff; i < padded_dim_count; i++) {
            padded_shape.SetDim(i, shape->GetDim(i - dim_diff));
        }
    }
    return padded_shape;
}

template <relation_op_type_t _op>
ppl::common::RetCode relation_ndarray_binary_op_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    uint8_t *dst)
{
    // pad input dim
    const int64_t dim_count = dst_shape->GetDimCount();
    if (dim_count > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    ppl::nn::TensorShape padded_tensor_shape0 = pad_shape(src0_shape, dim_count);
    ppl::nn::TensorShape padded_tensor_shape1 = pad_shape(src1_shape, dim_count);

    // prepare incs
    int64_t inc0[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t inc1[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t inc_out[PPL_X86_TENSOR_MAX_DIMS()] = {0};

    int64_t stride0    = 1;
    int64_t stride1    = 1;
    int64_t stride_out = 1;

    for (int64_t i = dim_count - 1; i >= 0; i--) {
        inc0[i]    = padded_tensor_shape0.GetDim(i) == 1 ? 0 : stride0;
        inc1[i]    = padded_tensor_shape1.GetDim(i) == 1 ? 0 : stride1;
        inc_out[i] = stride_out;

        stride0 *= padded_tensor_shape0.GetDim(i);
        stride1 *= padded_tensor_shape1.GetDim(i);
        stride_out *= dst_shape->GetDim(i);
    }

    return relation_ndarray_binary_op_recursive_int64<_op>(
        &padded_tensor_shape0, &padded_tensor_shape1, dst_shape, src0, src1, inc0, inc1, inc_out, 0, false, dst);
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_INT64_RELATION_RELATION_INT64_COMMON_H_
