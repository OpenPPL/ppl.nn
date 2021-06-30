#ifndef __ST_PPL_KERNEL_X86_INT64_ARITHMETIC_ARITHMETIC_KERNEL_INT64_H_
#define __ST_PPL_KERNEL_X86_INT64_ARITHMETIC_ARITHMETIC_KERNEL_INT64_H_

#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/arithmetic/arithmetic_common.h"

namespace ppl { namespace kernel { namespace x86 {

template <arithmetic_op_type_t _op>
inline int64_t arithmetic_scalar_kernel_int64(int64_t a, int64_t b);

template <>
inline int64_t arithmetic_scalar_kernel_int64<ARITHMETIC_ADD>(int64_t a, int64_t b)
{
    return a + b;
}
template <>
inline int64_t arithmetic_scalar_kernel_int64<ARITHMETIC_SUB>(int64_t a, int64_t b)
{
    return a - b;
}
template <>
inline int64_t arithmetic_scalar_kernel_int64<ARITHMETIC_MUL>(int64_t a, int64_t b)
{
    return a * b;
}
template <>
inline int64_t arithmetic_scalar_kernel_int64<ARITHMETIC_DIV>(int64_t a, int64_t b)
{
    return a / b;
}

struct parallel_block {
    int64_t id;
    int64_t start[PPL_X86_TENSOR_MAX_DIMS()];
    int64_t end[PPL_X86_TENSOR_MAX_DIMS()];
    int64_t idx[PPL_X86_TENSOR_MAX_DIMS()];
};

inline void pad_shape(
    const ppl::nn::TensorShape *shape,
    const int64_t padded_dim_count,
    int64_t *padded_shape)
{
    const int64_t dim_diff = padded_dim_count - shape->GetRealDimCount();
    for (int64_t i = 0; i < dim_diff; i++) {
        padded_shape[i] = 1;
    }
    for (int64_t i = dim_diff; i < padded_dim_count; i++) {
        padded_shape[i] = shape->GetDim(i - dim_diff);
    }
}

inline void idx2dims(
    const int64_t idx,
    const int64_t *shape,
    const int64_t dim_count,
    int64_t *dims)
{
    int64_t _idx = idx;
    for (int64_t i = dim_count - 1; i >= 0; i--) {
        dims[i] = _idx % shape[i];
        _idx /= shape[i];
    }
}

inline bool is_first_dim(parallel_block* block, const int64_t dim_idx)
{
    bool is_first = true;
    for (int64_t i = 0; i < dim_idx; i++) {
        if (block->idx[i] != block->start[i]) {
            is_first = false;
            break;
        }
    }
    return is_first;
}

inline bool is_last_dim(parallel_block* block, const int64_t dim_idx)
{
    bool is_last = true;
    for (int64_t i = 0; i < dim_idx; i++) {
        if (block->idx[i] != block->end[i]) {
            is_last = false;
            break;
        }
    }
    return is_last;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_INT64_ARITHMETIC_ARITHMETIC_KERNEL_INT64_H_
