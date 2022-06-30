// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef __ST_PPL_KERNEL_X86_BOOL_LOGICAL_BOOL_COMMON_H_
#define __ST_PPL_KERNEL_X86_BOOL_LOGICAL_BOOL_COMMON_H_

#include <immintrin.h>
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/logical/logical_common.h"

namespace ppl { namespace kernel { namespace x86 {

template <logical_op_type_t _op>
inline uint8_t logical_scalar_kernel_bool(uint8_t a, uint8_t b);

template <>
inline uint8_t logical_scalar_kernel_bool<LOGICAL_AND>(uint8_t a, uint8_t b)
{
    return a && b ? 1 : 0;
}

template <>
inline uint8_t logical_scalar_kernel_bool<LOGICAL_OR>(uint8_t a, uint8_t b)
{
    return a || b ? 1 : 0;
}

template <logical_op_type_t _op>
ppl::common::RetCode logical_eltwise_binary_op_bool(
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *src0,
    const uint8_t *src1,
    uint8_t *dst)
{
    const int64_t length = dst_shape->CalcElementsIncludingPadding();
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < length; i++) {
        dst[i] = logical_scalar_kernel_bool<_op>(src0[i], src1[i]);
    }
    return ppl::common::RC_SUCCESS;
}

template <logical_op_type_t _op>
ppl::common::RetCode logical_ndarray_binary_op_recursive_bool(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *src0,
    const uint8_t *src1,
    const uint64_t *inc0,
    const uint64_t *inc1,
    const uint64_t *inc_out,
    const uint64_t dim,
    const bool has_paralleled,
    uint8_t *dst)
{
    const int64_t length = dst_shape->GetDim(dim);
    if (dim == dst_shape->GetDimCount() - 1) { // last dim
        if (dst_shape->GetDim(dim) > 1 && !has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                dst[i] = logical_scalar_kernel_bool<_op>(src0[i * inc0[dim]], src1[i * inc1[dim]]);
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                dst[i] = logical_scalar_kernel_bool<_op>(src0[i * inc0[dim]], src1[i * inc1[dim]]);
            }
        }
    } else {
        if (dst_shape->GetDim(dim) > 1 && !has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < dst_shape->GetDim(dim); i++) {
                const uint8_t* p_src0 = src0 + i * inc0[dim];
                const uint8_t* p_src1 = src1 + i * inc1[dim];
                uint8_t* p_dst        = dst + i * inc_out[dim];
                logical_ndarray_binary_op_recursive_bool<_op>(
                    src0_shape, src1_shape, dst_shape, p_src0, p_src1, inc0, inc1, inc_out, dim + 1, true, p_dst);
            }
        } else {
            for (int64_t i = 0; i < dst_shape->GetDim(dim); i++) {
                const uint8_t* p_src0 = src0 + i * inc0[dim];
                const uint8_t* p_src1 = src1 + i * inc1[dim];
                uint8_t* p_dst        = dst + i * inc_out[dim];
                logical_ndarray_binary_op_recursive_bool<_op>(
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

template <logical_op_type_t _op>
ppl::common::RetCode logical_ndarray_binary_op_bool(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *src0,
    const uint8_t *src1,
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
    uint64_t inc0[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    uint64_t inc1[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    uint64_t inc_out[PPL_X86_TENSOR_MAX_DIMS()] = {0};

    uint64_t stride0    = 1;
    uint64_t stride1    = 1;
    uint64_t stride_out = 1;

    for (int64_t i = dim_count - 1; i >= 0; i--) {
        inc0[i]    = padded_tensor_shape0.GetDim(i) == 1 ? 0 : stride0;
        inc1[i]    = padded_tensor_shape1.GetDim(i) == 1 ? 0 : stride1;
        inc_out[i] = stride_out;

        stride0 *= padded_tensor_shape0.GetDim(i);
        stride1 *= padded_tensor_shape1.GetDim(i);
        stride_out *= dst_shape->GetDim(i);
    }

    return logical_ndarray_binary_op_recursive_bool<_op>(
        &padded_tensor_shape0, &padded_tensor_shape1, dst_shape, src0, src1, inc0, inc1, inc_out, 0, false, dst);
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_BOOL_LOGICAL_LOGICAL_BOOL_COMMON_H_
