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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_EXPAND_EXPAND_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_EXPAND_EXPAND_COMMON_H_

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <memory>

#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

template <typename eT>
ppl::common::RetCode expand_ndarray_recursive(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    const int64_t *stride_src,
    const int64_t *stride_dst,
    const uint32_t dim_idx,
    const bool has_paralleled,
    eT *dst)
{
    if (dim_idx == dst_shape->GetDimCount() - 1) { // last dim
        if (src_shape->GetDim(dim_idx) == dst_shape->GetDim(dim_idx)) { // no broadcast
            if (dst_shape->GetDim(dim_idx) > 1 && !has_paralleled) {
                PRAGMA_OMP_PARALLEL_FOR()
                for (int64_t i = 0; i < dst_shape->GetDim(dim_idx); i++) {
                    dst[i] = src[i];
                }
            } else {
                for (int64_t i = 0; i < dst_shape->GetDim(dim_idx); i++) {
                    dst[i] = src[i];
                }
            }
        } else { // broadcast
            const eT val = src[0];
            if (dst_shape->GetDim(dim_idx) > 1 && !has_paralleled) {
                PRAGMA_OMP_PARALLEL_FOR()
                for (int64_t i = 0; i < dst_shape->GetDim(dim_idx); i++) {
                    dst[i] = val;
                }
            } else {
                for (int64_t i = 0; i < dst_shape->GetDim(dim_idx); i++) {
                    dst[i] = val;
                }
            }
        }
    } else {
        if (src_shape->GetDim(dim_idx) == dst_shape->GetDim(dim_idx)) { // no broadcast
            if (dst_shape->GetDim(dim_idx) > 1 && !has_paralleled) {
                PRAGMA_OMP_PARALLEL_FOR()
                for (int64_t i = 0; i < dst_shape->GetDim(dim_idx); i++) {
                    expand_ndarray_recursive<eT>(
                        src_shape, dst_shape, src + i * stride_src[dim_idx], stride_src, stride_dst, dim_idx + 1, true, dst + i * stride_dst[dim_idx]);
                }
            } else {
                for (int64_t i = 0; i < dst_shape->GetDim(dim_idx); i++) {
                    expand_ndarray_recursive<eT>(
                        src_shape, dst_shape, src + i * stride_src[dim_idx], stride_src, stride_dst, dim_idx + 1, has_paralleled, dst + i * stride_dst[dim_idx]);
                }
            }
        } else { // broadcast
            expand_ndarray_recursive<eT>(src_shape, dst_shape, src, stride_src, stride_dst, dim_idx + 1, has_paralleled, dst);
            if (dst_shape->GetDim(dim_idx) - 1 > 1 && !has_paralleled) {
                PRAGMA_OMP_PARALLEL_FOR()
                for (int64_t i = 1; i < dst_shape->GetDim(dim_idx); i++) {
                    memcpy(dst + i * stride_dst[dim_idx], dst, stride_dst[dim_idx] * sizeof(eT));
                }
            } else {
                for (int64_t i = 1; i < dst_shape->GetDim(dim_idx); i++) {
                    memcpy(dst + i * stride_dst[dim_idx], dst, stride_dst[dim_idx] * sizeof(eT));
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

inline std::shared_ptr<ppl::common::TensorShape> pad_shape(
    const ppl::common::TensorShape *shape,
    const int64_t padded_dim_count)
{
    auto padded_shape = std::make_shared<ppl::common::TensorShape>(*shape);
    padded_shape->SetDimCount(padded_dim_count);
    if (shape->IsScalar()) {
        for (int64_t i = 0; i < padded_dim_count; i++) {
            padded_shape->SetDim(i, 1);
        }
    } else {
        const int32_t dim_diff = padded_dim_count - shape->GetDimCount();
        for (int64_t i = 0; i < dim_diff; i++) {
            padded_shape->SetDim(i, 1);
        }
        for (int64_t i = dim_diff; i < padded_dim_count; i++) {
            padded_shape->SetDim(i, shape->GetDim(i - dim_diff));
        }
    }
    return padded_shape;
}

template <typename eT>
ppl::common::RetCode expand_ndarray_common(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    eT *dst)
{
    const int64_t dim_count = dst_shape->GetDimCount();
    if (dim_count > PPL_RISCV_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    auto padded_input_shape = pad_shape(src_shape, dim_count);

    // prepare incs
    int64_t stride_src[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};
    int64_t stride_dst[PPL_RISCV_TENSOR_MAX_DIMS()] = {0};

    stride_src[dim_count - 1] = 1;
    stride_dst[dim_count - 1] = 1;
    for (int64_t i = dim_count - 2; i >= 0; i--) {
        stride_src[i] = stride_src[i + 1] * padded_input_shape->GetDim(i + 1);
        stride_dst[i] = stride_dst[i + 1] * dst_shape->GetDim(i + 1);
    }

    return expand_ndarray_recursive<eT>(padded_input_shape.get(), dst_shape, src, stride_src, stride_dst, 0, false, dst);
}

}}}; // namespace ppl::kernel::riscv

#endif // __ST_PPL_KERNEL_RISCV_COMMON_EXPAND_EXPAND_COMMON_H_