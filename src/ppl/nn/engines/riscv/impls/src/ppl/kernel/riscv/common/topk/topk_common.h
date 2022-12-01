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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_TOPK_TOPK_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_TOPK_TOPK_COMMON_H_

#include "ppl/kernel/riscv/common/internal_include.h"

#include <algorithm>
#include <functional>

namespace ppl { namespace kernel { namespace riscv {

enum sort_order_t {
    SMALLEST = 0,
    LARRGEST = 1
};

template <typename T, sort_order_t order>
struct element_t {
    T data;
    uint32_t idx;
    inline bool operator<(const element_t &e) const
    {
        if (order == SMALLEST) {
            return this->data < e.data || (this->data == e.data && this->idx < e.idx);
        } else {
            return this->data > e.data || (this->data == e.data && this->idx < e.idx);
        }
    }
};

template <typename T>
uint64_t topk_ndarray_get_buffer_bytes(
    const ppl::common::TensorShape *src_shape,
    const int32_t axis)
{
    const uint64_t axis_dim         = src_shape->GetDim(axis);
    const uint64_t temp_buffer_size = axis_dim * sizeof(element_t<T, SMALLEST>);
    return temp_buffer_size;
}

template <typename T, sort_order_t order, bool sorted>
ppl::common::RetCode topk_ndarray_kernel_common(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *value_shape,
    const ppl::common::TensorShape *indices_shape,
    const T *src,
    const int64_t k,
    const int32_t axis,
    void *temp_buffer,
    T *values,
    int64_t *indices)
{
    const uint32_t axis_dim = src_shape->GetDim(axis);

    int64_t outer_dim = 1;
    int64_t inner_dim = 1;

    for (int32_t i = 0; i < axis; i++) {
        outer_dim *= src_shape->GetDim(i);
    }
    for (uint32_t i = axis + 1; i < src_shape->GetDimCount(); i++) {
        inner_dim *= src_shape->GetDim(i);
    }

    const uint64_t temp_buffer_size = axis_dim * sizeof(element_t<T, order>);

    for (int64_t od = 0; od < outer_dim; od++) {
        for (int64_t id = 0; id < inner_dim; id++) {
            element_t<T, order> *l_temp = (element_t<T, order> *)temp_buffer;
            const T *l_src              = src + od * axis_dim * inner_dim + id;
            T *l_values                 = values + od * k * inner_dim + id;
            int64_t *l_ind              = indices + od * k * inner_dim + id;
            for (uint32_t i = 0; i < axis_dim; i++) {
                l_temp[i].data = l_src[i * inner_dim];
                l_temp[i].idx  = i;
            }
            std::nth_element(l_temp, l_temp + k, l_temp + axis_dim, std::less<element_t<T, order>>());
            if (sorted) {
                std::sort(l_temp, l_temp + k, std::less<element_t<T, order>>());
            }
            for (uint32_t i = 0; i < k; i++) {
                l_values[i * inner_dim] = l_temp[i].data;
                l_ind[i * inner_dim]    = l_temp[i].idx;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename T>
ppl::common::RetCode topk_ndarray_common(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *value_shape,
    const ppl::common::TensorShape *indices_shape,
    const T *src,
    const int64_t k,
    const int32_t axis,
    const int32_t largest,
    const int32_t sorted,
    void *temp_buffer,
    T *values,
    int64_t *indices)
{
    if (sorted) {
        if (largest) {
            return topk_ndarray_kernel_common<T, LARRGEST, true>(src_shape, value_shape, indices_shape, src, k, axis, temp_buffer, values, indices);
        } else {
            return topk_ndarray_kernel_common<T, SMALLEST, true>(src_shape, value_shape, indices_shape, src, k, axis, temp_buffer, values, indices);
        }
    } else {
        if (largest) {
            return topk_ndarray_kernel_common<T, LARRGEST, false>(src_shape, value_shape, indices_shape, src, k, axis, temp_buffer, values, indices);
        } else {
            return topk_ndarray_kernel_common<T, SMALLEST, false>(src_shape, value_shape, indices_shape, src, k, axis, temp_buffer, values, indices);
        }
    }
}

}}}; // namespace ppl::kernel::riscv

#endif // __ST_PPL_KERNEL_RISCV_COMMON_TOPK_TOPK_COMMON_H_