// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for arithmeticitional information
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

#include <vector>
#include <algorithm>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/memory.h"
#include "ppl/kernel/arm_server/reduce/neon/reduce_common.h"
#include "ppl/kernel/arm_server/reduce/neon/reduce_ndarray_common.h"
#include "ppl/kernel/arm_server/reduce/neon/reduce_nbcx_common.h"

#include "ppl/kernel/arm_server/reduce/neon/reduce_kernel.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, reduce_op_type_t op_type>
static ppl::common::RetCode reduce(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    const int32_t *axes,
    const int32_t num_axes,
    eT *dst)
{
    if (src_shape->CalcElementsExcludingPadding() == dst_shape->CalcElementsExcludingPadding()) { // no actual reduce happened, just copy
        return memory_copy(src, src_shape->CalcBytesIncludingPadding(), dst);
    }

    std::vector<int32_t> real_axes(num_axes); // change negative axes to positive & sort axes
    for (int64_t i = 0; i < num_axes; i++) {
        real_axes[i] = axes[i] >= 0 ? axes[i] : axes[i] + src_shape->GetDimCount();
    }
    std::sort(real_axes.begin(), real_axes.end());

    const auto data_format = src_shape->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        bool continous_reduce_axis = false;
        if (continous_reduce_axis) {
            // TODO: implement this
            return ppl::common::RC_UNSUPPORTED;
        } else {
            return reduce_ndarray_common<eT, op_type>(src_shape, dst_shape, src, real_axes.data(), num_axes, dst);
        }
    }

    // NBCX
    if (std::is_same<eT, float>::value) {
        if (data_format == ppl::common::DATAFORMAT_N4CX) {
            return reduce_nbcx_common<float, 4, op_type>(src_shape, dst_shape, (const float *)src, real_axes.data(), num_axes, (float *)dst);
        }
    }
#ifdef PPLNN_USE_ARMV8_2_FP16
    if (std::is_same<eT, __fp16>::value) {
        if (data_format == ppl::common::DATAFORMAT_N8CX) {
            return reduce_nbcx_common<__fp16, 8, op_type>(src_shape, dst_shape, (const __fp16 *)src, real_axes.data(), num_axes, (__fp16 *)dst);
        }
    }
#endif

    return ppl::common::RC_UNSUPPORTED;
}

template <reduce_op_type_t op_type>
static ppl::common::RetCode reduce_wrapper(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int32_t *axes,
    const int32_t num_axes,
    void *dst)
{
    const auto data_type = src_shape->GetDataType();
    switch (data_type) {
        case ppl::common::DATATYPE_FLOAT32: return reduce<float, op_type>(src_shape, dst_shape, (const float *)src, axes, num_axes, (float *)dst);
        case ppl::common::DATATYPE_INT64: return reduce<int64_t, op_type>(src_shape, dst_shape, (const int64_t *)src, axes, num_axes, (int64_t *)dst);
#ifdef PPLNN_USE_ARMV8_2_FP16
        case ppl::common::DATATYPE_FLOAT16: return reduce<__fp16, op_type>(src_shape, dst_shape, (const __fp16 *)src, axes, num_axes, (__fp16 *)dst);
#endif
        default: break;
    }
    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode reduce_max(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int32_t *axes,
    const int32_t num_axes,
    void *dst)
{
    return reduce_wrapper<REDUCE_MAX>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_min(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int32_t *axes,
    const int32_t num_axes,
    void *dst)
{
    return reduce_wrapper<REDUCE_MIN>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_sum(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int32_t *axes,
    const int32_t num_axes,
    void *dst)
{
    return reduce_wrapper<REDUCE_SUM>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_mean(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int32_t *axes,
    const int32_t num_axes,
    void *dst)
{
    return reduce_wrapper<REDUCE_MEAN>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_prod(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int32_t *axes,
    const int32_t num_axes,
    void *dst)
{
    return reduce_wrapper<REDUCE_PROD>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_sum_square(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int32_t *axes,
    const int32_t num_axes,
    void *dst)
{
    return reduce_wrapper<REDUCE_SUM_SQUARE>(src_shape, dst_shape, src, axes, num_axes, dst);
}

ppl::common::RetCode reduce_abs_sum(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const int32_t *axes,
    const int32_t num_axes,
    void *dst)
{
    return reduce_wrapper<REDUCE_ABS_SUM>(src_shape, dst_shape, src, axes, num_axes, dst);
}

}}}} // namespace ppl::kernel::arm_server::neon
