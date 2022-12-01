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

#include <math.h>
#include <vector>

#include "ppl/kernel/riscv/common/internal_include.h"
#include "ppl/kernel/riscv/common/resize2d/resize2d_ndarray_common.h"
#include "ppl/kernel/riscv/common/resize2d/resize2d_nbcx_common.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode resize2d_ndarray_pytorch_linear_floor_fp16(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const __fp16* src,
    const float scale_h,
    const float scale_w,
    __fp16* dst)
{
    return resize2d_ndarray_pytorch_linear_floor_common<__fp16>(src_shape, dst_shape, src, scale_h, scale_w, dst);
}

ppl::common::RetCode resize2d_ndarray_asymmetric_nearest_floor_fp16(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const __fp16* src,
    const float scale_h,
    const float scale_w,
    __fp16* dst)
{
    return resize2d_ndarray_asymmetric_nearest_floor_common<__fp16>(src_shape, dst_shape, src, scale_h, scale_w, dst);
}

ppl::common::RetCode resize2d_ndarray_pytorch_cubic_floor_fp16(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const __fp16* src,
    const float scale_h,
    const float scale_w,
    const float cubic_coeff_a,
    __fp16* dst)
{
    return resize2d_ndarray_pytorch_cubic_floor_common<__fp16>(src_shape, dst_shape, src, scale_h, scale_w, cubic_coeff_a, dst);
}

ppl::common::RetCode resize2d_nbcx_pytorch_linear_floor_fp16(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const __fp16* src,
    const float scale_h,
    const float scale_w,
    __fp16* dst)
{
    return resize2d_nbcx_pytorch_linear_floor_common<float16xm1_t, __fp16, 8>(src_shape, dst_shape, src, scale_h, scale_w, dst);
}

ppl::common::RetCode resize2d_nbcx_asymmetric_nearest_floor_fp16(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const __fp16* src,
    const float scale_h,
    const float scale_w,
    __fp16* dst)
{
    return resize2d_nbcx_asymmetric_nearest_floor_common<float16xm1_t, __fp16, 8>(src_shape, dst_shape, src, scale_h, scale_w, dst);
}

}}} // namespace ppl::kernel::riscv
