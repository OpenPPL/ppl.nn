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

#include "ppl/kernel/riscv/common/transpose/transpose_common.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode transpose_ndarray_fp32(
    const float* src,
    float* dst,

    const int32_t* perm,
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape)
{
    return transpose_ndarray<float>(src, dst, perm, src_shape, dst_shape);
}

ppl::common::RetCode transpose_ndarray_continous2d_fp32(
    const float* src,
    float* dst,

    const ppl::common::TensorShape* src_shape,
    const uint32_t axis0,
    const uint32_t axis1)
{
    return transpose_ndarray_continous2d<float>(src, dst, src_shape, axis0, axis1);
}

}}}; //  namespace ppl::kernel::riscv
