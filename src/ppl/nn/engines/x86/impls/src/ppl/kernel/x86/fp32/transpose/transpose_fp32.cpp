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

#include "ppl/kernel/x86/common/transpose/transpose_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode transpose_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *perm,
    float *dst)
{
    return transpose_ndarray<float>(src_shape, dst_shape, perm, src, dst);
}

ppl::common::RetCode transpose_ndarray_continous2d_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const uint32_t axis0,
    const uint32_t axis1,
    float *dst)
{
    return transpose_ndarray_continous2d<float>(src_shape, axis0, axis1, src, dst);
}

}}}; // namespace ppl::kernel::x86
