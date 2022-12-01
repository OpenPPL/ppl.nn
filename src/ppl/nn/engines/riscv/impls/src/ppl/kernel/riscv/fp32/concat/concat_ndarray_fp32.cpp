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

#include "ppl/kernel/riscv/common/concat/concat_common.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode concat_ndarray_fp32(
    const float** src_list,
    float* dst,

    const ppl::common::TensorShape** src_shape_list,
    const int32_t num_src,
    const int32_t c_axis)
{
    return concat_ndarray<float>(src_list, dst, src_shape_list, num_src, c_axis);
}

}}}; // namespace ppl::kernel::riscv
