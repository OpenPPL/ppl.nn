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

#include "ppl/kernel/riscv/common/internal_include.h"
#include "ppl/kernel/riscv/common/split/split_common.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode split_n8cx_fp16(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape** dst_shape_list,
    const __fp16* src,
    const int32_t slice_axis,
    const int32_t num_dst,
    __fp16** dst_list)
{
    return split_nxcx<__fp16, 8>(src_shape, dst_shape_list, src, slice_axis, num_dst, dst_list);
}

}}} // namespace ppl::kernel::riscv
