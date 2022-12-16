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

#include "ppl/kernel/x86/common/non_zero/non_zero_common.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode non_zero_ndarray_bool(
    const ppl::common::TensorShape *src_shape,
    const uint8_t *src,
    void *temp_buffer,
    int64_t *non_zero_num,
    int64_t *dst)
{
    return non_zero_ndarray_common<uint8_t>(src_shape, src, temp_buffer, non_zero_num, dst);
}

}}}; // namespace ppl::kernel::x86
