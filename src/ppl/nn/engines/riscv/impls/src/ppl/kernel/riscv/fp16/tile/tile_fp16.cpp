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

#include "ppl/kernel/riscv/common/tile/tile_common.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode tile_ndarray_fp16(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const __fp16 *src,
    const int64_t *repeats,
    __fp16 *dst)
{
    return tile_ndarray<__fp16>(src_shape, dst_shape, src, repeats, dst);
}

}}}; // namespace ppl::kernel::riscv