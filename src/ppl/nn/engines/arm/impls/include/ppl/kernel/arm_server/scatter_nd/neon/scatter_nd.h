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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_SCATTER_ND_NEON_SCATTER_ND_H_
#define __ST_PPL_KERNEL_ARM_SERVER_SCATTER_ND_NEON_SCATTER_ND_H_

#include "ppl/kernel/arm_server/common/general_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode scatter_nd_ndarray(
    const ppl::common::TensorShape *input_shape,
    const void *src,
    const void *updates,
    const int64_t *indices,
    const int32_t *strides,
    const int64_t src_length,
    const int64_t inner_dim,
    const int64_t num_indices,
    const int64_t indices_dim,
    void *dst);

}}}}; // namespace ppl::kernel::arm_server::neon

#endif
