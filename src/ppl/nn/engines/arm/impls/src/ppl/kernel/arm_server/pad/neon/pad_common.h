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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_PAD_NEON_PAD_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_PAD_NEON_PAD_COMMON_H_

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

enum pad_mode_type_t {
    PAD_MODE_CONSTANT = 0,
    PAD_MODE_REFLECT  = 1,
    PAD_MODE_EDGE     = 2
};

static inline int32_t get_reflect_idx(
    int32_t idx,
    int32_t length)
{
    while (idx < 0 || idx >= length) {
        if (idx < 0) {
            idx = -idx;
        }
        if (idx >= length) {
            idx = 2 * length - 2 - idx;
        }
    }
    return idx;
}

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_PAD_NEON_PAD_COMMON_H_