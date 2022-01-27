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

#include <string.h>

#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode memory_init(
    const void* src,
    const uint64_t sizeof_elem,
    const uint64_t num_elements,
    void* dst)
{
    if (sizeof_elem == 1) {
        const uint8_t val = ((uint8_t*)src)[0];
        for (uint64_t i = 0; i < num_elements; i++) {
            ((uint8_t*)dst)[i] = val;
        }
    } else if (sizeof_elem == 2) {
        const uint16_t val = ((uint16_t*)src)[0];
        for (uint64_t i = 0; i < num_elements; i++) {
            ((uint16_t*)dst)[i] = val;
        }
    } else if (sizeof_elem == 4) {
        const uint32_t val = ((uint32_t*)src)[0];
        for (uint64_t i = 0; i < num_elements; i++) {
            ((uint32_t*)dst)[i] = val;
        }
    } else if (sizeof_elem == 8) {
        const uint64_t val = ((uint64_t*)src)[0];
        for (uint64_t i = 0; i < num_elements; i++) {
            ((uint64_t*)dst)[i] = val;
        }
    } else {
        for (uint64_t i = 0; i < num_elements; i++) {
            for (uint64_t j = 0; j < sizeof_elem; j++) {
                ((uint8_t*)dst)[i * sizeof_elem + j] = ((uint8_t*)src)[j];
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode memory_copy(
    const void* src,
    const uint64_t num_bytes,
    void* dst)
{
    memcpy((uint8_t*)dst, (uint8_t*)src, num_bytes);
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv
