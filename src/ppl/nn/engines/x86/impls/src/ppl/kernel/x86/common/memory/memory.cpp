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

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode memory_init(
    const void *src,
    const uint64_t sizeof_elem,
    const uint64_t num_elements,
    void* dst)
{
    const int64_t n_elem = num_elements;
    if (sizeof_elem == 1) {
        const uint8_t val = ((uint8_t*)src)[0];
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t i = 0; i < n_elem; i++) {
            ((uint8_t*)dst)[i] = val;
        }
    } else if (sizeof_elem == 2) {
        const uint16_t val = ((uint16_t*)src)[0];
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t i = 0; i < n_elem; i++) {
            ((uint16_t*)dst)[i] = val;
        }
    } else if (sizeof_elem == 4) {
        const uint32_t val = ((uint32_t*)src)[0];
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t i = 0; i < n_elem; i++) {
            ((uint32_t*)dst)[i] = val;
        }
    } else if (sizeof_elem == 8) {
        const uint64_t val = ((uint64_t*)src)[0];
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t i = 0; i < n_elem; i++) {
            ((uint64_t*)dst)[i] = val;
        }
    } else {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t i = 0; i < n_elem; i++) {
            for (uint64_t j = 0; j < sizeof_elem; j++) {
                ((uint8_t*)dst)[i * sizeof_elem + j] = ((uint8_t*)src)[j];
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode memory_copy(
    const void *src,
    const uint64_t num_bytes,
    void* dst)
{
    const int64_t min_block_size = 128 * 1024;
    const int64_t num_block = min<int64_t>(PPL_OMP_MAX_THREADS(), div_up(num_bytes, min_block_size));
    const int64_t block_body = num_bytes / num_block;
    const int64_t block_tail = num_bytes % num_block;

    uint8_t *l_dst        = (uint8_t*)dst;
    const uint8_t * l_src = (const uint8_t*)src;

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t n = 0; n < num_block; ++n) {
        const int64_t block_start = n * block_body + (block_tail > n ? n : block_tail);
        const int64_t block_size = block_body + (block_tail > n ? 1 : 0);
        memcpy(l_dst + block_start, l_src + block_start, block_size);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
