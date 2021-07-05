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

#include "ppl/nn/utils/compact_buffer_manager.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

static inline uint64_t Align(uint64_t x, uint64_t n) {
    return (x + n - 1) & (~(n - 1));
}

RetCode CompactBufferManager::Realloc(uint64_t bytes, BufferDesc* buffer) {
    if (buffer->addr) {
        mgr_.Free(buffer->addr, buffer->desc);
    }

    if (bytes == 0) {
        buffer->addr = nullptr;
        buffer->desc = 0;
        return RC_SUCCESS;
    }

    bytes = Align(bytes, mgr_.GetAllocator()->GetAlignment());
    buffer->addr = mgr_.Alloc(bytes);
    if (!buffer->addr) {
        buffer->desc = 0;
        return RC_OUT_OF_MEMORY;
    }

    buffer->desc = bytes;
    return RC_SUCCESS;
}

void CompactBufferManager::Free(BufferDesc* buffer) {
    if (buffer->addr) {
        mgr_.Free(buffer->addr, buffer->desc);
        buffer->addr = nullptr;
    }
}

}}} // namespace ppl::nn::utils
