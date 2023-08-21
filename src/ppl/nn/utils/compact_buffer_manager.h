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

#ifndef _ST_HPC_PPL_NN_UTILS_COMPACT_BUFFER_MANAGER_H_
#define _ST_HPC_PPL_NN_UTILS_COMPACT_BUFFER_MANAGER_H_

#include "ppl/common/compact_addr_manager.h"
#include "ppl/nn/utils/buffer_manager.h"
#include <functional>

namespace ppl { namespace nn { namespace utils {

class CompactBufferManager final : public BufferManager {
public:
    CompactBufferManager(ppl::common::CompactAddrManager::Allocator* ar, uint64_t alignment)
        : BufferManager("CompactBufferManager"), alignment_(alignment), mgr_(ar) {
        get_buffered_bytes_ = [ar]() -> uint64_t {
            return ar->GetAllocatedSize();
        };
    }

    CompactBufferManager(ppl::common::CompactAddrManager::VMAllocator* vmr, uint64_t alignment)
        : BufferManager("CompactBufferManager"), alignment_(alignment), mgr_(vmr) {
        get_buffered_bytes_ = [vmr]() -> uint64_t {
            return vmr->GetAllocatedSize();
        };
    }

    uint64_t GetBufferedBytes() const override {
        return get_buffered_bytes_();
    }

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc* buffer) override;
    void Free(BufferDesc* buffer) override;

private:
    uint64_t alignment_;
    ppl::common::CompactAddrManager mgr_;
    std::function<uint64_t()> get_buffered_bytes_;
};

}}} // namespace ppl::nn::utils

#endif
