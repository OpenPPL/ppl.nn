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

#include "ppl/common/compact_memory_manager.h"
#include "ppl/nn/utils/buffer_manager.h"

namespace ppl { namespace nn { namespace utils {

class CompactBufferManager final : public BufferManager {
public:
    CompactBufferManager(ppl::common::Allocator* ar, uint64_t alignment, uint64_t block_size = 1048576)
        : BufferManager("CompactBufferManager"), alignment_(alignment), mgr_(ar, block_size) {}

    /**
       @brief replaces all blocks with a new single block.
       @note make sure that this manager is not used when calling Defragment().
    */
    ppl::common::RetCode Defragment() {
        return mgr_.Defragment();
    }

    /**
       @brief frees all blocks and reserve a new single block with size `bytes`.
       @note make sure that this manager is not used when calling `Reset()`.
       @param reserved_bytes if it is 0, all blocks are freed.
    */
    ppl::common::RetCode Reset(uint64_t reserved_bytes = 0) {
        return mgr_.Reset(reserved_bytes);
    }

    uint64_t GetAllocatedBytes() const override {
        return mgr_.GetAllocatedBytes();
    }

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc* buffer) override;
    void Free(BufferDesc* buffer) override;

private:
    uint64_t alignment_;
    ppl::common::CompactMemoryManager mgr_;
};

}}} // namespace ppl::nn::utils

#endif
