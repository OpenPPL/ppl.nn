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

#ifndef _ST_HPC_PPL_NN_UTILS_STACK_BUFFER_MANAGER_H_
#define _ST_HPC_PPL_NN_UTILS_STACK_BUFFER_MANAGER_H_

#include "ppl/common/allocator.h"
#include "ppl/nn/utils/buffer_manager.h"
#include <vector>

namespace ppl { namespace nn { namespace utils {

class StackBufferManager final : public BufferManager {
public:
    StackBufferManager(ppl::common::Allocator* ar, bool use_bestfit = false)
        : BufferManager("StackBufferManager"), use_bestfit_(use_bestfit), allocator_(ar) {
        buffer_list_.reserve(32);
        buffer_stack_.reserve(32);
    };
    ~StackBufferManager();

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc* buffer) override;
    void Free(BufferDesc* buffer) override;
    uint64_t GetAllocatedBytes() const override {
        return allocated_bytes_;
    }

private:
    struct BufferAddrAndSize {
        void* addr;
        uint64_t size;
    };

    inline void AllocManagedBuffer(uint64_t bytes, BufferAddrAndSize* managed_buffer) {
        managed_buffer->addr = allocator_->Alloc(bytes);
        if (managed_buffer->addr) {
            managed_buffer->size = bytes;
            allocated_bytes_ += bytes;
        } else {
            managed_buffer->size = 0;
        }
    }

    inline void ReallocManagedBuffer(uint64_t bytes, BufferAddrAndSize* managed_buffer) {
        if (bytes > managed_buffer->size) {
            allocator_->Free(managed_buffer->addr);
            allocated_bytes_ -= managed_buffer->size;
            AllocManagedBuffer(bytes, managed_buffer);
        }
    }

    // Find the best fit buffer which is larger than the required bytes
    inline int FindBestEmptyBuffer(uint64_t bytes) {
        int buffer_stack_size = buffer_stack_.size();
        uint64_t diff = bytes;
        uint64_t tmp_diff = diff;
        int res = buffer_stack_.back();
        int idx = buffer_stack_size - 1;
        for (int i = 0; i < buffer_stack_size; i++) {
            if (buffer_list_[buffer_stack_[i]].size > bytes) {
                tmp_diff = buffer_list_[buffer_stack_[i]].size - bytes;
            } else {
                tmp_diff = bytes - buffer_list_[buffer_stack_[i]].size;
            }
            if (tmp_diff < diff) {
                diff = tmp_diff;
                res = buffer_stack_[i];
                idx = i;
            }
        }
        buffer_stack_[idx] = buffer_stack_.back();
        buffer_stack_.pop_back();
        return res;
    }

private:
    const bool use_bestfit_;
    uint64_t allocated_bytes_ = 0;
    ppl::common::Allocator* allocator_;
    std::vector<int64_t> buffer_stack_;
    std::vector<BufferAddrAndSize> buffer_list_;
};

}}} // namespace ppl::nn::utils

#endif
