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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_RUNTIME_X86_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_RUNTIME_X86_DEVICE_H_

#include "ppl/nn/engines/x86/x86_device.h"
#include "ppl/nn/engines/x86/x86_options.h"
#include "ppl/nn/utils/stack_buffer_manager.h"
#include "ppl/nn/utils/compact_buffer_manager.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn { namespace x86 {

class RuntimeX86Device final : public X86Device {
private:
    static inline uint64_t Align(uint64_t x, uint64_t n) {
        return (x + n - 1) & (~(n - 1));
    }

public:
    RuntimeX86Device(uint64_t alignment, ppl::common::isa_t isa, uint32_t mm_policy) : X86Device(alignment, isa) {
        if (mm_policy == X86_MM_MRU) {
            buffer_manager_.reset(new utils::StackBufferManager(GetAllocator()));
        } else if (mm_policy == X86_MM_COMPACT) {
            buffer_manager_.reset(new utils::CompactBufferManager(GetAllocator()));
        }
    }

    ~RuntimeX86Device() {
        LOG(DEBUG) << "buffer manager[" << buffer_manager_->GetName() << "] allocates ["
                   << buffer_manager_->GetAllocatedBytes() << "] bytes.";
        buffer_manager_->Free(&shared_tmp_buffer_);
        buffer_manager_.reset();
    }

    ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) override {
        if (bytes > tmp_buffer_size_) {
            auto status = buffer_manager_->Realloc(bytes, &shared_tmp_buffer_);
            if (status == ppl::common::RC_SUCCESS) {
                tmp_buffer_size_ = bytes;
                *buffer = shared_tmp_buffer_;
            }
            return status;
        }

        *buffer = shared_tmp_buffer_;
        return ppl::common::RC_SUCCESS;
    }

    void FreeTmpBuffer(BufferDesc*) override {}

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc* buffer) override {
        bytes = Align(bytes, 256);
        return buffer_manager_->Realloc(bytes, buffer);
    }

    void Free(BufferDesc* buffer) override {
        buffer_manager_->Free(buffer);
    }

private:
    std::unique_ptr<utils::BufferManager> buffer_manager_;
    BufferDesc shared_tmp_buffer_;
    uint64_t tmp_buffer_size_ = 0;
};

}}} // namespace ppl::nn::x86

#endif
