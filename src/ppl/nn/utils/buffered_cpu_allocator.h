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

#ifndef _ST_HPC_PPL_NN_UTILS_BUFFERED_CPU_ALLOCATOR_H_
#define _ST_HPC_PPL_NN_UTILS_BUFFERED_CPU_ALLOCATOR_H_

#include "ppl/common/compact_addr_manager.h"
#include "ppl/common/retcode.h"

namespace ppl { namespace nn { namespace utils {

class BufferedCpuAllocator final : public ppl::common::CompactAddrManager::VMAllocator {
public:
    BufferedCpuAllocator();
    ~BufferedCpuAllocator();
    ppl::common::RetCode Init(uint64_t max_mem_bytes = UINT64_MAX);
    uint64_t Extend(uint64_t bytes) override;
    uintptr_t GetReservedBase() const override {
        return (uintptr_t)base_;
    }
    uint64_t GetAllocatedSize() const override {
        return (uintptr_t)cursor_ - (uintptr_t)base_;
    }

private:
    void* base_ = nullptr;
    void* cursor_ = nullptr;
    uint64_t addr_len_ = 0;

private:
    BufferedCpuAllocator(const BufferedCpuAllocator&) = delete;
    void operator=(const BufferedCpuAllocator&) = delete;
};

}}} // namespace ppl::nn::utils

#endif
