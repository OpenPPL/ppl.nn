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

#include "ppl/nn/utils/buffered_cpu_allocator.h"
#include "ppl/nn/common/logger.h"

#ifdef _MSC_VER
#include <windows.h>
#else
#include <string.h> // strerror
#include <sys/mman.h>
#endif

namespace ppl { namespace nn { namespace utils {

BufferedCpuAllocator::~BufferedCpuAllocator() {
    for (auto c = chunk_list_.begin(); c != chunk_list_.end(); ++c) {
#ifdef _MSC_VER
        VirtualFree(c->base, 0, MEM_RELEASE);
#else
        munmap(c->base, c->size);
#endif
    }
}

void* BufferedCpuAllocator::Alloc(uint64_t bytes) {
#ifdef _MSC_VER
    auto new_addr = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!new_addr) {
        static constexpr uint32_t max_msg_buf_size = 1024;
        char errmsg[max_msg_buf_size];
        FormatMessage(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), 0, errmsg,
                      max_msg_buf_size, nullptr);
        LOG(ERROR) << "VirtualAlloc [" << bytes << "] bytes failed: " << errmsg;
        return nullptr;
    }
#else
    /* tests show that trying to remap existing areas fails in almost all cases. we create a new mapping directly. */
    auto new_addr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (new_addr == MAP_FAILED) {
        LOG(ERROR) << "mmap [" << bytes << "] bytes failed: " << strerror(errno);
        return nullptr;
    }
#endif

    chunk_list_.push_back(ChunkInfo(new_addr, bytes));
    return new_addr;
}

}}} // namespace ppl::nn::utils
