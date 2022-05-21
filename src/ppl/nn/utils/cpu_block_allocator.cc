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

#include "ppl/nn/utils/cpu_block_allocator.h"
#include "ppl/nn/common/logger.h"
#include <errno.h>
#include <utility> // make_pair
using namespace std;

#ifdef _MSC_VER
#include <windows.h>
#else
#include <string.h> // strerror
#include <sys/mman.h>
#endif

namespace ppl { namespace nn { namespace utils {

static inline void DoFree(void* base, uint64_t bytes) {
#ifdef _MSC_VER
    (void)bytes;
    VirtualFree(base, 0, MEM_RELEASE);
#else
    munmap(base, bytes);
#endif
}

CpuBlockAllocator::~CpuBlockAllocator() {
    if (!addr2size_.empty()) {
        LOG(WARNING) << "[" << addr2size_.size() << "] block(s) are not freed.";
        for (auto x = addr2size_.begin(); x != addr2size_.end(); ++x) {
            DoFree(x->first, x->second);
        }
    }
}

void* CpuBlockAllocator::Alloc(uint64_t bytes) {
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

    addr2size_.insert(make_pair(new_addr, bytes));
    return new_addr;
}

void CpuBlockAllocator::Free(void* ptr) {
    auto ref = addr2size_.find(ptr);
    if (ref != addr2size_.end()) {
        DoFree(ref->first, ref->second);
        addr2size_.erase(ref);
    }
}

}}} // namespace ppl::nn::utils
