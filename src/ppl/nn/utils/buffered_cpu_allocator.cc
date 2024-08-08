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
#include <cstddef>
#include <windows.h>
#else
#include <unistd.h> // sysconf
#include <string.h> // strerror
#include <errno.h>
#include <sys/mman.h>
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

#ifdef _MSC_VER
static constexpr uint32_t g_max_msg_buf_size = 1024;
#endif

BufferedCpuAllocator::BufferedCpuAllocator() {
#ifdef _MSC_VER
    base_ = nullptr;
#else
    base_ = MAP_FAILED;
#endif
}

BufferedCpuAllocator::~BufferedCpuAllocator() {
#ifdef _MSC_VER
    if (base_) {
        VirtualFree(base_, 0, MEM_RELEASE);
    }
#else
    if (base_ != MAP_FAILED) {
        munmap(base_, addr_len_);
    }
#endif
}

RetCode BufferedCpuAllocator::Init(uint64_t max_mem_bytes) {
#ifdef _MSC_VER
    {
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) {
            char errmsg[g_max_msg_buf_size];
            FormatMessage(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), 0,
                          errmsg, g_max_msg_buf_size, nullptr);
            LOG(ERROR) << "get physical memory info failed: " << errmsg;
            return RC_OTHER_ERROR;
        }

        if (max_mem_bytes > status.ullTotalPhys) {
            max_mem_bytes = status.ullTotalPhys;
        }
    }

    base_ = VirtualAlloc(nullptr, max_mem_bytes, MEM_RESERVE, PAGE_READWRITE);
    if (!base_) {
        char errmsg[g_max_msg_buf_size];
        FormatMessage(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), 0, errmsg,
                      g_max_msg_buf_size, nullptr);
        LOG(ERROR) << "VirtualAlloc reserve [" << max_mem_bytes << "] bytes failed: " << errmsg;
        return RC_OTHER_ERROR;
    }
#else
    {
        uint64_t totalram = sysconf(_SC_PAGE_SIZE) * sysconf(_SC_PHYS_PAGES);
        if (max_mem_bytes > totalram) {
            max_mem_bytes = totalram;
        }
    }

    base_ = mmap(nullptr, max_mem_bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ == MAP_FAILED) {
        LOG(ERROR) << "mmap reserve [" << max_mem_bytes << "] bytes failed: " << strerror(errno);
        return RC_OTHER_ERROR;
    }
#endif

    cursor_ = base_;
    addr_len_ = max_mem_bytes;
    LOG(DEBUG) << "reserved [" << max_mem_bytes << "] bytes of virtual address from [" << base_ << "].";

    return RC_SUCCESS;
}

#define MIN_ALLOC_SIZE 65536

static inline uint64_t Align(uint64_t x, uint64_t n) {
    return (x + n - 1) & (~(n - 1));
}

uint64_t BufferedCpuAllocator::Extend(uint64_t bytes) {
    bytes = Align(bytes, MIN_ALLOC_SIZE);

#ifdef _MSC_VER
    auto new_addr = VirtualAlloc(cursor_, bytes, MEM_COMMIT, PAGE_READWRITE);
    if (!new_addr) {
        char errmsg[g_max_msg_buf_size];
        FormatMessage(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), 0, errmsg,
                      g_max_msg_buf_size, nullptr);
        LOG(ERROR) << "VirtualAlloc [" << bytes << "] bytes failed: " << errmsg;
        return 0;
    }
#else
    if (mprotect(cursor_, bytes, PROT_READ | PROT_WRITE) != 0) {
        LOG(ERROR) << "mprotect [" << bytes << "] bytes failed: " << strerror(errno);
        return 0;
    }
#endif

    cursor_ = (char*)cursor_ + bytes;
    return bytes;
}

}}} // namespace ppl::nn::utils
