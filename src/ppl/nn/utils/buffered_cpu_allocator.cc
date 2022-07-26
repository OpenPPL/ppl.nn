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
#include <utility> // make_pair
using namespace std;
using namespace ppl::common;

#ifdef _MSC_VER
#include <windows.h>
#else
#include <errno.h>
#include <sys/sysinfo.h> // sysinfo
#include <string.h> // strerror
#include <sys/mman.h>
#endif

namespace ppl { namespace nn { namespace utils {

#ifdef _MSC_VER
static constexpr uint32_t g_max_msg_buf_size = 1024;
#endif

BufferedCpuAllocator::~BufferedCpuAllocator() {
#ifdef _MSC_VER
    if (base_) {
        VirtualFree(base_, 0, MEM_RELEASE);
    }
#else
    if (base_ != MAP_FAILED) {
        munmap(base_, (char*)cursor_ - (char*)base_);
    }
#endif
}

RetCode BufferedCpuAllocator::Init() {
#ifdef _MSC_VER
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        char errmsg[g_max_msg_buf_size];
        FormatMessage(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), 0, errmsg,
                      g_max_msg_buf_size, nullptr);
        LOG(ERROR) << "get physical memory info failed: " << errmsg;
        return RC_OTHER_ERROR;
    }

    base_ = VirtualAlloc(nullptr, status.ullTotalPhys, MEM_RESERVE, PAGE_READWRITE);
    if (!base_) {
        char errmsg[g_max_msg_buf_size];
        FormatMessage(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), 0, errmsg,
                      g_max_msg_buf_size, nullptr);
        LOG(ERROR) << "VirtualAlloc reserve [" << status.ullTotalPhys << "] bytes failed: " << errmsg;
        return RC_OTHER_ERROR;
    }

    LOG(DEBUG) << "reserved [" << status.ullTotalPhys << "] bytes of virtual address from [" << base_ << "].";
#else
    struct sysinfo s;
    sysinfo(&s);

    base_ = mmap(nullptr, s.totalram, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ == MAP_FAILED) {
        LOG(ERROR) << "mmap reserve [" << s.totalram << "] bytes failed: " << strerror(errno);
        return RC_OTHER_ERROR;
    }

    LOG(DEBUG) << "reserved [" << s.totalram << "] bytes of virtual address from [" << base_ << "].";
#endif

    cursor_ = base_;
    return RC_SUCCESS;
}

void* BufferedCpuAllocator::Alloc(uint64_t bytes) {
#ifdef _MSC_VER
    auto new_addr = VirtualAlloc(cursor_, bytes, MEM_COMMIT, PAGE_READWRITE);
    if (!new_addr) {
        char errmsg[g_max_msg_buf_size];
        FormatMessage(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), 0, errmsg,
                      g_max_msg_buf_size, nullptr);
        LOG(ERROR) << "VirtualAlloc [" << bytes << "] bytes failed: " << errmsg;
        return nullptr;
    }
#else
    auto new_addr = cursor_;
    if (mprotect(cursor_, bytes, PROT_READ | PROT_WRITE) != 0) {
        LOG(ERROR) << "mprotect [" << bytes << "] bytes failed: " << strerror(errno);
        return nullptr;
    }
#endif

    cursor_ = (char*)cursor_ + bytes;
    return new_addr;
}

}}} // namespace ppl::nn::utils
