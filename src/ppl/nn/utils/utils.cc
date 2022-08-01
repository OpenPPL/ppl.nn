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

#ifdef _MSC_VER
#define NO_MINMAX
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#include <stdint.h>
static constexpr uint32_t g_max_msg_buf_size = 1024;
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#endif

#include "ppl/nn/utils/utils.h"
#include "ppl/common/destructor.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

#ifdef _MSC_VER
RetCode ReadFileContent(const char* fname, Buffer* buf, uint64_t offset, uint64_t length) {
    if (length == 0) {
        return RC_SUCCESS;
    }

    auto handle =
        CreateFile(fname, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        char errmsg[g_max_msg_buf_size];
        FormatMessage(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), 0, errmsg,
                      g_max_msg_buf_size, nullptr);
        LOG(ERROR) << "open file[" << fname << "] for reading failed: " << errmsg;
        return RC_OTHER_ERROR;
    }
    Destructor __d([handle]() -> void {
        CloseHandle(handle);
    });

    DWORD file_size_high = 0;
    DWORD file_size_low = GetFileSize(handle, &file_size_high);
    const uint64_t file_size = ((uint64_t)file_size_high << 32) + file_size_low;
    if (file_size == 0) {
        return RC_SUCCESS;
    }

    if (offset >= file_size) {
        LOG(ERROR) << "offset[" << offset << "] >= file size[" << file_size << "]";
        return RC_INVALID_VALUE;
    }

    auto available_bytes = file_size - offset;
    if (length > available_bytes) {
        length = available_bytes;
    }

    LONG distance_low = offset & 0xffffffff;
    LONG distance_high = offset >> 32;
    auto pos = SetFilePointer(handle, distance_low, &distance_high, FILE_BEGIN);
    if (pos != distance_low) {
        char errmsg[g_max_msg_buf_size];
        FormatMessage(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), 0, errmsg,
                      g_max_msg_buf_size, nullptr);
        LOG(ERROR) << "set offset[" << offset << "] of file[" << fname << "] failed: " << errmsg;
        return RC_INVALID_VALUE;
    }

    buf->Resize(length);
    DWORD bytes_read = 0;
    auto ok = ReadFile(handle, buf->GetData(), length, &bytes_read, nullptr);
    if (!ok) {
        char errmsg[g_max_msg_buf_size];
        FormatMessage(FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), 0, errmsg,
                      g_max_msg_buf_size, nullptr);
        LOG(ERROR) << "read file[" << fname << "] failed: " << errmsg;
        return RC_OTHER_ERROR;
    }
    if (bytes_read != length) {
        LOG(ERROR) << "[" << (uint32_t)bytes_read << "] bytes read != expected [" << length << "]";
        buf->Clear();
        return RC_INVALID_VALUE;
    }

    return RC_SUCCESS;
}
#else
RetCode ReadFileContent(const char* fname, Buffer* buf, uint64_t offset, uint64_t length) {
    if (length == 0) {
        return RC_SUCCESS;
    }

    int fd = open(fname, O_RDONLY);
    if (fd == -1) {
        LOG(ERROR) << "open file[" << fname << "] failed: " << strerror(errno);
        return RC_OTHER_ERROR;
    }
    Destructor __d([fd]() -> void {
        close(fd);
    });

    struct stat st;
    if (fstat(fd, &st) != 0) {
        LOG(ERROR) << "get size of file[" << fname << "] failed: " << strerror(errno);
        return RC_OTHER_ERROR;
    }
    if (st.st_size == 0) {
        return RC_SUCCESS;
    }

    if (offset >= (uint64_t)st.st_size) {
        LOG(ERROR) << "offset[" << offset << "] >= file size[" << st.st_size << "]";
        return RC_INVALID_VALUE;
    }
    auto pos = lseek(fd, offset, SEEK_SET);
    if ((uint64_t)pos != offset) {
        LOG(ERROR) << "seek to offset[" << offset << "] of file[" << fname << "] failed: " << strerror(errno);
        return RC_INVALID_VALUE;
    }

    auto available_bytes = st.st_size - offset;
    if (length > available_bytes) {
        length = available_bytes;
    }

    buf->Resize(length);
    auto bytes_read = read(fd, buf->GetData(), length);
    if (bytes_read < 0) {
        LOG(ERROR) << "read file[" << fname << "] failed: " << strerror(errno);
        buf->Clear();
        return RC_OTHER_ERROR;
    }
    if ((uint64_t)bytes_read != length) {
        LOG(ERROR) << "[" << bytes_read << "] bytes read != expected [" << length << "]";
        buf->Clear();
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}
#endif

}}} // namespace ppl::nn::utils
