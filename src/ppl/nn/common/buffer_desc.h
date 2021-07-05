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

#ifndef _ST_HPC_PPL_NN_COMMON_BUFFER_DESC_H_
#define _ST_HPC_PPL_NN_COMMON_BUFFER_DESC_H_

#include <stdint.h>
#include <functional>

namespace ppl { namespace nn {

struct BufferDesc final {
    BufferDesc(void* a = nullptr) : addr(a) {}

    /** pointer to data area */
    void* addr;

    /** used by engines with different meanings. this union is invalid if `addr` is nullptr. */
    union {
        uint64_t desc;
        void* info;
    };
};

class BufferDescGuard {
public:
    BufferDescGuard(BufferDesc* buffer, const std::function<void(BufferDesc*)>& deleter)
        : buffer_(buffer), deleter_(deleter) {}
    ~BufferDescGuard() {
        deleter_(buffer_);
    }

private:
    BufferDesc* buffer_;
    std::function<void(BufferDesc*)> deleter_;
};

}} // namespace ppl::nn

#endif
