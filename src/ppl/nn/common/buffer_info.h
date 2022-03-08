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

#ifndef _ST_HPC_PPL_NN_COMMON_BUFFER_INFO_H_
#define _ST_HPC_PPL_NN_COMMON_BUFFER_INFO_H_

#include "ppl/nn/common/device.h"

namespace ppl { namespace nn {

class BufferInfo final {
public:
    BufferInfo() : is_buffer_owner_(false), device_(nullptr) {}
    BufferInfo(BufferInfo&&);
    BufferInfo& operator=(BufferInfo&&);
    ~BufferInfo();

    bool IsBufferOwner() const {
        return is_buffer_owner_;
    }

    /**
       @brief set device used to manage buffer of this tensor
       @note fails when buffer_.addr is not null
    */
    ppl::common::RetCode SetDevice(Device* dev);

    Device* GetDevice() const {
        return device_;
    }

    /**
       @brief set a buffer `buf` allocated by `device` as this tensor's buffer.
       old buffer of this tensor will be freed or detached.
       @note if `device` is nullptr, make sure that `buf` can be read/written by device_.
    */
    void SetBuffer(const BufferDesc& buf, Device* device = nullptr, bool is_buffer_owner = false);

    /** @brief returns buffer_ to caller and reset buffer_. */
    BufferDesc DetachBuffer();

    /** @brief frees the internal buffer */
    void FreeBuffer();

    ppl::common::RetCode ReallocBuffer(const TensorShape&);

    template <typename T = void>
    T* GetBufferPtr() const {
        return static_cast<T*>(buffer_.addr);
    }

    BufferDesc& GetBufferDesc() {
        return buffer_;
    }
    const BufferDesc& GetBufferDesc() const {
        return buffer_;
    }

private:
    bool is_buffer_owner_;
    BufferDesc buffer_;
    Device* device_;

private:
    BufferInfo(const BufferInfo&) = delete;
    BufferInfo& operator=(const BufferInfo&) = delete;
};

}} // namespace ppl::nn

#endif
