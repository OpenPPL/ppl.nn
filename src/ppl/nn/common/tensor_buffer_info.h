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

#ifndef _ST_HPC_PPL_NN_COMMON_TENSOR_BUFFER_INFO_H_
#define _ST_HPC_PPL_NN_COMMON_TENSOR_BUFFER_INFO_H_

#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/common/buffer_info.h"

namespace ppl { namespace nn {

/** refer to `BufferInfo` for API descriptions */
class TensorBufferInfo final {
public:
    TensorBufferInfo() {}
    TensorBufferInfo(TensorBufferInfo&&) = default;
    TensorBufferInfo& operator=(TensorBufferInfo&&) = default;

    bool IsBufferOwner() const {
        return info_.IsBufferOwner();
    }

    ppl::common::RetCode SetDevice(Device* dev) {
        return info_.SetDevice(dev);
    }

    Device* GetDevice() const {
        return info_.GetDevice();
    }

    void SetBuffer(const BufferDesc& buf, Device* device = nullptr, bool is_buffer_owner = false) {
        return info_.SetBuffer(buf, device, is_buffer_owner);
    }

    BufferDesc DetachBuffer() {
        return info_.DetachBuffer();
    }

    void FreeBuffer() {
        info_.FreeBuffer();
    }

    ppl::common::RetCode ReallocBuffer() {
        return info_.ReallocBuffer(shape_);
    }

    template <typename T = void>
    T* GetBufferPtr() const {
        return info_.GetBufferPtr<T>();
    }

    BufferDesc& GetBufferDesc() {
        return info_.GetBufferDesc();
    }
    const BufferDesc& GetBufferDesc() const {
        return info_.GetBufferDesc();
    }

    TensorShape* GetShape() const {
        return &shape_;
    }

    void Reshape(const TensorShape& new_shape) {
        shape_ = new_shape;
    }

private:
    BufferInfo info_;
    mutable TensorShape shape_;

private:
    TensorBufferInfo(const TensorBufferInfo&) = delete;
    TensorBufferInfo& operator=(const TensorBufferInfo&) = delete;
};

}} // namespace ppl::nn

#endif
