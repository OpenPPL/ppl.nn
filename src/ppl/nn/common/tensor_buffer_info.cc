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

#include "ppl/nn/common/tensor_buffer_info.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

TensorBufferInfo::~TensorBufferInfo() {
    FreeBuffer();
}

TensorBufferInfo::TensorBufferInfo(TensorBufferInfo&& info) {
    is_buffer_owner_ = info.is_buffer_owner_;
    buffer_ = info.buffer_;
    device_ = info.device_;
    shape_ = std::move(info.shape_);

    info.buffer_.addr = nullptr;
    info.device_ = nullptr;
    info.is_buffer_owner_ = false;
}

TensorBufferInfo& TensorBufferInfo::operator=(TensorBufferInfo&& info) {
    if (is_buffer_owner_ && device_) {
        device_->Free(&buffer_);
    }

    is_buffer_owner_ = info.is_buffer_owner_;
    buffer_ = info.buffer_;
    device_ = info.device_;
    shape_ = std::move(info.shape_);

    info.buffer_.addr = nullptr;
    info.device_ = nullptr;
    info.is_buffer_owner_ = false;

    return *this;
}

RetCode TensorBufferInfo::SetDevice(Device* dev) {
    if (buffer_.addr) {
        LOG(ERROR) << "TensorBufferInfo::SetDevice() failed: buffer is not empty";
        return RC_PERMISSION_DENIED;
    }
    if (!dev) {
        LOG(ERROR) << "SetDevice failed: device is empty.";
        return RC_INVALID_VALUE;
    }

    device_ = dev;
    return RC_SUCCESS;
}

void TensorBufferInfo::SetBuffer(const BufferDesc& buf, Device* device, bool is_buffer_owner) {
    if (is_buffer_owner_ && device_) {
        device_->Free(&buffer_);
    }

    if (device) {
        device_ = device;
    }

    buffer_ = buf;
    is_buffer_owner_ = is_buffer_owner;
}

RetCode TensorBufferInfo::ReallocBuffer() {
    if (!device_) {
        LOG(ERROR) << "ReallocBuffer() failed: device not set.";
        return RC_PERMISSION_DENIED;
    }

    if (!is_buffer_owner_) {
        buffer_.addr = nullptr;
    }

    auto status = device_->Realloc(shape_, &buffer_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Realloc [" << shape_.GetBytesIncludingPadding() << "] bytes failed: " << GetRetCodeStr(status);
        return status;
    }

    is_buffer_owner_ = true;

    return RC_SUCCESS;
}

BufferDesc TensorBufferInfo::DetachBuffer() {
    auto ret = buffer_;
    buffer_.addr = nullptr;
    is_buffer_owner_ = false;
    return ret;
}

void TensorBufferInfo::FreeBuffer() {
    if (is_buffer_owner_ && device_) {
        device_->Free(&buffer_);
        is_buffer_owner_ = false;
    }

    buffer_.addr = nullptr;
}

}} // namespace ppl::nn
