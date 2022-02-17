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

#include "ppl/nn/utils/buffer_data_stream.h"
#include <cstring> // memcpy
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

RetCode BufferDataStream::Seek(uint64_t offset) {
    if (offset >= data_.size()) {
        return RC_OUT_OF_RANGE;
    }

    offset_ = offset;
    return RC_SUCCESS;
}

RetCode BufferDataStream::Write(const void* base, uint64_t bytes) {
    if (offset_ + bytes > data_.size()) {
        data_.resize(offset_ + bytes);
    }
    memcpy(data_.data() + offset_, base, bytes);
    offset_ += bytes;
    return RC_SUCCESS;
}

uint64_t BufferDataStream::Tell() const {
    return offset_;
}

uint64_t BufferDataStream::GetSize() const {
    return data_.size();
}

}}} // namespace ppl::nn::utils
