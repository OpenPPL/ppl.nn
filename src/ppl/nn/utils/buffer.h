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

#ifndef _ST_HPC_PPL_NN_UTILS_BUFFER_H_
#define _ST_HPC_PPL_NN_UTILS_BUFFER_H_

#include <cstring> // memcpy
#include <vector>

namespace ppl { namespace nn { namespace utils {

class Buffer final {
public:
    void Append(const void* data, uint64_t bytes) {
        auto offset = data_.size();
        data_.resize(data_.size() + bytes);
        memcpy(data_.data() + offset, data, bytes);
    }
    void Assign(const void* data, uint64_t bytes) {
        data_.resize(bytes);
        memcpy(data_.data(), data, bytes);
    }
    void Resize(uint64_t bytes) {
        data_.resize(bytes);
    }
    void Resize(uint64_t bytes, char c) {
        data_.resize(bytes, c);
    }
    void Reserve(uint64_t bytes) {
        data_.reserve(bytes);
    }
    uint64_t GetSize() const {
        return data_.size();
    }
    const void* GetData() const {
        return data_.data();
    }
    void* GetData() {
        return data_.data();
    }
    void Clear() {
        data_.clear();
    }
    bool operator==(const Buffer& rhs) const {
        return (data_ == rhs.data_);
    }

private:
    std::vector<char> data_;
};

}}} // namespace ppl::nn::utils

#endif
