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

#ifndef _ST_HPC_PPL_NN_UTILS_BUFFER_DATA_STREAM_H_
#define _ST_HPC_PPL_NN_UTILS_BUFFER_DATA_STREAM_H_

#include "ppl/nn/utils/data_stream.h"
#include <vector>

namespace ppl { namespace nn { namespace utils {

class BufferDataStream final : public DataStream {
public:
    ppl::common::RetCode Seek(uint64_t offset) override;
    ppl::common::RetCode Write(const void* base, uint64_t bytes) override;
    uint64_t Tell() const override;
    uint64_t GetSize() const override;

    const void* GetData() const {
        return data_.data();
    }

private:
    uint64_t offset_ = 0;
    std::vector<char> data_;
};

}}} // namespace ppl::nn::utils

#endif
