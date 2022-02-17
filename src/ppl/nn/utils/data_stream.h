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

#ifndef _ST_HPC_PPL_NN_UTILS_DATA_STREAM_H_
#define _ST_HPC_PPL_NN_UTILS_DATA_STREAM_H_

#include "ppl/common/retcode.h"

namespace ppl { namespace nn { namespace utils {

class DataStream {
public:
    virtual ~DataStream() {}

    /** @brief moves write point to `offset` */
    virtual ppl::common::RetCode Seek(uint64_t offset) = 0;

    /** @brief write `bytes` bytes pointed by `base` */
    virtual ppl::common::RetCode Write(const void* base, uint64_t bytes) = 0;

    /** @brief returns current offset */
    virtual uint64_t Tell() const = 0;

    /** @brief bytes written */
    virtual uint64_t GetSize() const = 0;
};

}}} // namespace ppl::nn::utils

#endif
