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

#ifndef _ST_HPC_PPL_NN_COMMON_TYPES_H_
#define _ST_HPC_PPL_NN_COMMON_TYPES_H_

#include <stdint.h>
#include <memory>
#include <functional>

namespace ppl { namespace nn {

enum {
    /** tensors that can be modified or reused */
    TENSORTYPE_NORMAL,
    /** tensors that are reserved and cannot be reused */
    TENSORTYPE_RESERVED,
};
typedef uint32_t tensortype_t;

typedef std::unique_ptr<void, std::function<void(void*)>> VoidPtr;

static const uint32_t INVALID_NODEID = UINT32_MAX;
static const uint32_t INVALID_EDGEID = UINT32_MAX;

typedef uint32_t nodeid_t;
typedef uint32_t edgeid_t;

}} // namespace ppl::nn

#endif
