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

#ifndef _ST_HPC_PPL_NN_UTILS_UTILS_H_
#define _ST_HPC_PPL_NN_UTILS_UTILS_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/ir/node.h"
#include "ppl/nn/utils/buffer.h"

namespace ppl { namespace nn { namespace utils {

static inline bool IsPplConverterNode(const ir::Node* node) {
    auto& type = node->GetType();
    return (type.name == "Converter" && type.domain == "pmx");
}

static inline ir::Node::Type MakePplConverterNodeType() {
    return ir::Node::Type("pmx", "Converter", 1);
}

ppl::common::RetCode ReadFileContent(const char* fname, Buffer* buf, uint64_t offset = 0, uint64_t length = UINT64_MAX);

}}} // namespace ppl::nn::utils

#endif
