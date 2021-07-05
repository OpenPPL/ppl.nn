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

#ifndef _ST_HPC_PPL_NN_MODELS_OP_INFO_MANAGER_H
#define _ST_HPC_PPL_NN_MODELS_OP_INFO_MANAGER_H

#include "ppl/common/retcode.h"
#include "ppl/nn/ir/graph.h"

namespace ppl { namespace nn {

typedef bool (*ParamEqualFunc)(void* param_0, void* param_1);

struct OpInfo {
    ParamEqualFunc param_equal;
};

class OpInfoManager {
public:
    static OpInfoManager* Instance() {
        static OpInfoManager mgr;
        return &mgr;
    }

    const OpInfo* Find(const std::string& domain, const std::string& op_type) const;
    void Register(const std::string& domain, const std::string& op_type, const OpInfo&);

private:
    std::map<std::string, std::map<std::string, OpInfo>> info_;
};

}} // namespace ppl::nn

#endif
