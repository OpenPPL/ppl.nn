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

#ifndef _ST_HPC_PPL_NN_PARAMS_PARAM_UTILS_MANAGER_H_
#define _ST_HPC_PPL_NN_PARAMS_PARAM_UTILS_MANAGER_H_

#include "ppl/nn/utils/op_info_manager.h"

namespace ppl { namespace nn {

struct ParamUtils final {
    bool (*equal)(const void* param_0, const void* param_1);
};

class ParamUtilsManager final {
public:
    static ParamUtilsManager* Instance() {
        static ParamUtilsManager mgr;
        return &mgr;
    }

    ppl::common::RetCode Register(const std::string& domain, const std::string& type, const utils::VersionRange& ver,
                                  const ParamUtils& item) {
        return mgr_.Register(domain, type, ver, item);
    }

    const ParamUtils* Find(const std::string& domain, const std::string& type, uint64_t version) const {
        return mgr_.Find(domain, type, version);
    }

private:
    utils::OpInfoManager<ParamUtils> mgr_;

private:
    ParamUtilsManager() {}
};

}} // namespace ppl::nn

#endif
