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

#include "passes/i8i8/quantization_pass.h"
#include "passes/i8i8/fuse_rms_norm_pass.h"

#include "opt_pass_manager.h"

#include "ppl/common/log.h"

namespace ppl { namespace nn { namespace llm { namespace cuda {

ppl::common::RetCode OptPassManager::Register(
    const std::string& domain,
    const std::string& name,
    const OptPass& pass)
{
    auto domain_ret = pass_all_.insert(
        std::make_pair(domain, std::unordered_map<std::string, OptPass>()));
    auto name_ret = domain_ret.first->second.insert(make_pair(name, pass));
    return name_ret.second ? ppl::common::RC_SUCCESS : ppl::common::RC_EXISTS;
}

OptPassStatus OptPassManager::Apply(const std::string& domain, const std::string& name, const OptKernelOptions& options) {
    auto domain_ret = pass_all_.find(domain);
    if (domain_ret != pass_all_.end()) {
        auto pass = domain_ret->second.find(name);
        if (pass != domain_ret->second.end()) {
            return pass->second(options);
        }
    }
    LOG(WARNING) << "OptPass[" << domain << ":" << name << "] not found";
    return {ppl::common::RC_NOT_FOUND, false};
}

ppl::common::RetCode OptPassManager::ApplyByDomain(const std::string& domain, const OptKernelOptions& options) {
    auto domain_it = pass_all_.find(domain);
    if (domain_it != pass_all_.end()) {
        bool modified = false;
        auto& domain_map = domain_it->second;
        do {
            modified = false;
            auto pass_it = domain_map.begin();
            while (pass_it != domain_map.end()) {
                auto ret = pass_it->second(options);
                if (ppl::common::RC_SUCCESS != ret.retcode) {
                    LOG(ERROR) << "OptPass [" << domain << ":" << pass_it->first << "] failed: " << ppl::common::GetRetCodeStr(ret.retcode);
                    return ret.retcode;
                }
                modified = modified || ret.graph_modified;
                ++pass_it;
            }
        } while(modified);
    } else {
        LOG(WARNING) << "OptPass domain[" << domain << "] not found";
    }

    return ppl::common::RC_SUCCESS;
}

OptPassManager::~OptPassManager() {
}

OptPassManager::OptPassManager() {
    Register("", "I8I8Quantization", i8i8::QuantizationPass);
    Register("i8i8.fuse", "FuseRMSNorm", i8i8::FuseRMSNormPass);
}


}}}} // namespace ppl::nn::llm::cuda
