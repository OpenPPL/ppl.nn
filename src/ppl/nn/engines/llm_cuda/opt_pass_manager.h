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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPT_PASS_MANAGER_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPT_PASS_MANAGER_H_

#include  "opt_pass.h"

namespace ppl { namespace nn { namespace llm { namespace cuda {

class OptPassManager {
public:
    static OptPassManager* GetInstance() {
        static OptPassManager mgr;
        return &mgr;
    }
    ~OptPassManager();

    ppl::common::RetCode Register(const std::string& domain, const std::string& name, const OptPass& pass);
    OptPassStatus Apply(const std::string& domain, const std::string& name, const OptKernelOptions& options);
    ppl::common::RetCode ApplyByDomain(const std::string& domain, const OptKernelOptions& options);

private:
    std::unordered_map<std::string, std::unordered_map<std::string, OptPass>> pass_all_;

private:
    OptPassManager();
};


}}}} // namespace ppl::nn::llm::cuda

#endif
