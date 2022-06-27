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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_RULE_MANAGER_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_RULE_MANAGER_H_

#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"
#include "ppl/nn/engines/arm/optimizer/opt_rule.h"

#include <string>
#include <vector>
#include <memory>

namespace ppl { namespace nn { namespace arm {

class OptRuleManager {
public:
    static OptRuleManager* Instance() {
        static OptRuleManager mgr;
        return &mgr;
    };
    ~OptRuleManager();

    // get max opt level according to arm engine options config
    OptRuleLevel GetMaxOptLevel(uint32_t graph_optimization_level);

    // apply single rule
    ppl::common::RetCode ApplyRule(const OptKernelOptions& options, const std::string& name) const;

    // apply rules filtered by optimize level, tag & name
    ppl::common::RetCode ApplyRules(const OptKernelOptions& options, const OptRuleLevel max_opt_level,
                                    const std::string& tag_filter = "", const std::string& name_filter = "") const;

    ppl::common::RetCode Register(OptRule* rule);
    ppl::common::RetCode Remove(const std::string& rule_name);

private:
    std::vector<std::shared_ptr<OptRule>> rule_all_;

private:
    OptRuleManager();
};

}}} // namespace ppl::nn::arm

#endif
