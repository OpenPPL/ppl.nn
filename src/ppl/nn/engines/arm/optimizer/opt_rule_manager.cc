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

#include "ppl/nn/engines/arm/optimizer/opt_rule_manager.h"

#include "ppl/nn/engines/arm/optimizer/rules/fuse_channel_shuffle.h"
#include "ppl/nn/engines/arm/optimizer/rules/fuse_conv_activation.h"
#include "ppl/nn/engines/arm/optimizer/rules/fuse_conv_eltwise.h"
#include "ppl/nn/engines/arm/optimizer/rules/fuse_arithmetic_relu.h"
#include "ppl/nn/engines/arm/optimizer/rules/fuse_batch_normalization_relu.h"

#include "ppl/nn/common/logger.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

OptRuleLevel OptRuleManager::GetMaxOptLevel(uint32_t graph_optimization_level) {
    switch (graph_optimization_level) {
        case ARM_OPT_DISABLE_ALL:
            return OPT_RULE_NO_OPT;
        case ARM_OPT_ENABLE_BASIC:
            return OPT_RULE_LEVEL_0;
        case ARM_OPT_ENABLE_EXTENDED:
            return OPT_RULE_LEVEL_1;
        case ARM_OPT_ENABLE_ALL:
            return OPT_RULE_LEVEL_2;
        default:
            break;
    }
    return OPT_RULE_NO_OPT;
}

ppl::common::RetCode OptRuleManager::ApplyRule(const OptKernelOptions& options, const std::string& name) const {
    // find rule
    for (auto r : rule_all_) {
        if (r->GetName() == name) {
            // rule found, do optimize
            bool graph_changed = false;
            do {
                graph_changed = r->Apply(options);
            } while (graph_changed);

            return RC_SUCCESS;
        }
    }

    LOG(ERROR) << "Failed to find OptRule " << name << ".";
    return RC_NOT_FOUND;
}

ppl::common::RetCode OptRuleManager::ApplyRules(const OptKernelOptions& options, const OptRuleLevel max_opt_level,
                                                const std::string& tag_filter, const std::string& name_filter) const {
    // filter rules by optimize level, tag & name
    std::vector<std::shared_ptr<OptRule>> filtered_rules;
    filtered_rules.reserve(rule_all_.size());
    for (auto r : rule_all_) {
        if (r->TagMatched(tag_filter) && r->NameMatched(name_filter) && r->GetLevel() <= max_opt_level) {
            filtered_rules.push_back(r);
        }
    }

    // apply optimize
    bool graph_changed = false;
    do {
        graph_changed = false;
        for (auto r : filtered_rules) {
            graph_changed |= r->Apply(options);
        }
    } while (graph_changed);

    return RC_SUCCESS;
}

ppl::common::RetCode OptRuleManager::Register(OptRule* rule) {
    rule_all_.emplace_back(rule);
    return RC_SUCCESS;
}

OptRuleManager::OptRuleManager() {
    rule_all_.emplace_back(new FuseChannelShuffleRule);
    rule_all_.emplace_back(new FuseConvActivationRule);
    rule_all_.emplace_back(new FuseConvEltwiseRule);
    rule_all_.emplace_back(new FuseBatchNormalizationReLURule);
}

OptRuleManager::~OptRuleManager() {}

}}} // namespace ppl::nn::arm
