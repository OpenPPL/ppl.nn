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

#include "ppl/nn/engines/x86/optimizer/opt_rule_manager.h"
#include "ppl/nn/common/logger.h"

#include "ppl/nn/engines/x86/optimizer/rules/fuse_conv_activation.h"
#include "ppl/nn/engines/x86/optimizer/rules/fuse_conv_eltwise.h"
#include "ppl/nn/engines/x86/optimizer/rules/fuse_gemm_activation.h"
#include "ppl/nn/engines/x86/optimizer/rules/fuse_arithmetic_relu.h"
#include "ppl/nn/engines/x86/optimizer/rules/fuse_batch_normalization_relu.h"
#include "ppl/nn/engines/x86/optimizer/rules/fuse_channel_shuffle.h"
#include "ppl/nn/engines/x86/optimizer/rules/fuse_swish.h"
#include "ppl/nn/engines/x86/optimizer/rules/layout_optimize.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode OptRuleManager::Register(const std::string& tag, const std::string& name, OptRule rule) {
    auto tag_ret = rule_all_.insert(std::make_pair(tag, std::map<std::string, OptRule>()));
    auto name_ret = tag_ret.first->second.insert(make_pair(name, rule));
    return name_ret.second ? ppl::common::RC_SUCCESS : ppl::common::RC_EXISTS;
}

void OptRuleManager::Remove(const std::string& tag, const std::string& name) {
    auto tag_ret = rule_all_.find(tag);
    if (tag_ret != rule_all_.end()) {
        auto& tag_rules = tag_ret->second;
        tag_rules.erase(name);
        if (tag_rules.empty()) {
            rule_all_.erase(tag_ret);
        }
    }
}

OptRule OptRuleManager::Find(const std::string& tag, const std::string& name) {
    auto tag_ret = rule_all_.find(tag);
    if (tag_ret != rule_all_.end()) {
        auto rule = tag_ret->second.find(name);
        if (rule != tag_ret->second.end()) {
            return rule->second;
        }
    }
    return nullptr;
}

bool OptRuleManager::Apply(const std::string& tag, const std::string& name, const OptKernelOptions& options) {
    auto rule = Find(tag, name);
    if (rule) {
        return rule(options);
    }
    return false;
}

void OptRuleManager::ApplyByTag(const std::string& tag, const OptKernelOptions& options) {
    auto tag_it = rule_all_.find(tag);
    if (tag_it != rule_all_.end()) {
        bool ret = false;
        auto& tag_rules_map = tag_it->second;
        do {
            ret = false;
            auto rule_it = tag_rules_map.begin();
            while (rule_it != tag_rules_map.end()) {
                ret = ret || rule_it->second(options);
                ++rule_it;
            }
        } while(ret);
    }
}

OptRuleManager::~OptRuleManager() {
}

#define REGISTER_OPT_RULE(tag, name, rule_func) Register(tag, name, rule_func)

OptRuleManager::OptRuleManager() {
    REGISTER_OPT_RULE("", "LayoutOptimize", LayoutOptimize);

    REGISTER_OPT_RULE("BeforeLayoutOptimize", "FuseChannelShuffle", FuseChannelShuffle);

    REGISTER_OPT_RULE("AfterLayoutOptimize", "FuseConvActivation", FuseConvActivation);
    REGISTER_OPT_RULE("AfterLayoutOptimize", "FuseConvEltwise", FuseConvEltwise);
    REGISTER_OPT_RULE("AfterLayoutOptimize", "FuseArithmeticReLU", FuseArithmeticReLU);
    REGISTER_OPT_RULE("AfterLayoutOptimize", "FuseBatchNormalizationReLU", FuseBatchNormalizationReLU);
    REGISTER_OPT_RULE("AfterLayoutOptimize", "FuseGemmActivation", FuseGemmActivation);
    REGISTER_OPT_RULE("AfterLayoutOptimize", "FuseSwish", FuseSwish);
}

}}} // namespace ppl::nn::x86
