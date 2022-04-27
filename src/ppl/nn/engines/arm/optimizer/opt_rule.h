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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_RULE_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_RULE_H_

#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"

#include <string>

namespace ppl { namespace nn { namespace arm {

enum OptRuleLevel {
    OPT_RULE_MUST_BE_DONE = -1,
    OPT_RULE_NO_OPT = 0,
    OPT_RULE_LEVEL_0 = 1,
    OPT_RULE_LEVEL_1 = 2,
    OPT_RULE_LEVEL_2 = 3,
};

class OptRule {
public:
    OptRule(){};
    virtual ~OptRule(){};

    void SetName(const std::string& name) {
        name_ = name;
    }

    void SetTag(const std::string& tag) {
        tag_ = tag;
    }

    void SetLevel(const OptRuleLevel level) {
        level_ = level;
    }

    const std::string& GetName(void) const {
        return name_;
    }

    const std::string& GetTag(void) const {
        return tag_;
    }

    const OptRuleLevel GetLevel(void) const {
        return level_;
    }

    bool NameMatched(const std::string& name) {
        return this->StringMatched(name_, name);
    }

    bool TagMatched(const std::string& tag) {
        return this->StringMatched(tag_, tag);
    }

    // apply graph optimize rule
    virtual bool Apply(const OptKernelOptions& options) = 0;

private:
    static bool StringMatched(const std::string& str, const std::string& pattern) {
        return str.find(pattern) != std::string::npos;
    }

protected:
    std::string name_;
    std::string tag_;
    OptRuleLevel level_;
};

}}} // namespace ppl::nn::arm

#endif
