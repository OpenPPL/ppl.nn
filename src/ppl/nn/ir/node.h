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

#ifndef _ST_HPC_PPL_NN_IR_NODE_H_
#define _ST_HPC_PPL_NN_IR_NODE_H_

#include "ppl/nn/ir/edge.h"
#include <string>
#include <stdint.h>

namespace ppl { namespace nn { namespace ir {

class Node final {
public:
    struct Type final {
        Type() {}
        Type(const std::string& d, const std::string& n, uint64_t v) : domain(d), name(n), version(v) {}
        Type(Type&&) = default;
        Type(const Type&) = default;

        Type& operator=(Type&&) = default;
        Type& operator=(const Type&) = default;

        bool operator==(const Type& tid) const {
            return (name == tid.name && domain == tid.domain && version == tid.version);
        }

        std::string domain;
        std::string name;
        uint64_t version;
    };

public:
    Node(nodeid_t id) : id_(id) {}

    nodeid_t GetId() const {
        return id_;
    }

    void SetName(const std::string& name) {
        name_ = name;
    }
    const std::string& GetName() const {
        return name_;
    }

    void SetType(const Type& type) {
        type_ = type;
    }
    void SetType(Type&& type) {
        type_ = std::move(type);
    }
    const Type& GetType() const {
        return type_;
    }

    // ----- //

    uint32_t GetInputCount() const {
        return inputs_.size();
    }
    edgeid_t GetInput(uint32_t idx) const {
        return inputs_[idx];
    }

    /** @note inputs may contain duplicated edges */
    void AddInput(edgeid_t eid) {
        inputs_.push_back(eid);
    }

    /**
       @brief replace `old_value` with `new_value` in inputs,
       keeping `new_value` is in the same place as `old_value`.
       @return the number of replacement
    */
    uint32_t ReplaceInput(edgeid_t old_value, edgeid_t new_value);

    // ----- //

    uint32_t GetOutputCount() const {
        return outputs_.size();
    }
    edgeid_t GetOutput(uint32_t idx) const {
        return outputs_[idx];
    }

    /** @note outputs don't contain duplicated edges */
    void AddOutput(edgeid_t);

    /**
       @brief replace `old_value` with `new_value` in outputs,
       keeping `new_value` is in the same place as `old_value`.
       @return the number of replacement
    */
    uint32_t ReplaceOutput(edgeid_t old_value, edgeid_t new_value);

    // ----- //

    uint32_t GetExtraInputCount() const {
        return extra_inputs_.size();
    }
    edgeid_t GetExtraInput(uint32_t idx) const {
        return extra_inputs_[idx];
    }

    /** @note extra inputs don't contain duplicated edges */
    void AddExtraInput(edgeid_t);

    /**
       @brief replaces `old_value` with `new_value` in extra inputs,
       keeping `new_value` is in the same place as `old_value`.
       @return the number of replacement
    */
    uint32_t ReplaceExtraInput(edgeid_t old_value, edgeid_t new_value);

private:
    const nodeid_t id_;
    std::string name_;
    Type type_;

    std::vector<edgeid_t> inputs_;
    std::vector<edgeid_t> outputs_;

    /** inputs for nodes containing subgraph(s), for example, If/Loop/Scan */
    std::vector<edgeid_t> extra_inputs_;

private:
    Node(const Node&) = delete;
    void operator=(const Node&) = delete;
};

}}} // namespace ppl::nn::ir

#endif
