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

#ifndef _ST_HPC_PPL_NN_OPTIMIZERS_SPECIAL_TYPE_GRAPH_PARTITIONER_H_
#define _ST_HPC_PPL_NN_OPTIMIZERS_SPECIAL_TYPE_GRAPH_PARTITIONER_H_

#include "ppl/nn/optimizers/graph_partitioner.h"
#include <unordered_set>
#include <vector>

namespace ppl { namespace nn {

class SpecialTypeGraphPartitioner final : public GraphPartitioner {
public:
    struct Type final {
        Type(const std::string& d = "", const std::string& n = "") : domain(d), name(n) {}
        Type(const Type&) = default;
        Type(Type&&) = default;

        bool operator==(const Type& t) const {
            return (name == t.name && domain == t.domain);
        }

        std::string domain;
        std::string name;
    };

public:
    void SetSpecialTypes(const std::vector<Type>& types) {
        special_types_.insert(types.begin(), types.end());
    }

    ppl::common::RetCode Partition(const std::vector<EngineImpl*>&, const ir::GraphTopo*,
                                   std::vector<std::pair<EngineImpl*, std::vector<nodeid_t>>>*) const override;

private:
    bool IsSpecialType(const ir::Node::Type& node_type) const {
        return (special_types_.find(Type(node_type.domain, node_type.name)) != special_types_.end());
    }

private:
    struct TypeHasher final {
        uint64_t operator()(const Type& t) const {
            return std::hash<std::string>()(t.domain) + std::hash<std::string>()(t.name);
        }
    };

    /** node types that need to be partitioned */
    std::unordered_set<Type, TypeHasher> special_types_;
};

}} // namespace ppl::nn

#endif
