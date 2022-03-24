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

#ifndef _ST_HPC_PPL_NN_RUNTIME_PARTIAL_RUNTIME_CREATOR_H_
#define _ST_HPC_PPL_NN_RUNTIME_PARTIAL_RUNTIME_CREATOR_H_

#include "ppl/nn/ir/graph_topo.h"
#include "ppl/nn/runtime/runtime_graph_info.h"
#include "ppl/nn/runtime/runtime_aux_info.h"
#include "ppl/nn/runtime/runtime_impl.h"
#include <memory>
#include <unordered_map>

namespace ppl { namespace nn {

class PartialRuntimeCreator final {
public:
    PartialRuntimeCreator() : topo_(nullptr) {}
    void Init(const ir::GraphTopo*, const std::shared_ptr<RuntimeGraphInfo>&, const std::shared_ptr<RuntimeAuxInfo>&);
    RuntimeImpl* Create(const char** begin_ops, uint32_t begin_op_num, const char** end_ops, uint32_t end_op_num,
                        const std::set<nodeid_t>& reserved_edgeids);

private:
    struct BeginEndOps final {
        std::vector<nodeid_t> begin;
        std::vector<nodeid_t> end;
        bool operator==(const BeginEndOps& rhs) const {
            return (begin == rhs.begin && end == rhs.end);
        }
    };

    struct PartialRuntimeResource final {
        std::shared_ptr<ir::GraphTopo> topo;
        std::shared_ptr<RuntimeAuxInfo> aux_info;
        std::set<nodeid_t> reserved_edgeids;
    };

    struct BeginEndOpsHash final {
        uint64_t operator()(const BeginEndOps& ops) const {
            uint64_t sum = 0;
            for (auto x = ops.begin.begin(); x != ops.begin.end(); ++x) {
                sum += *x;
            }
            for (auto x = ops.end.begin(); x != ops.end.end(); ++x) {
                sum += *x;
            }
            return sum;
        }
    };

    static void InitBeginEndOps(const char** begin_ops, uint32_t begin_op_num, const char** end_ops,
                                uint32_t end_op_num, const std::map<std::string, uint32_t>& name2idx, BeginEndOps* ops);
    static ppl::common::RetCode InitPartialRuntimeResource(const ir::GraphTopo* topo,
                                                           const std::set<nodeid_t>& reserved_edgeids,
                                                           const BeginEndOps& ops, PartialRuntimeResource* resource);

private:
    const ir::GraphTopo* topo_;
    std::shared_ptr<RuntimeGraphInfo> graph_info_;
    std::shared_ptr<RuntimeAuxInfo> aux_info_;
    std::map<std::string, nodeid_t> name2nid_;
    std::unordered_map<BeginEndOps, PartialRuntimeResource, BeginEndOpsHash> ops2resource_;

private:
    PartialRuntimeCreator(const PartialRuntimeCreator&) = delete;
    PartialRuntimeCreator& operator=(const PartialRuntimeCreator&) = delete;
};

}} // namespace ppl::nn

#endif
