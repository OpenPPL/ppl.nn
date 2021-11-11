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

#ifndef _ST_HPC_PPL_NN_IR_FULL_GRAPH_TOPO_H_
#define _ST_HPC_PPL_NN_IR_FULL_GRAPH_TOPO_H_

#include "ppl/nn/ir/graph_topo.h"

namespace ppl { namespace nn { namespace ir {

/**
   @class FullGraphTopo
   @brief topology of a complete graph.
*/
class FullGraphTopo final : public GraphTopo {
public:
    template <typename T>
    class Iter final : public GraphTopo::Iter<T> {
    public:
        Iter(const std::vector<std::unique_ptr<T>>* vec) : idx_(0), vec_(vec) {
            while (idx_ < vec_->size() && !vec_->at(idx_)) {
                ++idx_;
            }
        }
        Iter(const Iter&) = default;
        Iter(Iter&&) = default;
        bool IsValid() const override {
            return idx_ < vec_->size();
        }
        T* Get() override {
            return vec_->at(idx_).get();
        }
        const T* Get() const override {
            return vec_->at(idx_).get();
        }
        void Forward() override {
            while (true) {
                ++idx_;
                if (idx_ >= vec_->size()) {
                    return;
                }
                if (vec_->at(idx_)) {
                    return;
                }
            }
        }

    private:
        uint32_t idx_;
        const std::vector<std::unique_ptr<T>>* vec_;
    };

public:
    FullGraphTopo(const std::string& name) : GraphTopo(name) {}

    // ----- //

    /** @return {node, true} if insertion is successful, or {node, false} if `name` exists. */
    std::pair<Node*, bool> AddNode(const std::string& name) override;

    std::shared_ptr<GraphTopo::NodeIter> CreateNodeIter() const override {
        return std::make_shared<Iter<Node>>(&nodes_);
    }

    nodeid_t GetMaxNodeId() const override {
        return nodes_.size();
    }
    Node* GetNodeById(nodeid_t id) override;
    const Node* GetNodeById(nodeid_t id) const override;
    void DelNodeById(nodeid_t id) override;

    // ----- //

    /** @return {edge, true} if insertion is successful, or {edge, false} if `name` exists. */
    std::pair<Edge*, bool> AddEdge(const std::string& name) override;

    std::shared_ptr<GraphTopo::EdgeIter> CreateEdgeIter() const override {
        return std::make_shared<Iter<Edge>>(&edges_);
    }

    edgeid_t GetMaxEdgeId() const override {
        return edges_.size();
    }
    Edge* GetEdgeById(edgeid_t id) override;
    const Edge* GetEdgeById(edgeid_t id) const override;
    void DelEdgeById(edgeid_t) override;

private:
    std::vector<std::unique_ptr<Edge>> edges_;
    std::vector<std::unique_ptr<Node>> nodes_;

private:
    FullGraphTopo(const FullGraphTopo&) = delete;
    FullGraphTopo& operator=(const FullGraphTopo&) = delete;
};

}}} // namespace ppl::nn::ir

#endif
