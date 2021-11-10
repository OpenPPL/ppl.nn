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

#ifndef _ST_HPC_PPL_NN_IR_PARTIAL_GRAPH_TOPO_H_
#define _ST_HPC_PPL_NN_IR_PARTIAL_GRAPH_TOPO_H_

#include "ppl/nn/ir/graph_topo.h"
#include <map>

namespace ppl { namespace nn { namespace ir {

/**
   @class PartialGraphTopo
   @brief topology of part of a graph.
*/
class PartialGraphTopo final : public GraphTopo {
public:
    template <typename T>
    class Iter final : public GraphTopo::Iter<T> {
    public:
        Iter(const std::vector<T*>* vec) : idx_(0), vec_(vec) {
            while (idx_ < vec_->size() && !vec_->at(idx_)) {
                ++idx_;
            }
        }
        bool IsValid() const override {
            return idx_ < vec_->size();
        }
        T* Get() override {
            return vec_->at(idx_);
        }
        const T* Get() const override {
            return vec_->at(idx_);
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
        const std::vector<T*>* vec_;
    };

public:
    /**
       @param parent parent graph which this partial graph belongs to
       @param nodes nodes of this partial graph
    */
    PartialGraphTopo(GraphTopo* parent, const std::string& name, const std::vector<nodeid_t>& nodes);

    // ----- //

    /**
       @brief add a new node with `name`
       @return {node, true} if insertion is successful, or {node, false} if `name` exists.
       @note the first element may be null if `name` is in parent graph but not in this graph.
    */
    std::pair<Node*, bool> AddNode(const std::string& name) override;

    std::shared_ptr<GraphTopo::NodeIter> CreateNodeIter() const override {
        return std::make_shared<Iter<Node>>(&node_ptrs_);
    }

    nodeid_t GetMaxNodeId() const override {
        return parent_->GetMaxNodeId();
    }
    Node* GetNodeById(nodeid_t id) override;
    const Node* GetNodeById(nodeid_t id) const override;
    void DelNodeById(nodeid_t id) override;

    // ----- //

    /**
       @brief add a new edge with `name`
       @return {edge, true} if insertion is successful, or {edge, false} if `name` exists.
       @note the first element may be null if `name` is in parent graph but not in this graph.
    */
    std::pair<Edge*, bool> AddEdge(const std::string& name) override;

    std::shared_ptr<GraphTopo::EdgeIter> CreateEdgeIter() const override {
        return std::make_shared<Iter<Edge>>(&edge_ptrs_);
    }

    edgeid_t GetMaxEdgeId() const override {
        return parent_->GetMaxEdgeId();
    }
    Edge* GetEdgeById(edgeid_t id) override;
    const Edge* GetEdgeById(edgeid_t id) const override;
    void DelEdgeById(edgeid_t) override;

private:
    GraphTopo* parent_;

    /** node ptrs from parent */
    std::vector<Node*> node_ptrs_;

    /** edges that either in parent or in override_edges_ */
    std::vector<Edge*> edge_ptrs_;

    /**
       For example, the following graph `graphABC` is partitioned to 2 parts: AB and C.

               +---+
               | A |
               +---+
                / \
            ab /   \ ac
              /     \
           +---+   +===+
           | B |   | C |
           +---+   +===+

       We construct 2 PartialGraphTopos `subAB` and `subC`, where `subAB` has node AB and
       `subC` has C. Edge `ac`(with edgeid = 1) has 1 consumer C from the perspective of `graphABC`,
       while there is no consumer of Edge `ac` from the perspective of `subAB`. A new edge
       `ac`(with the same edgeid = 1 and name `ac`) will be created in `subAB` to override the one
       in `graphABC`.
    */
    std::map<edgeid_t, std::unique_ptr<Edge>> override_edges_;

private:
    PartialGraphTopo(const PartialGraphTopo&) = delete;
    PartialGraphTopo& operator=(const PartialGraphTopo&) = delete;
};

}}} // namespace ppl::nn::ir

#endif
