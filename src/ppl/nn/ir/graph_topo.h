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

#ifndef _ST_HPC_PPL_NN_IR_GRAPH_TOPO_H_
#define _ST_HPC_PPL_NN_IR_GRAPH_TOPO_H_

#include "ppl/common/types.h"
#include "ppl/common/retcode.h"
#include "ppl/nn/common/types.h"
#include "ppl/nn/ir/edge.h"
#include "ppl/nn/ir/node.h"
#include <memory>
#include <functional>

namespace ppl { namespace nn { namespace ir {

/** @brief graph topology */
class GraphTopo {
public:
    template <typename T>
    class Iter {
    public:
        virtual ~Iter() {}

        /** @brief tells if this iterator is valid */
        virtual bool IsValid() const = 0;

        virtual T* Get() = 0;
        virtual const T* Get() const = 0;

        /** @brief move to next valid item */
        virtual void Forward() = 0;
    };

public:
    typedef Iter<Node> NodeIter;
    typedef Iter<Edge> EdgeIter;

public:
    GraphTopo(const std::string& name) : name_(name) {}
    virtual ~GraphTopo() {}

    // ----- //

    const std::string& GetName() const {
        return name_;
    }

    // ----- //

    /**
       @return a pair that the first element is set to a pointer pointing to
       either the newly inserted node or to the node with the same name in the graph.
       the second element is set to true if a new node was inserted or false if `name` already existed.
       @note the first element of returned value may be null.
    */
    virtual std::pair<Node*, bool> AddNode(const std::string& name) = 0;

    /** @brief create an iterator for iterating all valid nodes. */
    virtual std::shared_ptr<NodeIter> CreateNodeIter() const = 0;

    /** @brief return the max node id that is greater than any used node id. */
    virtual nodeid_t GetMaxNodeId() const = 0;

    virtual Node* GetNodeById(nodeid_t id) = 0;
    virtual const Node* GetNodeById(nodeid_t id) const = 0;
    virtual void DelNodeById(nodeid_t id) = 0;

    Node* GetNodeByName(const std::string& name);
    const Node* GetNodeByName(const std::string& name) const;

    // ----- //

    /**
       @return a pair that the first element is set to a pointer pointing to
       either the newly inserted edge or to the edge with the same name in the graph.
       the second element is set to true if a new edge was inserted or false if `name` already existed.
       @note the first element of returned value may be null.
    */
    virtual std::pair<Edge*, bool> AddEdge(const std::string& name) = 0;

    /** @brief create an iterator for iterating all valid edges. */
    virtual std::shared_ptr<EdgeIter> CreateEdgeIter() const = 0;

    /** @brief return the max edge id that is greater than any used edge id. */
    virtual edgeid_t GetMaxEdgeId() const = 0;

    virtual Edge* GetEdgeById(edgeid_t) = 0;
    virtual const Edge* GetEdgeById(edgeid_t) const = 0;
    virtual void DelEdgeById(edgeid_t) = 0;

    Edge* GetEdgeByName(const std::string& name);
    const Edge* GetEdgeByName(const std::string& name) const;

    // ----- //

    /** @brief mark an edge as graph input edge, which is needed to be filled. */
    void MarkAsInput(edgeid_t);

    uint32_t GetInputCount() const {
        return inputs_.size();
    }
    edgeid_t GetInput(uint32_t idx) const {
        return inputs_[idx];
    }
    edgeid_t GetInput(const std::string& name) const;

    // ----- //

    /** @brief mark an edge as constant edge. */
    void MarkAsConstant(edgeid_t);

    uint32_t GetConstantCount() const {
        return constants_.size();
    }
    edgeid_t GetConstant(uint32_t idx) const {
        return constants_[idx];
    }
    edgeid_t GetConstant(const std::string& name) const;

    // ----- //

    /** @brief mark an edge as constant edge. */
    void MarkAsOutput(edgeid_t);

    uint32_t GetOutputCount() const {
        return outputs_.size();
    }
    edgeid_t GetOutput(uint32_t idx) const {
        return outputs_[idx];
    }
    edgeid_t GetOutput(const std::string& name) const;

    // ----- //

    /** @brief mark an edge, which is from outer scope, as extra input. */
    void MarkAsExtraInput(edgeid_t);

    uint32_t GetExtraInputCount() const {
        return extra_inputs_.size();
    }
    edgeid_t GetExtraInput(uint32_t idx) const {
        return extra_inputs_[idx];
    }
    edgeid_t GetExtraInput(const std::string& name) const;

    // ----- //

    /**
       @brief replace this graph with a new node named `node_name`.
       inputs/extra_inputs/constants of the graph are treated as new node's inputs, and
       outputs of the graph are treated as new node's outputs.
    */
    ppl::common::RetCode ReplaceWithNode(const std::string& node_name, const Node::Type& node_type);

    /** @brief find predecessors of the given node in this graph */
    std::vector<nodeid_t> FindPredecessors(nodeid_t) const;

    /** @brief find successors of the given node in this graph */
    std::vector<nodeid_t> FindSuccessors(nodeid_t) const;

    /**
       @brief topological sort
       @note this function does not modify this graph.
    */
    void TopologicalSort(const std::function<void(nodeid_t)>& callback) const;

protected:
    /** name of the graph */
    const std::string name_;

    /** ids of input edges that are needed to be filled. constant edges are not included. */
    std::vector<edgeid_t> inputs_;

    /** constant edge ids */
    std::vector<edgeid_t> constants_;

    /**
       inputs from outer scope. usually used if this graph is part of a node.
       for example, subgraphs of If/Loop/Scan.
    */
    std::vector<edgeid_t> extra_inputs_;

    /** output edge ids */
    std::vector<edgeid_t> outputs_;

private:
    GraphTopo(const GraphTopo&) = delete;
    GraphTopo& operator=(const GraphTopo&) = delete;
};

}}} // namespace ppl::nn::ir

#endif
