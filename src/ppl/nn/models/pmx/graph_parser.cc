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

#include "ppl/nn/models/pmx/graph_parser.h"
#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;
using namespace flatbuffers;

namespace ppl { namespace nn { namespace pmx {

static RetCode ParseGraphTopoEdges(const GraphTopo* fb_topo, ir::GraphTopo* topo) {
    for (uint32_t i = 0; i < fb_topo->edges()->size(); ++i) {
        auto fb_edge = fb_topo->edges()->Get(i);
        auto ret_pair = topo->AddEdge(fb_edge->name()->c_str());
        if (!ret_pair.second) {
            LOG(ERROR) << "duplicated edge[" << fb_edge->name()->c_str() << "].";
            return RC_EXISTS;
        }
    }
    return RC_SUCCESS;
}

static RetCode ParseGraphTopoNodes(const GraphTopo* fb_topo, ir::GraphTopo* topo) {
    for (uint32_t i = 0; i < fb_topo->nodes()->size(); ++i) {
        auto fb_node = fb_topo->nodes()->Get(i);
        auto ret_pair = topo->AddNode(fb_node->name()->c_str());
        if (!ret_pair.second) {
            LOG(ERROR) << "duplicated node[" << fb_node->name()->c_str() << "].";
            return RC_EXISTS;
        }

        auto node = ret_pair.first;

        auto fb_node_type = fb_node->type();
        node->SetType(
            ir::Node::Type(fb_node_type->domain()->c_str(), fb_node_type->name()->c_str(), fb_node_type->version()));

        for (auto e = fb_node->inputs()->begin(); e != fb_node->inputs()->end(); ++e) {
            auto edge = topo->GetEdge(*e);
            node->AddInput(edge->GetId());
            edge->AddConsumer(node->GetId());
        }
        for (auto e = fb_node->outputs()->begin(); e != fb_node->outputs()->end(); ++e) {
            auto edge = topo->GetEdge(*e);
            node->AddOutput(edge->GetId());
            if (edge->GetProducer() != INVALID_NODEID) {
                auto parent = topo->GetNode(edge->GetProducer());
                LOG(ERROR) << "multiple producer [" << parent->GetName() << "] and [" << node->GetName()
                           << "] of edge [" << edge->GetName() << "]";
                return RC_INVALID_VALUE;
            }
            edge->SetProducer(node->GetId());
        }
        for (auto e = fb_node->extra_inputs()->begin(); e != fb_node->extra_inputs()->end(); ++e) {
            auto edge = topo->GetEdge(*e);
            node->AddExtraInput(edge->GetId());
            edge->AddConsumer(node->GetId());
        }
    }

    return RC_SUCCESS;
}

static RetCode ParseGraphTopo(const GraphTopo* fb_topo, ir::GraphTopo* topo) {
    topo->SetName(fb_topo->name()->c_str());

    auto status = ParseGraphTopoEdges(fb_topo, topo);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphTopoEdges failed: " << GetRetCodeStr(status);
        return status;
    }

    status = ParseGraphTopoNodes(fb_topo, topo);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphTopoNodes failed: " << GetRetCodeStr(status);
        return status;
    }

    for (auto e = fb_topo->constants()->begin(); e != fb_topo->constants()->end(); ++e) {
        topo->MarkAsConstant(*e);
    }
    for (auto e = fb_topo->inputs()->begin(); e != fb_topo->inputs()->end(); ++e) {
        topo->MarkAsInput(*e);
    }
    for (auto e = fb_topo->outputs()->begin(); e != fb_topo->outputs()->end(); ++e) {
        topo->MarkAsOutput(*e);
    }
    for (auto e = fb_topo->extra_inputs()->begin(); e != fb_topo->extra_inputs()->end(); ++e) {
        topo->MarkAsExtraInput(*e);
    }

    return RC_SUCCESS;
}

static const datatype_t g_type_fb2ppl[] = {
    DATATYPE_UNKNOWN, DATATYPE_BOOL,     DATATYPE_INT8,  DATATYPE_UINT8,     DATATYPE_INT16,      DATATYPE_UINT16,
    DATATYPE_INT32,   DATATYPE_UINT32,   DATATYPE_INT64, DATATYPE_UINT64,    DATATYPE_FLOAT16,    DATATYPE_FLOAT32,
    DATATYPE_FLOAT64, DATATYPE_BFLOAT16, DATATYPE_INT4B, DATATYPE_COMPLEX64, DATATYPE_COMPLEX128,
};

static const dataformat_t g_format_fb2ppl[] = {
    DATAFORMAT_UNKNOWN, DATAFORMAT_NDARRAY, DATAFORMAT_NHWC8, DATAFORMAT_NHWC16, DATAFORMAT_N2CX,
    DATAFORMAT_N4CX,    DATAFORMAT_N8CX,    DATAFORMAT_N16CX, DATAFORMAT_N32CX,
};

static RetCode ParseGraphDataShapes(const GraphData* fb_data, map<edgeid_t, TensorShape>* shapes) {
    for (auto x = fb_data->shapes()->begin(); x != fb_data->shapes()->end(); ++x) {
        auto fb_shape = *x;

        auto ret_pair = shapes->insert(make_pair(fb_shape->edge_id(), TensorShape()));
        if (!ret_pair.second) {
            LOG(ERROR) << "duplicated shape of edgeid[" << fb_shape->edge_id() << "]";
            return RC_EXISTS;
        }

        TensorShape& shape = ret_pair.first->second;
        shape.SetDataType(g_type_fb2ppl[fb_shape->data_type()]);
        shape.SetDataFormat(g_format_fb2ppl[fb_shape->data_format()]);
        shape.SetDimCount(fb_shape->dims()->size());
        for (uint32_t i = 0; i < fb_shape->dims()->size(); ++i) {
            shape.SetDim(i, fb_shape->dims()->Get(i));
        }
        shape.CalcPadding();
    }
    return RC_SUCCESS;
}

static inline uint64_t Align(uint64_t v, uint64_t alignment) {
    return (v + alignment - 1) & (~(alignment - 1));
}

class PmxConstantVisitor final : public ConstantVisitor {
public:
    PmxConstantVisitor(const ir::GraphTopo* topo, const uint8_t* shared_data, const RuntimeGraphInfo* info,
                       const flatbuffers::Vector<flatbuffers::Offset<ppl::nn::pmx::Constant>>* fb_constants)
        : topo_(topo), shared_data_(shared_data), info_(info), fb_constants_(fb_constants) {}

    uint64_t CalcTotalBytes(uint64_t alignment) const override {
        uint64_t total_bytes = 0;
        for (auto y = fb_constants_->begin(); y != fb_constants_->end(); ++y) {
            auto fb_constant = *y;
            total_bytes += Align(fb_constant->data_bytes(), alignment);
        }
        return total_bytes;
    }

    RetCode ForEach(
        const function<RetCode(const ir::Edge*, const void*, uint64_t, const TensorShape&)>& f) const override {
        for (auto y = fb_constants_->begin(); y != fb_constants_->end(); ++y) {
            auto fb_constant = *y;
            auto edge = topo_->GetEdge(fb_constant->edge_id());

            auto shape_ref = info_->shapes.find(fb_constant->edge_id());
            if (shape_ref == info_->shapes.end()) {
                LOG(ERROR) << "cannot find shape of constant[" << edge->GetName() << "]";
                return RC_NOT_FOUND;
            }

            auto status =
                f(edge, shared_data_ + fb_constant->data_offset(), fb_constant->data_bytes(), shape_ref->second);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "exec callback for constant[" << edge->GetName() << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }
        return RC_SUCCESS;
    }

private:
    const ir::GraphTopo* topo_;
    const uint8_t* shared_data_;
    const RuntimeGraphInfo* info_;
    const flatbuffers::Vector<flatbuffers::Offset<ppl::nn::pmx::Constant>>* fb_constants_;
};

static RetCode ParseGraphDataPartitions(const GraphData* fb_data, const ir::GraphTopo* topo,
                                        const vector<EngineImpl*>& seq2engine, RuntimeGraphInfo* info) {
    auto fb_partitions = fb_data->partitions();
    info->partitions.reserve(fb_partitions->size());

    for (auto x = fb_partitions->begin(); x != fb_partitions->end(); ++x) {
        auto fb_partition = *x;
        auto engine = seq2engine[fb_partition->engine_id()];

        RuntimeGraphInfo::Partition partition;
        partition.engine = engine;
        DeserializationContext dummy_ctx;

        for (auto y = fb_partition->nodes()->begin(); y != fb_partition->nodes()->end(); ++y) {
            auto fb_node = *y;

            auto node = topo->GetNode(fb_node->node_id());
            if (!node) {
                LOG(ERROR) << "cannot find node of id[" << fb_node->node_id() << "] in partition of engine["
                           << engine->GetName() << "]";
                return RC_NOT_FOUND;
            }

            auto op = std::unique_ptr<OptKernel>(engine->CreateOptKernel(node));
            if (!op) {
                LOG(ERROR) << "create op for node[" << node->GetName() << "] failed.";
                return RC_NOT_FOUND;
            }

            auto status = op->DeserializeData(dummy_ctx, fb_node->data()->data(), fb_node->data()->size());
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "deserialize of op[" << node->GetName() << "] failed: " << GetRetCodeStr(status);
                return status;
            }

            partition.ops.emplace_back(std::move(op));
        }

        PmxConstantVisitor visitor(topo, fb_data->shared_data()->data(), info, fb_partition->constants());
        auto status = engine->LoadConstants(visitor, &partition.constants);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "LoadConstants of engine[" << engine->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        info->partitions.emplace_back(std::move(partition));
    }

    return RC_SUCCESS;
}

static RetCode ParseGraphData(const GraphData* fb_data, const ir::GraphTopo* topo,
                              const vector<EngineImpl*>& seq2engine, RuntimeGraphInfo* info) {
    auto status = ParseGraphDataShapes(fb_data, &info->shapes);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphDataShapes failed: " << GetRetCodeStr(status);
        return status;
    }

    status = ParseGraphDataPartitions(fb_data, topo, seq2engine, info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphDataPartitions failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode GraphParser::Parse(const Graph* fb_graph, const vector<EngineImpl*>& seq2engine, ir::GraphTopo* topo,
                           RuntimeGraphInfo* info) {
    auto status = ParseGraphTopo(fb_graph->topo(), topo);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphTopo failed: " << GetRetCodeStr(status);
        return status;
    }

    status = ParseGraphData(fb_graph->data(), topo, seq2engine, info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ParseGraphData failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx
