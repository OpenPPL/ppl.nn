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

#include "ppl/nn/models/pmx/generated/pmx_generated.h"
#include "ppl/nn/models/pmx/pmx_serializer.h"
#include "ppl/nn/models/pmx/serialization_info.h"
#include "ppl/nn/runtime/opt_kernel.h"
#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/buffer_data_stream.h"
#include <vector>
#include <memory>
#include <fstream>
using namespace std;
using namespace ppl::common;
using namespace flatbuffers;

namespace ppl { namespace nn { namespace pmx {

static RetCode CreateFbEngine(SerializationInfo* sinfo, const EngineImpl* engine, Offset<pmx::Engine>* fb_engine) {
    utils::BufferDataStream content;
    auto status = engine->SerializeData(&content);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "serialize data of engine[" << engine->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    auto fb_content = sinfo->builder.CreateVector<uint8_t>((const uint8_t*)content.GetData(), content.GetSize());
    auto fb_name = sinfo->builder.CreateString(engine->GetName());
    *fb_engine = pmx::CreateEngine(sinfo->builder, fb_name, fb_content);
    return RC_SUCCESS;
}

static RetCode CreateFbEngines(SerializationInfo* sinfo, const ir::GraphTopo* topo, const vector<EngineImpl*>& engines,
                               Offset<Vector<Offset<pmx::Engine>>>* fb_engines) {
    vector<Offset<pmx::Engine>> engine_vec;
    engine_vec.reserve(engines.size());

    for (uint32_t i = 0; i < engines.size(); ++i) {
        auto engine = engines[i];
        Offset<pmx::Engine> fb_engine;
        auto status = CreateFbEngine(sinfo, engine, &fb_engine);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "CreateFbEngine[" << engine->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
        engine_vec.emplace_back(std::move(fb_engine));
    }

    *fb_engines = sinfo->builder.CreateVector<Offset<pmx::Engine>>(engine_vec);
    return RC_SUCCESS;
}

static RetCode CreateFbEdges(SerializationInfo* sinfo, const ir::GraphTopo* topo,
                             Offset<Vector<Offset<pmx::Edge>>>* fb_edges) {
    const vector<edgeid_t>& seq2edgeid = sinfo->seq2edgeid;

    vector<Offset<pmx::Edge>> edges(seq2edgeid.size());
    for (uint32_t i = 0; i < seq2edgeid.size(); ++i) {
        auto edge = topo->GetEdgeById(seq2edgeid[i]);
        edges[i] = pmx::CreateEdgeDirect(sinfo->builder, edge->GetName().c_str());
    }

    *fb_edges = sinfo->builder.CreateVector<Offset<pmx::Edge>>(edges);
    return RC_SUCCESS;
}

static RetCode CreateFbNodes(SerializationInfo* sinfo, const ir::GraphTopo* topo,
                             Offset<Vector<Offset<pmx::Node>>>* fb_nodes) {
    const vector<nodeid_t>& seq2nodeid = sinfo->seq2nodeid;
    const vector<edgeid_t>& edgeid2seq = sinfo->edgeid2seq;

    vector<Offset<pmx::Node>> nodes(seq2nodeid.size());
    for (uint32_t i = 0; i < seq2nodeid.size(); ++i) {
        auto node = topo->GetNodeById(seq2nodeid[i]);

        auto& type = node->GetType();
        auto fb_type = pmx::CreateNodeTypeDirect(sinfo->builder, type.domain.c_str(), type.name.c_str(), type.version);

        vector<uint32_t> inputs(topo->GetInputCount());
        for (uint32_t i = 0; i < topo->GetInputCount(); ++i) {
            inputs[i] = edgeid2seq[topo->GetInput(i)];
        }

        vector<uint32_t> outputs(topo->GetOutputCount());
        for (uint32_t i = 0; i < topo->GetOutputCount(); ++i) {
            outputs[i] = edgeid2seq[topo->GetOutput(i)];
        }

        vector<uint32_t> extra_inputs(topo->GetExtraInputCount());
        for (uint32_t i = 0; i < topo->GetExtraInputCount(); ++i) {
            extra_inputs[i] = edgeid2seq[topo->GetExtraInput(i)];
        }

        nodes[i] =
            pmx::CreateNodeDirect(sinfo->builder, node->GetName().c_str(), fb_type, &inputs, &outputs, &extra_inputs);
    }

    *fb_nodes = sinfo->builder.CreateVector<Offset<pmx::Node>>(nodes);
    return RC_SUCCESS;
}

static RetCode CreateFbGraphTopo(SerializationInfo* sinfo, const ir::GraphTopo* topo, Offset<pmx::GraphTopo>* fb_topo) {
    auto fb_name = sinfo->builder.CreateString(topo->GetName().c_str());

    Offset<Vector<Offset<pmx::Edge>>> fb_edges;
    auto status = CreateFbEdges(sinfo, topo, &fb_edges);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "CreateFbEdges failed: " << GetRetCodeStr(status);
        return status;
    }

    Offset<Vector<Offset<pmx::Node>>> fb_nodes;
    status = CreateFbNodes(sinfo, topo, &fb_nodes);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "CreateFbNodes failed: " << GetRetCodeStr(status);
        return status;
    }

    const vector<edgeid_t>& edgeid2seq = sinfo->edgeid2seq;

    vector<uint32_t> constants(topo->GetConstantCount());
    for (uint32_t i = 0; i < topo->GetConstantCount(); ++i) {
        constants[i] = edgeid2seq[topo->GetConstant(i)];
    }
    auto fb_constants = sinfo->builder.CreateVector<uint32_t>(constants);

    vector<uint32_t> inputs(topo->GetInputCount());
    for (uint32_t i = 0; i < topo->GetInputCount(); ++i) {
        inputs[i] = edgeid2seq[topo->GetInput(i)];
    }
    auto fb_inputs = sinfo->builder.CreateVector<uint32_t>(inputs);

    vector<uint32_t> outputs(topo->GetOutputCount());
    for (uint32_t i = 0; i < topo->GetOutputCount(); ++i) {
        outputs[i] = edgeid2seq[topo->GetOutput(i)];
    }
    auto fb_outputs = sinfo->builder.CreateVector<uint32_t>(outputs);

    vector<uint32_t> extra_inputs(topo->GetInputCount());
    for (uint32_t i = 0; i < topo->GetExtraInputCount(); ++i) {
        extra_inputs[i] = edgeid2seq[topo->GetExtraInput(i)];
    }
    auto fb_extra_inputs = sinfo->builder.CreateVector<uint32_t>(extra_inputs);

    *fb_topo = pmx::CreateGraphTopo(sinfo->builder, fb_name, fb_edges, fb_nodes, fb_constants, fb_inputs, fb_outputs,
                                    fb_extra_inputs);
    return RC_SUCCESS;
}

static const pmx::DataType g_type_ppl2fb[] = {
    pmx::DataType_UNKNOWN,    pmx::DataType_UINT8,   pmx::DataType_UINT16,  pmx::DataType_UINT32,
    pmx::DataType_UINT64,     pmx::DataType_FLOAT16, pmx::DataType_FLOAT32, pmx::DataType_FLOAT64,
    pmx::DataType_BFLOAT16,   pmx::DataType_INT4B,   pmx::DataType_INT8,    pmx::DataType_INT16,
    pmx::DataType_INT32,      pmx::DataType_INT64,   pmx::DataType_BOOL,    pmx::DataType_COMPLEX64,
    pmx::DataType_COMPLEX128,
};

static const pmx::DataFormat g_format_ppl2fb[] = {
    pmx::DataFormat_UNKNOWN, pmx::DataFormat_NDARRAY, pmx::DataFormat_NHWC8,
    pmx::DataFormat_NHWC16,  pmx::DataFormat_N2CX,    pmx::DataFormat_N4CX,
    pmx::DataFormat_N8CX,    pmx::DataFormat_N16CX,   pmx::DataFormat_N32CX,
};

static RetCode CreateFbShapes(SerializationInfo* sinfo, const map<edgeid_t, TensorShape>& shapes,
                              Offset<Vector<Offset<pmx::Shape>>>* fb_shapes) {
    const vector<edgeid_t>& edgeid2seq = sinfo->edgeid2seq;

    vector<Offset<pmx::Shape>> shape_vec;
    shape_vec.reserve(shapes.size());

    for (auto it = shapes.begin(); it != shapes.end(); ++it) {
        const TensorShape& shape = it->second;

        vector<int64_t> dims;
        for (uint32_t j = 0; j < shape.GetRealDimCount(); ++j) {
            dims.push_back(shape.GetDim(j));
        }

        auto fb_shape =
            pmx::CreateShapeDirect(sinfo->builder, edgeid2seq[it->first], g_type_ppl2fb[shape.GetDataType()],
                                   g_format_ppl2fb[shape.GetDataFormat()], &dims);
        shape_vec.emplace_back(std::move(fb_shape));
    }

    *fb_shapes = sinfo->builder.CreateVector<Offset<pmx::Shape>>(shape_vec);
    return RC_SUCCESS;
}

static pair<uint64_t, uint64_t> FindOrInsertData(const vector<uint8_t>& data, vector<uint8_t>* shared_data,
                                                 vector<pair<uint64_t, uint64_t>>* shared_data_items) {
    for (auto o = shared_data_items->begin(); o != shared_data_items->end(); ++o) {
        if (data.size() != o->second) {
            continue;
        }
        if (memcmp(data.data(), shared_data->data() + o->first, o->second) == 0) {
            return *o;
        }
    }

    auto new_data_item = pair<uint64_t, uint64_t>(shared_data->size(), data.size());
    shared_data->resize(shared_data->size() + data.size());
    memcpy(shared_data->data() + new_data_item.first, data.data(), data.size());
    shared_data_items->push_back(new_data_item);
    return new_data_item;
}

static RetCode CreateFbConstants(SerializationInfo* sinfo, const ir::GraphTopo* topo,
                                 const map<edgeid_t, RuntimeConstantInfo>& constants,
                                 Offset<Vector<Offset<pmx::Constant>>>* fb_constants, vector<uint8_t>* shared_data,
                                 vector<pair<uint64_t, uint64_t>>* shared_data_items) {
    const vector<edgeid_t>& edgeid2seq = sinfo->edgeid2seq;

    vector<Offset<pmx::Constant>> constant_vec;
    constant_vec.reserve(constants.size());

    for (auto it = constants.begin(); it != constants.end(); ++it) {
        const RuntimeConstantInfo& info = it->second;

        TensorShape dst_shape = *info.GetShape();
        dst_shape.SetDataFormat(DATAFORMAT_NDARRAY);

        auto device = info.GetDevice();
        if (!device) {
            continue;
        }

        auto bytes = dst_shape.GetBytesIncludingPadding();
        if (bytes == 0) {
            continue;
        }

        vector<uint8_t> data(bytes);
        auto converter = device->GetDataConverter();
        auto status = converter->ConvertToHost(data.data(), dst_shape, info.GetBufferPtr(), *info.GetShape());
        if (status != RC_SUCCESS) {
            auto edge = topo->GetEdgeById(it->first);
            LOG(ERROR) << "copy data of tensor[" << edge->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        auto ret_pair = FindOrInsertData(data, shared_data, shared_data_items);
        auto fb_constant = pmx::CreateConstant(sinfo->builder, edgeid2seq[it->first], ret_pair.first, ret_pair.second);
        constant_vec.emplace_back(std::move(fb_constant));
    }

    *fb_constants = sinfo->builder.CreateVector<Offset<pmx::Constant>>(constant_vec);
    return RC_SUCCESS;
}

static RetCode CreateFbNodeInfo(SerializationInfo* sinfo, const OptKernel* op, Offset<pmx::NodeInfo>* fb_node_info) {
    utils::BufferDataStream content;
    auto status = op->SerializeData(&content);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "serialize data of op[" << op->GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    auto fb_content = sinfo->builder.CreateVector<uint8_t>((const uint8_t*)content.GetData(), content.GetSize());
    *fb_node_info = pmx::CreateNodeInfo(sinfo->builder, sinfo->nodeid2seq[op->GetNode()->GetId()], fb_content);

    return RC_SUCCESS;
}

static RetCode CreateFbPartitionNodes(SerializationInfo* sinfo, const vector<std::unique_ptr<OptKernel>>& ops,
                                      Offset<Vector<Offset<pmx::NodeInfo>>>* fb_nodes) {
    vector<Offset<pmx::NodeInfo>> node_info_list;
    for (uint32_t i = 0; i < ops.size(); ++i) {
        Offset<pmx::NodeInfo> fb_node_info;
        auto status = CreateFbNodeInfo(sinfo, ops[i].get(), &fb_node_info);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "create node info of [" << ops[i]->GetNode()->GetName()
                       << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        node_info_list.emplace_back(std::move(fb_node_info));
    }

    *fb_nodes = sinfo->builder.CreateVector<Offset<pmx::NodeInfo>>(node_info_list);
    return RC_SUCCESS;
}

static RetCode CreateFbPartition(SerializationInfo* sinfo, const RuntimeGraphInfo::Partition& partition,
                                 const ir::GraphTopo* topo, Offset<pmx::Partition>* fb_partition,
                                 vector<uint8_t>* shared_data, vector<pair<uint64_t, uint64_t>>* shared_data_items) {
    const map<EngineImpl*, uint32_t>& engine2seq = sinfo->engine2seq;
    auto ref = engine2seq.find(partition.engine);
    if (ref == engine2seq.end()) {
        LOG(ERROR) << "cannot find seq of engine[" << partition.engine->GetName() << "]";
        return RC_NOT_FOUND;
    }

    Offset<Vector<Offset<pmx::NodeInfo>>> fb_nodes;
    auto status = CreateFbPartitionNodes(sinfo, partition.ops, &fb_nodes);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "cannot create nodes for partition of engine[" << partition.engine->GetName() << "]";
        return status;
    }

    Offset<Vector<Offset<pmx::Constant>>> fb_constants;
    status = CreateFbConstants(sinfo, topo, partition.constants, &fb_constants, shared_data, shared_data_items);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "create constants failed: " << GetRetCodeStr(status);
        return status;
    }

    *fb_partition = pmx::CreatePartition(sinfo->builder, ref->second, fb_nodes, fb_constants);
    return RC_SUCCESS;
}

static RetCode CreateFbPartitions(SerializationInfo* sinfo, const vector<RuntimeGraphInfo::Partition>& partitions,
                                  const ir::GraphTopo* topo, Offset<Vector<Offset<pmx::Partition>>>* fb_partitions,
                                  vector<uint8_t>* shared_data, vector<pair<uint64_t, uint64_t>>* shared_data_items) {
    const map<EngineImpl*, uint32_t>& engine2seq = sinfo->engine2seq;

    vector<Offset<pmx::Partition>> partition_vec;
    partition_vec.reserve(partitions.size());

    for (auto p = partitions.begin(); p != partitions.end(); ++p) {
        auto engine_id_ref = engine2seq.find(p->engine);
        if (engine_id_ref == engine2seq.end()) {
            LOG(ERROR) << "cannot find id of engine[" << p->engine->GetName() << "]";
            return RC_NOT_FOUND;
        }

        Offset<pmx::Partition> fb_partition;
        auto status = CreateFbPartition(sinfo, *p, topo, &fb_partition, shared_data, shared_data_items);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "CreateFbPartition failed: " << GetRetCodeStr(status);
            return status;
        }

        partition_vec.emplace_back(std::move(fb_partition));
    }

    *fb_partitions = sinfo->builder.CreateVector<Offset<pmx::Partition>>(partition_vec);
    return RC_SUCCESS;
}

static RetCode CreateFbGraphData(SerializationInfo* sinfo, const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                                 Offset<pmx::GraphData>* fb_data) {
    Offset<Vector<Offset<pmx::Shape>>> fb_shapes;
    auto status = CreateFbShapes(sinfo, info.shapes, &fb_shapes);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "CreateFbShapes failed: " << GetRetCodeStr(status);
        return status;
    }

    vector<uint8_t> shared_data;
    vector<pair<uint64_t, uint64_t>> shared_data_items;
    Offset<Vector<Offset<pmx::Partition>>> fb_partitions;
    status = CreateFbPartitions(sinfo, info.partitions, topo, &fb_partitions, &shared_data, &shared_data_items);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "CreateFbPartition failed: " << GetRetCodeStr(status);
        return status;
    }

    auto fb_shared_data = sinfo->builder.CreateVector<uint8_t>(shared_data);
    *fb_data = CreateGraphData(sinfo->builder, fb_shapes, fb_partitions, fb_shared_data);
    return RC_SUCCESS;
}

static RetCode CreateFbGraph(SerializationInfo* sinfo, const ir::GraphTopo* topo, const RuntimeGraphInfo& info,
                             Offset<pmx::Graph>* fb_graph) {
    Offset<pmx::GraphTopo> fb_topo;
    auto status = CreateFbGraphTopo(sinfo, topo, &fb_topo);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "CreateFbGraphTopo failed: " << GetRetCodeStr(status);
        return status;
    }

    Offset<pmx::GraphData> fb_data;
    status = CreateFbGraphData(sinfo, topo, info, &fb_data);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "CreateFbGraphData failed: " << GetRetCodeStr(status);
        return status;
    }

    *fb_graph = pmx::CreateGraph(sinfo->builder, fb_topo, fb_data);
    return RC_SUCCESS;
}

static RetCode CreateFbModel(SerializationInfo* sinfo, const ir::GraphTopo* topo, const vector<EngineImpl*>& engines,
                             const RuntimeGraphInfo& info, Offset<pmx::Model>* fb_model) {
    Offset<Vector<Offset<pmx::Engine>>> fb_engines;
    auto status = CreateFbEngines(sinfo, topo, engines, &fb_engines);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "CreateFbEngines failed: " << GetRetCodeStr(status);
        return status;
    }

    Offset<pmx::Graph> fb_graph;
    status = CreateFbGraph(sinfo, topo, info, &fb_graph);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "CreateFbGraph failed: " << GetRetCodeStr(status);
        return status;
    }

    *fb_model = pmx::CreateModel(sinfo->builder, 1, fb_engines, fb_graph);
    return RC_SUCCESS;
}

static void InitSerializationInfo(const ir::GraphTopo* topo, const vector<RuntimeGraphInfo::Partition>& partitions,
                                  SerializationInfo* sinfo) {
    for (auto it = partitions.begin(); it != partitions.end(); ++it) {
        sinfo->engine2seq.insert(make_pair(it->engine, sinfo->engine2seq.size()));
    }

    sinfo->nodeid2seq.resize(topo->GetMaxNodeId(), INVALID_NODEID);
    sinfo->seq2nodeid.reserve(topo->GetMaxNodeId());
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        sinfo->nodeid2seq[node->GetId()] = sinfo->seq2nodeid.size();
        sinfo->seq2nodeid.push_back(node->GetId());
    }

    sinfo->edgeid2seq.resize(topo->GetMaxEdgeId(), INVALID_EDGEID);
    sinfo->seq2edgeid.reserve(topo->GetMaxEdgeId());
    for (auto it = topo->CreateEdgeIter(); it->IsValid(); it->Forward()) {
        auto edge = it->Get();
        sinfo->edgeid2seq[edge->GetId()] = sinfo->seq2edgeid.size();
        sinfo->seq2edgeid.push_back(edge->GetId());
    }
}

static RetCode WriteModel(const FlatBufferBuilder& builder, const std::string& filename) {
    ofstream ofs(filename, ios_base::out | ios_base::trunc);
    if (!ofs.is_open()) {
        LOG(ERROR) << "open output file [" << filename << "] failed.";
        return RC_OTHER_ERROR;
    }
    ofs.write((const char*)builder.GetBufferPointer(), builder.GetSize());
    ofs.close();
    return RC_SUCCESS;
}

RetCode PmxSerializer::Serialize(const string& output_file, const ir::GraphTopo* topo,
                                 const vector<EngineImpl*>& engines, const RuntimeGraphInfo& info) {
    LOG(WARNING) << "pmx format is under heavily developing and may change in the future. do not use it in production "
                    "environment.";

    SerializationInfo sinfo;
    InitSerializationInfo(topo, info.partitions, &sinfo);

    Offset<pmx::Model> fb_model;
    auto status = CreateFbModel(&sinfo, topo, engines, info, &fb_model);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "CreateFbModel failed: " << GetRetCodeStr(status);
        return status;
    }

    sinfo.builder.Finish(fb_model);
    status = WriteModel(sinfo.builder, output_file);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "WriteModel failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx
