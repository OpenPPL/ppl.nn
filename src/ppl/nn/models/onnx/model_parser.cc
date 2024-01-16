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

#include "ppl/nn/models/onnx/model_parser.h"
#include "ppl/nn/models/onnx/graph_parser.h"
#include "ppl/nn/common/logger.h"

// large proto file support
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

static bool ParseFromBinaryBuffer(const char* buf, uint64_t buf_len, google::protobuf::MessageLite* pb_model) {
    if (!buf) {
        LOG(ERROR) << "buf ptr is nullptr.";
        return false;
    }

    if (buf_len == 0) {
        LOG(ERROR) << "buf len is 0.";
        return false;
    }

    google::protobuf::io::CodedInputStream cis((uint8_t*)buf, buf_len);
#if GOOGLE_PROTOBUF_VERSION < 3011000
    cis.SetTotalBytesLimit(INT_MAX, INT_MAX);
#else
    cis.SetTotalBytesLimit(INT_MAX);
#endif
    return pb_model->ParseFromCodedStream(&cis);
}

static void ParseOpSets(const ::onnx::ModelProto& pb_model, map<string, uint64_t>* opset) {
    for (int i = 0; i < pb_model.opset_import_size(); ++i) {
        const string& domain = pb_model.opset_import(i).domain();
        uint64_t version = pb_model.opset_import(i).version();

        auto ref = opset->insert(make_pair(domain, 0));
        if (version > ref.first->second) {
            ref.first->second = version;
        }
    }
}

static RetCode CommonParse(
    const char* buf, uint64_t buf_len, const char* model_file_dir, Model* model,
    const function<RetCode(const ::onnx::GraphProto&, const map<string, uint64_t>& op_set, const char* model_file_dir,
                           ir::Graph* graph, map<pair<edgeid_t, uint32_t>, string>* axis_symbols)>& parse_graph_func) {
    ::onnx::ModelProto pb_model;
    if (!ParseFromBinaryBuffer(buf, buf_len, &pb_model)) {
        LOG(ERROR) << "load onnx model from model buffer failed.";
        return RC_OTHER_ERROR;
    }

    if (pb_model.graph().quantization_annotation_size() > 0) {
        LOG(ERROR) << "quantization in ONNX model is not supported now.";
        return RC_UNSUPPORTED;
    }

    ParseOpSets(pb_model, &model->opset);

    auto rc = parse_graph_func(pb_model.graph(), model->opset, model_file_dir, &model->graph, &model->axis_symbols);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "parse graph failed: " << GetRetCodeStr(rc);
        return rc;
    }

    auto topo = model->graph.topo.get();
    if (topo->GetExtraInputCount() > 0) {
        LOG(ERROR) << "unresolved extra input of graph[" << topo->GetName() << "]:";
        for (uint32_t i = 0; i < topo->GetExtraInputCount(); ++i) {
            LOG(ERROR) << "    -> " << topo->GetEdge(topo->GetExtraInput(i))->GetName();
        }
        return RC_NOT_FOUND;
    }

    return RC_SUCCESS;
}

RetCode ModelParser::Parse(const char* buf, uint64_t buf_len, const char* model_file_dir, Model* model) {
    return CommonParse(
        buf, buf_len, model_file_dir, model,
        [](const ::onnx::GraphProto& pb_graph, const map<string, uint64_t>& op_set, const char* model_file_dir,
           ir::Graph* graph, map<pair<edgeid_t, uint32_t>, string>* axis_symbols) -> RetCode {
            GraphParser graph_parser;
            return graph_parser.Parse(pb_graph, op_set, model_file_dir, graph, axis_symbols);
        });
}

RetCode ModelParser::Parse(const char* buf, uint64_t buf_len, const char* model_file_dir, const char** inputs,
                           uint32_t nr_input, const char** outputs, uint32_t nr_output, Model* model) {
    return CommonParse(
        buf, buf_len, model_file_dir, model,
        [inputs, nr_input, outputs, nr_output](const ::onnx::GraphProto& pb_graph, const map<string, uint64_t>& op_set,
                                               const char* model_file_dir, ir::Graph* graph,
                                               map<pair<edgeid_t, uint32_t>, string>* axis_symbols) -> RetCode {
            GraphParser graph_parser;
            return graph_parser.Parse(pb_graph, op_set, model_file_dir, inputs, nr_input, outputs, nr_output, graph,
                                      axis_symbols);
        });
}
}}} // namespace ppl::nn::onnx
