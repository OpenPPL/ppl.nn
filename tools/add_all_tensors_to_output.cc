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

#include "ppl/nn/models/onnx/generated/onnx.pb.h"

// large proto file support
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include <iostream>
#include <fstream>
using namespace std;

static bool ParseFromBinaryFile(const char* model_file, ::onnx::ModelProto* pb_model) {
    FILE* fp = fopen(model_file, "r");
    if (!fp) {
        cerr << "open onnx model file[" << model_file << "] failed." << endl;
        return false;
    }

    int fd = fileno(fp);
    google::protobuf::io::FileInputStream fis(fd);
    google::protobuf::io::CodedInputStream cis(&fis);
    cis.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
    bool ok = pb_model->ParseFromCodedStream(&cis);
    fclose(fp);

    return ok;
}

static void ParseNodeInfo(const ::onnx::NodeProto& pb_node, set<string>* res) {
    for (int i = 0; i < pb_node.input_size(); ++i) {
        res->insert(pb_node.input(i));
    }
    for (int i = 0; i < pb_node.output_size(); ++i) {
        res->insert(pb_node.output(i));
    }
}

static set<string> CollectAllTensors(const ::onnx::GraphProto& pb_graph) {
    set<string> res;
    for (int i = 0; i < pb_graph.node_size(); ++i) {
        auto& pb_node = pb_graph.node(i);
        ParseNodeInfo(pb_node, &res);
    }
    return res;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "usage: " << argv[0] << " input-onnx-model output-onnx-model" << endl;
        return -1;
    }

    ::onnx::ModelProto pb_model;
    if (!ParseFromBinaryFile(argv[1], &pb_model)) {
        cerr << "parse onnx model[" << argv[1] << "] failed." << endl;
        return -1;
    }

    auto pb_graph = pb_model.mutable_graph();
    auto tensors = CollectAllTensors(*pb_graph);

    for (int i = 0; i < pb_graph->output_size(); ++i) {
        auto& output = pb_graph->output(i);
        tensors.erase(output.name());
    }

    for (auto it = tensors.begin(); it != tensors.end(); ++it) {
        auto output = pb_graph->add_output();
        output->set_name(*it);
        output->mutable_type()->mutable_tensor_type(); // set tensor type
    }

    string content;
    pb_model.SerializeToString(&content);

    ofstream ofs(argv[2], ios_base::out | ios_base::binary | ios_base::trunc);
    if (!ofs.is_open()) {
        cerr << "open output file[" << argv[2] << "] failed." << endl;
        return -1;
    }

    ofs.write(content.data(), content.size());
    return 0;
}
