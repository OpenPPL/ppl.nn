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

#include "ppl/nn/models/onnx/graph_parser.h"
#include "gtest/gtest.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include <string>
#include <iostream>

using namespace std;

class GraphParserTest : public testing::Test {};

static void LoadModel(const string& onnx_file, ::onnx::ModelProto* pb_model) {
    FILE* fp = fopen(onnx_file.c_str(), "r");
    EXPECT_NE(nullptr, fp);

    int fd = fileno(fp);
    google::protobuf::io::FileInputStream fis(fd);
    google::protobuf::io::CodedInputStream cis(&fis);
#if GOOGLE_PROTOBUF_VERSION < 3011000
    cis.SetTotalBytesLimit(INT_MAX, INT_MAX);
#else
    cis.SetTotalBytesLimit(INT_MAX);
#endif
    EXPECT_TRUE(pb_model->ParseFromCodedStream(&cis));
    fclose(fp);
}

TEST_F(GraphParserTest, Parse_Test) {
    const string onnx_file = PPLNN_TESTDATA_DIR + string("/conv.onnx");
    ::onnx::ModelProto pb_model;
    LoadModel(onnx_file, &pb_model);

    ppl::nn::onnx::GraphParser graph_parser;
    ppl::nn::ir::Graph graph;
    map<string, uint64_t> op_sets = {{"", 11}};
    auto status = graph_parser.Parse(pb_model.graph(), op_sets, nullptr, &graph);
    EXPECT_EQ(status, ppl::common::RC_SUCCESS);
}

TEST_F(GraphParserTest, partial_parse_test) {
    const string onnx_file = PPLNN_TESTDATA_DIR + string("/mnasnet0_5.onnx");
    ::onnx::ModelProto pb_model;
    LoadModel(onnx_file, &pb_model);

    ppl::nn::onnx::GraphParser graph_parser;
    ppl::nn::ir::Graph graph;
    map<string, uint64_t> op_sets = {{"", 11}};

    const char* inputs[] = {"325", "479", "480"};
    const uint32_t nr_input = 3;
    const char* outputs[] = {"481", "490"};
    const uint32_t nr_output = 2;
    auto status = graph_parser.Parse(pb_model.graph(), op_sets, nullptr, inputs, nr_input, outputs, nr_output, &graph);
    EXPECT_EQ(status, ppl::common::RC_SUCCESS);

    auto topo = graph.topo.get();
    EXPECT_EQ(1, topo->GetInputCount());

    const char* expected_nodes[] = {"Conv_9", "Conv_10", "Relu_11", "Conv_12", "Relu_13", "Conv_14"};
    const uint32_t nr_node = 6;
    for (uint32_t i = 0; i < nr_node; ++i) {
        auto node_name = expected_nodes[i];
        auto node = topo->GetNode(node_name);
        EXPECT_NE(nullptr, node);
    }

    const char* expected_edges[] = {"328", "482", "483", "481", "484", "485", "486", "333",
                                    "487", "488", "489", "336", "490", "491", "492"};
    const uint32_t nr_edge = 15;
    for (uint32_t i = 0; i < nr_edge; ++i) {
        auto edge_name = expected_edges[i];
        auto edge = topo->GetEdge(edge_name);
        EXPECT_NE(nullptr, edge);
    }
}
