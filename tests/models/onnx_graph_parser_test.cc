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

TEST_F(GraphParserTest, Parse_Test) {
    const string onnx_file = PPLNN_TESTDATA_DIR + string("/conv.onnx");
    FILE* fp = fopen(onnx_file.c_str(), "r");
    if (!fp) {
        cout << "open onnx model file [" << onnx_file << "] failed!" << endl;
        return;
    }
    int fd = fileno(fp);
    google::protobuf::io::FileInputStream fis(fd);
    google::protobuf::io::CodedInputStream cis(&fis);
    cis.SetTotalBytesLimit(INT_MAX);
    ::onnx::ModelProto pb_model;
    if (!pb_model.ParseFromCodedStream(&cis)) {
        return;
    }
    fclose(fp);
    ppl::nn::onnx::GraphParser graph_parser;
    ppl::nn::ir::Graph graph;
    auto status = graph_parser.Parse(pb_model.graph(), &graph);
    EXPECT_EQ(status, ppl::common::RC_SUCCESS);
}
