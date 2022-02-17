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
#include "ppl/common/file_mapping.h"
#include "gtest/gtest.h"
#include <string>
using namespace std;
using namespace ppl::nn;
using namespace ppl::common;

class ModelParserTest : public testing::Test {};

TEST_F(ModelParserTest, TestModelParser) {
    ir::Graph graph;
    const string onnx_file = PPLNN_TESTDATA_DIR + string("/conv.onnx");
    FileMapping fm;
    EXPECT_EQ(RC_SUCCESS, fm.Init(onnx_file.c_str()));
    auto res = onnx::ModelParser::Parse(fm.Data(), fm.Size(), &graph);
    EXPECT_EQ(RC_SUCCESS, res);
}
