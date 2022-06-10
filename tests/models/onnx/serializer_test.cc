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
#include "ppl/nn/models/onnx/serializer.h"
#include "ppl/common/file_mapping.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

TEST(OnnxSerializerTest, serialize) {
    const string model_file(PPLNN_TESTDATA_DIR + string("/conv.onnx"));
    FileMapping fm;
    auto status = fm.Init(model_file.c_str(), FileMapping::READ);
    EXPECT_EQ(RC_SUCCESS, status);

    Model model;
    ModelParser model_parser;
    status = model_parser.Parse(fm.GetData(), fm.GetSize(), nullptr, &model);
    EXPECT_EQ(RC_SUCCESS, status);

    Serializer serializer;
    const string dst_model_file(PPLNN_TESTS_BUILD_DIR + string("/conv.onnx"));
    status = serializer.Serialize(dst_model_file, model);
    // remove(dst_model_file.c_str());
    EXPECT_EQ(RC_SUCCESS, status);
}
