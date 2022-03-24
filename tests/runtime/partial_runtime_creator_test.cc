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

#include "ppl/nn/runtime/partial_runtime_creator.h"
#include "ppl/nn/auxtools/to_graphviz.h"
#include "tests/runtime/create_runtime_graph_info.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::nn;
using namespace ppl::common;
using namespace ppl::nn::test;

class PartialRuntimeCreatorTest : public testing::Test {
protected:
    void SetUp() override {
        graph_info_ = CreateRuntimeGraphInfoForTest(&builder_, &engines_);

        auto topo = builder_.GetGraph()->topo.get();
        cout << "graph -> " << utils::ToGraphviz(topo) << endl;

        aux_info_ = make_shared<RuntimeAuxInfo>();
        auto status = GenerateRuntimeAuxInfo(topo, {}, aux_info_.get());
        EXPECT_EQ(RC_SUCCESS, status);
    }

protected:
    shared_ptr<RuntimeAuxInfo> aux_info_;
    shared_ptr<RuntimeGraphInfo> graph_info_;
    vector<unique_ptr<EngineImpl>> engines_;
    GraphBuilder builder_;
};

TEST_F(PartialRuntimeCreatorTest, partial1) {
    auto topo = builder_.GetGraph()->topo.get();

    PartialRuntimeCreator creator;
    creator.Init(topo, graph_info_, aux_info_);

    const char* begin_ops[] = {"c", "d", "h"};
    const char* end_ops[] = {"i"};

    auto runtime = creator.Create(begin_ops, 3, end_ops, 1, {});
    EXPECT_TRUE(runtime != nullptr);

    auto input_count = runtime->GetInputCount();
    EXPECT_EQ(3, input_count);

    set<string> expected_inputs = {"in2", "out1", "out2"};
    for (uint32_t i = 0; i < input_count; ++i) {
        auto in = runtime->GetInputTensorImpl(i);
        EXPECT_TRUE(in != nullptr);
        EXPECT_TRUE(expected_inputs.find(in->GetName()) != expected_inputs.end());
    }

    auto output_count = runtime->GetOutputCount();
    EXPECT_EQ(3, output_count);

    set<string> expected_outputs = {"out6", "out9", "out11"};
    for (uint32_t i = 0; i < output_count; ++i) {
        auto out = runtime->GetOutputTensorImpl(i);
        EXPECT_TRUE(out != nullptr);
        EXPECT_TRUE(expected_outputs.find(out->GetName()) != expected_outputs.end());
    }
}
