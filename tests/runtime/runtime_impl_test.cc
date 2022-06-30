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

#include "tests/ir/graph_builder.h"
#include "create_runtime_graph_info.h"
#include "ppl/nn/runtime/runtime_impl.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::nn;
using namespace ppl::common;
using namespace ppl::nn::test;

static void CreateRuntimeImpl(vector<unique_ptr<EngineImpl>>* engines, RuntimeImpl* rt) {
    GraphBuilder builder;
    auto graph_info = CreateRuntimeGraphInfoForTest(&builder, engines);
    EXPECT_NE(nullptr, graph_info);

    RetCode status;
    auto topo = builder.GetGraph()->topo;

    auto aux_info = make_shared<RuntimeAuxInfo>();
    status = aux_info->Init(topo.get(), {});
    EXPECT_EQ(RC_SUCCESS, status);

    RuntimeInitInfo init_info;
    status = init_info.Init(topo.get());
    EXPECT_EQ(RC_SUCCESS, status);

    status = rt->Init(topo, graph_info, aux_info, init_info, {});
    EXPECT_EQ(RC_SUCCESS, status);
}

TEST(RuntimeImplTest, init) {
    RuntimeImpl rt;
    vector<unique_ptr<EngineImpl>> engines;
    CreateRuntimeImpl(&engines, &rt);
}
