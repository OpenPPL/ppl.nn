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

#ifndef _ST_HPC_PPL_NN_TESTS_RUNTIME_CREATE_RUNTIME_GRAPH_INFO_H_
#define _ST_HPC_PPL_NN_TESTS_RUNTIME_CREATE_RUNTIME_GRAPH_INFO_H_

#include "ppl/nn/optimizers/utils.h"
#include "ppl/nn/optimizers/engine_graph_partitioner.h"
#include "tests/engines/tmp_engine.h"
#include "tests/ir/graph_builder.h"
#include "gtest/gtest.h"

namespace ppl { namespace nn { namespace test {

static inline std::shared_ptr<RuntimeGraphInfo> CreateRuntimeGraphInfoForTest(
    GraphBuilder* builder, std::vector<std::unique_ptr<EngineImpl>>* engines) {
    engines->emplace_back(std::unique_ptr<EngineImpl>(new TmpEngine1()));

    builder->AddNode("a", ir::Node::Type("test", "op1", 1), {"in1"}, {"out1", "out2"});
    builder->AddNode("b", ir::Node::Type("test", "op1", 1), {"out1", "out6", "out9", "out11"}, {"out3"});
    builder->AddNode("c", ir::Node::Type("test", "op1", 1), {"out1"}, {"out4"});
    builder->AddNode("d", ir::Node::Type("test", "op1", 1), {"out2"}, {"out5"});
    builder->AddNode("e", ir::Node::Type("test", "op1", 1), {"out4"}, {"out6", "out7"});
    builder->AddNode("f", ir::Node::Type("test", "op1", 1), {"out5"}, {"out8"});
    builder->AddNode("g", ir::Node::Type("test", "op1", 1), {"out7", "out8"}, {"out9"});
    builder->AddNode("h", ir::Node::Type("test", "op1", 1), {"in2"}, {"out10"});
    builder->AddNode("i", ir::Node::Type("test", "op1", 1), {"out9", "out10"}, {"out11"});
    builder->Finalize();
    auto graph = builder->GetGraph();

    utils::SharedResource resource;
    resource.engines.push_back(engines->at(0).get());
    resource.graph_partitioner = std::make_shared<EngineGraphPartitioner>();

    auto graph_info = std::make_shared<RuntimeGraphInfo>();
    auto status = utils::ProcessGraph(&resource, graph, graph_info.get());
    EXPECT_EQ(ppl::common::RC_SUCCESS, status);

    EXPECT_EQ(1, graph_info->partitions.size());
    auto& partition = graph_info->partitions[0];
    EXPECT_EQ(9, partition.ops.size());
    EXPECT_EQ(0, partition.constants.size());
    EXPECT_EQ(engines->at(0).get(), partition.engine);

    return graph_info;
}

}}} // namespace ppl::nn::test

#endif
