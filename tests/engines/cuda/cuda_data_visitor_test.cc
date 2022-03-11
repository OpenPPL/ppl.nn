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

#include "ppl/nn/models/onnx/onnx_runtime_builder_factory.h"
#include "ppl/nn/engines/cuda/engine_factory.h"
#include "ppl/nn/common/data_visitor.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::nn;
using namespace ppl::common;

TEST(CudaDataVisitorTest, get_constant) {
    auto engine = unique_ptr<Engine>(CudaEngineFactory::Create(CudaEngineOptions()));
    auto builder = unique_ptr<OnnxRuntimeBuilder>(OnnxRuntimeBuilderFactory::Create());

    auto ep = engine.get();
    auto status = builder->Init((PPLNN_TESTDATA_DIR + string("/conv.onnx")).c_str(), &ep, 1);
    EXPECT_EQ(RC_SUCCESS, status);

    status = builder->Preprocess();
    EXPECT_EQ(RC_SUCCESS, status);

    DataVisitor* visitor = nullptr;
    status = engine->Configure(CUDA_CONF_GET_DATA_VISITOR, &visitor);
    EXPECT_TRUE(visitor != nullptr);

    auto weight = visitor->GetConstant("conv1.weight");
    EXPECT_TRUE(weight != nullptr);

    auto shape = weight->GetShape();
    EXPECT_TRUE(shape != nullptr);
    EXPECT_GT(shape->GetBytesIncludingPadding(), 0);

    cout << "shape bytes of [conv1.weight] is " << shape->GetBytesIncludingPadding() << endl;
}
