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

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/utils/generic_cpu_device.h"
#include "tests/ir/graph_builder.h"
#include "gtest/gtest.h"
using namespace ppl::nn;
using namespace ppl::nn::test;
using namespace ppl::common;

class TensorImplTest : public testing::Test {
protected:
    void SetUp() override {
        builder_.AddNode("a", ir::Node::Type("test", "op1"), {"input_of_a"}, {"output_of_a"});
        builder_.AddNode("b", ir::Node::Type("test", "op2"), {"output_of_a"}, {"output_of_b"});
        builder_.AddNode("c", ir::Node::Type("test", "op3"), {"output_of_b"}, {"output_of_c"});
        builder_.Finalize();
    }

protected:
    GraphBuilder builder_;
};

TEST_F(TensorImplTest, empty) {
    auto topo = builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(1);
    EXPECT_NE(nullptr, edge);

    TensorImpl tensor(edge, EdgeObject::T_TENSOR);
    EXPECT_EQ(EdgeObject::T_TENSOR, tensor.GetObjectType());
    EXPECT_EQ(edge->GetName(), tensor.GetName());

    utils::GenericCpuDevice device;
    EXPECT_EQ(RC_SUCCESS, tensor.SetDevice(&device));
    EXPECT_EQ(&device, tensor.GetDevice());
    EXPECT_EQ(RC_SUCCESS, tensor.SetDevice(&device)); // buffer is empty
    EXPECT_FALSE(tensor.IsBufferOwner());
}

static inline int64_t GenRandDim() {
    static const uint32_t max_dim = 640;
    return rand() % max_dim + 1;
}

TEST_F(TensorImplTest, with_buffer) {
    auto topo = builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(1);
    EXPECT_NE(nullptr, edge);

    TensorImpl tensor(edge, EdgeObject::T_TENSOR);
    EXPECT_EQ(EdgeObject::T_TENSOR, tensor.GetObjectType());
    EXPECT_EQ(edge->GetName(), tensor.GetName());

    TensorShape& shape = tensor.GetShape();
    shape.Reshape({1, 3, GenRandDim(), GenRandDim()});
    shape.SetDataType(DATATYPE_FLOAT32);
    shape.SetDataFormat(DATAFORMAT_NDARRAY);

    utils::GenericCpuDevice device;
    EXPECT_EQ(RC_SUCCESS, tensor.SetDevice(&device));
    EXPECT_EQ(&device, tensor.GetDevice());

    EXPECT_EQ(RC_SUCCESS, tensor.ReallocBuffer());
    EXPECT_NE(RC_SUCCESS, tensor.SetDevice(&device)); // cannot set device if buffer is not empty
    tensor.FreeBuffer();
}

TEST_F(TensorImplTest, setbuffer) {
    auto topo = builder_.GetGraph()->topo.get();
    auto edge = topo->GetEdgeById(1);
    EXPECT_NE(nullptr, edge);

    TensorImpl tensor(edge, EdgeObject::T_TENSOR);
    EXPECT_EQ(EdgeObject::T_TENSOR, tensor.GetObjectType());
    EXPECT_EQ(edge->GetName(), tensor.GetName());

    TensorShape& shape = tensor.GetShape();
    shape.Reshape({1, 3, GenRandDim(), GenRandDim()});
    shape.SetDataType(DATATYPE_FLOAT32);
    shape.SetDataFormat(DATAFORMAT_NDARRAY);

    utils::GenericCpuDevice device;
    EXPECT_EQ(RC_SUCCESS, tensor.SetDevice(&device));

    auto buf = tensor.DetachBuffer();
    EXPECT_EQ(nullptr, tensor.GetBufferPtr());
    device.Free(&buf);
}
