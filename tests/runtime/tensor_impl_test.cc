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
#include <vector>
using namespace std;
using namespace ppl::nn;
using namespace ppl::nn::test;
using namespace ppl::common;

static inline int64_t GenRandDim() {
    static const uint32_t max_dim = 640;
    return rand() % max_dim + 1;
}

class TensorImplTest : public testing::Test {
protected:
    void SetUp() override {
        builder_.AddNode("a", ir::Node::Type("test", "op1", 1), {"input_of_a"}, {"output_of_a"});
        builder_.AddNode("b", ir::Node::Type("test", "op2", 1), {"output_of_a"}, {"output_of_b"});
        builder_.AddNode("c", ir::Node::Type("test", "op1", 1), {"output_of_b"}, {"output_of_c"});
        builder_.Finalize();
    }

protected:
    TensorImpl ConstructFp32TensorWithCpuDevice() {
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

        EXPECT_EQ(RC_SUCCESS, tensor.SetDevice(&cpu_device_));
        EXPECT_EQ(&cpu_device_, tensor.GetDevice());

        return tensor;
    }

protected:
    utils::GenericCpuDevice cpu_device_;
    GraphBuilder builder_;
};

TEST_F(TensorImplTest, empty) {
    auto tensor = ConstructFp32TensorWithCpuDevice();
    EXPECT_EQ(RC_SUCCESS, tensor.SetDevice(&cpu_device_)); // buffer is empty
    EXPECT_FALSE(tensor.IsBufferOwner());
}

TEST_F(TensorImplTest, with_buffer) {
    auto tensor = ConstructFp32TensorWithCpuDevice();
    EXPECT_EQ(RC_SUCCESS, tensor.ReallocBuffer());
    EXPECT_NE(RC_SUCCESS, tensor.SetDevice(&cpu_device_)); // cannot set device if buffer is not empty
    tensor.FreeBuffer();
}

TEST_F(TensorImplTest, setbuffer) {
    auto tensor = ConstructFp32TensorWithCpuDevice();
    auto buf = tensor.DetachBuffer();
    EXPECT_EQ(nullptr, tensor.GetBufferPtr());
    cpu_device_.Free(&buf);
}

TEST_F(TensorImplTest, SetAndGetBufferPtr) {
    auto tensor = ConstructFp32TensorWithCpuDevice();
    auto& shape = tensor.GetShape();
    auto nr_element = shape.GetElementsIncludingPadding();

    vector<float> buf(nr_element);
    buf[0] = 1.234;
    buf[3] = 3.1415926;
    tensor.SetBufferPtr(buf.data());
    EXPECT_EQ(buf.data(), tensor.GetBufferPtr());

    vector<float> buf2(nr_element);
    EXPECT_EQ(RC_SUCCESS, tensor.CopyToHost(buf2.data()));
    EXPECT_EQ(buf, buf2);
}
