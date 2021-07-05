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

#include "tensor_buffer_info_tools.h"
#include "ppl/nn/utils/generic_cpu_device.h"
#include "gtest/gtest.h"
using namespace ppl::nn;
using namespace ppl::common;
using namespace ppl::nn::test;

TEST(TensorBufferInfoTest, empty) {
    utils::GenericCpuDevice device;
    TensorBufferInfo info;
    EXPECT_EQ(RC_SUCCESS, info.SetDevice(&device));
    EXPECT_EQ(&device, info.GetDevice());
    EXPECT_EQ(RC_SUCCESS, info.SetDevice(&device));
    EXPECT_FALSE(info.IsBufferOwner());
}

TEST(TensorBufferInfoTest, with_buffer) {
    utils::GenericCpuDevice device;
    auto info = GenRandomTensorBufferInfo(&device);
    EXPECT_EQ(&device, info.GetDevice());
    EXPECT_NE(RC_SUCCESS, info.SetDevice(&device)); // cannot set device if buffer is not empty
    info.FreeBuffer();
}

TEST(TensorBufferInfoTest, setbuffer) {
    utils::GenericCpuDevice device;
    TensorBufferInfo info;

    BufferDesc buffer;
    device.Realloc(1000, &buffer);
    info.SetBuffer(buffer, &device, true);
    EXPECT_EQ(buffer.addr, info.GetBufferPtr());

    auto buf = info.DetachBuffer();
    EXPECT_EQ(nullptr, info.GetBufferPtr());
    device.Free(&buf);
}
