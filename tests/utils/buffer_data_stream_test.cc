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

#include "ppl/nn/utils/buffer_data_stream.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::nn::utils;
using namespace ppl::common;

TEST(BufferDataStreamTest, all) {
    BufferDataStream bds;
    const string content("Hello, world!");
    auto ret = bds.Write(content.data(), content.size());
    EXPECT_EQ(RC_SUCCESS, ret);
    EXPECT_EQ(content.size(), bds.Tell());
    EXPECT_EQ(content.size(), bds.GetSize());

    EXPECT_EQ(RC_SUCCESS, bds.Seek(2));
    EXPECT_EQ(2, bds.Tell());

    auto data = static_cast<const char*>(bds.GetData());
    EXPECT_EQ('H', data[0]);
    EXPECT_EQ(',', data[5]);
}
