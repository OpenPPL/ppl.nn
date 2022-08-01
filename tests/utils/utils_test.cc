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

#include "ppl/nn/utils/utils.h"
#include "gtest/gtest.h"
using namespace ppl::nn::utils;
using namespace ppl::common;

TEST(UtilsUtilsTest, ReadFileContent_normal) {
    Buffer buf;
    auto rc = ReadFileContent(__FILE__, &buf);
    EXPECT_EQ(RC_SUCCESS, rc);

    auto data = (const char*)buf.GetData();
    EXPECT_NE(nullptr, data);
    EXPECT_EQ('/', data[0]);
    EXPECT_EQ('c', data[5]);
}

TEST(UtilsUtilsTest, ReadFileContent_offset) {
    Buffer buf;
    auto rc = ReadFileContent(__FILE__, &buf, 5, 2);
    EXPECT_EQ(RC_SUCCESS, rc);
    EXPECT_EQ(2, buf.GetSize());

    auto data = (const char*)buf.GetData();
    EXPECT_NE(nullptr, data);
    EXPECT_EQ('c', data[0]);
    EXPECT_EQ('e', data[1]);
}

TEST(UtilsUtilsTest, ReadFileContent_error) {
    Buffer buf;
    auto rc = ReadFileContent(__FILE__, &buf, 0xffffffff, 1);
    EXPECT_NE(RC_SUCCESS, rc);
}
