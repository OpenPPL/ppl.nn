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

#include "ppl/nn/utils/buffered_cpu_allocator.h"
#include "gtest/gtest.h"
using namespace ppl::nn::utils;
using namespace ppl::common;

constexpr uint32_t TEST_PAGE_SIZE = 4096;

TEST(BufferedCpuAllocatorTest, all) {
    BufferedCpuAllocator ar;
    EXPECT_EQ(RC_SUCCESS, ar.Init());
    auto ptr = ar.Alloc(TEST_PAGE_SIZE);
    EXPECT_NE(nullptr, ptr);
    ar.Free(ptr);
}
