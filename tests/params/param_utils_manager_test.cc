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

#include "ppl/nn/params/param_utils_manager.h"
#include "gtest/gtest.h"
using namespace ppl::nn;
using namespace ppl::common;

static bool TestParamEqualFunc(const void* param_0, const void* param_1) {
    return true;
}

TEST(ParamUtilsManagerTest, misc) {
    ParamUtils info;
    info.equal = TestParamEqualFunc;
    auto mgr = ParamUtilsManager::Instance();
    mgr->Register("domain", "type", utils::VersionRange(0, 0), info);
    auto ret = mgr->Find("domain", "type", 0);
    EXPECT_NE(nullptr, ret);
    EXPECT_EQ(&TestParamEqualFunc, ret->equal);
}
