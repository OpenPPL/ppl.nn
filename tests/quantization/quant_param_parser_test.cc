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

#include "ppl/nn/quantization/quant_param_parser.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::nn;
using namespace ppl::common;

TEST(QuantParamParserTest, misc) {
    QuantParamInfo info;
    const string test_conf = PPLNN_TESTDATA_DIR + string("/quant_test.json");
    auto status = QuantParamParser::ParseFile(test_conf.c_str(), &info);
    EXPECT_EQ(RC_SUCCESS, status);

    auto item_iter = info.tensor_params.find("input.1");
    EXPECT_NE(info.tensor_params.end(), item_iter);
    auto field_iter = item_iter->second.fields.find("algorithm");
    EXPECT_NE(item_iter->second.fields.end(), field_iter);
    EXPECT_EQ("KL", field_iter->second.content);
}
