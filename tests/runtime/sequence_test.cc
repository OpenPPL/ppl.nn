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

#include "ppl/nn/runtime/sequence.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::nn;

struct Value {
    Value() {}
    Value(int vv) : v(vv) {}
    int v;
};

TEST(SequenceTest, misc) {
    vector<Value> values = {
        363, 521, 16556, 5345, 974,
    };

    Sequence<Value> seq(nullptr);
    for (auto x = values.begin(); x != values.end(); ++x) {
        seq.EmplaceBack(Value(x->v));
    }

    EXPECT_EQ(values.size(), seq.GetElementCount());
    for (uint32_t i = 0; i < seq.GetElementCount(); ++i) {
        auto element = seq.GetElement(i);
        EXPECT_NE(nullptr, element);
        EXPECT_EQ(values[i].v, element->v);
    }
}
