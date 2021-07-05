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

#include "ppl/nn/ir/node.h"
#include "ppl/nn/utils/vector_utils.h"
using namespace std;

namespace ppl { namespace nn { namespace ir {

static uint32_t DoReplace(edgeid_t* vec, uint32_t size, edgeid_t old_value, edgeid_t new_value) {
    uint32_t counter = 0;
    for (uint32_t i = 0; i < size; ++i) {
        if (vec[i] == old_value) {
            vec[i] = new_value;
            ++counter;
        }
    }
    return counter;
}

uint32_t Node::ReplaceInput(edgeid_t old_value, edgeid_t new_value) {
    return DoReplace(inputs_.data(), inputs_.size(), old_value, new_value);
}

void Node::AddOutput(edgeid_t eid) {
    utils::VectorAddUnique(outputs_, eid);
}

uint32_t Node::ReplaceOutput(edgeid_t old_value, edgeid_t new_value) {
    return DoReplace(outputs_.data(), outputs_.size(), old_value, new_value);
}

void Node::AddExtraInput(edgeid_t eid) {
    utils::VectorAddUnique(extra_inputs_, eid);
}

uint32_t Node::ReplaceExtraInput(edgeid_t old_value, edgeid_t new_value) {
    return DoReplace(extra_inputs_.data(), extra_inputs_.size(), old_value, new_value);
}

}}} // namespace ppl::nn::ir
