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

#ifndef _ST_HPC_PPL_NN_RUNTIME_TENSOR_SEQUENCE_H_
#define _ST_HPC_PPL_NN_RUNTIME_TENSOR_SEQUENCE_H_

#include "ppl/nn/runtime/edge_object.h"
#include "ppl/nn/common/tensor_buffer_info.h"
#include <vector>

namespace ppl { namespace nn {

class TensorSequence;

template <>
struct EdgeObjectType<TensorSequence> final {
    static const uint32_t value = EdgeObject::T_TENSOR_SEQUENCE;
};

class TensorSequence final : public EdgeObject {
public:
    TensorSequence(const ir::Edge* edge) : EdgeObject(edge, EdgeObjectType<TensorSequence>::value) {}

    uint32_t GetElementCount() const {
        return elements_.size();
    }
    TensorBufferInfo* GetElement(uint32_t idx) {
        return &elements_[idx];
    }
    const TensorBufferInfo* GetElement(uint32_t idx) const {
        return &elements_[idx];
    }
    void EmplaceBack(TensorBufferInfo&& value) {
        elements_.emplace_back(std::move(value));
    }

private:
    std::vector<TensorBufferInfo> elements_;
};

}} // namespace ppl::nn

#endif
