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

#ifndef _ST_HPC_PPL_NN_RUNTIME_EDGE_OBJECT_H_
#define _ST_HPC_PPL_NN_RUNTIME_EDGE_OBJECT_H_

#include "ppl/nn/ir/edge.h"
#include "ppl/nn/runtime/barrier.h"

namespace ppl { namespace nn {

class EdgeObject {
public:
    /** EdgeObject types */
    enum {
        T_UNKNOWN,
        T_EDGE_OBJECT,
        T_TENSOR,
        T_TENSOR_SEQUENCE,
    };

public:
    EdgeObject(const ir::Edge* edge, uint32_t type) : edge_(edge), type_(type) {}
    virtual ~EdgeObject() {}
    EdgeObject(EdgeObject&&) = default;
    EdgeObject& operator=(EdgeObject&&) = default;
    EdgeObject(const EdgeObject&) = default;
    EdgeObject& operator=(const EdgeObject&) = default;

    const ir::Edge* GetEdge() const {
        return edge_;
    }
    uint32_t GetObjectType() const {
        return type_;
    }

    void SetBarrier(Barrier* b) {
        barrier_ = b;
    }
    Barrier* GetBarrier() const {
        return barrier_;
    }

private:
    const ir::Edge* edge_;
    uint32_t type_;
    Barrier* barrier_ = nullptr;
};

template <typename T>
struct EdgeObjectType final {
    static const uint32_t value = EdgeObject::T_UNKNOWN;
};

template <>
struct EdgeObjectType<EdgeObject> final {
    static const uint32_t value = EdgeObject::T_EDGE_OBJECT;
};

}} // namespace ppl::nn

#endif
