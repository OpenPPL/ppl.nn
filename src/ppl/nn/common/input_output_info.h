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

#ifndef _ST_HPC_PPL_NN_COMMON_INPUT_OUTPUT_INFO_H_
#define _ST_HPC_PPL_NN_COMMON_INPUT_OUTPUT_INFO_H_

#include "ppl/nn/ir/node.h"
#include "ppl/nn/runtime/edge_object.h"

namespace ppl { namespace nn {

/**
   @class InputOutputInfo
   @brief wrapper for getting input/output tensors of a kernel
*/
class InputOutputInfo {
public:
    /** @brief a wrapper for getting the corresponding object of the specified id and type. */
    class AcquireObject {
    public:
        virtual ~AcquireObject() {}
        virtual EdgeObject* Acquire(edgeid_t eid, uint32_t etype) = 0;
    };

public:
    virtual ~InputOutputInfo() {}

    void SetNode(const ir::Node* node) {
        node_ = node;
    }

    /** @brief sets a getter class which requires an edgeid `eid` and `etype` and returns the corresponding object. */
    void SetAcquireObject(AcquireObject* getter) {
        getter_ = getter;
    }

    uint32_t GetInputCount() const {
        return node_->GetInputCount();
    }

    template <typename T>
    T* GetInput(uint32_t idx) const {
        auto eid = node_->GetInput(idx);
        return static_cast<T*>(getter_->Acquire(eid, EdgeObjectType<T>::value));
    }

    uint32_t GetExtraInputCount() const {
        return node_->GetExtraInputCount();
    }

    template <typename T>
    T* GetExtraInput(uint32_t idx) const {
        auto eid = node_->GetExtraInput(idx);
        return static_cast<T*>(getter_->Acquire(eid, EdgeObjectType<T>::value));
    }

    uint32_t GetOutputCount() const {
        return node_->GetOutputCount();
    }

    template <typename T>
    T* GetOutput(uint32_t idx) const {
        auto eid = node_->GetOutput(idx);
        return static_cast<T*>(getter_->Acquire(eid, EdgeObjectType<T>::value));
    }

protected:
    const ir::Node* node_ = nullptr;
    AcquireObject* getter_ = nullptr;
};

}} // namespace ppl::nn

#endif
