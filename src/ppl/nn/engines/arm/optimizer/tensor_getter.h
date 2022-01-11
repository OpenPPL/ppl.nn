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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_TENSOR_GETTER_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_TENSOR_GETTER_H_

#include "ppl/nn/common/input_output_info.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include <map>
#include <memory>

namespace ppl { namespace nn { namespace arm {

class TensorGetter final : public InputOutputInfo::AcquireObject {
public:
    TensorGetter(std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensor_impls) : tensor_impls_(tensor_impls) {}

    EdgeObject* Acquire(edgeid_t eid, uint32_t etype) override {
        auto iter = tensor_impls_->find(eid);
        if (iter == tensor_impls_->end()) {
            return nullptr;
        }
        return iter->second.get();
    }

private:
    std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensor_impls_;
};

}}} // namespace ppl::nn::arm

#endif
