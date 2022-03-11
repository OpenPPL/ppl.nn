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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_DATA_VISITOR_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_DATA_VISITOR_H_

#include "ppl/nn/common/data_visitor.h"
#include "ppl/nn/runtime/tensor_impl.h"

namespace ppl { namespace nn { namespace cuda {

class CudaDataVisitor final : public DataVisitor {
public:
    CudaDataVisitor(const std::map<std::string, TensorImpl>* n2c) : name2constant_(n2c) {}
    Tensor* GetConstant(const char* name) const override {
        auto ref = name2constant_->find(name);
        if (ref == name2constant_->end()) {
            return nullptr;
        }
        return const_cast<TensorImpl*>(&ref->second);
    }

private:
    const std::map<std::string, TensorImpl>* name2constant_;
};

}}} // namespace ppl::nn::cuda

#endif
