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

#include <ppl/nn/engines/cuda/module/cuda_module.h>

namespace ppl { namespace nn { namespace cuda {

void CUDAModule::SaveToFile() {
}

CUfunction CUDAModule::GetKernelFunc() {
    if (module_ == nullptr) {
        cuModuleLoadDataEx(&module_, source_code_.second.c_str(), 0, 0 , 0);
    }
    CUfunction func;
    cuModuleGetFunction(&func, module_, this->source_code_.first.c_str());
    return func;
}

CUfunction CUDAModuleWrapper::GetKernelFunc() {
    return module_->GetKernelFunc();
}
}}} // namespace ppl::nn::cuda
