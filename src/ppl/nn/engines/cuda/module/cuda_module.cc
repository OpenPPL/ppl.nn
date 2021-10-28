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

CUfunction CUDAModule::GetKernelFunc() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (module_ == nullptr) {
        PPL_CUDA_SAFE_CALL(cuModuleLoadDataEx(&module_, source_code_.second.c_str(), 0, 0, 0));
    }
    if (func_ == nullptr) {
        PPL_CUDA_SAFE_CALL(cuModuleGetFunction(&func_, module_, this->source_code_.first.c_str()));
    }
    return func_;
}

CUfunction CUDAModule::GetKernelFunc(std::string name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (module_ == nullptr) {
        PPL_CUDA_SAFE_CALL(cuModuleLoadDataEx(&module_, source_code_.second.c_str(), 0, 0, 0));
    }
    CUfunction function;
    PPL_CUDA_SAFE_CALL(cuModuleGetFunction(&function, module_, name.c_str()));
    return function;
}
void CUDAModule::SetSourceCode(std::string name, std::string code) {
    source_code_ = std::make_pair<std::string, std::string>(std::move(name), std::move(code));
}
CUfunction CUDAModuleWrapper::GetKernelFunc() {
    return module_->GetKernelFunc();
}
CUDAModuleWrapper* CUDAModuleManager::FindModuleByNodeId(nodeid_t id) {
    auto mod = this->module_.find(id);
    if (mod != this->module_.end()) {
        return mod->second;
    }
    return nullptr;
}
void CUDAModuleManager::InsertModule(std::pair<nodeid_t, CUDAModuleWrapper*> mod) {
    this->module_.emplace(mod);
}

}}} // namespace ppl::nn::cuda