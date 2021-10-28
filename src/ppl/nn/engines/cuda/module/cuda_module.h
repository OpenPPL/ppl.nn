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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_MODULE_CUDA_MODULE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_MODULE_CUDA_MODULE_H_

#include <unordered_map>
#include <map>
#include <vector>
#include <mutex>

#include <nvrtc.h>
#include <cuda.h>

#include "ppl/nn/engines/cuda/module/cuda_thread_config.h"
#include "ppl/nn/engines/cuda/module/cuda_compiler.h"

#include "ppl/nn/engines/cuda/cuda_device.h"
#include "ppl/nn/common/types.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn { namespace cuda {

class CUDAModule {
public:
    ~CUDAModule() {
        if (module_ != nullptr) {
            cuModuleUnload(module_);
        }
    }
    // Get the kernel function from the CUmodule
    CUfunction GetKernelFunc();
    CUfunction GetKernelFunc(std::string name);
    void SetCuModule(CUmodule module) {
        this->module_ = module;
    }
    void SetCuModule(CUfunction func) {
        this->func_ = func;
    }
    void SetSourceCode(std::string, std::string);

    std::pair<std::string, std::string> GetSourceCode() {
        return source_code_;
    }

private:
    // source code, <name, code>
    std::unordered_map<std::string, std::string> code_list_;
    // opt source code
    std::pair<std::string, std::string> source_code_;
    std::mutex mutex_;
    CUmodule module_ = nullptr;
    CUfunction func_ = nullptr;
};

class CUDAModuleWrapper {
public:
    ~CUDAModuleWrapper() {
        if (module_ != nullptr) {
            delete module_;
        }
    }
    // Initilize the cuda func wrapper
    void Init(CUDAModule* module, std::string func_name, CudaDevice* device) {
        module_ = module;
        func_name_ = func_name;
        device_ = device;
        std::fill(cuda_thread_config_.thread_config, cuda_thread_config_.thread_config + 6, 1);
        cuda_thread_config_.dyn_shmem_size = 0;
    }
    // Invoke the cuda kernel
    void Run(void** args) {
        CUfunction func = module_->GetKernelFunc();
        cudaStream_t stream = device_->GetStream();
        cuLaunchKernel(func, cuda_thread_config_.GridDim(0), cuda_thread_config_.GridDim(1),
                       cuda_thread_config_.GridDim(2), cuda_thread_config_.BlockDim(0), cuda_thread_config_.BlockDim(1),
                       cuda_thread_config_.BlockDim(2), cuda_thread_config_.dyn_shmem_size, stream, args, nullptr);
    }
    // Get the kernel func from CUDAModule
    CUfunction GetKernelFunc();

private:
    // module
    CUDAModule* module_ = nullptr;
    // Name of function
    std::string func_name_;
    // Kernel Luanch Parameters
    CUDAThreadConfig cuda_thread_config_;
    // Device in PPL CUDA
    CudaDevice* device_;
};

using ModuleMap = std::map<nodeid_t, CUDAModuleWrapper*>;

class CUDAModuleManager {
public:
    ~CUDAModuleManager() {
        for (auto iter : module_) {
            if (iter.second != nullptr) {
                delete iter.second;
            }
        }
    }
    CUDAModuleWrapper* FindModuleByNodeId(nodeid_t id);
    void InsertModule(std::pair<nodeid_t, CUDAModuleWrapper*> mod);
    ModuleMap* GetModule() {
        return &(this->module_);
    }

private:
    ModuleMap module_;
};

}}} // namespace ppl::nn::cuda

#endif