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


#include <unordered_map>
#include <vector>
#include <mutex>

#include <nvrtc.h>
#include <cuda.h>

#include "cuda_thread_config.h"
#include "ppl/nn/engines/cuda/cuda_device.h"

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

    void SaveToFile();

private:
    // source code, <name, code>
    std::unordered_map<std::string, std::string> code_list_;
    // opt source code
    std::pair<std::string, std::string> source_code_;

    std::mutex mutex_;

    CUmodule module_ = nullptr;

};

class CUDAModuleWrapper {
public:
    // Initilize the cuda func wrapper
    void Init(CUDAModule* module, std::string func_name) {
        module_ = module;
        func_name_ = func_name_;
        std::fill(cuda_thread_config_.thread_config, cuda_thread_config_.thread_config + 6, 1);
        cuda_thread_config_.dyn_shmem_size = 0;
    }
    // Invoke the cuda kernel 
    void Run(void **args) {
        CUfunction func = module_->GetKernelFunc();
        cudaStream_t stream = device->GetStream();
        CUresult result = cuLaunchKernel(func, cuda_thread_config_.GridDim(0), cuda_thread_config_.GridDim(1), 
                                    cuda_thread_config_.GridDim(2), cuda_thread_config_.BlockDim(0), 
                                    cuda_thread_config_.BlockDim(1), cuda_thread_config_.BlockDim(2),
                                    cuda_thread_config_.dyn_shmem_size, stream, args, nullptr);
    }
    // Get the kernel func from CUDAModule
    CUfunction GetKernelFunc();
private:
    // module
    CUDAModule *module_;
    // Name of function 
    std::string func_name_;
    // Kernel Luanch Parameters
    CUDAThreadConfig cuda_thread_config_;
    // Device in PPL CUDA 
    CudaDevice *device;
};

class CUDAModuleManager {

private:
    std::vector<CUDAModule> module_;
};

}}} // namespace ppl::nn::cuda
