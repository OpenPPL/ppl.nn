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

#include "ppl/nn/engines/cuda/module/cuda_compiler.h"

namespace ppl { namespace nn { namespace cuda {

std::string CUDANVRTCCompile(std::pair<string, string> code, std::vector<const char*> compile_params, int device,
                             bool include) {
    std::string ptx_code;
#ifdef PPLNN_ENABLE_CUDA_JIT
    std::vector<const char*> cuda_compile_params{};
    std::vector<std::string> params;
    auto arch = PPLCudaGetDeviceArch(device);
    std::string compile_arch = "-arch=compute_" + std::to_string(arch.first) + std::to_string(arch.second);
    params.push_back(compile_arch);
    if (include) {
        std::string cuda_include = "--include-path=" + CUDAIncludePath();
        params.push_back(cuda_include);
    }
    for (auto& iter : params) {
        cuda_compile_params.push_back(iter.c_str());
    }
    for (auto& iter : compile_params) {
        cuda_compile_params.push_back(iter);
    }
    nvrtcProgram program;
    PPL_NVRTC_SAFE_CALL(nvrtcCreateProgram(&program, code.second.c_str(), code.first.c_str(), 0, nullptr, nullptr));
    nvrtcResult compile_res = nvrtcCompileProgram(program, cuda_compile_params.size(), cuda_compile_params.data());
    if (compile_res != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        std::string log;
        log.resize(log_size);
        nvrtcGetProgramLog(program, &log[0]);
        LOG(ERROR) << log;
    }

    size_t ptx_size = 0;
    PPL_NVRTC_SAFE_CALL(nvrtcGetPTXSize(program, &ptx_size));
    ptx_code.resize(ptx_size);

    PPL_NVRTC_SAFE_CALL(nvrtcGetPTX(program, &ptx_code[0]));
    PPL_NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));
    cudaDeviceSynchronize();
#endif
    return ptx_code;
}

}}} // namespace ppl::nn::cuda