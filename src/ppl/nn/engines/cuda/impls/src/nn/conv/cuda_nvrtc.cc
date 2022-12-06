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

#include "cuda_nvrtc.h"

static std::mutex mutex_;

std::string CUDAIncludePathImpl() {
#if defined(_WIN32) || defined(_WIN64)
    const std::string delimiter = "\\";
#else
    const std::string delimiter = "/";
#endif

    std::string include_path;
    const char* cuda_path = "CUDA_PATH";
    const char* cuda_path_env = std::getenv(cuda_path);
    if (cuda_path_env != nullptr) {
        include_path = cuda_path_env + delimiter + "include";
        return include_path;
    }
#if defined(__linux__)
    struct stat st;
    include_path = "/usr/local/cuda/include";
    if (stat(include_path.c_str(), &st) == 0) {
        return include_path;
    }

    if (stat("/usr/include/cuda.h", &st) == 0) {
        return "/usr/include";
    }
#endif
    return include_path;
}

CUfunction GetKernelFuncImpl(CUmodule module_ptr, const std::string& ptx_code, std::string name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (module_ptr == nullptr) {
        PPL_CUDA_SAFE_CALL(cuModuleLoadDataEx(&module_ptr, ptx_code.c_str(), 0, 0, 0));
    }
    CUfunction function;
    PPL_CUDA_SAFE_CALL(cuModuleGetFunction(&function, module_ptr, name.c_str()));
    return function;
}

std::string CUDANVRTCCompileImpl(std::pair<std::string, std::string> code, std::vector<const char*> compile_params, const cudaDeviceProp& device_prop,
                             bool include) {
    std::string ptx_code;
#ifdef PPLNN_ENABLE_CUDA_JIT
    std::vector<const char*> cuda_compile_params{};
    std::vector<std::string> params;
    std::string compile_arch = "-arch=compute_" + std::to_string(device_prop.major) + std::to_string(device_prop.minor);
    params.push_back(compile_arch);
    std::string macro = "-DPPLNN_ENABLE_CUDA_JIT=ON";
    params.push_back(macro);
    if (include) {
        std::string cuda_include = "--include-path=" + CUDAIncludePathImpl();
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
