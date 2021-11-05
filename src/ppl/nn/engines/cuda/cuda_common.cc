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

#include "ppl/nn/engines/cuda/cuda_common.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace nn {

std::pair<int, int> PPLCudaGetDeviceArch(int device) {
    int major = 6, minor = 0;
    cudaError_t e1 = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaError_t e2 = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    std::pair<int, int> res;
    if (e1 == cudaSuccess && e2 == cudaSuccess) {
        res = std::pair<int, int>(std::move(major), std::move(minor));
    } else {
        LOG(ERROR) << "Can Not Get Correct Arch For Device";
        res = std::pair<int, int>(std::move(6), std::move(0));
    }
    return res;
}

std::string CUDAIncludePath() {
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

bool PPLCudaComputeCapabilityRequired(int major, int minor, int device) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);
    return device_prop.major > major || (device_prop.major == major && device_prop.minor >= minor);
}

bool PPLCudaComputeCapabilityEqual(int major, int minor, int device) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);
    return (device_prop.major == major && device_prop.minor == minor);
}

}} // namespace ppl::nn
