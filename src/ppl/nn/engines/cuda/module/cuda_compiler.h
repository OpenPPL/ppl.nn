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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_MODULE_CUDA_COMPILER_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_MODULE_CUDA_COMPILER_H_

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <nvrtc.h>

#include "ppl/nn/common/logger.h"

#include "ppl/nn/engines/cuda/cuda_common.h"

using namespace std;

#define PPL_NVRTC_SAFE_CALL(x)                                                                        \
    do {                                                                                              \
        nvrtcResult result = x;                                                                       \
        if (result != NVRTC_SUCCESS) {                                                                \
            std::cerr << "\nerror: " #x " failed with error " << nvrtcGetErrorString(result) << '\n'; \
            exit(1);                                                                                  \
        }                                                                                             \
    } while (0)

#define PPL_CUDA_SAFE_CALL(x)                                                 \
    do {                                                                      \
        CUresult result = x;                                                  \
        if (result != CUDA_SUCCESS) {                                         \
            const char* msg;                                                  \
            cuGetErrorName(result, &msg);                                     \
            std::cerr << "\nerror: " #x " failed with error " << msg << '\n'; \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define PPL_RUNTIME_SAFE_CALL(x)                                              \
    do {                                                                      \
        cudaError_t result = x;                                               \
        if (result != cudaSuccess) {                                          \
            const char* msg = cudaGetErrorName(result);                       \
            std::cerr << "\nerror: " #x " failed with error " << msg << '\n'; \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

namespace ppl { namespace nn { namespace cuda {
std::string CUDANVRTCCompile(std::pair<string, string> code, std::vector<const char*> compile_params, int device,
                             bool include);
}}} // namespace ppl::nn::cuda

#endif