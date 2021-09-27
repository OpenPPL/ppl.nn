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

std::string CUDANVRTCCompile(std::pair<std::string, std::string> code, std::vector<const char*> compile_params) {
    nvrtcProgram program;
    PPL_NVRTC_SAFE_CALL(nvrtcCreateProgram(&program, code.second.c_str(), code.first.c_str(), 0, nullptr, nullptr));
    PPL_NVRTC_SAFE_CALL(nvrtcCompileProgram(program, compile_params.size(), compile_params.data()));
    
    std::string ptx_code;
    size_t ptx_size = 0;
    PPL_NVRTC_SAFE_CALL(nvrtcGetPTXSize(program, &ptx_size));
    ptx_code.resize(ptx_size);

    PPL_NVRTC_SAFE_CALL(nvrtcGetPTX(program, &ptx_code[0]));
    PPL_NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));
    return ptx_code;
}
    
}}} // namespace ppl::nn::cuda