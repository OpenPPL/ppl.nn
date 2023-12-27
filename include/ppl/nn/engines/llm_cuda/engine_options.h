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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_ENGINE_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_ENGINE_OPTIONS_H_

#include "options.h"
#include "ppl/nn/common/common.h"
#include <stdint.h>
#include <cuda_runtime.h>

namespace ppl { namespace nn { namespace llm { namespace cuda {

struct PPLNN_PUBLIC EngineOptions final {
    uint32_t device_id = 0;
    uint32_t quant_method = QUANT_METHOD_NONE;
    uint32_t cublas_layout_hint = CUBLAS_LAYOUT_DEFAULT;

    uint32_t mm_policy = MM_COMPACT;

    /** used by runtime if != 0 */
    cudaStream_t runtime_stream = 0;

    /**
       if `runtime_device` is not nullptr, Runtime will use this device for memory management,
       `mm_policy` and `runtime_stream` are ignored.
    */
    DeviceContext* runtime_device = nullptr;
};

}}}} // namespace ppl::nn::llm::cuda

#endif
