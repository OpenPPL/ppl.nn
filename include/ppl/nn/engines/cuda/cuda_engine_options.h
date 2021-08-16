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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_ENGINE_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_ENGINE_OPTIONS_H_

#include "ppl/nn/common/common.h"
#include "ppl/nn/engines/cuda/cuda_options.h"
#include <stdint.h>

namespace ppl { namespace nn {

struct PPLNN_PUBLIC CudaEngineOptions final {
    uint32_t device_id = 0;
    uint32_t mm_policy = CUDA_MM_COMPACT;
};

}} // namespace ppl::nn

#endif
