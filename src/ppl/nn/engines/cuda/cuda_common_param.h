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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_COMMON_PARAM_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_COMMON_PARAM_H_

#include <stdint.h>
#include <string>
#include <vector>

#include "ppl/common/types.h"
#include "ppl/nn/common/types.h"
namespace ppl { namespace nn { namespace cuda {

struct CudaTensorQuant {
    ppl::common::dataformat_t format = ppl::common::DATAFORMAT_UNKNOWN;
    ppl::common::datatype_t type = ppl::common::DATATYPE_UNKNOWN;
    bool per_channel = false;
    uint32_t bit_width = 0;
    std::vector<float> scale{0.1f};
    std::vector<float> zero_point{0.0f};
};

struct CudaCommonParam {
    std::vector<CudaTensorQuant>* cuda_tensor_info;
    void* module = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
