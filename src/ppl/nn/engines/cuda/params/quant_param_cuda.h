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

#ifndef _ST_HPC_PPL_NN_QUANTIZATION_QUANT_PARAM_CUDA_H_
#define _ST_HPC_PPL_NN_QUANTIZATION_QUANT_PARAM_CUDA_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace cuda {

struct QuantParamCuda {
    QuantParamCuda(int i_z = 0, int o_z = 0, float i_s = 1.f, float o_s = 1.f) :
        i_zero_point(i_z), o_zero_point(o_z), i_step(i_s), o_step(o_s) {}
    int i_zero_point = 0;
    int o_zero_point = 0;
    float i_step = 1.0f;
    float o_step = 1.0f;
};

}}} // namespace ppl::nn::cuda

#endif
