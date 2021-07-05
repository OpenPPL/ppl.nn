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

#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_GEMM_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_GEMM_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct GemmParam {
    int32_t num_output;
    int32_t bias_term; // 0 or 1

    // version onnx
    float alpha;
    float beta;
    int32_t transA;
    int32_t transB;
    int32_t N; // for converted mat B

    bool operator==(const GemmParam& p) const {
        return this->num_output == p.num_output && this->bias_term == p.bias_term && this->alpha == p.alpha &&
            this->beta == p.beta && this->transA == p.transA && this->transB == p.transB && this->N == p.N;
    }
};

}}} // namespace ppl::nn::common

#endif
