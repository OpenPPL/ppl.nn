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

#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_RANDOMUNIFORM_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_RANDOMUNIFORM_PARAM_H_

#include <stdint.h>
#include <vector>
#include "ppl/nn/ir/attr.h"

namespace ppl { namespace nn { namespace onnx {

struct RandomUniformParam final : public ir::TypedAttr<RandomUniformParam> {
    int32_t dtype;
    float high;
    float low;
    std::vector<float> seed;
    std::vector<int32_t> shape;

    bool operator==(const RandomUniformParam& p) const {
        return this->dtype == p.dtype && this->high == p.high && this->low == p.low
            && this->seed == p.seed && this->shape == p.shape;
    }
};

}}} // namespace ppl::nn::onnx

#endif
