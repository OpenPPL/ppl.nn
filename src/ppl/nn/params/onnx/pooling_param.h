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

#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_POOLING_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_POOLING_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace onnx {

struct PoolingParam final : public ir::TypedAttr<PoolingParam> {
    int32_t auto_pad;
    int32_t ceil_mode;
    int32_t storage_order = 0; // MaxPool
    std::vector<int32_t> dilations;
    std::vector<int32_t> kernel_shape;
    std::vector<int32_t> pads;
    std::vector<int32_t> strides;

    enum { POOLING_MAX = 0, POOLING_AVERAGE_EXCLUDE = 1, POOLING_AVERAGE_INCLUDE = 2 };
    int32_t mode; // AveragePool, corresponding to `count_include_pad`

    int32_t global_pooling;

    bool operator==(const PoolingParam& p) const {
        return this->kernel_shape == p.kernel_shape && this->dilations == p.dilations && this->strides == p.strides &&
            this->pads == p.pads && this->mode == p.mode && this->ceil_mode == p.ceil_mode &&
            this->global_pooling == p.global_pooling;
    }
};

}}} // namespace ppl::nn::onnx

#endif
