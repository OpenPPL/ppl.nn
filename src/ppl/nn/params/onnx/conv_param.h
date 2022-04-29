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

#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_CONV_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_CONV_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace onnx {

struct ConvParam final : public ir::TypedAttr<ConvParam> {
    enum { NOSET = 0, SAME_UPPER, SAME_LOWER, VALID };
    uint32_t auto_pad;
    int32_t group;
    std::vector<int32_t> kernel_shape;
    std::vector<int32_t> dilations;
    std::vector<int32_t> strides;
    std::vector<int32_t> pads;

    bool operator==(const ConvParam& p) const {
        return (auto_pad == p.auto_pad && group == p.group && kernel_shape == p.kernel_shape &&
                dilations == p.dilations && strides == p.strides && pads == p.pads);
    }
};

}}} // namespace ppl::nn::onnx

#endif
