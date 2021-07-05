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

#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_CONVTRANSPOSE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_CONVTRANSPOSE_PARAM_H_

#include <stdint.h>
#include <string>
#include <vector>

namespace ppl { namespace nn { namespace common {

struct ConvTransposeParam {
    std::string auto_pad;
    int64_t group;
    std::vector<int32_t> dilations;
    std::vector<int32_t> kernel_shape;
    std::vector<int32_t> pads;
    std::vector<int32_t> strides;
    std::vector<int32_t> output_padding;
    std::vector<int32_t> output_shape;

    bool operator==(const ConvTransposeParam& p) const {
        return this->auto_pad == p.auto_pad && this->group == p.group && this->dilations == p.dilations &&
            this->kernel_shape == p.kernel_shape && this->pads == p.pads && this->strides == p.strides &&
            this->output_padding == p.output_padding && this->output_shape == p.output_shape;
    }
};

}}} // namespace ppl::nn::common

#endif
