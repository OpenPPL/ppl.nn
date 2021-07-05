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

#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_DECONVOLUTION_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_DECONVOLUTION_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct DeconvolutionParam {
    int32_t bias_term; // 0 or 1
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t hole_h;
    int32_t hole_w;
    int32_t group;
    int32_t num_output;

    bool operator==(const DeconvolutionParam& p) const {
        return this->bias_term == p.bias_term && this->kernel_h == p.kernel_h && this->kernel_w == p.kernel_w &&
            this->stride_h == p.stride_h && this->stride_w == p.stride_w && this->pad_h == p.pad_h &&
            this->pad_w == p.pad_w && this->hole_h == p.hole_h && this->hole_w == p.hole_w && this->group == p.group &&
            this->num_output == p.num_output;
    }
};

}}} // namespace ppl::nn::common

#endif
