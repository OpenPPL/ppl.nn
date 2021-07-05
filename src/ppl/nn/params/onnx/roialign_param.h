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

#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_ROIALIGN_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_ROIALIGN_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct ROIAlignParam {
    enum { ONNXROIAlignMode_AVG = 0, ONNXROIAlignMode_MAX = 1 };

    int32_t mode;
    int32_t output_height;
    int32_t output_width;
    int32_t sampling_ratio;
    float spatial_scale;

    bool operator==(const ROIAlignParam& p) const {
        return this->mode == p.mode && this->output_height == p.output_height && this->output_width == p.output_width &&
            this->sampling_ratio == p.sampling_ratio && this->spatial_scale == p.spatial_scale;
    }
};

}}} // namespace ppl::nn::common

#endif
