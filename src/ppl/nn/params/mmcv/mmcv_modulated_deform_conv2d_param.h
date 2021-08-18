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

#ifndef _ST_HPC_PPL_NN_PARAMS_MMCV_MMCV_MODULATED_DEFORM_CONV2D_PARAM_H
#define _ST_HPC_PPL_NN_PARAMS_MMCV_MMCV_MODULATED_DEFORM_CONV2D_PARAM_H

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct MMCVModulatedDeformConv2dParam {
    int64_t kernel_size[2]; // written in op ctx
    int64_t stride[2];
    int64_t padding[2];
    int64_t dilation[2];
    int64_t groups;
    int64_t deform_groups;

    int64_t channels;  // written in op ctx
    int64_t num_output; // written in op ctx
    int64_t bias_term; // written in op ctx, for multi-input layer fusion

    bool operator==(const MMCVModulatedDeformConv2dParam& p) const {
        return false; // has attr written in op ctx
    }
};

}}} // namespace ppl::nn::common

#endif
