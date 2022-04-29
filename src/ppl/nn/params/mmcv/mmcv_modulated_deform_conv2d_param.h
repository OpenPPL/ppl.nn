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

#include "ppl/nn/ir/attr.h"
#include <stdint.h>

namespace ppl { namespace nn { namespace mmcv {

struct MMCVModulatedDeformConv2dParam final : public ir::TypedAttr<MMCVModulatedDeformConv2dParam> {
    int64_t stride[2];
    int64_t padding[2];
    int64_t dilation[2];
    int64_t groups;
    int64_t deform_groups;

    bool operator==(const MMCVModulatedDeformConv2dParam& p) const {
        return (stride[0] == p.stride[0] && stride[1] == p.stride[1] && padding[0] == p.padding[0] &&
                padding[1] == p.padding[1] && dilation[0] == p.dilation[0] && dilation[1] == p.dilation[1] &&
                groups == p.groups && deform_groups == p.deform_groups);
    }
};

}}} // namespace ppl::nn::mmcv

#endif
