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

#ifndef _ST_HPC_PPL_NN_PARAMS_MMCV_MMCV_GRIDSAMPLE_PARAM_H
#define _ST_HPC_PPL_NN_PARAMS_MMCV_MMCV_GRIDSAMPLE_PARAM_H

#include "ppl/nn/ir/attr.h"
#include <stdint.h>

namespace ppl { namespace nn { namespace mmcv {

struct MMCVGridSampleParam final : public ir::TypedAttr<MMCVGridSampleParam> {
    int64_t align_corners;
    int64_t interpolation_mode;
    int64_t padding_mode;

    bool operator==(const MMCVGridSampleParam& p) const {
        return this->align_corners == p.align_corners && this->interpolation_mode == p.interpolation_mode &&
            this->padding_mode == p.padding_mode;
    }
};

}}} // namespace ppl::nn::mmcv

#endif
