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

#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/resize.h"
using namespace flatbuffers;

namespace ppl { namespace nn { namespace pmx { namespace onnx {

Offset<ResizeParam> SerializeResizeParam(const ppl::nn::common::ResizeParam& param, FlatBufferBuilder* builder) {
    return CreateResizeParam(*builder, static_cast<ResizeCoordTransMode>(param.coord_trans_mode), param.cubic_coeff_a,
                             param.exclude_outside, param.extrapolation_value, static_cast<ResizeMode>(param.mode),
                             static_cast<ResizeNearestMode>(param.nearest_mode));
}

void DeserializeResizeParam(const ResizeParam& fb_param, ppl::nn::common::ResizeParam* param) {
    param->coord_trans_mode = static_cast<uint32_t>(fb_param.coord_trans_mode());
    param->cubic_coeff_a = fb_param.cubic_coeff_a();
    param->exclude_outside = fb_param.exclude_outside();
    param->extrapolation_value = fb_param.extrapolation_value();
    param->mode = static_cast<uint32_t>(fb_param.mode());
    param->nearest_mode = static_cast<uint32_t>(fb_param.nearest_mode());
}

}}}} // namespace ppl::nn::pmx::onnx
