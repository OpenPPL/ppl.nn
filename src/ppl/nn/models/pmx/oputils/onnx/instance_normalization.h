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

#ifndef _ST_HPC_PPL_NN_MODELS_PMX_OPUTILS_ONNX_INSTANCE_NORMALIZATION_H_
#define _ST_HPC_PPL_NN_MODELS_PMX_OPUTILS_ONNX_INSTANCE_NORMALIZATION_H_

#include "ppl/nn/models/pmx/generated/onnx_op_generated.h"
#include "ppl/nn/params/onnx/instance_normalization_param.h"

namespace ppl { namespace nn { namespace pmx { namespace onnx {

flatbuffers::Offset<InstanceNormalizationParam> SerializeInstanceNormalizationParam(
    const ppl::nn::onnx::InstanceNormalizationParam&, flatbuffers::FlatBufferBuilder*);
void DeserializeInstanceNormalizationParam(const InstanceNormalizationParam&, ppl::nn::onnx::InstanceNormalizationParam*);

}}}} // namespace ppl::nn::pmx::onnx

#endif
