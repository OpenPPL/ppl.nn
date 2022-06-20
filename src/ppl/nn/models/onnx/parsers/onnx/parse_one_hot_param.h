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

#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_PARSERS_PARSE_ONE_HOT_PARAM_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_PARSERS_PARSE_ONE_HOT_PARAM_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/params/onnx/one_hot_param.h"
#include "ppl/nn/models/onnx/param_parser_extra_args.h"
#include "ppl/nn/models/onnx/generated/onnx.pb.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseOneHotParam(const ::onnx::NodeProto&, const ParamParserExtraArgs&, ir::Node*, ir::Attr*);

ppl::common::RetCode PackOneHotParam(const ir::Node*, const ir::Attr*, ::onnx::NodeProto*);

}}} // namespace ppl::nn::onnx

#endif
