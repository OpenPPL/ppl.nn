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

#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_PARAM_PARSER_MANAGER_H
#define _ST_HPC_PPL_NN_MODELS_ONNX_PARAM_PARSER_MANAGER_H

#include "ppl/common/retcode.h"
#include "ppl/nn/ir/attr.h"
#include "ppl/nn/utils/op_info_manager.h"
#include "ppl/nn/models/onnx/param_parser_extra_args.h"
#include "ppl/nn/models/onnx/generated/onnx.pb.h"

namespace ppl { namespace nn { namespace onnx {

typedef ir::Attr* (*CreateParamFunc)();
typedef ppl::common::RetCode (*ParseParamFunc)(const ::onnx::NodeProto&, const ParamParserExtraArgs&, ir::Node*,
                                               ir::Attr*);
typedef void (*DeleteParamFunc)(ir::Attr*);

struct ParserInfo final {
    CreateParamFunc create_param;
    ParseParamFunc parse_param;
    DeleteParamFunc destroy_param;
};

class ParamParserManager final {
public:
    static ParamParserManager* Instance() {
        static ParamParserManager mgr;
        return &mgr;
    }

    ppl::common::RetCode Register(const std::string& domain, const std::string& type, const utils::VersionRange&,
                                  const ParserInfo&);
    const ParserInfo* Find(const std::string& domain, const std::string& type, uint64_t version) const;

private:
    utils::OpInfoManager<ParserInfo> mgr_;

private:
    ParamParserManager();
};

}}} // namespace ppl::nn::onnx

#endif
