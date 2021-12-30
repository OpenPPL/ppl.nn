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

#include "ppl/nn/quantization/quant_param_parser.h"
#include "ppl/nn/common/logger.h"
#include <fstream>
#include <sstream>
using namespace std;
using namespace ppl::common;

#include "rapidjson/document.h"
#include "rapidjson/error/error.h"

namespace ppl { namespace nn {

static RetCode ParseParam(const rapidjson::Value& v, QuantParam* param) {
    if (!v.IsObject()) {
        LOG(ERROR) << "value is not an object.";
        return RC_INVALID_VALUE;
    }

    for (auto it = v.MemberBegin(); it != v.MemberEnd(); ++it) {
        const string key(it->name.GetString(), it->name.GetStringLength());
        QuantParam::Value value;
        if (it->value.IsBool()) {
            auto field_value = it->value.GetBool();
            value.content.assign((const char*)&field_value, sizeof(field_value));
        } else if (it->value.IsDouble()) {
            auto field_value = it->value.GetDouble();
            value.content.assign((const char*)&field_value, sizeof(field_value));
        } else if (it->value.IsInt64()) {
            auto field_value = it->value.GetInt64();
            value.content.assign((const char*)&field_value, sizeof(field_value));
        } else if (it->value.IsString()) {
            value.content.assign(it->value.GetString(), it->value.GetStringLength());
        } else if (it->value.IsArray()) {
            value.content.clear();
            for (auto iter = it->value.GetArray().Begin(); iter != it->value.GetArray().End(); ++iter) {
                auto field_value = iter->GetDouble();
                value.content.append((const char*)&field_value, sizeof(field_value));
            }
        } else {
            LOG(ERROR) << "unsupported json value type.";
            return RC_UNSUPPORTED;
        }

        param->fields.insert(make_pair(key, value));
    }

    return RC_SUCCESS;
}

static RetCode ReadFileContent(const char* fname, string* buf) {
    ifstream ifile;

    ifile.open(fname, ios_base::in);
    if (!ifile.is_open()) {
        LOG(ERROR) << "open quant file[" << fname << "] failed.";
        return RC_NOT_FOUND;
    }

    stringstream ss;
    ss << ifile.rdbuf();
    *buf = ss.str();

    ifile.close();
    return RC_SUCCESS;
}

RetCode QuantParamParser::ParseBuffer(const char* buf, QuantParamInfo* info) {
    rapidjson::Document d;

    d.Parse(buf);
    if (d.HasParseError()) {
        LOG(ERROR) << "parse quant file failed: position[" << d.GetErrorOffset() << "], code[" << d.GetParseError()
                   << "]";
        return RC_INVALID_VALUE;
    }

    if (!d.IsObject()) {
        LOG(ERROR) << "quant file content is not an object.";
        return RC_INVALID_VALUE;
    }

    for (auto it = d.MemberBegin(); it != d.MemberEnd(); ++it) {
        const string object_name(it->name.GetString(), it->name.GetStringLength());
        if (!it->value.IsObject()) {
            LOG(ERROR) << "value of object[" << object_name << "] is not an object.";
            return RC_INVALID_VALUE;
        }
        const bool is_tensor_set = (object_name == "quant_info");
        const bool is_node_set = (object_name == "op_info");
        if (!is_tensor_set && !is_node_set) {
            LOG(ERROR) << "name of object[" << object_name << "] is not meaningful.";
            return RC_INVALID_VALUE;
        }
        auto& quant_info = is_tensor_set ? info->tensor_params : info->node_params;
        for (auto iter = it->value.MemberBegin(); iter != it->value.MemberEnd(); ++iter) {
            const string quant_name(iter->name.GetString(), iter->name.GetStringLength());
            if (!iter->value.IsObject()) {
                LOG(ERROR) << "value of sub-object[" << quant_name << "] is not an object.";
                return RC_INVALID_VALUE;
            }
            QuantParam param;
            auto status = ParseParam(iter->value, &param);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "ParseParam of [" << quant_name << "] failed: " << GetRetCodeStr(status);
                return status;
            }
            quant_info.insert(make_pair(quant_name, param));
        }
    }

    return RC_SUCCESS;
}

RetCode QuantParamParser::ParseFile(const char* fname, QuantParamInfo* info) {
    string buf;
    auto status = ReadFileContent(fname, &buf);
    if (status != RC_SUCCESS) {
        return status;
    }

    status = ParseBuffer(buf.c_str(), info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse quant file[" << fname << "] failed: " << GetRetCodeStr(status);
    }

    return status;
}

}} // namespace ppl::nn
