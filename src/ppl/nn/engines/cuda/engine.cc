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

#include "ppl/nn/engines/cuda/engine.h"

#include <stdarg.h>

#include "ppl/nn/engines/cuda/engine_context.h"
#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/engines/cuda/optimizer/opt_graph.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode CudaEngine::Init(const CudaEngineOptions& options) {
    // TODO implement other options
    return device_.Init(options, MM_LESS_MEMORY);
}

EngineContext* CudaEngine::CreateEngineContext(const string&, const EngineContextOptions& options) {
    auto ctx = unique_ptr<CudaEngineContext>(new CudaEngineContext(GetName()));
    CudaEngineOptions cuda_options;
    cuda_options.device_id = device_.GetDeviceId();
    auto status = ctx->Init(cuda_options, options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init CudaEngineContext failed: " << GetRetCodeStr(status);
        return nullptr;
    }

    return ctx.release();
}

bool CudaEngine::CanRunOp(const ir::Node* node) const {
    auto& type = node->GetType();
    return (OptKernelCreatorManager::Instance()->Find(type.domain, type.name) != nullptr);
}

RetCode CudaEngine::DoOptimize(ir::Graph* graph, utils::SharedResource* resource, RuntimePartitionInfo* info) {
    OptGraph opt_graph(graph, info, resource, &cuda_flags_);
    auto status = opt_graph.DoOptimize(&device_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "OptGraph DoOptimeize failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode CudaEngine::ProcessGraph(utils::SharedResource* resource, ir::Graph* graph, RuntimePartitionInfo* info) {
    auto status = DoOptimize(graph, resource, info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "DoOptimize failed: " << GetRetCodeStr(status);
        return status;
    }

    status = utils::LoadConstants(*graph, &device_, &info->constants);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "LoadConstants failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

/* -------------------------------------------------------------------------- */

RetCode CudaEngine::SetOutputFormat(CudaEngine* engine, va_list args) {
    const char* edge_formats = va_arg(args, const char*);
    std::string temp_name[2] = {"", ""};
    uint32_t flag = 0;

    if (edge_formats[0] == '\0')
        return RC_SUCCESS;

    for (uint32_t i = 0;; i++) {
        if (edge_formats[i] == ',') {
            flag = 1;
        } else if (edge_formats[i] == ';' || edge_formats[i] == '\0') {
            datatype_t temp_type = DATATYPE_UNKNOWN;
            for (int j = 1; j < DATATYPE_MAX; ++j) {
                if (temp_name[1] == GetDataTypeStr(j)) {
                    temp_type = j;
                }
            }
            if (temp_type == DATATYPE_UNKNOWN) {
                LOG(ERROR) << "Incorrect data type.";
                return RC_INVALID_VALUE;
            }

            engine->cuda_flags_.output_formats.emplace(temp_name[0], temp_type);
            temp_name[0] = "";
            temp_name[1] = "";
            flag = 0;

            if (edge_formats[i] == '\0') {
                break;
            }
        } else {
            temp_name[flag] = temp_name[flag] + edge_formats[i];
        }
    }

    if (temp_name[0] != "" || temp_name[1] != "") {
        LOG(ERROR) << "The sizes of edge name and data format are not equal.";
        return RC_INVALID_VALUE;
    }
    return RC_SUCCESS;
}

RetCode CudaEngine::SetOutputType(CudaEngine* engine, va_list args) {
    const char* edge_types = va_arg(args, const char*);
    std::string temp_name[2] = {"", ""};
    uint32_t flag = 0;

    if (edge_types[0] == '\0')
        return RC_SUCCESS;

    for (uint32_t i = 0;; i++) {
        if (edge_types[i] == ',') {
            flag = 1;
        } else if (edge_types[i] == ';' || edge_types[i] == '\0') {
            datatype_t temp_type = DATATYPE_UNKNOWN;
            for (int j = 1; j < DATATYPE_MAX; ++j) {
                if (temp_name[1] == GetDataTypeStr(j)) {
                    temp_type = j;
                }
            }
            if (temp_type == DATATYPE_UNKNOWN) {
                LOG(ERROR) << "Incorrect data type.";
                return RC_INVALID_VALUE;
            }

            engine->cuda_flags_.output_types.emplace(temp_name[0], temp_type);
            temp_name[0] = "";
            temp_name[1] = "";
            flag = 0;

            if (edge_types[i] == '\0') {
                break;
            }
        } else {
            temp_name[flag] = temp_name[flag] + edge_types[i];
        }
    }

    if (temp_name[0] != "" || temp_name[1] != "") {
        LOG(ERROR) << "The sizes of edge name and data type are not equal.";
        return RC_INVALID_VALUE;
    }
    return RC_SUCCESS;
}

RetCode CudaEngine::SetCompilerInputDims(CudaEngine* engine, va_list args) {
    const char* str = va_arg(args, const char*);
    int count = 0;
    bool load_dims = true;
    std::string temp_name = "";
    std::vector<uint32_t> temp_dims;

    for (uint32_t i = 0; str[i] != '\0'; i++) {
        if (str[i] == ',') { // has input edge name
            load_dims = false;
            break;
        }
    }

    for (uint32_t i = 0; str[i] != '\0'; i++) {
        if (str[i] == ';') {
            engine->cuda_flags_.input_dims.emplace(temp_name, temp_dims);
            temp_name = "";
            temp_dims.clear();
            load_dims = false;
            count = 0;
        }
        if (str[i] == '_') {
            temp_dims.push_back(count);
            count = 0;
        } else if (str[i] == ',') { // swap to load dims
            load_dims = true;
        } else if (load_dims) {
            if (str[i] < '0' || str[i] > '9') {
                LOG(ERROR) << "Invalid input dims";
                return RC_INVALID_VALUE;
            }
            count = count * 10 + (uint32_t)(str[i] - '0');
        } else {
            temp_name = temp_name + str[i];
        }
    }
    temp_dims.push_back(count);

    if (temp_dims.size() == 1 && temp_dims[0] == 0) {
        LOG(WARNING) << "Default input dims for dynamic graph are 1_3_224_224, we recommend using '--dims' to set a "
                        "suitable training shape.";
    } else {
        engine->cuda_flags_.input_dims[temp_name] = temp_dims;
    }
    return RC_SUCCESS;
}

RetCode CudaEngine::SetUseDefaultAlgorithms(CudaEngine* engine, va_list args) {
    auto flag = va_arg(args, uint32_t);
    engine->cuda_flags_.quick_select = (flag > 0);
    return RC_SUCCESS;
}

RetCode CudaEngine::SetQuantization(CudaEngine* engine, va_list args) {
    const char* json_file = va_arg(args, const char*);
    if (json_file && json_file[0] != '\0') {
        QuantParamParser parser;
        parser.Parse(json_file, &engine->cuda_flags_.quant_info);
        LOG(INFO) << "Quant tensor size: " << engine->cuda_flags_.quant_info.tensor_params.size();
    }
    return RC_SUCCESS;
}

CudaEngine::ConfHandlerFunc CudaEngine::conf_handlers_[] = {
    CudaEngine::SetOutputFormat, // CUDA_CONF_SET_OUTPUT_DATA_FORMAT
    CudaEngine::SetOutputType, // CUDA_CONF_SET_OUTPUT_TYPE
    CudaEngine::SetCompilerInputDims, // CUDA_CONF_SET_COMPILER_INPUT_SHAPE
    CudaEngine::SetUseDefaultAlgorithms, // CUDA_CONF_USE_DEFAULT_ALGORITHMS
};

RetCode CudaEngine::Configure(uint32_t option, ...) {
    if (option >= CUDA_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << CUDA_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}}} // namespace ppl::nn::cuda
