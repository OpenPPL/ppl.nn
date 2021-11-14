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

#include "ppl/nn/engines/common/onnx/loop_op.h"
#include "ppl/nn/engines/common/onnx/loop_kernel.h"
#include "ppl/nn/params/onnx/loop_param.h"
#include "ppl/nn/optimizers/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace common {

RetCode LoopOp::Init(utils::SharedResource* resource, LoopParam* loop_param, LoopConcatOutputFunc concat_output_func) {
    auto status = utils::ProcessGraph(resource, &loop_param->graph, &graph_info_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ProcessGraph failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GenerateRuntimeAuxInfo(loop_param->graph.topo.get(), &aux_info_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo failed: " << GetRetCodeStr(status);
        return status;
    }

    graph_ = loop_param->graph;
    resource_ = resource;
    concat_output_func_ = concat_output_func;

    return RC_SUCCESS;
}

KernelImpl* LoopOp::CreateKernelImpl() const {
    auto kernel = unique_ptr<LoopKernel>(new LoopKernel(node_));
    auto status = kernel->SetExecutionInfo(graph_.topo, &graph_info_, &aux_info_, resource_, concat_output_func_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "SetExecutionInfo of kernel[" << kernel->GetName() << "] failed: " << GetRetCodeStr(status);
        return nullptr;
    }

    return kernel.release();
}

}}} // namespace ppl::nn::common
