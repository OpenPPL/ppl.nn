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

namespace ppl { namespace nn { namespace onnx {

LoopOp::~LoopOp() {
    // make sure that engines are released at last.
    topo_.reset();
    graph_info_.Clear();
    engines_.clear();
}

RetCode LoopOp::Init(const utils::SharedResource& resource, ppl::nn::onnx::LoopParam* loop_param,
                     LoopConcatOutputFunc concat_output_func) {
    utils::SharedResource new_resource;
    for (auto x = resource.engines.begin(); x != resource.engines.end(); ++x) {
        auto e = (*x)->Create();
        if (!e) {
            LOG(ERROR) << "create instance of engine[" << (*x)->GetName() << "] failed.";
            return RC_OTHER_ERROR;
        }
        engines_.emplace_back(unique_ptr<EngineImpl>(e));
        new_resource.engines.push_back(e);
    }
    new_resource.graph_partitioner = resource.graph_partitioner;

    auto status = utils::ProcessGraph(new_resource, &loop_param->graph, &graph_info_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ProcessGraph failed: " << GetRetCodeStr(status);
        return status;
    }

    status = aux_info_.Init(loop_param->graph.topo.get(), {});
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo failed: " << GetRetCodeStr(status);
        return status;
    }

    status = init_info_.Init(loop_param->graph.topo.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeInitInfo failed: " << GetRetCodeStr(status);
        return status;
    }

    topo_ = loop_param->graph.topo;
    concat_output_func_ = concat_output_func;

    return RC_SUCCESS;
}

KernelImpl* LoopOp::CreateKernelImpl() const {
    auto kernel = unique_ptr<LoopKernel>(new LoopKernel(node_));
    auto status = kernel->SetExecutionInfo(topo_, &graph_info_, &aux_info_, &init_info_, concat_output_func_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "SetExecutionInfo of kernel[" << kernel->GetName() << "] failed: " << GetRetCodeStr(status);
        return nullptr;
    }

    return kernel.release();
}

}}} // namespace ppl::nn::onnx
