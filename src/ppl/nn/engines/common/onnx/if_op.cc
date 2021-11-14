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

#include "ppl/nn/engines/common/onnx/if_kernel.h"
#include "ppl/nn/engines/common/onnx/if_op.h"
#include "ppl/nn/params/onnx/if_param.h"
#include "ppl/nn/optimizers/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace common {

RetCode IfOp::Init(utils::SharedResource* resource, IfParam* if_param) {
    extra_inputs_of_then_graph_ = if_param->then_extra_input_indices_in_parent_node;
    extra_inputs_of_else_graph_ = if_param->else_extra_input_indices_in_parent_node;

    auto status = utils::ProcessGraph(resource, &if_param->then_branch, &then_info_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ProcessGraph then_branch of kernel [" << node_->GetName()
                   << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GenerateRuntimeAuxInfo(if_param->then_branch.topo.get(), &then_aux_info_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo for then_branch of kernel[" << node_->GetName()
                   << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    status = utils::ProcessGraph(resource, &if_param->else_branch, &else_info_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ProcessGraph else_branch failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GenerateRuntimeAuxInfo(if_param->else_branch.topo.get(), &else_aux_info_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo for else_branch of kernel[" << node_->GetName()
                   << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    resource_ = resource;
    then_graph_ = if_param->then_branch;
    else_graph_ = if_param->else_branch;

    return RC_SUCCESS;
}

KernelImpl* IfOp::CreateKernelImpl() const {
    auto kernel = unique_ptr<IfKernel>(new IfKernel(node_));
    auto status = kernel->SetExecutionInfo(then_graph_.topo, &then_info_, &then_aux_info_, &extra_inputs_of_then_graph_,
                                           else_graph_.topo, &else_info_, &else_aux_info_, &extra_inputs_of_else_graph_,
                                           resource_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "SetExecutionInfo of kernel[" << kernel->GetName() << "] failed:" << GetRetCodeStr(status);
        return nullptr;
    }

    return kernel.release();
}

}}} // namespace ppl::nn::common
