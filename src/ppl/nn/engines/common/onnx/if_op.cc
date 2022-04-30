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

namespace ppl { namespace nn { namespace onnx {

IfOp::~IfOp() {
    // make sure that engines are released at last.
    then_topo_.reset();
    then_info_.Clear();
    else_topo_.reset();
    else_info_.Clear();

    then_engines_.clear();
    else_engines_.clear();
}

static RetCode InitNewSharedResource(const utils::SharedResource& resource, utils::SharedResource* new_resource,
                                     vector<unique_ptr<EngineImpl>>* engines) {
    for (auto x = resource.engines.begin(); x != resource.engines.end(); ++x) {
        auto e = (*x)->Create();
        if (!e) {
            LOG(ERROR) << "create instance of engine[" << (*x)->GetName() << "] failed.";
            return RC_OTHER_ERROR;
        }
        engines->emplace_back(unique_ptr<EngineImpl>(e));
        new_resource->engines.push_back(e);
    }
    new_resource->graph_partitioner = resource.graph_partitioner;
    return RC_SUCCESS;
}

RetCode IfOp::Init(const utils::SharedResource& resource, ppl::nn::onnx::IfParam* if_param) {
    extra_inputs_of_then_graph_ = if_param->then_extra_input_indices_in_host_node;
    extra_inputs_of_else_graph_ = if_param->else_extra_input_indices_in_host_node;

    utils::SharedResource new_then_resource;
    auto status = InitNewSharedResource(resource, &new_then_resource, &then_engines_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init SharedResource failed: " << GetRetCodeStr(status);
        return status;
    }

    status = utils::ProcessGraph(new_then_resource, &if_param->then_branch, &then_info_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ProcessGraph then_branch of kernel [" << node_->GetName()
                   << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    status = then_aux_info_.Init(if_param->then_branch.topo.get(), {});
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo for then_branch of kernel[" << node_->GetName()
                   << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    status = then_init_info_.Init(if_param->then_branch.topo.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeInitInfo for then_branch of kernel[" << node_->GetName()
                   << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    utils::SharedResource new_else_resource;
    status = InitNewSharedResource(resource, &new_else_resource, &else_engines_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init SharedResource failed: " << GetRetCodeStr(status);
        return status;
    }

    status = utils::ProcessGraph(new_else_resource, &if_param->else_branch, &else_info_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ProcessGraph else_branch failed: " << GetRetCodeStr(status);
        return status;
    }

    status = else_aux_info_.Init(if_param->else_branch.topo.get(), {});
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo for else_branch of kernel[" << node_->GetName()
                   << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    status = else_init_info_.Init(if_param->else_branch.topo.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeInitInfo for else_branch of kernel[" << node_->GetName()
                   << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    then_topo_ = if_param->then_branch.topo;
    else_topo_ = if_param->else_branch.topo;

    return RC_SUCCESS;
}

KernelImpl* IfOp::CreateKernelImpl() const {
    auto kernel = unique_ptr<IfKernel>(new IfKernel(node_));
    auto status = kernel->SetExecutionInfo(then_topo_, &then_info_, &then_aux_info_, &then_init_info_,
                                           &extra_inputs_of_then_graph_, else_topo_, &else_info_, &else_aux_info_,
                                           &else_init_info_, &extra_inputs_of_else_graph_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "SetExecutionInfo of kernel[" << kernel->GetName() << "] failed:" << GetRetCodeStr(status);
        return nullptr;
    }

    return kernel.release();
}

}}} // namespace ppl::nn::onnx
