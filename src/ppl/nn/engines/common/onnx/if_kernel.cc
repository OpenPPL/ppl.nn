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
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace common {

template <typename T>
void EmptyDeleter(T*) {}

RetCode IfKernel::SetExecutionInfo(const shared_ptr<ir::GraphTopo>& then_topo, const RuntimeGraphInfo* then_info,
                                   const RuntimeAuxInfo* then_aux_info,
                                   const vector<uint32_t>* extra_inputs_of_then_branch,
                                   const shared_ptr<ir::GraphTopo>& else_topo, const RuntimeGraphInfo* else_info,
                                   const RuntimeAuxInfo* else_aux_info,
                                   const vector<uint32_t>* extra_inputs_of_else_branch,
                                   utils::SharedResource* resource) {
    auto status = then_branch_.Init(then_topo,
                                    shared_ptr<const RuntimeGraphInfo>(then_info, EmptyDeleter<const RuntimeGraphInfo>),
                                    shared_ptr<const RuntimeAuxInfo>(then_aux_info, EmptyDeleter<const RuntimeAuxInfo>),
                                    shared_ptr<utils::SharedResource>(resource, EmptyDeleter<utils::SharedResource>));
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init if kernel[" << GetName() << "] then_branch failed: " << GetRetCodeStr(status);
        return status;
    }

    status = else_branch_.Init(else_topo,
                               shared_ptr<const RuntimeGraphInfo>(else_info, EmptyDeleter<const RuntimeGraphInfo>),
                               shared_ptr<const RuntimeAuxInfo>(else_aux_info, EmptyDeleter<const RuntimeAuxInfo>),
                               shared_ptr<utils::SharedResource>(resource, EmptyDeleter<utils::SharedResource>));
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init if kernel[" << GetName() << "] else_branch failed: " << GetRetCodeStr(status);
        return status;
    }

    extra_inputs_of_then_branch_ = extra_inputs_of_then_branch;
    extra_inputs_of_else_branch_ = extra_inputs_of_else_branch;

    return RC_SUCCESS;
}

static RetCode SetExtraInputs(const KernelExecContext& ctx, const vector<uint32_t>* extra_inputs, RuntimeImpl* subgraph,
                              Device* tmp_cpu_device) {
    for (uint32_t i = 0; i < subgraph->GetExtraInputCount(); ++i) {
        auto dst = subgraph->GetExtraInputTensorImpl(i);
        auto src = ctx.GetExtraInput<TensorImpl>(extra_inputs->at(i));
        if (!src) {
            LOG(ERROR) << "cannot find extra input[" << dst->GetName() << "] from node inputs.";
            return RC_NOT_FOUND;
        }

        dst->GetShape() = src->GetShape();

        if (dst->GetDevice() == src->GetDevice()) {
            dst->SetBuffer(src->GetBufferDesc());
        } else {
            auto status = utils::CopyTensorBuffer(*src, dst, tmp_cpu_device);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy tensor from [" << src->GetName() << "] to [" << dst->GetName()
                           << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode SetOutputs(RuntimeImpl* subgraph, KernelExecContext* ctx, Device* tmp_cpu_device) {
    for (uint32_t i = 0; i < subgraph->GetOutputCount(); ++i) {
        auto src = subgraph->GetOutputTensorImpl(i);
        auto dst = ctx->GetOutput<TensorImpl>(i);

        dst->GetShape() = src->GetShape();

        // outputs are already synchronized by subgraph->Sync()
        auto status = utils::CopyTensorBuffer(*src, dst, tmp_cpu_device);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy from tensor[" << src->GetName() << "] to tensor[" << dst->GetName()
                       << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

static RetCode SyncAllInputs(KernelExecContext* ctx) {
    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        auto barrier = ctx->GetInputBarrier(i);
        if (barrier) {
            auto status = barrier->Sync();
            if (status != RC_SUCCESS) {
                auto e = ctx->GetInput<EdgeObject>(i);
                LOG(ERROR) << "sync EdgeObject[" << e->GetEdge()->GetName() << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }
    }

    for (uint32_t i = 0; i < ctx->GetExtraInputCount(); ++i) {
        auto barrier = ctx->GetExtraInputBarrier(i);
        if (barrier) {
            auto status = barrier->Sync();
            if (status != RC_SUCCESS) {
                auto e = ctx->GetExtraInput<EdgeObject>(i);
                LOG(ERROR) << "sync EdgeObject[" << e->GetEdge()->GetName() << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }
    }

    return RC_SUCCESS;
}

RetCode IfKernel::DoExecute(KernelExecContext* ctx) {
    auto status = SyncAllInputs(ctx);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "sync inputs of if kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    bool cond;
    status = ctx->GetInput<TensorImpl>(0)->CopyToHost(&cond);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "If op[" << GetName() << "] get condition value failed: " << GetRetCodeStr(status);
        return status;
    }

    RuntimeImpl* subgraph = nullptr;
    const vector<uint32_t>* extra_inputs = nullptr;
    if (cond) {
        subgraph = &then_branch_;
        extra_inputs = extra_inputs_of_then_branch_;
    } else {
        subgraph = &else_branch_;
        extra_inputs = extra_inputs_of_else_branch_;
    }

    utils::GenericCpuDevice tmp_cpu_device;
    status = SetExtraInputs(*ctx, extra_inputs, subgraph, &tmp_cpu_device);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "SetExtraInputs for If kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    status = subgraph->Run();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "if kernel[" << GetName() << "] Run() failed: " << GetRetCodeStr(status);
        return status;
    }

    status = subgraph->Sync();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "sync if kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    return SetOutputs(subgraph, ctx, &tmp_cpu_device);
}

}}} // namespace ppl::nn::common
