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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPT_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPT_KERNEL_H_

#include "llm_cuda_device.h"

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "ppl/nn/runtime/opt_kernel.h"
#include "ppl/nn/utils/shared_resource.h"
#include "ppl/nn/engines/llm_cuda/engine_options.h"

namespace ppl { namespace nn { namespace llm { namespace cuda {

struct OptKernelOptions final {
    const utils::SharedResource* resource = nullptr;
    ir::Graph* graph = nullptr;
    LlmCudaDevice* device = nullptr;
    RuntimePartitionInfo* partition_info = nullptr;
    const EngineOptions* engine_options = nullptr;
};

class LlmCudaOptKernel : public OptKernel {
public:
    LlmCudaOptKernel(const ir::Node* node) : OptKernel(node) {}
    virtual ppl::common::RetCode Init(const OptKernelOptions&);

protected:
    virtual ppl::common::RetCode DoInit(const OptKernelOptions&) = 0;

    template <typename T>
    ppl::common::RetCode GenericLoadParam(const OptKernelOptions& options, std::shared_ptr<T>* param) const {
        auto node = GetNode();
        auto graph_data = options.graph->data.get();

        auto param_ref = graph_data->attrs.find(node->GetId());
        if (param_ref == graph_data->attrs.end()) {
            return ppl::common::RC_NOT_FOUND;
        }

        *param = std::static_pointer_cast<T>(param_ref->second);
        return ppl::common::RC_SUCCESS;
    }

    template <typename KernelType, typename ParamType>
    KernelType* CreateKernelImplWithParam(const ParamType param) const {
        auto kernel = new KernelType(GetNode());
        auto status = kernel->Init();
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "init kernel for ["
                        << this->GetNode()->GetName()
                        << "] failed: " << ppl::common::GetRetCodeStr(status);
            delete kernel;
            return nullptr;
        }
        kernel->SetParam(param);
        kernel->SetReshapeFunc([this](InputOutputInfo* info) -> ppl::common::RetCode {
            auto rc = infer_type_and_format_func_(info);
            if (ppl::common::RC_SUCCESS != rc)
                return rc;
            return infer_dims_func_(info);
        });
        return kernel;
    }

    template <typename KernelType>
    KernelType* CreateKernelImplWithoutParam() const {
        auto kernel = new KernelType(GetNode());
        auto status = kernel->Init();
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "init kernel for ["
                        << this->GetNode()->GetName()
                        << "] failed: " << ppl::common::GetRetCodeStr(status);
            delete kernel;
            return nullptr;
        }
        kernel->SetReshapeFunc([this](InputOutputInfo* info) -> ppl::common::RetCode {
            auto rc = infer_type_and_format_func_(info);
            if (ppl::common::RC_SUCCESS != rc)
                return rc;
            return infer_dims_func_(info);
        });
        return kernel;
    }

    static ppl::common::RetCode GenericInferDims(InputOutputInfo* info) {
        auto& in_shape0 = *info->GetInput<TensorImpl>(0)->GetShape();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto& out_shape = *info->GetOutput<TensorImpl>(i)->GetShape();
            if (in_shape0.IsScalar()) {
                out_shape.ReshapeAsScalar();
            } else {
                out_shape.Reshape(in_shape0.GetDims(), in_shape0.GetDimCount());
            }
        }
        return ppl::common::RC_SUCCESS;
    }

    static ppl::common::RetCode GenericInferTypeAndFormat(InputOutputInfo* info) {
        auto& in_shape0 = *info->GetInput<TensorImpl>(0)->GetShape();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = info->GetOutput<TensorImpl>(i)->GetShape();
            out_shape->SetDataType(in_shape0.GetDataType());
            out_shape->SetDataFormat(in_shape0.GetDataFormat());
        }
        return ppl::common::RC_SUCCESS;
    }

    std::function<ppl::common::RetCode(InputOutputInfo*)> infer_dims_func_;
    std::function<ppl::common::RetCode(InputOutputInfo*)> infer_type_and_format_func_;
};

}}}} // namespace ppl::nn::llm::cuda

#endif
