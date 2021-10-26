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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPT_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPT_KERNEL_H_

#include "ppl/nn/runtime/opt_kernel.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/engines/x86/x86_device.h"
#include "ppl/nn/engines/x86/x86_common_param.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include <functional>

namespace ppl { namespace nn { namespace utils {
struct SharedResource;
}}} // namespace ppl::nn::utils

namespace ppl { namespace nn { namespace x86 {

struct OptKernelOptions {
    utils::SharedResource* resource = nullptr;
    ir::GraphData* graph_data = nullptr;
    ir::GraphTopo* graph_topo = nullptr;
    X86Device* device = nullptr;
    RuntimePartitionInfo *info = nullptr;
    std::map<edgeid_t, std::unique_ptr<TensorImpl>> *tensors = nullptr;
};

class X86OptKernel : public OptKernel {
public:
    X86OptKernel(const ir::Node* node);
    virtual ~X86OptKernel() {}

    virtual ppl::common::RetCode Init(const OptKernelOptions&) = 0;

    void InferType(InputOutputInfo* info) const {
        if (infer_type_func_) {
            infer_type_func_(info);
        }
    }

    ppl::common::RetCode InferDims(InputOutputInfo* info) const {
        if (infer_dims_func_) {
            return infer_dims_func_(info);
        }
        return ppl::common::RC_NOT_FOUND;
    }

    virtual ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                              std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                              std::vector<ppl::common::dataformat_t>* selected_output_formats) {
        return ppl::common::RC_SUCCESS;
    }

    virtual ppl::common::RetCode SelectAlgorithm(const InputOutputInfo&, const OptKernelOptions&) {
        return ppl::common::RC_SUCCESS;
    }

    void SetOutputDataFormat(uint32_t idx, ppl::common::dataformat_t format) {
        common_param_.output_formats[idx] = format;
    }

protected:
    template <typename T>
    ppl::common::RetCode GenericLoadParam(const OptKernelOptions& options, std::shared_ptr<T>* param) const {
        auto node = GetNode();
        auto graph_data = options.graph_data;

        auto param_ref = graph_data->attrs.find(node->GetId());
        if (param_ref == graph_data->attrs.end()) {
            return ppl::common::RC_NOT_FOUND;
        }

        *param = std::static_pointer_cast<T>(param_ref->second);
        return ppl::common::RC_SUCCESS;
    }

    template <typename KernelType, typename ParamType>
    KernelType* CreateKernelImplWithParam(const ParamType* param) const {
        auto kernel = new KernelType(GetNode());
        kernel->SetParam(param);
        kernel->SetCommonParam(&common_param_);
        kernel->SetReshapeFunc([this](InputOutputInfo* info) -> ppl::common::RetCode {
            infer_type_func_(info);
            for (uint32_t i = 0; i < info->GetOutputCount(); i++) {
                info->GetOutput<TensorImpl>(i)->GetShape().SetDataFormat(common_param_.output_formats[i]);
            }
            return infer_dims_func_(info);
        });
        return kernel;
    }

    template <typename KernelType>
    KernelType* CreateKernelImplWithoutParam() const {
        auto kernel = new KernelType(GetNode());
        kernel->SetCommonParam(&common_param_);
        kernel->SetReshapeFunc([this](InputOutputInfo* info) -> ppl::common::RetCode {
            infer_type_func_(info);
            for (uint32_t i = 0; i < info->GetOutputCount(); i++) {
                info->GetOutput<TensorImpl>(i)->GetShape().SetDataFormat(common_param_.output_formats[i]);
            }
            return infer_dims_func_(info);
        });
        return kernel;
    }

    static ppl::common::RetCode GenericInferDims(InputOutputInfo* info) {
        auto& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto& out_shape = info->GetOutput<TensorImpl>(i)->GetShape();
            if (in_shape0.IsScalar()) {
                out_shape.ReshapeAsScalar();
            } else {
                out_shape.Reshape(in_shape0.GetDims(), in_shape0.GetDimCount());
            }
        }
        return ppl::common::RC_SUCCESS;
    }

    static void GenericInferType(InputOutputInfo* info) {
        auto& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = &info->GetOutput<TensorImpl>(i)->GetShape();
            out_shape->SetDataType(in_shape0.GetDataType());
        }
    }

protected:
    std::function<void(InputOutputInfo*)> infer_type_func_;
    std::function<ppl::common::RetCode(InputOutputInfo*)> infer_dims_func_;
    X86CommonParam common_param_;
};

}}} // namespace ppl::nn::x86

#endif
