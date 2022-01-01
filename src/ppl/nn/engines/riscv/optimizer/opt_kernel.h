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

#ifndef _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIMIZER_OPT_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIMIZER_OPT_KERNEL_H_

#include "ppl/nn/runtime/opt_kernel.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/engines/riscv/riscv_device.h"
#include "ppl/nn/engines/riscv/riscv_common_param.h"
#include "ppl/nn/engines/riscv/utils/macros.h"
#include "ppl/nn/engines/riscv/riscv_engine_options.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include <functional>

namespace ppl { namespace nn { namespace utils {
struct SharedResource;
}}} // namespace ppl::nn::utils

namespace ppl { namespace nn { namespace riscv {

struct OptKernelOptions {
    utils::SharedResource* resource = nullptr;
    ir::GraphData* graph_data = nullptr;
    ir::GraphTopo* graph_topo = nullptr;
    RiscvDevice* device = nullptr;
    nn::RuntimePartitionInfo* info = nullptr;
    std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors = nullptr;
    RiscvEngineOptions* engine_options = nullptr;
};

class RiscvOptKernel : public OptKernel {
public:
    RiscvOptKernel(const ir::Node* node);
    virtual ~RiscvOptKernel() {}

    virtual ppl::common::RetCode Init(const OptKernelOptions&) = 0;

    ppl::common::RetCode InferDims(InputOutputInfo* info) const {
        if (infer_dims_func_) {
            return infer_dims_func_(info);
        }
        return ppl::common::RC_NOT_FOUND;
    }
    void InferType(InputOutputInfo* info) const {
        if (infer_type_func_) {
            infer_type_func_(info);
        }
    }

    virtual ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                              std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                              std::vector<ppl::common::dataformat_t>* selected_output_formats) {
        auto input_format = selected_input_formats->at(0);
        for (int64_t i = 0; i < selected_output_formats->size(); i += 1) {
            selected_output_formats->at(i) = input_format;
        }
        return ppl::common::RC_SUCCESS;
    }

    virtual ppl::common::RetCode SelectDataType(const InputOutputInfo& info,
                                                std::vector<ppl::common::datatype_t>* selected_input_data_types,
                                                std::vector<ppl::common::datatype_t>* selected_output_data_types) {
        auto input_type = selected_input_data_types->at(0);
        for (int64_t i = 0; i < selected_output_data_types->size(); i += 1) {
            selected_output_data_types->at(i) = input_type;
        }
        return ppl::common::RC_SUCCESS;
    }

    virtual ppl::common::RetCode SelectAlgorithm(const InputOutputInfo&, const OptKernelOptions&) {
        return ppl::common::RC_SUCCESS;
    }

    void SetOutputDataFormat(uint32_t idx, ppl::common::dataformat_t format) {
        common_param_.output_formats[idx] = format;
    }

    void SetOutputDataType(uint32_t idx, ppl::common::datatype_t type) {
        common_param_.output_types[idx] = type;
    }

    virtual ppl::common::RetCode OmitConstantsData(std::map<edgeid_t, int64_t>* constants_data_refcount) {
        return ppl::common::RC_SUCCESS;
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
                info->GetOutput<TensorImpl>(i)->GetShape()->SetDataFormat(common_param_.output_formats[i]);
                info->GetOutput<TensorImpl>(i)->GetShape()->SetDataType(common_param_.output_types[i]);
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
                info->GetOutput<TensorImpl>(i)->GetShape()->SetDataFormat(common_param_.output_formats[i]);
                info->GetOutput<TensorImpl>(i)->GetShape()->SetDataType(common_param_.output_types[i]);
            }
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

    static void GenericInferType(InputOutputInfo* info) {
        auto& in_shape0 = *info->GetInput<TensorImpl>(0)->GetShape();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = info->GetOutput<TensorImpl>(i)->GetShape();
            out_shape->SetDataType(in_shape0.GetDataType());
        }
    }

protected:
    std::function<void(InputOutputInfo*)> infer_type_func_;
    std::function<ppl::common::RetCode(InputOutputInfo*)> infer_dims_func_;
    RiscvCommonParam common_param_;
};

}}} // namespace ppl::nn::riscv

#endif
