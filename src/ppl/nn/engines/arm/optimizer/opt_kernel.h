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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_KERNEL_H_

#include "ppl/nn/runtime/opt_kernel.h"

#include <functional>

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "ppl/nn/engines/arm/arm_device.h"
#include "ppl/nn/engines/arm/arm_common_param.h"
#include "ppl/nn/engines/arm/utils/macros.h"
#include "ppl/nn/engines/arm/engine_options.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/generated/onnx_op_generated.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/arm/pmx/generated/arm_op_params_generated.h"
#endif

namespace ppl { namespace nn { namespace utils {
struct SharedResource;
}}} // namespace ppl::nn::utils

namespace ppl { namespace nn { namespace arm {

struct OptKernelOptions final {
    const utils::SharedResource* resource = nullptr;
    ir::GraphData* graph_data = nullptr;
    ir::GraphTopo* graph_topo = nullptr;
    ArmDevice* device = nullptr;
    RuntimePartitionInfo* info = nullptr;
    std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors = nullptr;
    EngineOptions* engine_options = nullptr;
};

class ArmOptKernel : public OptKernel {
public:
    ArmOptKernel(const ir::Node* node);
    virtual ~ArmOptKernel() {}

    virtual ppl::common::RetCode Init(const OptKernelOptions&) = 0;

    ppl::common::RetCode InferDims(InputOutputInfo* info) const {
        if (infer_dims_func_) {
            return infer_dims_func_(info);
        }
        return ppl::common::RC_NOT_FOUND;
    }
    void InferTypes(InputOutputInfo* info) const {
        if (infer_type_func_) {
            infer_type_func_(info);
        }
    }

    virtual ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                              std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                              std::vector<ppl::common::dataformat_t>* selected_output_formats) {
        return ppl::common::RC_SUCCESS;
    }

    virtual ppl::common::RetCode SelectDataType(const InputOutputInfo& info,
                                                std::vector<ppl::common::datatype_t>* selected_input_types,
                                                std::vector<ppl::common::datatype_t>* selected_output_types,
                                                const ppl::common::datatype_t preferred_fp_datatype) {
        GenericSelectDataType(info, selected_input_types, selected_output_types, preferred_fp_datatype);
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

    virtual void SetAllocator(ppl::common::Allocator*) { }

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
            for (uint32_t i = 0; i < info->GetOutputCount(); i++) {
                info->GetOutput<TensorImpl>(i)->GetShape()->SetDataType(common_param_.output_types[i]);
            }
            infer_type_func_(info);
            for (uint32_t i = 0; i < info->GetOutputCount(); i++) {
                info->GetOutput<TensorImpl>(i)->GetShape()->SetDataFormat(common_param_.output_formats[i]);
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
            for (uint32_t i = 0; i < info->GetOutputCount(); i++) {
                info->GetOutput<TensorImpl>(i)->GetShape()->SetDataType(common_param_.output_types[i]);
            }
            infer_type_func_(info);
            for (uint32_t i = 0; i < info->GetOutputCount(); i++) {
                info->GetOutput<TensorImpl>(i)->GetShape()->SetDataFormat(common_param_.output_formats[i]);
            }
            auto status = infer_dims_func_(info);
            return status;
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

    static void GenericSelectDataType(const InputOutputInfo& info,
                                      std::vector<ppl::common::datatype_t>* selected_input_types,
                                      std::vector<ppl::common::datatype_t>* selected_output_types,
                                      const ppl::common::datatype_t preferred_fp_datatype) {
        for (uint32_t i = 0; i < info.GetInputCount(); i++) {
            const auto input_datatype = info.GetInput<ppl::nn::TensorImpl>(i)->GetShape()->GetDataType();
            if (input_datatype == ppl::common::DATATYPE_FLOAT16 || input_datatype == ppl::common::DATATYPE_FLOAT32) {
                selected_input_types->at(i) = preferred_fp_datatype;
            } else {
                selected_input_types->at(i) = input_datatype;
            }
        }
        for (uint32_t i = 0; i < info.GetOutputCount(); i++) {
            selected_output_types->at(i) = selected_input_types->at(0);
        }
    }

    static void PassiveInferType(InputOutputInfo* info) {
        auto& in_shape0 = *info->GetInput<TensorImpl>(0)->GetShape();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = info->GetOutput<TensorImpl>(i)->GetShape();
            if (out_shape->GetDataType() == ppl::common::DATATYPE_UNKNOWN) {
                out_shape->SetDataType(in_shape0.GetDataType());
            }
        }
    }

#ifdef PPLNN_ENABLE_PMX_MODEL
    virtual ppl::nn::pmx::onnx::OpParamType GetOptParamType(void) const {
        return ppl::nn::pmx::onnx::OpParamType_NONE;
    }

    virtual flatbuffers::Offset<void> SerializeOptParam(flatbuffers::FlatBufferBuilder*) const {
        return flatbuffers::Offset<void>();
    }

    virtual ppl::common::RetCode DeserializeOptParam(const ppl::nn::pmx::onnx::OpParam*) {
        return ppl::common::RC_SUCCESS;
    }

    virtual ppl::nn::pmx::arm::PrivateDataType GetPrivateDataType(void) const {
        return ppl::nn::pmx::arm::PrivateDataType_NONE;
    }

    virtual flatbuffers::Offset<void> SerializePrivateData(flatbuffers::FlatBufferBuilder*) const {
        return flatbuffers::Offset<void>();
    }
    
    virtual ppl::common::RetCode DeserializePrivateData(const ppl::nn::pmx::arm::OpData*) {
        return ppl::common::RC_SUCCESS;
    }


    ppl::common::RetCode SerializeData(const pmx::SerializationContext&, utils::DataStream* ds) const override {
        flatbuffers::FlatBufferBuilder arm_data_builder;
        auto fp_output_info = ppl::nn::pmx::arm::CreateOutputInfoDirect(arm_data_builder, &common_param_.output_types, &common_param_.output_formats);
        auto fb_arm_op_data = ppl::nn::pmx::arm::CreateOpData(arm_data_builder, fp_output_info, GetPrivateDataType(), SerializePrivateData(&arm_data_builder));
        ppl::nn::pmx::arm::FinishOpDataBuffer(arm_data_builder, fb_arm_op_data);

        flatbuffers::FlatBufferBuilder op_builder;
        auto fb_data = op_builder.CreateVector(arm_data_builder.GetBufferPointer(), arm_data_builder.GetSize());
        auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, GetOptParamType(), SerializeOptParam(&op_builder), fb_data);
        ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);

        return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
    }

    ppl::common::RetCode DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t) override {
        auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);
        auto status = DeserializeOptParam(fb_op_param);
        if (status != ppl::common::RC_SUCCESS) {
            return status;
        }

        auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
        ppl::nn::pmx::utils::Fbvec2Stdvec(arm_op_data->output_info()->dtype(), &common_param_.output_types);
        ppl::nn::pmx::utils::Fbvec2Stdvec(arm_op_data->output_info()->dformat(), &common_param_.output_formats);
        return DeserializePrivateData(arm_op_data);
    }
#endif

protected:
    std::function<void(InputOutputInfo*)> infer_type_func_;
    std::function<ppl::common::RetCode(InputOutputInfo*)> infer_dims_func_;
public:
    ArmCommonParam common_param_;
};

}}} // namespace ppl::nn::arm

#endif
