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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/gemm_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/gemm_kernel.h"
#include "ppl/nn/engines/arm/kernels/onnx/fc_kernel.h"
#include "ppl/nn/engines/arm/utils/data_trans.h"
#include "ppl/nn/oputils/onnx/reshape_gemm.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/gemm.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/arm/pmx/generated/arm_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

GemmOp::GemmOp(const ir::Node* node) : ArmOptKernel(node), fc_param_(nullptr) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeGemm(info, param_.get());
    };

    infer_type_func_ = GenericInferType;
}

GemmOp::~GemmOp() {
    if (fc_param_ != nullptr) {
        if (fc_param_->mgr != nullptr) {
            fc_param_->mgr->release_cvt_weights();
            delete fc_param_->mgr;
        }
        delete fc_param_;
    }
}

RetCode GemmOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

ppl::common::RetCode GemmOp::SelectAlgorithm(const InputOutputInfo& info, const OptKernelOptions& options) {
    auto node = GetNode();
    auto graph_data = options.graph_data;

    auto weight_data_it = graph_data->constants.find(node->GetInput(1));
    int64_t weight_len = weight_data_it->second.data.GetSize() / sizeof(float);
    void* weight_data = nullptr;
    if (weight_data_it != graph_data->constants.end()) {
        weight_data = weight_data_it->second.data.GetData();
    }

    void* bias_data = nullptr;
    int64_t bias_len = 0;
    if (node->GetInputCount() == 3) {
        auto bias_data_it = graph_data->constants.find(node->GetInput(2));
        if (bias_data_it != graph_data->constants.end()) {
            bias_len = bias_data_it->second.data.GetSize() / sizeof(float);
            bias_data = bias_data_it->second.data.GetData();
        }
    }

    if (!param_->transA && param_->transB && weight_data != nullptr) {
        if (!fc_param_) {
            fc_param_ = new FCParam;
        }
        if (!fc_param_) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }

        const ir::Shape& weight_shape = graph_data->shapes.find(node->GetInput(1))->second;
        fc_param_->param.num_output = weight_shape.dims[0];
        fc_param_->param.channels = weight_shape.dims[1];
        fc_param_->param.fuse_flag = 0;

        auto input_format = info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
        auto dtype = info.GetInput<TensorImpl>(0)->GetShape()->GetDataType();

        fc_param_->algo_info = ppl::kernel::arm_server::neon::fc_algo_selector::select_algo(
            input_format, fc_param_->param, dtype, options.device->GetISA());

        if (fc_param_->algo_info.algo_type == ppl::kernel::arm_server::neon::fc_algo::unknown) {
            LOG(INFO) << "FC select algorithm failed, use fallback kernel";
        } else {
            fc_param_->mgr = ppl::kernel::arm_server::neon::fc_algo_selector::gen_algo(
                fc_param_->param, fc_param_->algo_info, options.device->GetAllocator());

            if (bias_data != nullptr) {
                if (dtype == ppl::common::DATATYPE_FLOAT32) {
                    fc_param_->mgr->gen_cvt_weights(weight_data, bias_data, dtype);
#ifdef PPLNN_USE_ARMV8_2_FP16
                } else if (dtype == ppl::common::DATATYPE_FLOAT16) {
                    vector<__fp16> weight_data_fp16;
                    weight_data_fp16.resize(weight_len * sizeof(__fp16));
                    Fp32ToFp16((const float*)weight_data, weight_len, weight_data_fp16.data());

                    vector<__fp16> bias_data_fp16;
                    bias_data_fp16.resize(bias_len * sizeof(__fp16));
                    Fp32ToFp16((const float*)bias_data, bias_len, bias_data_fp16.data());

                    fc_param_->mgr->gen_cvt_weights(weight_data_fp16.data(), bias_data_fp16.data(), dtype);
#endif
                }
            } else {
                if (dtype == ppl::common::DATATYPE_FLOAT32) {
                    std::vector<float> zero_bias(fc_param_->param.num_output, 0.0f);
                    fc_param_->mgr->gen_cvt_weights(weight_data, zero_bias.data(), dtype);
#ifdef PPLNN_USE_ARMV8_2_FP16
                } else if (dtype == ppl::common::DATATYPE_FLOAT16) {
                    vector<__fp16> weight_data_fp16;
                    weight_data_fp16.resize(weight_len * sizeof(__fp16));
                    Fp32ToFp16((const float*)weight_data, weight_len, weight_data_fp16.data());

                    std::vector<__fp16> zero_bias(fc_param_->param.num_output, 0.0f);
                    fc_param_->mgr->gen_cvt_weights(weight_data_fp16.data(), zero_bias.data(), dtype);
#endif
                }
            }
        }
    } else {
        LOG(INFO) << "FC select algorithm failed, use fallback kernel";
    }

    return RC_SUCCESS;
}

ppl::common::RetCode GemmOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                                          vector<dataformat_t>* selected_output_formats) {
    // auto input_datatype = info.GetInput<TensorImpl>(0)->GetShape()->GetDataType();
    selected_input_formats->at(0) = selected_output_formats->at(0) = DATAFORMAT_NDARRAY;
    return RC_SUCCESS;
}

ppl::common::RetCode GemmOp::SelectDataType(const InputOutputInfo& info,
                                            std::vector<ppl::common::datatype_t>* selected_input_types,
                                            std::vector<ppl::common::datatype_t>* selected_output_types,
                                            const ppl::common::datatype_t preferred_fp_datatype) {
    GenericSelectDataType(info, selected_input_types, selected_output_types, preferred_fp_datatype);
    for (uint32_t i = 1; i < info.GetInputCount(); i++) {
        selected_input_types->at(i) = info.GetInput<TensorImpl>(i)->GetShape()->GetDataType();
    }
    return RC_SUCCESS;
}

bool GemmOp::TryFuseReLU() {
    gemm_fuse_relu_ = true;
    if (fc_param_ && fc_param_->algo_info.algo_type != ppl::kernel::arm_server::neon::fc_algo::unknown) {
        ppl::kernel::arm_server::neon::fc_param param = fc_param_->mgr->param();
        param.fuse_flag |= ppl::kernel::arm_server::neon::fc_fuse_flag::relu;
        fc_param_->mgr->set_param(param);
    }
    return true;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode GemmOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    if (!fc_param_) { 
        flatbuffers::FlatBufferBuilder builder;
        auto fb_fuse_data = ppl::nn::pmx::arm::CreateFusionData(builder, (gemm_fuse_relu_ ? 1 : 0));
        auto fp_output_info = ppl::nn::pmx::arm::CreateOutputInfoDirect(builder, &common_param_.output_types, &common_param_.output_formats);
        auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(builder, fp_output_info, ppl::nn::pmx::arm::PrivateDataType_FusionData, fb_fuse_data.Union());
        ppl::nn::pmx::arm::FinishOpDataBuffer(builder, fb_op_data);

        flatbuffers::FlatBufferBuilder op_builder;
        auto fb_param = ppl::nn::pmx::onnx::SerializeGemmParam(*param_.get(), &op_builder);
        auto fb_data = op_builder.CreateVector(builder.GetBufferPointer(), builder.GetSize());
        auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_GemmParam, fb_param.Union(), fb_data);
        ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
        return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
    }

    flatbuffers::FlatBufferBuilder fc_internal_builder;
    auto algo_info = fc_param_->algo_info;
    auto fb_algo_info = ppl::nn::pmx::arm::CreateFCAlgoInfo(fc_internal_builder,
                                                            algo_info.algo_type,
                                                            algo_info.dtype,
                                                            algo_info.isa);
    auto mgr = fc_param_->mgr;
    auto fb_exec_info = ppl::nn::pmx::arm::CreateFCExecInfo(fc_internal_builder,
                                                            fc_internal_builder.CreateVector<uint8_t>((const uint8_t*)mgr->cvt_filter(), mgr->cvt_filter_size()),
                                                            mgr->cvt_filter_size(),
                                                            fc_internal_builder.CreateVector<uint8_t>((const uint8_t*)mgr->cvt_bias(), mgr->cvt_bias_size()),
                                                            mgr->cvt_bias_size());
    LOG(WARNING) << mgr->cvt_filter_size();
    LOG(WARNING) << mgr->cvt_bias_size();
    auto fb_param_info = ppl::nn::pmx::arm::CreateFCParamInfo(fc_internal_builder, 
                                                              fc_param_->param.num_output, 
                                                              fc_param_->param.channels, 
                                                              fc_param_->param.fuse_flag);
    auto fb_fc_data = ppl::nn::pmx::arm::CreateFullConnectData(fc_internal_builder, fb_algo_info, fb_exec_info, fb_param_info);
    auto fp_output_info = ppl::nn::pmx::arm::CreateOutputInfoDirect(fc_internal_builder, &common_param_.output_types, &common_param_.output_formats);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(fc_internal_builder, fp_output_info, ppl::nn::pmx::arm::PrivateDataType_FullConnectData, fb_fc_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(fc_internal_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_param = ppl::nn::pmx::onnx::SerializeGemmParam(*param_.get(), &op_builder);
    auto fb_data = op_builder.CreateVector(fc_internal_builder.GetBufferPointer(), fc_internal_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_GemmParam, fb_param.Union(), fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode GemmOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {

    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);

    auto fb_gemm_param = fb_op_param->value_as_GemmParam();
    if (!fb_gemm_param) {
        return ppl::common::RC_INVALID_VALUE;
    }
    param_ = std::make_shared<ppl::nn::onnx::GemmParam>();
    DeserializeGemmParam(*fb_gemm_param, param_.get());

    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_op_data->output_info()->dtype(), &common_param_.output_types);
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_op_data->output_info()->dformat(), &common_param_.output_formats);
    
    auto arm_fc_data = arm_op_data->value_as_FullConnectData();
    if (arm_fc_data) {
        if (fc_param_) {
            delete fc_param_;
        }
        fc_param_ = new FCParam;
        if (!fc_param_) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }

        auto algo_info = arm_fc_data->algo_info();
        auto exec_info = arm_fc_data->exec_info();
        auto param_info = arm_fc_data->param_info();


        fc_param_->algo_info.algo_type = algo_info->algo_type();
        fc_param_->algo_info.dtype = algo_info->dtype();
        fc_param_->algo_info.isa = algo_info->isa();

        fc_param_->param.channels = param_info->channels();
        fc_param_->param.num_output = param_info->num_output();
        fc_param_->param.fuse_flag = param_info->fuse_flag();

        auto mgr = new ppl::kernel::arm_server::neon::fc_manager(fc_param_->param, allocator_);
        uint64_t cvt_filter_size = exec_info->cvt_filter_size();
        uint64_t cvt_bias_size = exec_info->cvt_bias_size();

        void *cvt_filter_ptr = allocator_->Alloc(cvt_filter_size);
        void *cvt_bias_ptr = allocator_->Alloc(cvt_bias_size);
        memcpy(cvt_filter_ptr, exec_info->cvt_filter()->data(), cvt_filter_size);
        memcpy(cvt_bias_ptr, exec_info->cvt_bias()->data(), cvt_bias_size);
        mgr->set_cvt_filter(cvt_filter_ptr, cvt_filter_size);
        mgr->set_cvt_bias(cvt_bias_ptr, cvt_bias_size);

        LOG(WARNING) << cvt_filter_size;
        LOG(WARNING) << cvt_bias_size;


        fc_param_->mgr = mgr;
        return RC_SUCCESS;
    }
    else {
        auto arm_fuse_data = arm_op_data->value_as_FusionData();
        if (arm_fuse_data) {
            gemm_fuse_relu_ = (arm_fuse_data->fuse_relu() == 1);
        }
        return RC_SUCCESS;
    }

    return RC_INVALID_VALUE;
}

#endif

KernelImpl* GemmOp::CreateKernelImpl() const {
    if (fc_param_ && fc_param_->algo_info.algo_type != ppl::kernel::arm_server::neon::fc_algo::unknown) {
        return CreateKernelImplWithParam<FCKernel>(fc_param_);
    } else {
        auto kernel = CreateKernelImplWithParam<GemmKernel>(param_.get());
        kernel->SetFuseReLU(gemm_fuse_relu_);
        return kernel;
    }
}

}}} // namespace ppl::nn::arm
