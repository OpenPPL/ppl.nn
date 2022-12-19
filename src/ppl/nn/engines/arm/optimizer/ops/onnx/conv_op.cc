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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/conv_op.h"

#include <cstring>

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/arm/kernels/onnx/conv2d_kernel.h"
#include "ppl/nn/engines/arm/optimizer/algos/conv2d_algo_selector.h"
#include "ppl/nn/engines/arm/utils/data_trans.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/oputils/onnx/reshape_conv.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/conv.h"
#endif

using namespace std;
using namespace ppl::common;
using namespace ppl::kernel::arm_server::neon;

namespace ppl { namespace nn { namespace arm {

ConvOp::ConvOp(const ir::Node* node) : ArmOptKernel(node), conv2d_param_(nullptr) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (info->GetInput<TensorImpl>(1)->GetShape()->GetDimCount() == 1) {  // cvt filters
            if (conv2d_param_ == nullptr) {
                LOG(ERROR) << "Cannot infer dimensions for " << GetNode()->GetName() << " outputs: ";
                LOG(ERROR) << "    missing filters dimensions";
                return RC_INVALID_VALUE;
            }
            auto x = info->GetInput<TensorImpl>(0)->GetShape();
            auto y = info->GetOutput<TensorImpl>(0)->GetShape();
            auto num_output = conv2d_param_->param.num_output;

            y->SetDimCount(x->GetDimCount());
            y->SetDim(0, x->GetDim(0));
            y->SetDim(1, num_output);

            const int32_t kernel_dims = (int32_t)x->GetDimCount() - 2;
            for (int32_t i = 0; i < kernel_dims; ++i) {
                const int32_t j = i + 2;
                const int32_t kernel_shape_eff = (param_->kernel_shape[i] - 1) * param_->dilations[i] + 1;
                const int64_t out_dim =
                    (x->GetDim(j) + param_->pads[i] + param_->pads[i + kernel_dims] - kernel_shape_eff) / param_->strides[i] + 1;
                if (out_dim <= 0) {
                    LOG(DEBUG) << "ERROR: output dim[" << out_dim << "] < 0.";
                    return RC_INVALID_VALUE;
                }
                y->SetDim(j, out_dim);
            }
            y->CalcPadding();

            return RC_SUCCESS;
        } else {
            return onnx::ReshapeConv(info, param_.get());
        }
    };

    infer_type_func_ = GenericInferType;
}

ConvOp::~ConvOp() {
    if (conv2d_param_ != nullptr) {
        if (conv2d_param_->mgr != nullptr) {
            conv2d_param_->mgr->release_cvt_weights();
            delete conv2d_param_->mgr;
        }
        if (conv2d_param_->fallback_mgr != nullptr) {
            conv2d_param_->fallback_mgr->release_cvt_weights();
            delete conv2d_param_->fallback_mgr;
        }
        delete conv2d_param_;
    }
}

RetCode ConvOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

ppl::common::RetCode ConvOp::SelectAlgorithm(const InputOutputInfo& info, const OptKernelOptions& options) {
    auto node = GetNode();
    auto graph_data = options.graph_data;

    auto weight_data_it = graph_data->constants.find(node->GetInput(1));
    if (weight_data_it == graph_data->constants.end()) {
        LOG(INFO) << "ConvOp constant weight not found, will use conv runtime.";
        return ppl::common::RC_SUCCESS;
    }

    float* weight_data = (float*)weight_data_it->second.data.GetData();
    int64_t weight_len = weight_data_it->second.data.GetSize() / sizeof(float);

    float* bias_data = nullptr;
    int64_t bias_len = 0;
    if (node->GetInputCount() == 3) {
        auto bias_data_it = graph_data->constants.find(node->GetInput(2));
        if (bias_data_it == graph_data->constants.end()) {
            LOG(INFO) << "ConvOp constant weight not found, will use conv runtime.";
            return ppl::common::RC_SUCCESS;
        }
        bias_data = (float*)bias_data_it->second.data.GetData();
        bias_len = bias_data_it->second.data.GetSize() / sizeof(float);
    }

    const ir::Shape& weight_shape = graph_data->shapes.find(node->GetInput(1))->second;
    const int64_t kernel_dims = weight_shape.dims.size() - 2;

    // Check Param
    const ppl::nn::onnx::ConvParam& conv_param = *param_;
    for (int64_t i = 0; i < kernel_dims; ++i) {
        if (conv_param.pads[i] != conv_param.pads[i + kernel_dims]) {
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    if (kernel_dims == 2) {
        if (!conv2d_param_) {
            conv2d_param_ = new Convolution2DParam;
        }
        if (!conv2d_param_) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }

        const int32_t num_output = weight_shape.dims[0];
        const int32_t channels = weight_shape.dims[1] * param_->group;

        ppl::kernel::arm_server::neon::conv2d_param& conv2d_kernel_param = conv2d_param_->param;
        conv2d_kernel_param.kernel_h = conv_param.kernel_shape[0];
        conv2d_kernel_param.kernel_w = conv_param.kernel_shape[1];
        conv2d_kernel_param.stride_h = conv_param.strides[0];
        conv2d_kernel_param.stride_w = conv_param.strides[1];
        conv2d_kernel_param.pad_h = conv_param.pads[0];
        conv2d_kernel_param.pad_w = conv_param.pads[1];
        conv2d_kernel_param.dilation_h = conv_param.dilations[0];
        conv2d_kernel_param.dilation_w = conv_param.dilations[1];
        conv2d_kernel_param.group = conv_param.group;
        conv2d_kernel_param.num_output = num_output;
        conv2d_kernel_param.channels = channels;
        conv2d_kernel_param.fuse_flag = 0;

        conv2d_param_->mgr = conv2d_algo_selector::fast_gen_algo(
            *info.GetInput<TensorImpl>(0)->GetShape(), options.engine_options->forward_precision,
            options.engine_options->dynamic_tuning_level, options.engine_options->winograd_level, 
            options.device->GetISA(), conv2d_param_->param, options.device->GetAllocator());

        if (conv2d_param_->mgr == nullptr) {
            LOG(ERROR) << "No algorithm selected.";
            return ppl::common::RC_UNSUPPORTED;
        }

        auto selected_algo = conv2d_param_->mgr->algo_info();
        if (selected_algo.algo_type == ppl::kernel::arm_server::neon::conv2d_algo::unknown) {
            LOG(ERROR) << "Unsupported algorithm type: " << selected_algo.algo_type;
            return ppl::common::RC_UNSUPPORTED;
        }
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
        LOG(INFO) << "Op " << node->GetName() << " selected conv algorithm: "
                  << ppl::kernel::arm_server::neon::get_conv_algo_str(selected_algo.algo_type);
#endif

        // Note: If the filter is reused, generate a new input edge to hold cvt_filter which may differ due to different algo/sched_param.
        ppl::nn::TensorBufferInfo * new_filter = &options.info->constants[node->GetInput(1)];
        if (new_filter && new_filter->IsBufferOwner() && new_filter->GetBufferPtr<void>()) {
            auto edge = options.graph_topo->AddEdge(node->GetName() + "_Input_Cvt_Filter").first;
            edge->AddConsumer(node->GetId());

            TensorImpl* tensor = new TensorImpl(edge, TENSORTYPE_RESERVED);
            utils::IrShape2TensorShape(options.graph_data->shapes[node->GetInput(1)], tensor->GetShape());
            options.tensors->emplace(edge->GetId(), unique_ptr<TensorImpl>(tensor));

            const_cast<ir::Node*>(node)->ReplaceInput(node->GetInput(1), edge->GetId());
            new_filter = &options.info->constants[node->GetInput(1)];
        }
        new_filter->SetDevice(options.device);

        // Note: If the bias is reused, check it in generate_cvt_weights and simply reuse it as they are the same.
        // CAVEAT: change this if new algorithms with different cvt_bias are added.
        ppl::nn::TensorBufferInfo * new_bias = nullptr;
        if (bias_data != nullptr) {
            new_bias = &options.info->constants[node->GetInput(2)];
            new_bias->SetDevice(options.device);
        }

        ppl::common::RetCode normal_cvt_weights_ret = ppl::common::RC_SUCCESS;
        if (selected_algo.data_type == ppl::common::DATATYPE_FLOAT32) {
            ppl::common::TensorShape cvt_filter_shape, cvt_bias_shape;
            normal_cvt_weights_ret = conv2d_param_->mgr->generate_cvt_weights_shapes(cvt_filter_shape, cvt_bias_shape);
            if (normal_cvt_weights_ret != ppl::common::RC_SUCCESS) {
                return normal_cvt_weights_ret;
            }
            
            if (new_bias && new_bias->IsBufferOwner() && new_bias->GetBufferPtr()) {
                bias_data = nullptr;
            } else if (bias_data && new_bias) {
                new_bias->Reshape(cvt_bias_shape);
                new_bias->ReallocBuffer();
            }

            new_filter->Reshape(cvt_filter_shape);
            new_filter->ReallocBuffer();

            void *cvt_filter_buffer = (new_filter) ? new_filter->GetBufferPtr() : nullptr;
            void *cvt_bias_buffer   = (new_bias  ) ? new_bias->GetBufferPtr()   : nullptr;

            normal_cvt_weights_ret = conv2d_param_->mgr->generate_cvt_weights(weight_data, bias_data, cvt_filter_buffer, cvt_bias_buffer);
            if (normal_cvt_weights_ret != ppl::common::RC_SUCCESS) {
                return normal_cvt_weights_ret;
            }
#ifdef PPLNN_USE_ARMV8_2_FP16
        } else if (selected_algo.data_type == ppl::common::DATATYPE_FLOAT16) {
            ppl::common::TensorShape cvt_filter_shape, cvt_bias_shape;
            normal_cvt_weights_ret = conv2d_param_->mgr->generate_cvt_weights_shapes(cvt_filter_shape, cvt_bias_shape);
            if (normal_cvt_weights_ret != ppl::common::RC_SUCCESS) {
                return normal_cvt_weights_ret;
            }

            vector<__fp16> weight_data_fp16;
            weight_data_fp16.resize(weight_len * sizeof(__fp16));
            Fp32ToFp16(weight_data, weight_len, weight_data_fp16.data());

            vector<__fp16> bias_data_fp16;
            __fp16 * origin_bias_data_fp16 = nullptr;
            if (bias_data != nullptr) {
                bias_data_fp16.resize(bias_len * sizeof(__fp16));
                Fp32ToFp16(bias_data, bias_len, bias_data_fp16.data());
                origin_bias_data_fp16 = bias_data_fp16.data();
            }

            if (new_bias && new_bias->IsBufferOwner() && new_bias->GetBufferPtr()) {
                origin_bias_data_fp16 = nullptr;
            } else if (origin_bias_data_fp16 && new_bias) {
                new_bias->Reshape(cvt_bias_shape);
                new_bias->ReallocBuffer();
            }

            new_filter->Reshape(cvt_filter_shape);
            new_filter->ReallocBuffer();

            void *cvt_filter_buffer = (new_filter) ? new_filter->GetBufferPtr() : nullptr;
            void *cvt_bias_buffer   = (new_bias  ) ? new_bias->GetBufferPtr()   : nullptr;

            normal_cvt_weights_ret = conv2d_param_->mgr->generate_cvt_weights(weight_data_fp16.data(), origin_bias_data_fp16, cvt_filter_buffer, cvt_bias_buffer);
            if (normal_cvt_weights_ret != ppl::common::RC_SUCCESS) {
                return normal_cvt_weights_ret;
            }
#endif
        } else {
            LOG(ERROR) << "Unsupported data type: " << selected_algo.data_type;
            return ppl::common::RC_UNSUPPORTED;
        }
        if (ppl::common::RC_SUCCESS != normal_cvt_weights_ret) {
            LOG(ERROR) << "algo " << selected_algo.algo_type << " cvt weights failed.";
        }
    } else {
        LOG(ERROR) << "Unsupported kernel dim: " << kernel_dims;
        return ppl::common::RC_UNSUPPORTED;
    }

    return RC_SUCCESS;
}

RetCode ConvOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                             vector<dataformat_t>* selected_output_formats) {
    if (conv2d_param_ && conv2d_param_->mgr &&
        conv2d_param_->mgr->algo_info().algo_type != ppl::kernel::arm_server::neon::conv2d_algo::unknown) {
        selected_input_formats->at(0) = conv2d_param_->mgr->algo_info().input_format;
        selected_output_formats->at(0) = conv2d_param_->mgr->algo_info().output_format;
        return RC_SUCCESS;
    }
    return RC_INVALID_VALUE;
}
RetCode ConvOp::SelectDataType(const InputOutputInfo& info, std::vector<ppl::common::datatype_t>* selected_input_types,
                               std::vector<ppl::common::datatype_t>* selected_output_types,
                               const ppl::common::datatype_t preferred_fp_datatype) {
    GenericSelectDataType(info, selected_input_types, selected_output_types, preferred_fp_datatype);
    for (uint32_t i = 1; i < info.GetInputCount(); i++) {
        selected_input_types->at(i) = info.GetInput<TensorImpl>(i)->GetShape()->GetDataType();
    }
    return RC_SUCCESS;
}

bool ConvOp::TryFuseReLU(void) {
    if (!conv2d_param_ || !conv2d_param_->mgr ||
        conv2d_param_->mgr->algo_info().algo_type == ppl::kernel::arm_server::neon::conv2d_algo::unknown) {
        return false;
    }
    ppl::kernel::arm_server::neon::conv2d_param param = conv2d_param_->mgr->get_param();
    param.fuse_flag |= ppl::kernel::arm_server::neon::conv_fuse_flag::RELU;
    conv2d_param_->mgr->set_param(param);
    return true;
}

bool ConvOp::TryFuseReLU6(void) {
    if (!conv2d_param_ || !conv2d_param_->mgr ||
        conv2d_param_->mgr->algo_info().algo_type == ppl::kernel::arm_server::neon::conv2d_algo::unknown) {
        return false;
    }
    ppl::kernel::arm_server::neon::conv2d_param param = conv2d_param_->mgr->get_param();
    param.fuse_flag |= ppl::kernel::arm_server::neon::conv_fuse_flag::RELU;
    param.fuse_flag |= ppl::kernel::arm_server::neon::conv_fuse_flag::RELU6;
    conv2d_param_->mgr->set_param(param);
    return true;
}

bool ConvOp::TryFuseSum(void) {
    if (!conv2d_param_ || !conv2d_param_->mgr ||
        conv2d_param_->mgr->algo_info().algo_type == ppl::kernel::arm_server::neon::conv2d_algo::unknown) {
        return false;
    }
    ppl::kernel::arm_server::neon::conv2d_param param = conv2d_param_->mgr->get_param();
    if (param.fuse_flag) { // already fused sum, relu or relu6
        return false;
    }
    param.fuse_flag |= ppl::kernel::arm_server::neon::conv_fuse_flag::SUM;
    conv2d_param_->mgr->set_param(param);
    return true;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode ConvOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder conv_builder;
    auto mgr = conv2d_param_->mgr;
    std::vector<int64_t> algo_sp = mgr->get_schedule_param();

    auto fb_algo_info = ppl::nn::pmx::arm::CreateConvAlgoInfo(conv_builder,
                                                              mgr->get_algo_type(),
                                                              mgr->algo_info().data_type,
                                                              mgr->algo_info().isa,
                                                              conv_builder.CreateVector<int64_t>(algo_sp));
    auto fb_param_info = ppl::nn::pmx::arm::CreateConvParamInfo(conv_builder, 
                                                                conv2d_param_->param.num_output, 
                                                                conv2d_param_->param.channels, 
                                                                conv2d_param_->mgr->get_param().pad_type, 
                                                                conv2d_param_->mgr->get_param().fuse_flag,
                                                                (!mgr->is_zero_bias()) ? 1 : 0);
    
    auto fb_conv_data = ppl::nn::pmx::arm::CreateConvData(conv_builder, fb_algo_info, fb_param_info);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(conv_builder, ppl::nn::pmx::arm::PrivateDataType_ConvData, fb_conv_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(conv_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_param = ppl::nn::pmx::onnx::SerializeConvParam(*param_.get(), &op_builder);
    auto fb_data = op_builder.CreateVector(conv_builder.GetBufferPointer(), conv_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_ConvParam, fb_param.Union(), fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode ConvOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    if (conv2d_param_) {
        delete conv2d_param_;
    }
    conv2d_param_ = new Convolution2DParam;
    if (!conv2d_param_) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);
    auto fb_conv_param = fb_op_param->value_as_ConvParam();

    param_ = std::make_shared<ppl::nn::onnx::ConvParam>();
    DeserializeConvParam(*fb_conv_param, param_.get());

    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());

    auto arm_conv_data = arm_op_data->value_as_ConvData();
    auto algo_info = arm_conv_data->algo_info();
    auto param_info = arm_conv_data->param_info();

    common_param_.output_types.resize(1);
    common_param_.output_formats.resize(1);
    common_param_.output_types[0] = algo_info->dtype();
    common_param_.output_formats[0] = (algo_info->dtype() == DATATYPE_FLOAT32) ? DATAFORMAT_N4CX : DATAFORMAT_N8CX;

    conv2d_param &conv2d_kernel_param = conv2d_param_->param;
    conv2d_kernel_param.kernel_h = param_->kernel_shape[0];
    conv2d_kernel_param.kernel_w = param_->kernel_shape[1];
    conv2d_kernel_param.stride_h = param_->strides[0];
    conv2d_kernel_param.stride_w = param_->strides[1];
    conv2d_kernel_param.dilation_h = param_->dilations[0];
    conv2d_kernel_param.dilation_w = param_->dilations[1];
    conv2d_kernel_param.pad_h = param_->pads[0];
    conv2d_kernel_param.pad_w = param_->pads[1];
    conv2d_kernel_param.channels = param_info->channels();
    conv2d_kernel_param.num_output = param_info->num_output();
    conv2d_kernel_param.group = param_->group;
    conv2d_kernel_param.fuse_flag = param_info->fuse_type();
    conv2d_kernel_param.pad_type = param_info->pad_type();

    auto mgr = conv2d_algo::generate_conv_mgr(algo_info->algo_type(), algo_info->dtype(), conv2d_kernel_param, allocator_);
    mgr->set_algo_info({ .algo_type = algo_info->algo_type(),
                         .isa = algo_info->isa(),
                         .data_type = algo_info->dtype()
                       });
    
    std::vector<int64_t> sp;
    ppl::nn::pmx::utils::Fbvec2Stdvec(algo_info->sched_param(), &sp);
    mgr->set_schedule_param(sp);
    
    const auto & shapes = *ctx.shapes;
    const auto & constants = *ctx.constants;

    uint32_t cvt_filter_id = GetNode()->GetInput(1);
    uint64_t cvt_filter_size = shapes.at(cvt_filter_id).CalcBytesExcludingPadding();
    void *cvt_filter_ptr = constants.at(cvt_filter_id).GetBufferPtr<void>();
    mgr->set_cvt_filter(cvt_filter_ptr, cvt_filter_size);

    bool has_bias = (param_info->has_bias() == 1);
    void *cvt_bias_ptr;
    if (has_bias) {
        uint32_t cvt_bias_id = GetNode()->GetInput(2);
        uint64_t cvt_bias_size = shapes.at(cvt_bias_id).CalcBytesExcludingPadding();
        cvt_bias_ptr = constants.at(cvt_bias_id).GetBufferPtr<void>();
        mgr->set_cvt_bias(cvt_bias_ptr, cvt_bias_size);
    } else {
        uint64_t cvt_bias_size = conv2d_kernel_param.num_output * ppl::common::GetSizeOfDataType(algo_info->dtype());
        cvt_bias_size = (cvt_bias_size + 15) & (~15);
        cvt_bias_ptr = mgr->get_allocator()->Alloc(cvt_bias_size);
        memset(cvt_bias_ptr, 0, cvt_bias_size);
        mgr->set_cvt_bias(cvt_bias_ptr, cvt_bias_size, true);
    }

    conv2d_param_->mgr = mgr;
    conv2d_param_->fallback_mgr = nullptr;

    return RC_SUCCESS;
}

#endif

KernelImpl* ConvOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<Conv2dKernel>(conv2d_param_);
}

}}} // namespace ppl::nn::arm
