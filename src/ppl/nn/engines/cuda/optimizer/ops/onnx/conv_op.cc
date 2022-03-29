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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/conv_op.h"

#include "ppl/nn/engines/cuda/kernels/onnx/conv_hmma_kernel.h"
#include "ppl/nn/engines/cuda/kernels/onnx/conv_imma_kernel.h"
#include "ppl/nn/engines/cuda/kernels/onnx/conv_depthwise_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_conv.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/conv.h"
#include "ppl/nn/engines/cuda/pmx/generated/cuda_op_params_generated.h"
#include "ppl/nn/models/pmx/oputils/onnx/conv.h"
#endif

namespace ppl { namespace nn { namespace cuda {

ConvOp::ConvOp(const ir::Node* node) : CudaOptKernel(node) {

    infer_type_func_ = [this](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        if (type == DATATYPE_INT8) {
            auto in_edge_id = info->GetInput<TensorImpl>(0)->GetEdge()->GetId();
            auto& in_quant = quant->at(in_edge_id);
            auto out_edge_id = info->GetOutput<TensorImpl>(0)->GetEdge()->GetId();
            auto& out_quant = quant->at(out_edge_id);
            if (in_quant.type != DATATYPE_INT8 || out_quant.type != DATATYPE_INT8) {
                return RC_INVALID_VALUE;
            }
            info->GetInput<TensorImpl>(0)->GetShape()->SetDataType(in_quant.type);
            info->GetOutput<TensorImpl>(0)->GetShape()->SetDataType(out_quant.type);

            // Copy quant info skipping input0
            for (uint32_t i = 1; i < info->GetInputCount(); ++i) {
                auto in_edge_id = info->GetInput<TensorImpl>(i)->GetEdge()->GetId();
                auto& in_quant = quant->at(in_edge_id);
                auto in_shape = info->GetInput<TensorImpl>(i)->GetShape();
                if (i == 1 && in_quant.type != DATATYPE_UNKNOWN) {
                    in_shape->SetDataType(in_quant.type);
                    continue;
                }
                if (i == 2 && param_.extra_param.algo_info.has_bias) {
                    in_shape->SetDataType(ppl::common::DATATYPE_FLOAT32);
                    continue;
                }
                in_shape->SetDataType(out_quant.type);
            }
            return ppl::common::RC_SUCCESS;
        }
        type = ppl::common::DATATYPE_FLOAT16;
        return InferDefaultType(info, type);
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto inshape = info->GetInput<TensorImpl>(0)->GetShape();
        if (inshape->GetDimCount() < 4) {
            inshape->Reshape(info->GetInput<TensorImpl>(1)->GetShape()->GetDims(), 4);
        }
        auto status = oputils::ReshapeConv(info, &(param_.param));
        if (info->GetOutputCount() > 1 && param_.extra_param.fuse_info.channel_offset >= 0) {
            auto postshape = info->GetOutput<TensorImpl>(1);
            postshape->GetShape()->Reshape(info->GetInput<TensorImpl>(0)->GetShape()->GetDims(),
                                           info->GetInput<TensorImpl>(0)->GetShape()->GetRealDimCount());
            postshape->GetShape()->SetDim(1, param_.extra_param.fuse_info.channel_size);
        }
        return status;
    };
}

void ConvOp::CopyParam(void*& param) {
    if (param == nullptr) {
        param = new CudaConvParam();
    }
    *(CudaConvParam*)param = param_;
    return;
}

RetCode ConvOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ConvParam>(options, &param_.param);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    param_.extra_param.algo_info.has_bias = GetNode()->GetInputCount() > 2 ? true : false;

    return RC_SUCCESS;
}

RetCode ConvOp::Finalize(const OptKernelOptions& options) {
    param_ = *((CudaConvParam*)options.param);

    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

KernelImpl* ConvOp::CreateKernelImpl() const {
    if (param_.extra_param.algo_info.algo_type == "TuringHMMAImpgemm") {
        return CreateKernelImplWithParam<ConvHmmaKernel>(&param_);
    } else if (param_.extra_param.algo_info.algo_type == "TuringIMMAImpgemm") {
        return CreateKernelImplWithParam<ConvImmaKernel>(&param_);
    } else if (param_.extra_param.algo_info.algo_type == "DepthwiseDirect") {
        return CreateKernelImplWithParam<ConvDepthwiseKernel>(&param_);
    } else if (param_.extra_param.algo_info.algo_type == "DepthwiseDirectInt8") {
        return CreateKernelImplWithParam<ConvDepthwiseKernel>(&param_);
    }
    return nullptr;
}

#ifdef PPLNN_ENABLE_PMX_MODEL
static RetCode SerializePrivateData(const pmx::SerializationContext& ctx, const ConvExtraParam& extra_param, const std::string& ptx_code, flatbuffers::FlatBufferBuilder* builder) {
    auto& tiles = extra_param.algo_info.tiles;
    vector<int32_t> tiles_vec{tiles.m_cta,
                              tiles.n_cta,
                              tiles.k_cta,
                              tiles.m_warp,
                              tiles.n_warp,
                              tiles.k_per_step,
                              tiles.k_per_set,
                              tiles.flt_size,
                              tiles.flt_pad_size,
                              tiles.cta_size_in_thd,
                              tiles.buf};
    auto fb_algo_info = pmx::cuda::CreateConvAlgoInfoDirect(*builder,
                                                            extra_param.algo_info.algo_type.c_str(),
                                                            extra_param.algo_info.algo_name.c_str(),
                                                            &tiles_vec,
                                                            extra_param.algo_info.kid,
                                                            extra_param.algo_info.splitk,
                                                            extra_param.algo_info.splitf,
                                                            extra_param.algo_info.is_initializer_weight,
                                                            extra_param.algo_info.has_bias);

    std::vector<flatbuffers::Offset<flatbuffers::String>> fb_types;
    std::vector<pmx::cuda::FuseAttrs> fb_attrs;
    for (uint32_t i = 0; i < extra_param.fuse_info.types.size(); ++i) {
        auto type = extra_param.fuse_info.types[i];
        auto fb_type = builder->CreateString(type);
        fb_types.push_back(fb_type);
        pmx::cuda::FuseAttrs fb_attr;
        if (type == "Clip") {
            auto attr = extra_param.fuse_info.fuse_attrs[i];
            auto clip_attr = (ClipParam*)attr;
            fb_attr = pmx::cuda::FuseAttrs(clip_attr->min_val, clip_attr->max_val, 0);
        } else if (type == "LeakyRelu") {
            auto attr = extra_param.fuse_info.fuse_attrs[i];
            auto leaky_attr = (LeakyReluParam*)attr;
            fb_attr = pmx::cuda::FuseAttrs(0, 0, leaky_attr->alpha);
        } else {
            fb_attr = pmx::cuda::FuseAttrs(0, 0, 0);
        }
        fb_attrs.push_back(fb_attr);
    }

    // edge_id has reset in sequence
    edgeid_t fb_concat_edge_it = -1;
    if (extra_param.fuse_info.concat_edge_id != -1) {
        fb_concat_edge_it = ctx.eid2seq[extra_param.fuse_info.concat_edge_id];
    }
    auto fb_fuse_info = pmx::cuda::CreateConvFusionInfoDirect(*builder,
                                                              &fb_types,
                                                              &extra_param.fuse_info.input_inds,
                                                              &fb_attrs,
                                                              extra_param.fuse_info.channel_size,
                                                              extra_param.fuse_info.channel_offset,
                                                              fb_concat_edge_it);

    auto fb_code = builder->CreateVector((const uint8_t*)ptx_code.data(), ptx_code.size());
    auto fb_conv_param = pmx::cuda::CreateConvParam(*builder, fb_algo_info, fb_fuse_info, fb_code);
    auto fb_op_param = pmx::cuda::CreateOpParam(*builder, pmx::cuda::OpParamType_ConvParam, fb_conv_param.Union());
    pmx::cuda::FinishOpParamBuffer(*builder, fb_op_param);
    return RC_SUCCESS;
}

static RetCode DeserializePrivateData(const void* fb_param, uint64_t size, std::string& ptx_code, ConvExtraParam* extra_param) {
    auto fb_op_param = pmx::cuda::GetOpParam(fb_param);
    auto fb_conv_param = fb_op_param->value_as_ConvParam();

    auto fb_algo_info = fb_conv_param->algo_info();
    extra_param->algo_info.algo_type = fb_algo_info->algo_type()->c_str();
    extra_param->algo_info.algo_name = fb_algo_info->algo_name()->c_str();
    extra_param->algo_info.kid = fb_algo_info->kid();
    extra_param->algo_info.splitk = fb_algo_info->splitk();
    extra_param->algo_info.splitf = fb_algo_info->splitf();
    extra_param->algo_info.is_initializer_weight = fb_algo_info->is_initializer_weight();
    extra_param->algo_info.has_bias = fb_algo_info->has_bias();
    auto& tiles = extra_param->algo_info.tiles;
    std::vector<int> tiles_vec;
    ppl::nn::pmx::utils::Fbvec2Stdvec(fb_algo_info->tiles(), &tiles_vec);
    tiles.m_cta = tiles_vec[0];
    tiles.n_cta = tiles_vec[1];
    tiles.k_cta = tiles_vec[2];
    tiles.m_warp = tiles_vec[3];
    tiles.n_warp = tiles_vec[4];
    tiles.k_per_step = tiles_vec[5];
    tiles.k_per_set = tiles_vec[6];
    tiles.flt_size = tiles_vec[7];
    tiles.flt_pad_size = tiles_vec[8];
    tiles.cta_size_in_thd = tiles_vec[9];
    tiles.buf = tiles_vec[10];

    auto fb_fuse_info = fb_conv_param->fuse_info();
    auto fb_types = fb_fuse_info->types();
    auto fb_input_inds = fb_fuse_info->input_inds();
    auto fb_fuse_attrs = fb_fuse_info->fuse_attrs();

    extra_param->fuse_info.types.clear();
    extra_param->fuse_info.input_inds.clear();
    extra_param->fuse_info.fuse_attrs.clear();

    for (uint32_t i = 0; i < fb_types->size(); ++i) {
        std::string str = fb_types->Get(i)->str();
        extra_param->fuse_info.types.push_back(str);
        auto ind = fb_input_inds->Get(i);
        extra_param->fuse_info.input_inds.push_back(ind);
        auto attr = fb_fuse_attrs->Get(i);
        if (str == "Clip") {
            ClipParam clip;
            clip.max_val = attr->clip_max();
            clip.min_val = attr->clip_min();
            extra_param->fuse_info.fuse_attrs.push_back(std::move(&clip));
        } else if (str == "LeakyRelu") {
            LeakyReluParam leaky;
            leaky.alpha = attr->leaky_alpha();
            extra_param->fuse_info.fuse_attrs.push_back(std::move(&leaky));
        } else {
            extra_param->fuse_info.fuse_attrs.push_back(nullptr);
        }
    }

    extra_param->fuse_info.channel_offset = fb_fuse_info->channel_offset();
    extra_param->fuse_info.channel_size = fb_fuse_info->channel_size();
    extra_param->fuse_info.concat_edge_id = fb_fuse_info->concat_edge_id();

    ptx_code.assign((const char*)fb_conv_param->gene_code()->data(), fb_conv_param->gene_code()->size());
    return RC_SUCCESS;
}

RetCode ConvOp::SerializeData(const pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    CUDAModule* module = static_cast<CUDAModule*>(GetCommparamModule());
    auto ptx_code = module->GetSourceCode().second;

    flatbuffers::FlatBufferBuilder private_data_builder;
    auto status = SerializePrivateData(ctx, param_.extra_param, ptx_code, &private_data_builder);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "SerializePrivateData of op[" << GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = pmx::onnx::SerializeConvParam(param_.param, &builder);
    auto fb_data = builder.CreateVector(private_data_builder.GetBufferPointer(), private_data_builder.GetSize());
    auto fb_root = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_ConvParam, fb_param.Union(), fb_data);
    pmx::onnx::FinishOpParamBuffer(builder, fb_root);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

RetCode ConvOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
    auto fb_op_param = pmx::onnx::GetOpParam(base);
    auto fb_conv_param = fb_op_param->value_as_ConvParam();

    pmx::onnx::DeserializeConvParam(*fb_conv_param, &param_.param);

    // CUDAModule* module = static_cast<CUDAModule*>(GetCommparamModule());
    // auto ptx_code = module->GetSourceCode().second;
    std::string ptx_code = "";
    auto fb_data = fb_op_param->data_();
    auto status = DeserializePrivateData(fb_data->data(), fb_data->size(), ptx_code, &param_.extra_param);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "DeserializePrivateData of op[" << GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    CUDAModule* cuda_module = new CUDAModule(); // delete later
    cuda_module->SetSourceCode(param_.extra_param.algo_info.algo_name, ptx_code);
    auto cuda_common_param = GetCommparam();
    cuda_common_param->module = (void*)cuda_module;
    return RC_SUCCESS;
}
#endif

}}} // namespace ppl::nn::cuda
