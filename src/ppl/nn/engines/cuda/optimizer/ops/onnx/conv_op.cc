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
#include "ppl/nn/engines/cuda/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/conv.h"
#include "ppl/nn/engines/cuda/pmx/generated/cuda_op_params_generated.h"
#endif

namespace ppl { namespace nn { namespace cuda {
ConvOp::~ConvOp() {
    for (uint32_t i = 0; i < param_.extra_param.fuse_info.fuse_attrs.size(); ++i) {
        free(param_.extra_param.fuse_info.fuse_attrs[i]);
    }
#ifdef PPLNN_ENABLE_PMX_MODEL
    if (pmx_module_created_) {
        auto cuda_common_param = GetCommparam();
        if (cuda_common_param->module) {
            delete (CUDAModule*)cuda_common_param->module;
            cuda_common_param->module = nullptr;
        }
        pmx_module_created_ = false;
    }
#endif
}

void ConvOp::CopyParam(void*& param) {
    if (param == nullptr) {
        param = new CudaConvParam();
    }
    *(CudaConvParam*)param = param_;
    return;
}

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
                if (i == 2 && param_.extra_param.bias_term) {
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
        if (inshape->GetDimCount() != 4) {
            LOG(DEBUG) << "error input shape dims " << inshape->GetDimCount();
            return ppl::common::RC_INVALID_VALUE;
        }
        auto status = onnx::ReshapeConv(info, &(param_.param));
        if (info->GetOutputCount() > 1 && param_.extra_param.fuse_info.channel_offset >= 0) {
            auto postshape = info->GetOutput<TensorImpl>(1);
            postshape->GetShape()->Reshape(info->GetInput<TensorImpl>(0)->GetShape()->GetDims(),
                                           info->GetInput<TensorImpl>(0)->GetShape()->GetRealDimCount());
            postshape->GetShape()->SetDim(1, param_.extra_param.fuse_info.channel_size);
        }
        return status;
    };
}

RetCode ConvOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ConvParam>(options, &param_.param);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    param_.extra_param.bias_term = GetNode()->GetInputCount() > 2 ? true : false;

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
    } else if (param_.extra_param.algo_info.algo_type == "TuringIMMAImpgemm" || \
    param_.extra_param.algo_info.algo_type == "CutlassHConv") {
        return CreateKernelImplWithParam<ConvImmaKernel>(&param_);
    } else if (param_.extra_param.algo_info.algo_type == "DepthwiseDirect") {
        return CreateKernelImplWithParam<ConvDepthwiseKernel>(&param_);
    } else if (param_.extra_param.algo_info.algo_type == "DepthwiseDirectInt8") {
        return CreateKernelImplWithParam<ConvDepthwiseKernel>(&param_);
    }
    return nullptr;
}

#ifdef PPLNN_ENABLE_PMX_MODEL
RetCode ConvOp::SerializeData(const pmx::SerializationContext& ctx, utils::DataStream* ds) const {
#ifdef PPLNN_ENABLE_CUDA_JIT
    CUDAModule* module = static_cast<CUDAModule*>(GetCommparamModule());
    std::string ptx_code;
    if (module) { // for depthwise conv, no jit support
        ptx_code = module->GetSourceCode().second;
    }
#else
    std::string ptx_code;
#endif

    flatbuffers::FlatBufferBuilder private_data_builder;
    auto status = pmx::cuda::SerializePrivateData<ConvExtraParam>(ctx, param_.extra_param, ptx_code, &private_data_builder);
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
    auto status = pmx::cuda::DeserializePrivateData<ConvExtraParam>(fb_data->data(), fb_data->size(), ptx_code, &param_.extra_param);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "DeserializePrivateData of op[" << GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

#ifdef PPLNN_ENABLE_CUDA_JIT
    if (!ptx_code.empty()) { // for depthwise conv, no jit support
        CUDAModule* cuda_module = new CUDAModule(); // delete later
        cuda_module->SetSourceCode(param_.extra_param.algo_info.algo_name, ptx_code);
        auto cuda_common_param = GetCommparam();
        if (cuda_common_param->module) delete (CUDAModule*)cuda_common_param->module;
        cuda_common_param->module = (void*)cuda_module;
        pmx_module_created_ = true;
    }
#endif
    return RC_SUCCESS;
}
#endif

}}} // namespace ppl::nn::cuda
