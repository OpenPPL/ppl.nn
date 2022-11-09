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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/convtranspose_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/convtranspose_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_convtranspose.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/cuda/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/conv_transpose.h"
#include "ppl/nn/engines/cuda/pmx/generated/cuda_op_params_generated.h"
#endif

namespace ppl { namespace nn { namespace cuda {

ConvTransposeOp::~ConvTransposeOp() {
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

ConvTransposeOp::ConvTransposeOp(const ir::Node* node) : CudaOptKernel(node) {

    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        type = ppl::common::DATATYPE_FLOAT16;
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeConvTranspose(info, &(param_.param));
    };
}

RetCode ConvTransposeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ConvTransposeParam>(options, &param_.param);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    
    return RC_SUCCESS;
}

RetCode ConvTransposeOp::Finalize(const OptKernelOptions& options) {
    param_ = *((CudaConvTransposeParam*)options.param);
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

void ConvTransposeOp::CopyParam(void*& param) {
    if (param == nullptr) {
        param = new CudaConvTransposeParam();
    }
    *(CudaConvTransposeParam*)param = param_;
    return;
}

KernelImpl* ConvTransposeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ConvTransposeKernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
RetCode ConvTransposeOp::SerializeData(const pmx::SerializationContext& ctx, utils::DataStream* ds) const {
#ifdef PPLNN_ENABLE_CUDA_JIT
    CUDAModule* module = static_cast<CUDAModule*>(GetCommparamModule());
    std::string ptx_code;
    if (module) { // for specific algo, may no jit support
        ptx_code = module->GetSourceCode().second;
    }
#else
    std::string ptx_code;
#endif

    flatbuffers::FlatBufferBuilder private_data_builder;
    auto status = pmx::cuda::SerializePrivateData<ConvTransposeExtraParam>(ctx, param_.extra_param, ptx_code, &private_data_builder);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "SerializePrivateData of op[" << GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = pmx::onnx::SerializeConvTransposeParam(param_.param, &builder);
    auto fb_data = builder.CreateVector(private_data_builder.GetBufferPointer(), private_data_builder.GetSize());
    auto fb_root = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_ConvTransposeParam, fb_param.Union(), fb_data);
    pmx::onnx::FinishOpParamBuffer(builder, fb_root);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

RetCode ConvTransposeOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
    auto fb_op_param = pmx::onnx::GetOpParam(base);
    auto fb_convtranspose_param = fb_op_param->value_as_ConvTransposeParam();

    pmx::onnx::DeserializeConvTransposeParam(*fb_convtranspose_param, &param_.param);

    // CUDAModule* module = static_cast<CUDAModule*>(GetCommparamModule());
    // auto ptx_code = module->GetSourceCode().second;
    std::string ptx_code = "";
    auto fb_data = fb_op_param->data_();
    auto status = pmx::cuda::DeserializePrivateData<ConvTransposeExtraParam>(fb_data->data(), fb_data->size(), ptx_code, &param_.extra_param);
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
