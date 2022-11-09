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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/matmul_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/matmul_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_matmul.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/cuda/pmx/utils.h"
#include "ppl/nn/engines/cuda/pmx/generated/cuda_op_params_generated.h"
#include "ppl/nn/models/pmx/oputils/onnx/gemm.h"
#endif

namespace ppl { namespace nn { namespace cuda {
MatMulOp::~MatMulOp() {
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

RetCode MatMulOp::Init(const OptKernelOptions& options) {
    param_.param.alpha = 1;
    param_.param.beta = 1;
    param_.param.transA = 0;
    param_.param.transB = 0;

    return RC_SUCCESS;
}

MatMulOp::MatMulOp(const ir::Node* node) : CudaOptKernel(node) {
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

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeMatMul(info, nullptr);
    };
}

RetCode MatMulOp::Finalize(const OptKernelOptions& options) {
    param_ = *((CudaGemmParam*)options.param);

    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

void MatMulOp::CopyParam(void*& param) {
    if (param == nullptr) {
        param = new CudaGemmParam();
    }
    *(CudaGemmParam*)param = param_;
    return;
}

KernelImpl* MatMulOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MatMulKernel>(&param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
RetCode MatMulOp::SerializeData(const pmx::SerializationContext& ctx, utils::DataStream* ds) const {
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
    auto status = pmx::cuda::SerializePrivateData<GemmExtraParam>(ctx, param_.extra_param, ptx_code, &private_data_builder);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "SerializePrivateData of op[" << GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = pmx::onnx::SerializeGemmParam(param_.param, &builder);
    auto fb_data = builder.CreateVector(private_data_builder.GetBufferPointer(), private_data_builder.GetSize());
    auto fb_root = pmx::onnx::CreateOpParam(builder, pmx::onnx::OpParamType_GemmParam, fb_param.Union(), fb_data);
    pmx::onnx::FinishOpParamBuffer(builder, fb_root);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

RetCode MatMulOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
    auto fb_op_param = pmx::onnx::GetOpParam(base);
    auto fb_conv_param = fb_op_param->value_as_GemmParam();

    pmx::onnx::DeserializeGemmParam(*fb_conv_param, &param_.param);

    // CUDAModule* module = static_cast<CUDAModule*>(GetCommparamModule());
    // auto ptx_code = module->GetSourceCode().second;
    std::string ptx_code = "";
    auto fb_data = fb_op_param->data_();
    auto status = pmx::cuda::DeserializePrivateData<GemmExtraParam>(fb_data->data(), fb_data->size(), ptx_code, &param_.extra_param);
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
