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
#include "ppl/nn/engines/cuda/kernels/onnx/conv_depthwise_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_convolution.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {
ConvOp::~ConvOp() {
    for (uint32_t i = 0; i < param_.extra_param.fuse_info.fuse_attrs.size(); ++i) {
        free(param_.extra_param.fuse_info.fuse_attrs[i]);
    }
}

void ConvOp::CopyParam(void*& param) {
    if (param == nullptr) {
        param = new CudaConvParam();
    }
    *(CudaConvParam*)param = param_;
    return;
}

RetCode ConvOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ConvolutionParam>(options, &param_.param);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    param_.param.bias_term = GetNode()->GetInputCount() > 2 ? 1 : 0;

    infer_type_func_ = [this](InputOutputInfo* info, datatype_t type) -> RetCode {
        if (type == ppl::common::DATATYPE_UNKNOWN) {
            type = ppl::common::DATATYPE_FLOAT16;
        }
        return InferDefaultType(info, type);
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto inshape = &info->GetInput<TensorImpl>(0)->GetShape();
        if (inshape->GetDimCount() < 4) {
            inshape->Reshape(info->GetInput<TensorImpl>(1)->GetShape().GetDims(), 4);
        }
        auto status = oputils::ReshapeConvolution(info, &(param_.param));
        if (info->GetOutputCount() > 1 && param_.extra_param.fuse_info.channel_offset >= 0) {
            auto postshape = info->GetOutput<TensorImpl>(1);
            postshape->GetShape().Reshape(info->GetInput<TensorImpl>(0)->GetShape().GetDims(),
                                          info->GetInput<TensorImpl>(0)->GetShape().GetRealDimCount());
            postshape->GetShape().SetDim(1, param_.extra_param.fuse_info.channel_size);
        }
        return status;
    };

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
    } else if (param_.extra_param.algo_info.algo_type == "DepthwiseDirect") {
        return CreateKernelImplWithParam<ConvDepthwiseKernel>(&param_);
    }
    return nullptr;
}

}}} // namespace ppl::nn::cuda
