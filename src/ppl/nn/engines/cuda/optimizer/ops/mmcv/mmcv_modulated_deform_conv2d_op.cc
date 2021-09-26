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

#include "ppl/nn/engines/cuda/optimizer/ops/mmcv/mmcv_modulated_deform_conv2d_op.h"
#include "ppl/nn/engines/cuda/kernels/mmcv/mmcv_modulated_deform_conv2d_kernel.h"
#include "ppl/nn/oputils/mmcv/reshape_mmcv_modulated_deform_conv2d.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode MMCVModulatedDeformConv2dOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeMMCVModulatedDeformConv2d(info, param_.get());
    };

    infer_type_func_ = [this](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        return InferInheritedType(info);
    };

    return RC_SUCCESS;
}

KernelImpl* MMCVModulatedDeformConv2dOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MMCVModulatedDeformConv2dKernel>(param_.get());
}

}}} // namespace ppl::nn::cuda
