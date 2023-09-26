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

#include "online_quantize_rms_norm_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/i8i8/online_quantize_rms_norm_kernel.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

RetCode I8I8OnlineQuantizeRMSNormOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<RMSNormParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto input_shape = info->GetInput<TensorImpl>(0)->GetShape();
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();

        output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
        output_shape->SetDataType(DATATYPE_INT8);

        for (uint32_t i = 1; i < info->GetOutputCount(); ++i) {
            output_shape = info->GetOutput<TensorImpl>(i)->GetShape();
            output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
            output_shape->SetDataType(input_shape->GetDataType());
        }

        return RC_SUCCESS;
    };
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto input_shape = info->GetInput<TensorImpl>(0)->GetShape();
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        auto scale_shape = info->GetOutput<TensorImpl>(1)->GetShape();

        output_shape->Reshape(input_shape->GetDims(), input_shape->GetDimCount());
        const int64_t dim_count = input_shape->GetDimCount();
        const int64_t real_axis = param_->axis > 0 ? param_->axis : (param_->axis + dim_count);
        scale_shape->Reshape(input_shape->GetDims(), real_axis);

        if (param_->skip_term) {
            output_shape = info->GetOutput<TensorImpl>(2)->GetShape();
            output_shape->Reshape(input_shape->GetDims(), input_shape->GetDimCount());
        }

        return RC_SUCCESS;
    };

    return RC_SUCCESS;
}

KernelImpl* I8I8OnlineQuantizeRMSNormOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<I8I8OnlineQuantizeRMSNormKernel>(param_.get());
}



}}}}} // namespace ppl::nn::llm::cuda::pmx
