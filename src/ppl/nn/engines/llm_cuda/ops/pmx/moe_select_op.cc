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

#include "moe_select_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/moe_select_kernel.h"
#include "ppl/nn/oputils/pmx/reshape_moe_select.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {


RetCode MoeSelectOp::CommonInit() {
    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto input_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
        auto output_shape0 = info->GetOutput<TensorImpl>(0)->GetShape();
        auto output_shape1 = info->GetOutput<TensorImpl>(1)->GetShape();
        auto output_shape2 = info->GetOutput<TensorImpl>(2)->GetShape();
        auto output_shape3 = info->GetOutput<TensorImpl>(3)->GetShape();

        output_shape0->SetDataType(input_shape0->GetDataType());
        output_shape0->SetDataFormat(input_shape0->GetDataFormat());
        output_shape1->SetDataType(input_shape0->GetDataType());
        output_shape1->SetDataFormat(input_shape0->GetDataFormat());

        output_shape2->SetDataType(DATATYPE_INT64);
        output_shape2->SetDataFormat(input_shape0->GetDataFormat());

        output_shape3->SetDataType(DATATYPE_INT64);
        output_shape3->SetDataFormat(input_shape0->GetDataFormat());

        return ppl::common::RC_SUCCESS;
    };


    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return nn::pmx::ReshapeMoeSelect(info, param_.get());
    };
    return RC_SUCCESS;
}

RetCode MoeSelectOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<MoeSelectParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    return CommonInit();
}

KernelImpl* MoeSelectOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MoeSelectKernel>(param_.get());
}


}}}}} // namespace ppl::nn::llm::cuda::pmx
