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

#include "online_quantize_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/i8i8/online_quantize_kernel.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/engines/llm_cuda/engine.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_i8i8_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

RetCode I8I8OnlineQuantizeOp::CommonInit() {
    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto input_shape = info->GetInput<TensorImpl>(0)->GetShape();
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        auto scale_shape = info->GetOutput<TensorImpl>(1)->GetShape();

        output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
        output_shape->SetDataType(DATATYPE_INT8);

        scale_shape->SetDataFormat(DATAFORMAT_NDARRAY);
        scale_shape->SetDataType(input_shape->GetDataType());

        return RC_SUCCESS;
    };
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto input_shape = info->GetInput<TensorImpl>(0)->GetShape();
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        auto scale_shape = info->GetOutput<TensorImpl>(1)->GetShape();

        output_shape->Reshape(input_shape->GetDims(), input_shape->GetDimCount());
        scale_shape->Reshape(input_shape->GetDims(), input_shape->GetDimCount() - 1);

        return RC_SUCCESS;
    };
    return RC_SUCCESS;
}

RetCode I8I8OnlineQuantizeOp::DoInit(const OptKernelOptions& options) {
    return CommonInit();
}

KernelImpl* I8I8OnlineQuantizeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<I8I8OnlineQuantizeKernel>();
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode I8I8OnlineQuantizeOp::SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream* ds) const {
    return RC_SUCCESS;
}

ppl::common::RetCode I8I8OnlineQuantizeOp::DeserializeData(const ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    return CommonInit();
}
#endif

}}}}} // namespace ppl::nn::llm::cuda::pmx
