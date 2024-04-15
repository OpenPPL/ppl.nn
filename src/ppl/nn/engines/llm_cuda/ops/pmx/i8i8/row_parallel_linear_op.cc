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

#include "row_parallel_linear_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/i8i8/row_parallel_linear_kernel.h"
#include "ppl/nn/oputils/pmx/reshape_row_parallel_linear.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/engines/llm_cuda/engine.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

RetCode I8I8RowParallelLinearOp::CommonInit() {
    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
        output_shape->SetDataType(DATATYPE_FLOAT16);
        return RC_SUCCESS;
    };
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return nn::pmx::ReshapeRowParallelLinear(info, param_.get(), nccl_param_->size);
    };
    return RC_SUCCESS;
}

RetCode I8I8RowParallelLinearOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::pmx::RowParallelLinearParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    nccl_param_ = options.device->GetTensorParallelNcclParam();

    return CommonInit();
}

KernelImpl* I8I8RowParallelLinearOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<I8I8RowParallelLinearKernel>(param_.get());
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode I8I8RowParallelLinearOp::SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = pmx::CreateRowParallelLinearParam(builder, 
        param_.get()->in_features,
        param_.get()->out_features,
        param_.get()->bias_term,
        param_.get()->input_is_parallel);
    auto fb_op_param = pmx::CreateOpParam(builder, pmx::OpParamType_RowParallelLinearParam, fb_param.Union());
    pmx::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode I8I8RowParallelLinearOp::DeserializeData(const ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = pmx::GetOpParam(base);
    auto fb_param = fb_op_param->value_as_RowParallelLinearParam();
    param_ = make_shared<ppl::nn::pmx::RowParallelLinearParam>();
    param_.get()->in_features       = fb_param->in_features();
    param_.get()->out_features      = fb_param->out_features();
    param_.get()->bias_term         = fb_param->bias_term();
    param_.get()->input_is_parallel = fb_param->input_is_parallel();
    
    nccl_param_ = dynamic_cast<LlmCudaEngine*>(ctx.engine)->GetTensorParallelNcclParam();

    return CommonInit();
}
#endif


}}}}} // namespace ppl::nn::llm::cuda::pmx
