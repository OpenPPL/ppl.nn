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

#include "parallel_embedding_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/parallel_embedding_kernel.h"
#include "ppl/nn/oputils/pmx/reshape_parallel_embedding.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

RetCode ParallelEmbeddingOp::CommonInit() {
    infer_type_and_format_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto weight_shape = info->GetInput<TensorImpl>(1)->GetShape();
        auto output_shape = info->GetOutput<TensorImpl>(0)->GetShape();

        output_shape->SetDataFormat(DATAFORMAT_NDARRAY);
        output_shape->SetDataType(weight_shape->GetDataType());

        return RC_SUCCESS;
    };
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return nn::pmx::ReshapeParallelEmbedding(info, param_.get(), nccl_param_->size);
    };
    return RC_SUCCESS;
}

RetCode ParallelEmbeddingOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ParallelEmbeddingParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }
    nccl_param_ = options.device->GetTensorParallelNcclParam();

    return CommonInit();
}

KernelImpl* ParallelEmbeddingOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ParallelEmbeddingKernel>(param_.get());
}



}}}}} // namespace ppl::nn::llm::cuda::pmx
