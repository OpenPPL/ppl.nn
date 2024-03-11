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

#include "position_index_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/dynamic_batching/position_index_kernel.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

RetCode DynamicBatchingPositionIndexOp::CommonInit() {
    infer_type_and_format_func_ = GenericInferTypeAndFormat;
    infer_dims_func_ = GenericInferDims;
    return RC_SUCCESS;
}

RetCode DynamicBatchingPositionIndexOp::DoInit(const OptKernelOptions& options) {
    return CommonInit();
}

KernelImpl* DynamicBatchingPositionIndexOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<DynamicBatchingPositionIndexKernel>();
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode DynamicBatchingPositionIndexOp::SerializeData(const ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    return RC_SUCCESS;
}

ppl::common::RetCode DynamicBatchingPositionIndexOp::DeserializeData(const ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    return CommonInit();
}
#endif


}}}}} // namespace ppl::nn::llm::cuda::pmx
