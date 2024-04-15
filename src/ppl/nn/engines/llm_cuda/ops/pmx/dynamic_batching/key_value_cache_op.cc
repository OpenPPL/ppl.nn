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

#include "key_value_cache_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/pmx/dynamic_batching/key_value_cache_kernel.h"
#include "ppl/nn/oputils/pmx/reshape_dynamic_batching_key_value_cache.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace pmx {

RetCode DynamicBatchingKeyValueCacheOp::CommonInit() {
    infer_type_and_format_func_ = GenericInferTypeAndFormat;
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto kvstarts = info->GetInput<TensorImpl>(3);
        int64_t kv_length = 0;

        if (!kvstarts->GetBufferPtr()) {
            ppl::nn::pmx::ReshapeDynamicBatchingKeyValueCache(info, param_.get(), kv_length);
            return RC_NOT_FOUND;
        }

        auto kv_length_addr = kvstarts->GetBufferPtr<int64_t>() + kvstarts->GetShape()->GetDim(0) - 1;
        auto kv_length_desc = ppl::nn::BufferDesc(kv_length_addr);
        auto status = kvstarts->GetDevice()->CopyToHost(&kv_length, kv_length_desc, sizeof(kv_length));

        if (status != RC_SUCCESS) {
            LOG(ERROR) << "kvstarts->GetDevice()->CopyToHost() failed: " << GetRetCodeStr(status);
            return status;
        }

        return ppl::nn::pmx::ReshapeDynamicBatchingKeyValueCache(info, param_.get(), kv_length);
    };
    return RC_SUCCESS;
}

RetCode DynamicBatchingKeyValueCacheOp::DoInit(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::pmx::KeyValueCacheParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenericLoadParam failed: " << GetRetCodeStr(status);
        return status;
    }

    LOG(ERROR) << "currently do not support this op";
    return ppl::common::RC_UNSUPPORTED;

    return CommonInit();
}

KernelImpl* DynamicBatchingKeyValueCacheOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<DynamicBatchingKeyValueCacheKernel>(param_.get());
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode DynamicBatchingKeyValueCacheOp::SerializeData(const ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_param = pmx::CreateKeyValueCacheParam(builder, 
        param_.get()->num_layer,
        param_.get()->layer_idx,
        param_.get()->quant_bit,
        param_.get()->quant_group,
        param_.get()->num_repeat,
        param_.get()->cache_mode,
        param_.get()->cache_layout);
    auto fb_op_param = pmx::CreateOpParam(builder, pmx::OpParamType_KeyValueCacheParam, fb_param.Union());
    pmx::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode DynamicBatchingKeyValueCacheOp::DeserializeData(const ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = pmx::GetOpParam(base);
    auto fb_param = fb_op_param->value_as_KeyValueCacheParam();
    param_ = make_shared<ppl::nn::pmx::KeyValueCacheParam>();
    param_.get()->num_layer    = fb_param->num_layer();
    param_.get()->layer_idx    = fb_param->layer_idx();
    param_.get()->quant_bit    = fb_param->quant_bit();
    param_.get()->quant_group  = fb_param->quant_group();
    param_.get()->num_repeat   = fb_param->num_repeat();
    param_.get()->cache_mode   = fb_param->cache_mode();
    param_.get()->cache_layout = fb_param->cache_layout();
    
    return CommonInit();
}
#endif


}}}}} // namespace ppl::nn::llm::cuda::pmx
