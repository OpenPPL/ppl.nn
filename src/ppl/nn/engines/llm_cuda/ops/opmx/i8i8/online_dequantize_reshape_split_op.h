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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPS_OPMX_I8I8_ONLINE_DEQUANTIZE_RESHAPE_SPLIT_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPS_OPMX_I8I8_ONLINE_DEQUANTIZE_RESHAPE_SPLIT_OP_H_

#include "online_dequantize_op.h"
#include "ppl/nn/params/onnx/split_param.h"
#include "ppl/nn/engines/llm_cuda/opt_kernel.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

class I8I8OnlineDequantizeReshapeSplitOp final : public LlmCudaOptKernel {
public:
    I8I8OnlineDequantizeReshapeSplitOp(const ir::Node* node) : LlmCudaOptKernel(node) {}

    struct Param {
        std::shared_ptr<ppl::nn::onnx::SplitParam> split_param;
        std::vector<int64_t> split;
        std::vector<int64_t> shape;
        bool bias_term = false;
    };

    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode DoInit(const OptKernelOptions&) override;

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream*) const override;
    ppl::common::RetCode DeserializeData(const ppl::nn::pmx::DeserializationContext&, const void*, uint64_t) override;
#endif

    Param* GetParam() { return &param_; }
    const Param* GetParam() const { return &param_; }

private:
    ppl::common::RetCode CommonInit();
    
    Param param_;
};

}}}}} // namespace ppl::nn::llm::cuda::opmx

#endif
