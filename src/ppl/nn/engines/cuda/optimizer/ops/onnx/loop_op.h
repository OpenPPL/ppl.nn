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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_LOOP_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_LOOP_OP_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/engines/common/onnx/loop_op.h"

namespace ppl { namespace nn { namespace cuda {

class LoopOp final : public CudaOptKernel {
public:
    LoopOp(const ir::Node* node) : CudaOptKernel(node), op_(node) {}

    ppl::common::RetCode Init(const OptKernelOptions& options) override;

    ppl::common::RetCode Finalize(const OptKernelOptions&) override {
        return ppl::common::RC_SUCCESS;
    }

    KernelImpl* CreateKernelImpl() const override;
#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode SerializeData(const pmx::SerializationContext&, utils::DataStream*) const override {
        return ppl::common::RC_UNSUPPORTED;
    };
    ppl::common::RetCode DeserializeData(const pmx::DeserializationContext&, const void*, uint64_t) override {
        return ppl::common::RC_UNSUPPORTED;
    };
#endif

private:
    onnx::LoopOp op_;
};

}}} // namespace ppl::nn::cuda

#endif
