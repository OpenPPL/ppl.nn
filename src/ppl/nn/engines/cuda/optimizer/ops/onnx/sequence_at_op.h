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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_SEQUENCE_AT_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPS_ONNX_SEQUENCE_AT_OP_H_

#include "ppl/nn/engines/common/onnx/sequence_at_op.h"

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace cuda {

class SequenceAtOp final : public CudaOptKernel {
public:
    SequenceAtOp(const ir::Node* node) : CudaOptKernel(node), op_(node) {}

    ppl::common::RetCode Init(const OptKernelOptions&) override {
        infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant,
                                  datatype_t type) -> RetCode {
            if (type == DATATYPE_UNKNOWN) {
                return InferInheritedType(info);
            } else if (type == DATATYPE_INT8) {
                auto status = CopyQuantType(info, quant);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Set sequence quantization failed.";
                    return RC_INVALID_VALUE;
                }
            }
            return InferDefaultType(info, type);
        };

        infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
            auto in_shape = &info->GetInput<TensorImpl>(0)->GetShape();
            for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
                auto out_shape = &info->GetOutput<TensorImpl>(0)->GetShape();
                out_shape->Reshape(in_shape->GetDims(), in_shape->GetRealDimCount());
            }
            return ppl::common::RC_SUCCESS;
        };
        return ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode Finalize(const OptKernelOptions&) override {
        return ppl::common::RC_SUCCESS;
    }

    KernelImpl* CreateKernelImpl() const override {
        return op_.CreateKernelImpl();
    }

private:
    common::SequenceAtOp op_;
};

}}} // namespace ppl::nn::cuda

#endif
