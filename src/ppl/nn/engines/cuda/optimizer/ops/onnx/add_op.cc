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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/add_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/add_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_add.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode AddOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

AddOp::AddOp(const ir::Node* node) : CudaOptKernel(node) {

    infer_type_func_ = [this](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        uint64_t mask = 0;
        for (uint32_t i = 0; i < info->GetInputCount(); ++i) {        
            auto in_tensor = info->GetInput<TensorImpl>(i);
            if (in_tensor->GetType() == TENSORTYPE_RESERVED)
                mask |= 1<<i;
        }
        ppl::common::RetCode status;
        if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
        } else {
            status = InferHighestType(info, type, mask);
        }
        return status;
    };

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeAdd(info, nullptr);
    };
}

RetCode AddOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* AddOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<AddKernel>();
}

}}} // namespace ppl::nn::cuda
