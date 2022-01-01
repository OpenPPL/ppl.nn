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

#include "ppl/nn/engines/riscv/optimizer/ops/ppl/reorder_op.h"
#include "ppl/nn/engines/riscv/kernels/ppl/reorder_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode ReorderOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto& input = *info->GetInput<TensorImpl>(0)->GetShape();
        auto& output = *info->GetOutput<TensorImpl>(0)->GetShape();
        if (input.GetDimCount() != 4 && input.GetDimCount() != 2) {
            return RC_UNSUPPORTED;
        }

        if (input.IsScalar()) {
            output.ReshapeAsScalar();
        } else {
            output.Reshape(input.GetDims(), input.GetDimCount());
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

KernelImpl* ReorderOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ReorderKernel>();
}

}}} // namespace ppl::nn::riscv
