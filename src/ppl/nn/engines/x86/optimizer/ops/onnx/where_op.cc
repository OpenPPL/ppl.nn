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

#include "ppl/nn/engines/x86/optimizer/ops/onnx/where_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/where_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_where.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode WhereOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeWhere(info, nullptr);
    };

    infer_type_func_ = [](InputOutputInfo* info) -> void {
        GenericInferType(info);
        info->GetOutput<TensorImpl>(0)->GetShape()->SetDataType(info->GetInput<TensorImpl>(1)->GetShape()->GetDataType());
    };

    return RC_SUCCESS;
}

KernelImpl* WhereOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<WhereKernel>();
}

}}} // namespace ppl::nn::x86
