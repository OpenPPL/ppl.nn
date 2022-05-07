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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/tile_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/tile_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_tile.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

TileOp::TileOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        auto ret = onnx::ReshapeTile(info, nullptr);
        if (ret != RC_SUCCESS) {
            return ret;
        }
        auto A = info->GetInput<TensorImpl>(0)->GetShape();
        auto B = info->GetInput<TensorImpl>(1)->GetShape();
        auto C = info->GetOutput<TensorImpl>(0)->GetShape();

        if (A->GetDataFormat() != DATAFORMAT_NDARRAY || B->GetDataFormat() != DATAFORMAT_NDARRAY ||
            C->GetDataFormat() != DATAFORMAT_NDARRAY) {
            LOG(ERROR) << "only support ndarray now.";
            return RC_UNSUPPORTED;
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;
}

RetCode TileOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

KernelImpl* TileOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<TileKernel>();
}

}}} // namespace ppl::nn::arm
