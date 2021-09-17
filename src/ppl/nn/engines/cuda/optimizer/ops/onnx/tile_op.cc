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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/tile_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/tile_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_tile.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode TileOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [this](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        auto shape = &info->GetInput<TensorImpl>(1)->GetShape();
        shape->SetDataType(ppl::common::DATATYPE_INT64);
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        const TensorShape& shape = info->GetInput<TensorImpl>(0)->GetShape();
        uint32_t dim_count = shape.GetDimCount();
        auto repeat = info->GetInput<TensorImpl>(1);
        if (repeat->GetBufferPtr<void>() == nullptr) {
            return RC_NOT_FOUND;
        }
        if (repeat->GetShape().GetDimCount() != 1 || repeat->GetShape().GetDim(0) != dim_count ||
            repeat->GetShape().GetDataType() != DATATYPE_INT64) {
            return RC_INVALID_VALUE;
        }

        unique_ptr<int64_t[]> repeat_data(new int64_t[repeat->GetShape().GetElementsIncludingPadding()]);
        auto status = repeat->CopyToHost(repeat_data.get());
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Copy repeat data failed: " << GetRetCodeStr(status);
            return status;
        }

        return oputils::ReshapeTile(info, nullptr, repeat_data.get());
    };

    return RC_SUCCESS;
}

RetCode TileOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* TileOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<TileKernel>();
}

}}} // namespace ppl::nn::cuda
