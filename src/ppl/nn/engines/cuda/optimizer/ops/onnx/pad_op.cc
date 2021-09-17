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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/pad_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/pad_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_pad.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

RetCode PadOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<PadParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

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
        shape->SetDataType(DATATYPE_INT64);
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        const TensorShape& shape = info->GetInput<TensorImpl>(0)->GetShape();
        uint32_t dim_count = shape.GetDimCount();
        auto pad = info->GetInput<TensorImpl>(1);
        if (pad->GetShape().GetDimCount() != 1 || pad->GetShape().GetDim(0) != 2 * dim_count ||
            pad->GetShape().GetDataType() != DATATYPE_INT64) {
            return RC_INVALID_VALUE;
        }

        int pad_elems = pad->GetShape().GetElementsIncludingPadding();
        unique_ptr<int64_t[]> pad_data(new int64_t[pad_elems]);
        for (int it = 0; it < pad_elems; pad_data[it] = 0, ++it)
            ;
        if (pad->GetBufferPtr() != nullptr) {
            auto status = pad->CopyToHost(pad_data.get());
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy pad data failed: " << GetRetCodeStr(status);
                return status;
            }
        }

        return oputils::ReshapePad(info, &param_, pad_data.get(), pad_data.get() + dim_count);
    };

    return RC_SUCCESS;
}

RetCode PadOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* PadOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<PadKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
