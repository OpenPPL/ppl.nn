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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/less_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/less_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_less.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode LessOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        auto& in0_shape = info->GetInput<TensorImpl>(0)->GetShape();
        auto& in1_shape = info->GetInput<TensorImpl>(1)->GetShape();
        auto& out_shape = info->GetOutput<TensorImpl>(0)->GetShape();
        out_shape.SetDataType(DATATYPE_BOOL);
        if (type == DATATYPE_INT8) {
            auto in0_edge_id = info->GetInput<TensorImpl>(0)->GetEdge()->GetId();
            auto& in0_quant = quant->at(in0_edge_id);
            auto in1_edge_id = info->GetInput<TensorImpl>(1)->GetEdge()->GetId();
            auto& in1_quant = quant->at(in1_edge_id);
            if (in0_quant.type != DATATYPE_INT8 || in1_quant.type != DATATYPE_INT8) { // Do quantization
                return RC_INVALID_VALUE;
            }
            in0_shape.SetDataType(DATATYPE_INT8);
            in1_shape.SetDataType(DATATYPE_INT8);
        }

        if (in0_shape.GetDataType() != DATATYPE_INT8) {
            in1_shape.SetDataType(in0_shape.GetDataType());
        } else {
            in0_shape.SetDataType(in1_shape.GetDataType());
        }
        return RC_SUCCESS;
    };

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return oputils::ReshapeLess(info, nullptr);
    };

    return RC_SUCCESS;
}

RetCode LessOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* LessOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<LessKernel>();
}

}}} // namespace ppl::nn::cuda
