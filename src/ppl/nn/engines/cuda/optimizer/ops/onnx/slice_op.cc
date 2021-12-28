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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/slice_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/slice_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_slice.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode SliceOp::Init(const OptKernelOptions& options) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        for (uint32_t i = 1; i < 5; ++i) {
            if (info->GetInputCount() >= i) {
                auto shape = &info->GetInput<TensorImpl>(i)->GetShape();
                shape->SetDataType(ppl::common::DATATYPE_INT64);
            }
        }
        return status;
    };

    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        SliceKernelParam kernel_param;

        const TensorShape& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
        int dim_count = in_shape0.GetDimCount();
        int input_count = info->GetInputCount();
        { // starts
            auto input = info->GetInput<TensorImpl>(1);
            if (input->GetBufferPtr() == nullptr) {
                return RC_NOT_FOUND;
            }
            auto status = input->CopyToHost(kernel_param.starts);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy starts input failed: " << GetRetCodeStr(status);
                return status;
            }
        }
        { // ends
            auto input = info->GetInput<TensorImpl>(2);
            if (input->GetBufferPtr() == nullptr) {
                return RC_NOT_FOUND;
            }
            auto status = input->CopyToHost(kernel_param.ends);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy ends input failed: " << GetRetCodeStr(status);
                return status;
            }
        }
        if (input_count >= 4) { // axes
            auto input = info->GetInput<TensorImpl>(3);
            if (input->GetBufferPtr() == nullptr) {
                return RC_NOT_FOUND;
            }
            auto status = input->CopyToHost(kernel_param.axes);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy axes input failed: " << GetRetCodeStr(status);
                return status;
            }
            kernel_param.axes_num = input->GetShape().GetElementsIncludingPadding();
        } else {
            for (int it = 0; it < dim_count; ++it) {
                kernel_param.axes[it] = it;
            }
            kernel_param.axes_num = dim_count;
        }
        if (input_count >= 5) { // steps
            auto input = info->GetInput<TensorImpl>(4);
            if (input->GetBufferPtr() == nullptr) {
                return RC_NOT_FOUND;
            }
            auto status = input->CopyToHost(kernel_param.steps);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy steps input failed: " << GetRetCodeStr(status);
                return status;
            }
        } else {
            for (int it = 0; it < dim_count; ++it) {
                kernel_param.steps[it] = 1;
            }
        }

        for (int it = 0; it < kernel_param.axes_num; ++it) {
            int64_t axis = kernel_param.axes[it];
            int64_t start_val = kernel_param.starts[it];
            int64_t end_val = kernel_param.ends[it];
            // int step_val = kernel_param.steps[it];
            // int cur_dim_size = in_shape0.GetDim(axis);
            int cur_dim_size = in_shape0.GetDim((axis + dim_count) % dim_count);
            if (start_val == INT_MIN) {
                start_val = 0;
            }
            if (start_val == INT_MAX || start_val > cur_dim_size) {
                start_val = cur_dim_size;
            }
            if (start_val < 0) {
                start_val = cur_dim_size + start_val;
            }
            if (end_val == INT_MIN) {
                end_val = 0;
            }
            if (end_val == INT_MAX || end_val > cur_dim_size) {
                end_val = cur_dim_size;
            }
            if (end_val < 0) {
                end_val = cur_dim_size + end_val;
            }
            kernel_param.starts[it] = start_val;
            kernel_param.ends[it] = end_val;
        }
        return oputils::ReshapeSlice(info, kernel_param.starts, kernel_param.ends, kernel_param.axes,
                                     kernel_param.steps);
    };

    return RC_SUCCESS;
}

RetCode SliceOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* SliceOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<SliceKernel>();
}

}}} // namespace ppl::nn::cuda
