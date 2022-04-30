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

#include "ppl/nn/engines/x86/optimizer/ops/onnx/slice_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/slice_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_slice.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode SliceOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (info->GetOutputCount() != 1) {
            LOG(ERROR) << "output count[" << info->GetOutputCount() << "] != 1.";
            return RC_INVALID_VALUE;
        }

        if (info->GetInput<TensorImpl>(0)->GetShape()->GetDataType() != DATATYPE_FLOAT32 &&
            info->GetInput<TensorImpl>(0)->GetShape()->GetDataType() != DATATYPE_INT64) {
            LOG(ERROR) << "unsupported data type: "
                       << GetDataTypeStr(info->GetInput<TensorImpl>(0)->GetShape()->GetDataType());
            return RC_UNSUPPORTED;
        }

        if (info->GetInputCount() > 1) {
            if (info->GetInputCount() < 3 || info->GetInputCount() > 5) {
                LOG(ERROR) << "input count[" << info->GetInputCount() << "] is out of range[3, 5].";
                return RC_INVALID_VALUE;
            }
            for (uint32_t i = 1; i < info->GetInputCount(); i++) {
                if (info->GetInput<TensorImpl>(i)->GetShape()->GetDataType() != DATATYPE_INT64) {
                    LOG(ERROR) << "starts, ends, axes & steps only support int64 now.";
                    return RC_UNSUPPORTED;
                }
            }
            for (uint32_t i = 1; i < info->GetInputCount(); i++) {
                // starts, ends, axes, steps must be 1-D tensor
                auto in_shape = info->GetInput<TensorImpl>(i)->GetShape();
                if (in_shape->GetDimCount() != 1) {
                    LOG(ERROR) << "input[" << i << "]'s dim count[" << in_shape->GetDimCount() << "] != 1.";
                    return RC_INVALID_VALUE;
                }
            }

            const uint32_t axes_num = info->GetInput<TensorImpl>(1)->GetShape()->GetDimCount();

            for (uint32_t i = 2; i < info->GetInputCount(); i++) {
                // starts, end, axes, steps must have same length except for not defined
                auto in_shape = info->GetInput<TensorImpl>(i)->GetShape();
                if (in_shape->GetDim(0) != axes_num) {
                    LOG(ERROR) << "input[" << i << "]'s dim[0]'s value[" << in_shape->GetDim(0) << "] != axes_num["
                               << axes_num << "].";
                    return RC_INVALID_VALUE;
                }
            }

            this->aux_param_.starts.resize(axes_num);
            this->aux_param_.ends.resize(axes_num);
            this->aux_param_.axes.resize(axes_num);
            this->aux_param_.steps.resize(axes_num, 1);

            // prepare starts, ends, axes, steps
            auto starts = info->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
            auto ends = info->GetInput<TensorImpl>(2)->GetBufferPtr<int64_t>();

            if (!starts) {
                LOG(ERROR) << "input[1] is empty.";
                return RC_NOT_FOUND;
            }
            if (!ends) {
                LOG(ERROR) << "input[2] is empty.";
                return RC_NOT_FOUND;
            }

            for (uint32_t i = 0; i < axes_num; ++i) {
                this->aux_param_.starts[i] = starts[i];
                this->aux_param_.ends[i] = ends[i];
            }

            const int64_t* axes = nullptr;
            auto axes_tensor = info->GetInputCount() > 3 ? info->GetInput<TensorImpl>(3) : nullptr;
            if (axes_tensor) {
                axes = axes_tensor->GetBufferPtr<int64_t>();
                if (!axes) {
                    LOG(ERROR) << "`axes` is empty.";
                    return RC_NOT_FOUND;
                }
                for (uint32_t i = 0; i < axes_num; ++i) {
                    this->aux_param_.axes[i] = axes[i];
                }
            } else {
                for (uint32_t i = 0; i < axes_num; i++) {
                    this->aux_param_.axes[i] = i;
                }
            }

            const int64_t* steps = nullptr;
            vector<int64_t> steps_vec;
            auto steps_tensor = info->GetInputCount() > 4 ? info->GetInput<TensorImpl>(4) : nullptr;
            if (steps_tensor) {
                steps = steps_tensor->GetBufferPtr<int64_t>();
                if (!steps) {
                    LOG(ERROR) << "`steps` is empty.";
                    return RC_NOT_FOUND;
                }
                for (uint32_t i = 0; i < axes_num; ++i) {
                    this->aux_param_.steps[i] = steps[i];
                }
            }
        } else {
            auto p = param_.get();
            const uint32_t axes_num = p->starts.size();

            this->aux_param_.starts.resize(axes_num);
            this->aux_param_.ends.resize(axes_num);
            this->aux_param_.axes.resize(axes_num);
            this->aux_param_.steps.resize(axes_num, 1);

            for (uint32_t i = 0; i < axes_num; ++i) {
                this->aux_param_.starts[i] = p->starts[i];
            }

            if (p->ends.size() == axes_num) {
                for (uint32_t i = 0; i < axes_num; ++i) {
                    this->aux_param_.ends[i] = p->ends[i];
                }
            } else {
                LOG(ERROR) << "ends.size[" << p->ends.size() << "] != axes_num[" << axes_num << "].";
            }

            if (p->axes.size()) {
                if (p->axes.size() != axes_num) {
                    LOG(ERROR) << "axes.size[" << p->axes.size() << "] != axes_num[" << axes_num << "].";
                }
                for (uint32_t i = 0; i < axes_num; ++i) {
                    this->aux_param_.axes[i] = p->axes[i];
                }
            } else {
                for (uint32_t i = 0; i < axes_num; i++) {
                    this->aux_param_.axes[i] = i;
                }
            }
        }

        auto ret = onnx::ReshapeSlice(info, this->aux_param_.starts.data(), this->aux_param_.ends.data(),
                                      this->aux_param_.axes.data(), this->aux_param_.steps.data(),
                                      this->aux_param_.starts.size());
        if (ret != RC_SUCCESS) {
            return ret;
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

KernelImpl* SliceOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SliceKernel>(&aux_param_);
}

}}} // namespace ppl::nn::x86
