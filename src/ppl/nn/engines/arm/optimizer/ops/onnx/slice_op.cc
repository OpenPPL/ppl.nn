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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/slice_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/slice_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_slice.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

SliceOp::SliceOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        // check parameters
        if (info->GetOutputCount() != 1) {
            LOG(ERROR) << "output count[" << info->GetOutputCount() << "] != 1.";
            return RC_INVALID_VALUE;
        }

        if (info->GetInput<TensorImpl>(0)->GetShape()->GetDataType() != DATATYPE_FLOAT32 &&
            info->GetInput<TensorImpl>(0)->GetShape()->GetDataType() != DATATYPE_FLOAT16 &&
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

            // check data type support
            for (size_t i = 1; i < info->GetInputCount(); i++) {
                if (info->GetInput<TensorImpl>(i)->GetShape()->GetDataType() != DATATYPE_INT64) {
                    LOG(ERROR) << "starts, ends, axes & steps only support int64 now.";
                    return RC_UNSUPPORTED;
                }
            }

            // check data format support
            for (size_t i = 0; i < info->GetInputCount(); i++) {
                if (info->GetInput<TensorImpl>(i)->GetShape()->GetDataFormat() != DATAFORMAT_NDARRAY) {
                    LOG(ERROR) << "unsupported data format: "
                               << GetDataFormatStr(info->GetInput<TensorImpl>(i)->GetShape()->GetDataFormat());
                    return RC_UNSUPPORTED;
                }
            }

            for (size_t i = 1; i < info->GetInputCount(); i++) {
                if (info->GetInput<TensorImpl>(i)->GetShape()->GetDimCount() !=
                    1) { // starts, ends, axes, steps must be 1-D tensor
                    return RC_INVALID_VALUE;
                }
            }

            const uint32_t axes_num = info->GetInput<TensorImpl>(1)->GetShape()->GetDim(0);
            for (size_t i = 2; i < info->GetInputCount(); i++) {
                if (info->GetInput<TensorImpl>(i)->GetShape()->GetDim(0) !=
                    axes_num) { // starts, end, axes, steps must have same length except for not defined
                    return RC_INVALID_VALUE;
                }
            }

            this->slice_aux_param_.starts.resize(axes_num);
            this->slice_aux_param_.ends.resize(axes_num);
            this->slice_aux_param_.axes.resize(axes_num);
            this->slice_aux_param_.steps.resize(axes_num, 1);

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
                this->slice_aux_param_.starts[i] = starts[i];
                this->slice_aux_param_.ends[i] = ends[i];
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
                    this->slice_aux_param_.axes[i] = axes[i];
                }
            } else {
                for (uint32_t i = 0; i < axes_num; i++) {
                    this->slice_aux_param_.axes[i] = i;
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
                    this->slice_aux_param_.steps[i] = steps[i];
                }
            }
        } else {
            auto p = param_.get();
            const uint32_t axes_num = p->starts.size();

            this->slice_aux_param_.starts.resize(axes_num);
            this->slice_aux_param_.ends.resize(axes_num);
            this->slice_aux_param_.axes.resize(axes_num);
            this->slice_aux_param_.steps.resize(axes_num, 1);

            for (uint32_t i = 0; i < axes_num; ++i) {
                this->slice_aux_param_.starts[i] = p->starts[i];
            }

            if (p->ends.size() == axes_num) {
                for (uint32_t i = 0; i < axes_num; ++i) {
                    this->slice_aux_param_.ends[i] = p->ends[i];
                }
            } else {
                LOG(ERROR) << "ends.size[" << p->ends.size() << "] != axes_num[" << axes_num << "].";
            }

            if (p->axes.size()) {
                if (p->axes.size() != axes_num) {
                    LOG(ERROR) << "axes.size[" << p->axes.size() << "] != axes_num[" << axes_num << "].";
                }
                for (uint32_t i = 0; i < axes_num; ++i) {
                    this->slice_aux_param_.axes[i] = p->axes[i];
                }
            } else {
                for (uint32_t i = 0; i < axes_num; i++) {
                    this->slice_aux_param_.axes[i] = i;
                }
            }
        }

        auto ret = onnx::ReshapeSlice(info, this->slice_aux_param_.starts.data(), this->slice_aux_param_.ends.data(),
                                      this->slice_aux_param_.axes.data(), this->slice_aux_param_.steps.data(),
                                      this->slice_aux_param_.starts.size());
        if (ret != RC_SUCCESS) {
            return ret;
        }
        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;
}

RetCode SliceOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

ppl::common::RetCode SliceOp::SelectDataType(const InputOutputInfo& info,
                                             std::vector<ppl::common::datatype_t>* selected_input_types,
                                             std::vector<ppl::common::datatype_t>* selected_output_types,
                                             const ppl::common::datatype_t preferred_fp_datatype) {
    GenericSelectDataType(info, selected_input_types, selected_output_types, preferred_fp_datatype);
    for (int64_t i = 1; i < info.GetInputCount(); i++) {
        selected_input_types->at(i) = DATATYPE_INT64;
    }
    return RC_SUCCESS;
}

KernelImpl* SliceOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SliceKernel>(&slice_aux_param_);
}

}}} // namespace ppl::nn::arm
