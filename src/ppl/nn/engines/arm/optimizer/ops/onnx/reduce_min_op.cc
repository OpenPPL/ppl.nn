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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/reduce_min_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/reduce_min_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_reduce.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ReduceMinOp::ReduceMinOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeReduce(info, param_.get());
    };

    infer_type_func_ = GenericInferType;
}

RetCode ReduceMinOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode ReduceMinOp::SelectFormat(const InputOutputInfo& info,
                                  std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                  std::vector<ppl::common::dataformat_t>* selected_output_formats) {
    const TensorShape& input_shape = *info.GetInput<TensorImpl>(0)->GetShape();
    ppl::common::dataformat_t selected_data_format = ppl::common::DATAFORMAT_NDARRAY;
    const int64_t dim_count = input_shape.GetDimCount();

    if (dim_count > 0) { // dims has been infered
        if (input_shape.GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) { // for NBCX
            if (param_->keepdims == true) {
                selected_data_format = input_shape.GetDataFormat();
            } else {
                const int64_t remain_dim_count = dim_count - param_->axes.size();
                if (remain_dim_count >= 3) {
                    bool no_reduce_on_batch_channel_dim = true;
                    for (auto axis : param_->axes) {
                        if (axis == 0 || axis + dim_count == 0 || axis == 1 || axis + dim_count == 1) {
                            no_reduce_on_batch_channel_dim = false;
                            break;
                        }
                    }
                    if (no_reduce_on_batch_channel_dim) {
                        selected_data_format = input_shape.GetDataFormat();
                    }
                }
            }
        }
    }

    selected_input_formats->at(0) = selected_output_formats->at(0) = selected_data_format;
    return RC_SUCCESS;
}

KernelImpl* ReduceMinOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ReduceMinKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
