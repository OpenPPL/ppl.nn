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

#include "ppl/nn/oputils/onnx/reshape_squeeze.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
#include <set>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ReshapeSqueeze(InputOutputInfo* info, const ir::Attr* arg, const int64_t* axes) {
    uint32_t axes_size = 0;
    if (axes) {
        axes_size = info->GetInput<TensorImpl>(1)->GetShape()->GetDim(0);
    }

    if (info->GetInputCount() > 2 || info->GetOutputCount() != 1) {
        LOG(DEBUG) << "ERROR: input count[" << info->GetInputCount() << "] > 2 or output count["
                   << info->GetOutputCount() << "] != 1.";
        return RC_INVALID_VALUE;
    }

    const TensorShape& input = *info->GetInput<TensorImpl>(0)->GetShape();
    auto output = info->GetOutput<TensorImpl>(0)->GetShape();

    if (input.IsScalar()) {
        if (axes_size == 0 || (axes_size == 1 && (axes[0] == 0 || axes[0] == -1))) {
            output->ReshapeAsScalar();
        } else {
            LOG(DEBUG) << "ERROR: axes in parameter are invalid.";
            return RC_INVALID_VALUE;
        }
    } else {
        // check parameters
        if (axes_size > input.GetDimCount()) {
            LOG(DEBUG) << "ERROR: axes' size[" << axes_size << "] != input[0]'s dim count["
                       << input.GetDimCount() << "].";
            return RC_INVALID_VALUE;
        }
        for (uint32_t i = 0; i < axes_size; i++) {
            if (axes[i] >= (int32_t)input.GetDimCount() || axes[i] < -(int32_t)input.GetDimCount()) {
                LOG(DEBUG) << "ERROR: axes[" << i << "] is out of range[" << -(int)input.GetDimCount() << ", "
                           << input.GetDimCount() << "].";
                return RC_INVALID_VALUE;
            }
        }
        // calc real axes
        set<uint32_t> real_axes;
        if (axes_size == 0) { // squeeze axes is not set, default squeeze all 1 dims
            for (size_t i = 0; i < input.GetDimCount(); i++) {
                if (input.GetDim(i) == 1) {
                    real_axes.insert(i);
                }
            }
        } else {
            for (uint32_t i = 0; i < axes_size; i++) { // change negative dim to positive
                real_axes.insert(axes[i] >= 0 ? axes[i] : axes[i] + input.GetDimCount());
            }
            for (auto it = real_axes.begin(); it != real_axes.end(); ++it) { // check if all squeeze dims are 1
                if (input.GetDim(*it) != 1) {
                    LOG(DEBUG) << "ERROR: input[0]'s dim[" << *it << "]'s value[" << input.GetDim(*it) << "] != 1.";
                    return RC_INVALID_VALUE;
                }
            }
        }
        // set output axes
        vector<int64_t> output_axes;
        for (size_t i = 0; i < input.GetDimCount(); i++) {
            if (real_axes.find(i) == real_axes.end()) {
                output_axes.push_back(input.GetDim(i));
            }
        }
        if (output_axes.empty()) {
            output->ReshapeAsScalar();
        } else {
            output->Reshape(output_axes);
        }
        output->CalcPadding();
    }

    return RC_SUCCESS;
}

RetCode ReshapeSqueeze(InputOutputInfo* info, const ir::Attr* arg) {
    if (info->GetInputCount() > 2) {
        LOG(DEBUG) << "ERROR: input count[" << info->GetInputCount() << "] > 2.";
        return RC_INVALID_VALUE;
    }
    if (info->GetInputCount() == 1) {
        return onnx::ReshapeSqueeze(info, arg, nullptr);
    }

    auto axes = info->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
    return ReshapeSqueeze(info, arg, axes);
}

}}} // namespace ppl::nn::onnx
