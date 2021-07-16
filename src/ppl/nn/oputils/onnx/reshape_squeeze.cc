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
#include <set>
using namespace std;
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeSqueeze(InputOutputInfo* info, const void* arg) {
    auto param = (const SqueezeParam*)arg;
    if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }

    const TensorShape& input = info->GetInput<TensorImpl>(0)->GetShape();
    auto output = &info->GetOutput<TensorImpl>(0)->GetShape();

    if (input.IsScalar()) {
        if (param->axes.empty() || (param->axes.size() == 1 && (param->axes[0] == 0 || param->axes[0] == -1))) {
            output->ReshapeAsScalar();
        } else {
            return RC_INVALID_VALUE;
        }
    } else {
        // check parameters
        if (param->axes.size() > input.GetDimCount()) {
            return RC_INVALID_VALUE;
        }
        for (uint32_t i = 0; i < param->axes.size(); i++) {
            if (param->axes[i] >= (int32_t)input.GetDimCount() || param->axes[i] < -(int32_t)input.GetDimCount()) {
                return RC_INVALID_VALUE;
            }
        }
        // calc real axes
        set<uint32_t> real_axes;
        if (param->axes.empty()) { // squeeze axes is not set, default squeeze all 1 dims
            for (size_t i = 0; i < input.GetDimCount(); i++) {
                if (input.GetDim(i) == 1) {
                    real_axes.insert(i);
                }
            }
        } else {
            for (uint32_t i = 0; i < param->axes.size(); i++) { // change negative dim to positive
                real_axes.insert(param->axes[i] >= 0 ? param->axes[i] : param->axes[i] + input.GetDimCount());
            }
            for (auto it = real_axes.begin(); it != real_axes.end(); ++it) { // check if all squeeze dims are 1
                if (input.GetDim(*it) != 1) {
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
            output->Reshape({1});
        } else {
            output->Reshape(output_axes);
        }
        output->CalcPadding();
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
