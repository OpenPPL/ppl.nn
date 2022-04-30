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

#include "ppl/nn/oputils/onnx/reshape_slice.h"
#include <limits.h>
#include <cmath>
#include "ppl/common/log.h"
using namespace std;
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ReshapeSlice(InputOutputInfo* info, const int64_t* starts, const int64_t* ends, const int64_t* axes,
                     const int64_t* steps, const int64_t axes_num) {
    const TensorShape& shape = *info->GetInput<TensorImpl>(0)->GetShape();
    int dim_count = shape.GetDimCount();

    vector<int64_t> output_dim(dim_count);
    for (int it = 0; it < dim_count; ++it) {
        output_dim[it] = shape.GetDim(it);
    }

    for (int it = 0; it < axes_num; ++it) {
        int64_t start_val = starts[it];
        int64_t end_val = ends[it];
        int64_t axis = axes[it];
        int64_t step_val = steps[it];
        if (axis < -dim_count || axis >= dim_count) {
            LOG(DEBUG) << "ERROR: axis[" << axis << "] is out of range[" << -dim_count << ", " << dim_count << "].";
            return RC_INVALID_VALUE;
        }
        axis = axis < 0 ? axis + dim_count : axis;
        if (step_val == 0) {
            LOG(DEBUG) << "ERROR: step is 0.";
            return RC_INVALID_VALUE;
        }

        int64_t cur_dim_size = output_dim[axis];
        if (start_val == LONG_MIN)
            start_val = 0;
        if (start_val == LONG_MAX || start_val > cur_dim_size)
            start_val = cur_dim_size;
        if (start_val < 0)
            start_val = cur_dim_size + start_val;
        if (end_val == LONG_MAX || end_val > cur_dim_size)
            end_val = cur_dim_size;
        if (end_val < 0) {
            if (-end_val > cur_dim_size) {
                end_val = -1;
            } else {
                end_val = cur_dim_size + end_val;
            }
        }
        output_dim[axis] = (int)(std::ceil(((double)end_val - start_val) / step_val));
    }
    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(output_dim);

    return RC_SUCCESS;
}

RetCode ReshapeSlice(InputOutputInfo* info) {
    if (info->GetInputCount() < 3 || info->GetInputCount() > 5) {
        LOG(DEBUG) << "ERROR: input count[" << info->GetInputCount() << "] is out of range[3, 5].";
        return RC_INVALID_VALUE;
    }
    if (info->GetOutputCount() != 1) {
        LOG(DEBUG) << "ERROR: output count[" << info->GetOutputCount() << "] != 1.";
        return RC_INVALID_VALUE;
    }

    for (size_t i = 1; i < info->GetInputCount(); i++) {
        // starts, ends, axes, steps must be 1-D tensor
        auto in_shape = info->GetInput<TensorImpl>(i)->GetShape();
        if (in_shape->GetDimCount() != 1) {
            LOG(DEBUG) << "ERROR: input[" << i << "]'s dim count[" << in_shape->GetDimCount() << "] != 1.";
            return RC_INVALID_VALUE;
        }
    }
    const int axes_num = info->GetInput<TensorImpl>(1)->GetShape()->GetDim(0);
    for (size_t i = 2; i < info->GetInputCount(); i++) {
        // starts, end, axes, steps must have same length except for not defined
        auto in_shape = info->GetInput<TensorImpl>(i)->GetShape();
        if (in_shape->GetDim(0) != axes_num) {
            LOG(DEBUG) << "ERROR: input[" << i << "]'s dim[0]'s value[" << in_shape->GetDim(0) << "] != axes_num["
                       << axes_num << "].";
            return RC_INVALID_VALUE;
        }
    }

    // prepare starts, ends, axes, steps
    auto starts = info->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
    auto ends = info->GetInput<TensorImpl>(2)->GetBufferPtr<int64_t>();

    if (!starts) {
        LOG(DEBUG) << "ERROR: input[1] is empty.";
        return RC_NOT_FOUND;
    }
    if (!ends) {
        LOG(DEBUG) << "ERROR: input[2] is empty.";
        return RC_NOT_FOUND;
    }

    const int64_t* axes = nullptr;
    vector<int64_t> axes_vec;
    auto axes_tensor = info->GetInputCount() > 3 ? info->GetInput<TensorImpl>(3) : nullptr;
    if (axes_tensor) {
        axes = axes_tensor->GetBufferPtr<int64_t>();
        if (!axes) {
            LOG(DEBUG) << "ERROR: `axes` is empty.";
            return RC_NOT_FOUND;
        }
    } else {
        axes_vec.resize(axes_num);
        for (int i = 0; i < axes_num; i++) {
            axes_vec[i] = i;
        }
        axes = axes_vec.data();
    }

    const int64_t* steps = nullptr;
    vector<int64_t> steps_vec;
    auto steps_tensor = info->GetInputCount() > 4 ? info->GetInput<TensorImpl>(4) : nullptr;
    if (steps_tensor) {
        steps = steps_tensor->GetBufferPtr<int64_t>();
        if (!steps) {
            LOG(DEBUG) << "ERROR: `steps` is empty.";
            return RC_NOT_FOUND;
        }
    } else {
        steps_vec.resize(axes_num, 1);
        steps = steps_vec.data();
    }

    return ReshapeSlice(info, starts, ends, axes, steps, axes_num);
}

}}} // namespace ppl::nn::onnx
