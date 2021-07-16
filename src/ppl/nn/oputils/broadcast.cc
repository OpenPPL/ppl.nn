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

#include "ppl/nn/oputils/broadcast.h"
#include <deque>
#include <algorithm>
using namespace ppl::common;

namespace ppl { namespace nn { namespace oputils {

void MultiDirectionalBroadCaster::CalcBroadCast(void) {
    TensorShape* shape_min;
    TensorShape* shape_max;
    if (input_tensor_shape0_.GetDimCount() < input_tensor_shape1_.GetDimCount()) {
        shape_min = &input_tensor_shape0_;
        shape_max = &input_tensor_shape1_;
    } else {
        shape_min = &input_tensor_shape1_;
        shape_max = &input_tensor_shape0_;
    }
    const int32_t min_dims = shape_min->GetDimCount();
    const int32_t max_dims = shape_max->GetDimCount();

    output_tensor_shape_ = input_tensor_shape0_; // copy for datatype & dataformat

    if (shape_min->IsScalar()) { // one is scalar, will always broadcast
        if (shape_max->IsScalar()) {
            output_tensor_shape_.ReshapeAsScalar();
        } else {
            output_tensor_shape_.Reshape(shape_max->GetDims(), shape_max->GetDimCount());
        }
    } else if (min_dims == max_dims) { // same dimsCount
        for (int i = 0; i < min_dims; i++) {
            if (shape_min->GetDim(i) != shape_max->GetDim(i) && // shape not same
                shape_min->GetDim(i) != 1 && shape_max->GetDim(i) != 1) { // neither is 1
                can_broadcast_ = false;
                return;
            }
        }
        output_tensor_shape_.SetDimCount(min_dims); // can broadcast
        for (int i = 0; i < min_dims; i++) {
            int64_t out_dim_val = (shape_min->GetDim(i) == 0 || shape_max->GetDim(i) == 0)
                ? 0
                : std::max(shape_min->GetDim(i), shape_max->GetDim(i));
            output_tensor_shape_.SetDim(i, out_dim_val);
        }
    } else { // dimsCount not same
        TensorShape shape_min_pad = *shape_min;
        shape_min_pad.SetDimCount(max_dims);

        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - min_dims;
        for (int i = 0; i < offset; i++) {
            shape_min_pad.SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            shape_min_pad.SetDim(i, shape_min->GetDim(i - offset));
        }

        for (int i = 0; i < max_dims; i++) {
            if (shape_min_pad.GetDim(i) != shape_max->GetDim(i) && // shape not same
                shape_min_pad.GetDim(i) != 1 && shape_max->GetDim(i) != 1) { // neither is 1
                can_broadcast_ = false;
                return;
            }
        }
        output_tensor_shape_.SetDimCount(max_dims); // can broadcast
        for (int i = 0; i < max_dims; i++) {
            // output_tensor_shape_.SetDim(i, std::max(shape_min_pad.GetDim(i), shape_max->GetDim(i)));
            uint64_t output_dim; // dim may be 0
            if (shape_min_pad.GetDim(i) == 1) {
                output_dim = shape_max->GetDim(i);
            } else if (shape_max->GetDim(i) == 1) {
                output_dim = shape_min_pad.GetDim(i);
            } else {
                output_dim = shape_min_pad.GetDim(i);
            }
            output_tensor_shape_.SetDim(i, output_dim);
        }
    }

    can_broadcast_ = true;
    output_tensor_shape_.CalcPadding();
}

void MultiInputBroadCaster::CalcBroadCast(void) {
    if (input_tensor_shapes_.size() <= 1) {
        can_broadcast_ = true;
        return;
    }

    TensorShape output_shape = input_tensor_shapes_[0];
    for (size_t i = 1; i < input_tensor_shapes_.size(); i++) {
        MultiDirectionalBroadCaster two_tensor_broad_caster;
        two_tensor_broad_caster.SetInputTensorShapes(output_shape, input_tensor_shapes_[i]);
        if (!two_tensor_broad_caster.CanBroadCast()) {
            can_broadcast_ = false;
            return;
        }
        output_shape = two_tensor_broad_caster.OutputTensorShape();
    }

    can_broadcast_ = true;
    output_tensor_shape_ = output_shape;
}

void MatMulBroadCaster::CalcBroadCast(void) {
    can_broadcast_ = false;

    if (input_tensor_shape0_.GetRealDimCount() == 0 || input_tensor_shape1_.GetRealDimCount() == 0) { // no shape tensor
        return;
    }
    if (input_tensor_shape0_.IsScalar() || input_tensor_shape1_.IsScalar()) { // scalar matmul is not allowed
        return;
    }

    if (input_tensor_shape0_.GetDimCount() == 2 && input_tensor_shape1_.GetDimCount() == 2) { // normal gemm
        const uint32_t m = input_tensor_shape0_.GetDim(0);
        const uint32_t k0 = input_tensor_shape0_.GetDim(1);
        const uint32_t k1 = input_tensor_shape1_.GetDim(0);
        const uint32_t n = input_tensor_shape1_.GetDim(1);
        if (k0 == k1) {
            can_broadcast_ = true;
            output_tensor_shape_ = input_tensor_shape0_; // copy for datatype & dataformat
            output_tensor_shape_.Reshape({m, n});
        }
        return;
    }

    std::deque<uint32_t> input_dims0(input_tensor_shape0_.GetDims(),
                                     input_tensor_shape0_.GetDims() + input_tensor_shape0_.GetDimCount());
    std::deque<uint32_t> input_dims1(input_tensor_shape1_.GetDims(),
                                     input_tensor_shape1_.GetDims() + input_tensor_shape1_.GetDimCount());
    bool pad0 = false;
    bool pad1 = false;
    if (input_dims0.size() == 1) { // if A is 1-D Tensor, pad 1 to front
        pad0 = true;
        input_dims0.push_front(1);
    }
    if (input_dims1.size() == 1) { // if B is 1-D Tensor, pad 1 to end
        pad1 = true;
        input_dims1.push_back(1);
    }

    const int64_t max_dim_count = std::max(input_dims0.size(), input_dims1.size());
    while ((int64_t)input_dims0.size() < max_dim_count) {
        input_dims0.push_front(1);
    }
    while ((int64_t)input_dims1.size() < max_dim_count) {
        input_dims1.push_front(1);
    }

    std::vector<int64_t> output_dims(max_dim_count);

    // process last 2 dims
    const uint32_t m = input_dims0[max_dim_count - 2];
    const uint32_t k0 = input_dims0[max_dim_count - 1];
    const uint32_t k1 = input_dims1[max_dim_count - 2];
    const uint32_t n = input_dims1[max_dim_count - 1];
    if (k0 != k1) {
        return;
    }
    output_dims[max_dim_count - 2] = m;
    output_dims[max_dim_count - 1] = n;

    for (int64_t i = 0; i < max_dim_count - 2; i++) { // process higher dims
        if (input_dims0[i] == input_dims1[i]) {
            output_dims[i] = input_dims0[i];
        } else if (input_dims0[i] == 1) {
            output_dims[i] = input_dims1[i];
        } else if (input_dims1[i] == 1) {
            output_dims[i] = input_dims0[i];
        } else {
            return;
        }
    }

    if (pad0) { // remove 1-D Tensor padding
        output_dims.erase(output_dims.end() - 2);
    }
    if (pad1) {
        output_dims.erase(output_dims.end() - 1);
    }

    can_broadcast_ = true;
    output_tensor_shape_ = input_tensor_shape0_; // copy for datatype & dataformat
    output_tensor_shape_.Reshape(output_dims);
}

}}} // namespace ppl::nn::oputils
