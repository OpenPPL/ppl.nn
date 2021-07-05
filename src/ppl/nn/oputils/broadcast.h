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

#ifndef _ST_HPC_PPL_NN_OPUTILS_BROADCAST_H_
#define _ST_HPC_PPL_NN_OPUTILS_BROADCAST_H_

#include "ppl/nn/common/tensor_shape.h"
#include <vector>

namespace ppl { namespace nn { namespace oputils {

class BroadCaster {
public:
    BroadCaster() {}
    virtual ~BroadCaster() {}

    void SetInputTensorShape0(const TensorShape& input_tensor_shape0) {
        input_tensor_shape0_ = input_tensor_shape0;
    }
    void SetInputTensorShape1(const TensorShape& input_tensor_shape1) {
        input_tensor_shape1_ = input_tensor_shape1;
    }
    const TensorShape& InputTensorShape0() const {
        return input_tensor_shape0_;
    }
    const TensorShape& InputTensorShape1() const {
        return input_tensor_shape1_;
    }

    virtual void CalcBroadCast(void) = 0;

    void SetInputTensorShapes(const TensorShape& input_tensor_shape0, const TensorShape& input_tensor_shape1) {
        input_tensor_shape0_ = input_tensor_shape0;
        input_tensor_shape1_ = input_tensor_shape1;
        CalcBroadCast(); // will calculate broad cast automatically
    }

    /* functions below are valid only when calcBroadCast() has already called. */

    bool CanBroadCast(void) {
        return can_broadcast_;
    }

    const TensorShape& OutputTensorShape(void) const {
        return output_tensor_shape_;
    }

protected:
    bool can_broadcast_ = false;
    TensorShape input_tensor_shape0_;
    TensorShape input_tensor_shape1_;
    TensorShape output_tensor_shape_;
};

/**
 * MultiDirectionalBroadCaster
 * Used for ops such as Add, Sub, Mul and etc.
 * Input tensor 0 & input tensor 1 will both occur broadcasting
 * Details see: https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
 */
class MultiDirectionalBroadCaster : public BroadCaster {
public:
    MultiDirectionalBroadCaster() {}
    virtual ~MultiDirectionalBroadCaster() {}

    void CalcBroadCast(void) override;
};

/**
 * MultiInputBroadCaster
 * Used for ops such as Max, Min, Sum and etc.
 * All input tensors may occur broadcasting
 * Details see: https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
 */
class MultiInputBroadCaster {
public:
    void CalcBroadCast(void);
    bool CanBroadCast() {
        return can_broadcast_;
    }

    TensorShape InputTensorShape(uint32_t idx) const {
        if (idx < input_tensor_shapes_.size()) {
            return input_tensor_shapes_[idx];
        }
        return TensorShape();
    }

    void SetInputTensorShape(uint32_t idx, const TensorShape& shape) {
        if (idx < input_tensor_shapes_.size()) {
            input_tensor_shapes_[idx] = shape;
        }
    }

    void PushBackInputTensorShape(const TensorShape& shape) {
        input_tensor_shapes_.push_back(shape);
    }

    void SetInputTensorShapes(const std::vector<TensorShape>& shapes) {
        input_tensor_shapes_ = shapes;
        CalcBroadCast();
    }

    const TensorShape& OutputTensorShape(void) const {
        return output_tensor_shape_;
    }

    void ClearInputTensorShapes(void) {
        input_tensor_shapes_.clear();
    }

private:
    bool can_broadcast_ = false;
    std::vector<TensorShape> input_tensor_shapes_;
    TensorShape output_tensor_shape_;
};

/**
 * MatMulBroadCaster
 * Used for Matmul.
 * Details see: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
 */
class MatMulBroadCaster : public BroadCaster {
public:
    MatMulBroadCaster() {}
    virtual ~MatMulBroadCaster() {}

    void CalcBroadCast(void) override;
};

}}} // namespace ppl::nn::oputils

#endif
