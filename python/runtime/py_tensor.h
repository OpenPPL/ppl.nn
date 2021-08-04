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

#ifndef _ST_HPC_PPL_NN_PYTHON_PY_TENSOR_H_
#define _ST_HPC_PPL_NN_PYTHON_PY_TENSOR_H_

#include "../common/py_ndarray.h"
#include "ppl/nn/runtime/tensor.h"
#include "pybind11/pybind11.h"

namespace ppl { namespace nn { namespace python {

class PyTensor final {
public:
    PyTensor(Tensor* tensor) : tensor_(tensor) {}
    PyTensor(PyTensor&&) = default;
    PyTensor& operator=(PyTensor&&) = default;
    Tensor* GetPtr() const {
        return tensor_;
    }
    const char* GetName() const {
        return tensor_->GetName();
    }
    const TensorShape& GetConstShape() const {
        return tensor_->GetShape();
    }
    ppl::common::RetCode ConvertFromHost(const pybind11::buffer&);
    PyNdArray ConvertToHost() const;

private:
    Tensor* tensor_;
};

}}} // namespace ppl::nn::python

#endif
