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

#include "ppl/nn/common/tensor_shape.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
using namespace std;

namespace ppl { namespace nn { namespace python {

void RegisterTensorShape(pybind11::module* m) {
    pybind11::class_<TensorShape>(*m, "TensorShape")
        .def(pybind11::init<>())
        .def(pybind11::init<const TensorShape&>())
        .def("GetDims",
             [](TensorShape& shape) -> vector<int64_t> {
                 auto dim_count = shape.GetRealDimCount();
                 vector<int64_t> dims(dim_count);
                 for (uint32_t i = 0; i < dim_count; ++i) {
                     dims[i] = shape.GetDim(i);
                 }
                 return dims;
             })
        .def("GetDataType", &TensorShape::GetDataType)
        .def("GetDataFormat", &TensorShape::GetDataFormat)
        .def("IsScalar", &TensorShape::IsScalar);
}

}}} // namespace ppl::nn::python
