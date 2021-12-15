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

#include "py_runtime.h"
#include "py_tensor.h"
#include "../common/py_device_context.h"
#include "pybind11/pybind11.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace python {

void RegisterRuntime(pybind11::module* m) {
    pybind11::class_<PyRuntime>(*m, "Runtime")
        .def("__bool__",
             [](const PyRuntime& runtime) -> bool {
                 return (runtime.ptr.get());
             })
        .def("GetInputCount",
             [](const PyRuntime& runtime) -> uint32_t {
                 return runtime.ptr->GetInputCount();
             })
        .def("GetInputTensor",
             [](const PyRuntime& runtime, uint32_t idx) -> PyTensor {
                 return PyTensor(runtime.ptr->GetInputTensor(idx));
             })
        .def("Run",
             [](const PyRuntime& runtime) -> RetCode {
                 return runtime.ptr->Run();
             })
        .def("GetOutputCount",
             [](const PyRuntime& runtime) -> uint32_t {
                 return runtime.ptr->GetOutputCount();
             })
        .def("GetOutputTensor",
             [](const PyRuntime& runtime, uint32_t idx) -> PyTensor {
                 return PyTensor(runtime.ptr->GetOutputTensor(idx));
             })
        .def("GetDeviceContextCount",
             [](const PyRuntime& runtime) -> uint32_t {
                 return runtime.ptr->GetDeviceContextCount();
             })
        .def("GetDeviceContext", [](const PyRuntime& runtime, uint32_t idx) -> PyDeviceContext {
            return PyDeviceContext(runtime.ptr->GetDeviceContext(idx));
        });
}

}}} // namespace ppl::nn::python
