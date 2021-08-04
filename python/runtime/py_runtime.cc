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
#include "ppl/nn/runtime/policy_defs.h"
#include "pybind11/pybind11.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace python {

void RegisterRuntime(pybind11::module* m) {
    pybind11::class_<PyRuntime>(*m, "Runtime")
        .def("__bool__",
             [](const PyRuntime& runtime) -> bool {
                 return (runtime.GetPtr());
             })
        .def("GetInputCount", &PyRuntime::GetInputCount)
        .def("GetInputTensor", &PyRuntime::GetInputTensor)
        .def("Run", &PyRuntime::Run)
        .def("Sync", &PyRuntime::Sync)
        .def("GetOutputCount", &PyRuntime::GetOutputCount)
        .def("GetOutputTensor", &PyRuntime::GetOutputTensor);

    m->attr("MM_BETTER_PERFORMANCE") = (uint32_t)MM_BETTER_PERFORMANCE;
    m->attr("MM_LESS_MEMORY") = (uint32_t)MM_LESS_MEMORY;
}

}}} // namespace ppl::nn::python
