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

#include "../../engines/py_engine.h"
#include "../../runtime/py_runtime.h"
#include "py_pmx_runtime_builder.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace python {

void RegisterPmxRuntimeBuilder(pybind11::module* m) {
    pybind11::class_<PyPmxRuntimeBuilder>(*m, "RuntimeBuilder")
        .def("__bool__",
             [](const PyPmxRuntimeBuilder& builder) -> bool {
                 return (builder.ptr.get());
             })
        .def("InitFromFile",
             [](PyPmxRuntimeBuilder& builder, const char* model_file, const vector<PyEngine>& engines) -> RetCode {
                 vector<shared_ptr<Engine>> engine_list(engines.size());
                 vector<Engine*> engine_ptrs(engines.size());
                 for (uint32_t i = 0; i < engines.size(); ++i) {
                     engine_list[i] = engines[i].ptr;
                     engine_ptrs[i] = engines[i].ptr.get();
                 }

                 builder.engines = std::move(engine_list);
                 return builder.ptr->Init(model_file, engine_ptrs.data(), engine_ptrs.size());
             })
        .def("Preprocess",
             [](PyPmxRuntimeBuilder& builder) -> RetCode {
                 return builder.ptr->Preprocess();
             })
        .def("CreateRuntime",
             [](PyPmxRuntimeBuilder& builder) -> PyRuntime {
                 return PyRuntime(builder.engines, builder.ptr->CreateRuntime());
             })
        .def("Serialize",
             [](const PyPmxRuntimeBuilder& builder, const char* output_file, const char* fmt) -> RetCode {
                 return builder.ptr->Serialize(output_file, fmt);
             });
}

}}} // namespace ppl::nn::python
