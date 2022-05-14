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
#include "py_runtime_builder.h"
#include "py_runtime_builder_resources.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;

namespace ppl { namespace nn { namespace python { namespace pmx {

void RegisterRuntimeBuilder(pybind11::module* m) {
    pybind11::class_<PyRuntimeBuilder>(*m, "RuntimeBuilder")
        .def("__bool__",
             [](const PyRuntimeBuilder& builder) -> bool {
                 return (builder.ptr.get());
             })
        .def("LoadModelFromFile",
             [](PyRuntimeBuilder& builder, const char* model_file) -> RetCode {
                 return builder.ptr->LoadModel(model_file);
             })
        .def("SetResources",
             [](PyRuntimeBuilder& builder, const PyRuntimeBuilderResources& resources) -> RetCode {
                 vector<shared_ptr<Engine>> engines;
                 for (auto e = resources.engines.begin(); e != resources.engines.end(); ++e) {
                     engines.push_back(e->ptr);
                 }

                 vector<Engine*> engine_ptrs(engines.size());
                 for (uint32_t i = 0; i < engines.size(); ++i) {
                     engine_ptrs[i] = engines[i].get();
                 }
                 RuntimeBuilder::Resources r;
                 r.engines = engine_ptrs.data();
                 r.engine_num = engine_ptrs.size();

                 builder.engines = std::move(engines);
                 return builder.ptr->SetResources(r);
             })
        .def("Preprocess",
             [](PyRuntimeBuilder& builder) -> RetCode {
                 return builder.ptr->Preprocess();
             })
        .def("CreateRuntime",
             [](PyRuntimeBuilder& builder) -> PyRuntime {
                 return PyRuntime(builder.engines, builder.ptr->CreateRuntime());
             })
        .def("Serialize",
             [](const PyRuntimeBuilder& builder, const char* output_file, const char* fmt) -> RetCode {
                 return builder.ptr->Serialize(output_file, fmt);
             });
}

}}}} // namespace ppl::nn::python::pmx
