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

#include "py_onnx_runtime_builder.h"
#include "../../engines/py_engine.h"
#include "ppl/nn/models/onnx/onnx_runtime_builder_factory.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
using namespace std;

namespace ppl { namespace nn { namespace python {

class PyOnnxRuntimeBuilderFactory final {
public:
    static PyOnnxRuntimeBuilder CreateFromFile(const char* model_file, vector<PyEngine>& engines) {
        vector<Engine*> engine_ptrs(engines.size());
        for (uint32_t i = 0; i < engines.size(); ++i) {
            engine_ptrs[i] = engines[i].GetPtr();
        }
        return PyOnnxRuntimeBuilder(
            engines, OnnxRuntimeBuilderFactory::Create(model_file, engine_ptrs.data(), engine_ptrs.size()));
    }
};

void RegisterOnnxRuntimeBuilderFactory(pybind11::module* m) {
    pybind11::class_<PyOnnxRuntimeBuilderFactory>(*m, "OnnxRuntimeBuilderFactory")
        .def_static("CreateFromFile", &PyOnnxRuntimeBuilderFactory::CreateFromFile);
}

}}} // namespace ppl::nn::python
