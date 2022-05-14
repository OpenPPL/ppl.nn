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

#include "py_engine.h"
#include "ppl/nn/engines/cuda/engine_factory.h"
#include "pybind11/pybind11.h"
using namespace ppl::nn::cuda;

namespace ppl { namespace nn { namespace python { namespace cuda {

class PyEngineFactory final {
public:
    static PyCudaEngine Create(const EngineOptions& options) {
        return PyCudaEngine(EngineFactory::Create(options));
    }
};

void RegisterEngineFactory(pybind11::module* m) {
    pybind11::class_<PyEngineFactory>(*m, "EngineFactory")
        .def_static("Create", &PyEngineFactory::Create);
}

}}}} // namespace ppl::nn::python::cuda
