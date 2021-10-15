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
#include "pybind11/pybind11.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace python {

void RegisterEngine(pybind11::module* m) {
    pybind11::class_<PyEngine>(*m, "Engine")
#ifdef PPLNN_USE_X86
        .def(pybind11::init<PyX86Engine>())
#endif
#ifdef PPLNN_USE_CUDA
        .def(pybind11::init<PyCudaEngine>())
#endif
        .def("__bool__", [](const PyEngine& engine) -> bool {
            return (engine.ptr.get());
        });
}

}}} // namespace ppl::nn::python
