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

#include "py_x86_engine.h"
#include "ppl/common/retcode.h"
#include "ppl/nn/engines/x86/x86_options.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace python {

RetCode PyX86Engine::Configure(uint32_t option, const pybind11::args& args) {
    if (option == X86_CONF_DISABLE_AVX512) {
        return engine_->Configure(X86_CONF_DISABLE_AVX512);
    }

    if (option == X86_CONF_DISABLE_AVX_FMA3) {
        return engine_->Configure(X86_CONF_DISABLE_AVX_FMA3);
    }

    LOG(ERROR) << "unsupported option: " << option;
    return RC_UNSUPPORTED;
}

void RegisterX86Engine(pybind11::module* m) {
    pybind11::class_<PyX86Engine>(*m, "X86Engine")
        .def("__bool__",
             [](const PyX86Engine& engine) -> bool {
                 return (engine.GetInnerPtr().get());
             })
        .def("GetName", &PyX86Engine::GetName)
        .def("Configure", &PyX86Engine::Configure);

    m->attr("X86_CONF_DISABLE_AVX512") = (uint32_t)X86_CONF_DISABLE_AVX512;
    m->attr("X86_CONF_DISABLE_AVX_FMA3") = (uint32_t)X86_CONF_DISABLE_AVX_FMA3;
}

}}} // namespace ppl::nn::python
