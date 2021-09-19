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

#ifdef PPLNN_USE_X86

#include "ppl/nn/engines/x86/x86_engine_options.h"
#include "pybind11/pybind11.h"

namespace ppl { namespace nn { namespace python {

void RegisterX86EngineOptions(pybind11::module* m) {
    pybind11::class_<X86EngineOptions>(*m, "X86EngineOptions")
        .def(pybind11::init<>())
        .def_readwrite("mm_policy", &X86EngineOptions::mm_policy);

    m->attr("X86_MM_COMPACT") = (uint32_t)X86_MM_COMPACT;
    m->attr("X86_MM_MRU") = (uint32_t)X86_MM_MRU;
}

}}} // namespace ppl::nn::python

#endif
