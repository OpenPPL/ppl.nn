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

#include "ppl/nn/engines/llm_cuda/engine_options.h"
#include "ppl/nn/engines/llm_cuda/options.h"
#include "pybind11/pybind11.h"
using namespace ppl::nn::llm::cuda;

namespace ppl { namespace nn { namespace python { namespace llm { namespace cuda {

void RegisterEngineOptions(pybind11::module* m) {
    pybind11::class_<EngineOptions>(*m, "EngineOptions")
        .def(pybind11::init<>())
        .def_readwrite("device_id", &EngineOptions::device_id)
        .def_readwrite("quant_method", &EngineOptions::quant_method)
        .def_readwrite("cublas_layout_hint", &EngineOptions::cublas_layout_hint)
        .def_readwrite("mm_policy", &EngineOptions::mm_policy);

    m->attr("MM_PLAIN") = (uint32_t)MM_PLAIN;
    m->attr("MM_COMPACT") = (uint32_t)MM_COMPACT;
    m->attr("MM_BEST_FIT") = (uint32_t)MM_BEST_FIT;

    m->attr("QUANT_METHOD_ONLINE_I8I8") = (uint32_t)QUANT_METHOD_ONLINE_I8I8;
    m->attr("QUANT_METHOD_ONLINE_I4F16") = (uint32_t)QUANT_METHOD_ONLINE_I4F16;

    m->attr("CUBLAS_LAYOUT_AMPERE") = (uint32_t)CUBLAS_LAYOUT_AMPERE;
}

}}}}} // namespace ppl::nn::python::llm::cuda
