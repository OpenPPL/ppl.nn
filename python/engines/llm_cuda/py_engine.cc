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
#include "ppl/common/retcode.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/llm_cuda/options.h"
using namespace ppl::common;
using namespace ppl::nn::llm::cuda;

#include <map>
using namespace std;

namespace ppl { namespace nn { namespace python { namespace llm { namespace cuda {

typedef RetCode (*ConfigFunc)(PyLlmCudaEngine&, uint32_t option, const pybind11::args& args);

static const map<uint32_t, ConfigFunc> g_opt2func = {};

void RegisterEngine(pybind11::module* m) {
    pybind11::class_<PyLlmCudaEngine, PyEngine>(*m, "Engine")
        .def("__bool__",
             [](const PyLlmCudaEngine& engine) -> bool {
                 return (engine.ptr.get());
             })
        .def("Configure",
             [](PyLlmCudaEngine& engine, uint32_t option, const pybind11::args& args) -> RetCode {
                 auto it = g_opt2func.find(option);
                 if (it == g_opt2func.end()) {
                     LOG(ERROR) << "unsupported option: " << option;
                     return RC_UNSUPPORTED;
                 }
                 return it->second(engine, option, args);
             });
}

}}}}} // namespace ppl::nn::python::llm::cuda
