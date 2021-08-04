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

#include "py_cuda_engine.h"
#include "ppl/common/retcode.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/cuda_options.h"
using namespace ppl::common;

#include <map>
using namespace std;

namespace ppl { namespace nn { namespace python {

static RetCode SetUseDefaultAlgorithms(Engine* engine, uint32_t option, const pybind11::args&) {
    return engine->Configure(option);
}

typedef RetCode (*ConfigFunc)(Engine*, uint32_t option, const pybind11::args& args);

static const map<uint32_t, ConfigFunc> g_opt2func = {
    {cuda::CUDA_CONF_USE_DEFAULT_ALGORITHMS, SetUseDefaultAlgorithms},
};

RetCode PyCudaEngine::Configure(uint32_t option, const pybind11::args& args) {
    auto it = g_opt2func.find(option);
    if (it == g_opt2func.end()) {
        LOG(ERROR) << "unsupported option: " << option;
        return RC_UNSUPPORTED;
    }

    return it->second(engine_.get(), option, args);
}

void RegisterCudaEngine(pybind11::module* m) {
    pybind11::class_<PyCudaEngine>(*m, "CudaEngine")
        .def("__bool__",
             [](const PyCudaEngine& engine) -> bool {
                 return (engine.GetInnerPtr().get());
             })
        .def("GetName", &PyCudaEngine::GetName)
        .def("Configure", &PyCudaEngine::Configure);

    m->attr("CUDA_CONF_USE_DEFAULT_ALGORITHMS") = (uint32_t)cuda::CUDA_CONF_USE_DEFAULT_ALGORITHMS;
}

}}} // namespace ppl::nn::python
