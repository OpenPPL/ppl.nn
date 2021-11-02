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

#ifdef PPLNN_USE_CUDA

#include "py_cuda_engine.h"
#include "ppl/common/retcode.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/array.h"
#include "ppl/nn/engines/cuda/cuda_options.h"
using namespace ppl::common;

#include <map>
using namespace std;

namespace ppl { namespace nn { namespace python {

static RetCode GenericSetOption(Engine* engine, uint32_t option, const pybind11::args&) {
    return engine->Configure(option);
}

/**
   @param args a list containing shapes of each input tensor.
   example: [[1, 3, 224, 224], [1, 3, 80, 80], ...]
*/
static RetCode SetInputDims(Engine* engine, uint32_t option, const pybind11::args& args) {
    if (args.size() != 1) {
        LOG(ERROR) << "expected for 1 parameter but got [" << args.size() << "].";
        return RC_INVALID_VALUE;
    }

    auto dims_vec = args[0].cast<vector<vector<int64_t>>>();

    vector<utils::Array<int64_t>> input_dims(dims_vec.size());
    for (uint32_t i = 0; i < dims_vec.size(); ++i) {
        auto& arr = input_dims[i];
        arr.base = dims_vec[i].data();
        arr.size = dims_vec[i].size();
    }
    return engine->Configure(option, input_dims.data(), input_dims.size());
}

/**
   @param args a json buffer
*/
static RetCode ImportAlgorithms(Engine* engine, uint32_t option, const pybind11::args& args) {
    if (args.size() != 1) {
        LOG(ERROR) << "expected for 1 parameter but got [" << args.size() << "].";
        return RC_INVALID_VALUE;
    }

    auto buffer = args[0].cast<string>();
    return engine->Configure(option, buffer.c_str());
}

static RetCode ExportAlgorithms(Engine* engine, uint32_t option, const pybind11::args& args) {
    if (args.size() != 1) {
        LOG(ERROR) << "expected for 1 parameter but got [" << args.size() << "].";
        return RC_INVALID_VALUE;
    }

    auto fname = args[0].cast<string>();
    return engine->Configure(option, fname.c_str());
}

typedef RetCode (*ConfigFunc)(Engine*, uint32_t option, const pybind11::args& args);

static const map<uint32_t, ConfigFunc> g_opt2func = {
    {CUDA_CONF_USE_DEFAULT_ALGORITHMS, GenericSetOption},
    {CUDA_CONF_SET_INPUT_DIMS, SetInputDims},
    {CUDA_CONF_IMPORT_ALGORITHMS, ImportAlgorithms},
    {CUDA_CONF_EXPORT_ALGORITHMS, ExportAlgorithms},
};

void RegisterCudaEngine(pybind11::module* m) {
    pybind11::class_<PyCudaEngine>(*m, "CudaEngine")
        .def("__bool__",
             [](const PyCudaEngine& engine) -> bool {
                 return (engine.ptr.get());
             })
        .def("Configure",
             [](PyCudaEngine& engine, uint32_t option, const pybind11::args& args) -> RetCode {
                 auto it = g_opt2func.find(option);
                 if (it == g_opt2func.end()) {
                     LOG(ERROR) << "unsupported option: " << option;
                     return RC_UNSUPPORTED;
                 }
                 return it->second(engine.ptr.get(), option, args);
             });

    m->attr("CUDA_CONF_USE_DEFAULT_ALGORITHMS") = (uint32_t)CUDA_CONF_USE_DEFAULT_ALGORITHMS;
    m->attr("CUDA_CONF_SET_INPUT_DIMS") = (uint32_t)CUDA_CONF_SET_INPUT_DIMS;
    m->attr("CUDA_CONF_IMPORT_ALGORITHMS") = (uint32_t)CUDA_CONF_IMPORT_ALGORITHMS;
    m->attr("CUDA_CONF_EXPORT_ALGORITHMS") = (uint32_t)CUDA_CONF_EXPORT_ALGORITHMS;
}

}}} // namespace ppl::nn::python

#endif
