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
#include "ppl/common/types.h"
#include "ppl/common/file_mapping.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/array.h"
#include "ppl/nn/engines/cuda/options.h"
using namespace ppl::common;
using namespace ppl::nn::cuda;

#include <map>
using namespace std;

namespace ppl { namespace nn { namespace python { namespace cuda {

static RetCode GenericSetOption(PyCudaEngine& engine, uint32_t option, const pybind11::args&) {
    return engine.ptr->Configure(option);
}

/**
   @param args a list containing shapes of each input tensor.
   example: [[1, 3, 224, 224], [1, 3, 80, 80], ...]
*/
static RetCode SetInputDims(PyCudaEngine& engine, uint32_t option, const pybind11::args& args) {
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
    return engine.ptr->Configure(option, input_dims.data(), input_dims.size());
}

/**
   @param args a json buffer
*/
static void CudaSaveAlgoInfo(const char* data, uint64_t bytes, void* arg) {
    auto fname = (const char*)arg;
    auto fp = fopen(fname, "w");
    if (!fp) {
        LOG(ERROR) << "open [" << fname << "] for exporting algo info failed.";
        return;
    }

    auto ret = fwrite(data, bytes, 1, fp);
    if (ret != 1) {
        LOG(ERROR) << "write algo info to [" << fname << "] failed.";
    }

    fclose(fp);
}

static RetCode ImportAlgorithmsFromBuffer(PyCudaEngine& engine, uint32_t option, const pybind11::args& args) {
    if (args.size() != 1) {
        LOG(ERROR) << "expected for 1 parameter but got [" << args.size() << "].";
        return RC_INVALID_VALUE;
    }

    auto fname = args[0].cast<string>();
    FileMapping fm;
    auto rc = fm.Init(fname.c_str(), FileMapping::READ);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "mapping algorithms file[" << fname << "] failed: " << fm.GetErrorMessage();
        return rc;
    }

    return engine.ptr->Configure(ENGINE_CONF_IMPORT_ALGORITHMS_FROM_BUFFER, fm.GetData(), fm.GetSize());
}

static RetCode SetExportAlgorithmsHandler(PyCudaEngine& engine, uint32_t option, const pybind11::args& args) {
    if (args.size() != 1) {
        LOG(ERROR) << "expected for 1 parameter but got [" << args.size() << "].";
        return RC_INVALID_VALUE;
    }

    // save file name in PyCudaEngine
    engine.export_algo_file = args[0].cast<string>();
    return engine.ptr->Configure(ENGINE_CONF_SET_EXPORT_ALGORITHMS_HANDLER, CudaSaveAlgoInfo,
                                 engine.export_algo_file.c_str());
}

static RetCode SetKernelType(PyCudaEngine& engine, uint32_t option, const pybind11::args& args) {
    if (args.size() != 1) {
        LOG(ERROR) << "expected for 1 parameter but got [" << args.size() << "].";
        return RC_INVALID_VALUE;
    }

    return engine.ptr->Configure(option, args[0].cast<datatype_t>());
}

static RetCode SetQuantInfo(PyCudaEngine& engine, uint32_t option, const pybind11::args& args) {
    if (args.size() != 1) {
        LOG(ERROR) << "expected for 1 parameter but got [" << args.size() << "].";
        return RC_INVALID_VALUE;
    }

    auto json_str = args[0].cast<string>();
    return engine.ptr->Configure(option, json_str.data(), json_str.size());
}

typedef std::map<std::string, std::string> MapOfString;
typedef std::map<std::string, const void*> MapOfPointer;
static RetCode RefitConstantWeights(PyCudaEngine& engine, uint32_t option, const pybind11::args& args) {
    if (pybind11::len(args) != 2) {
        LOG(ERROR) << "expected for 2 parameter but got [" << pybind11::len(args) << "].";
        return RC_INVALID_VALUE;
    }

    MapOfString torch2onnx;
    auto py_torch2onnx = args[0].cast<pybind11::dict>();
    for (std::pair<pybind11::handle, pybind11::handle> it : py_torch2onnx)
    {
        auto key = it.first.cast<std::string>();
        auto value = it.second.cast<std::string>();
        torch2onnx[key] = value;
    }
    MapOfPointer name2val;
    auto py_name2val = args[1].cast<pybind11::dict>();
    for (std::pair<pybind11::handle, pybind11::handle> it : py_name2val)
    {
        auto key = it.first.cast<std::string>();
        auto value = it.second.cast<pybind11::buffer>();
        name2val[key] = (const void*)value.request().ptr;
    }

    return engine.ptr->Configure(option, &torch2onnx, &name2val);
}

typedef RetCode (*ConfigFunc)(PyCudaEngine&, uint32_t option, const pybind11::args& args);

static const map<uint32_t, ConfigFunc> g_opt2func = {
    {ENGINE_CONF_USE_DEFAULT_ALGORITHMS, GenericSetOption},
    {ENGINE_CONF_SET_INPUT_DIMS, SetInputDims},
    {ENGINE_CONF_IMPORT_ALGORITHMS_FROM_BUFFER, ImportAlgorithmsFromBuffer},
    {ENGINE_CONF_SET_EXPORT_ALGORITHMS_HANDLER, SetExportAlgorithmsHandler},
    {ENGINE_CONF_SET_KERNEL_TYPE, SetKernelType},
    {ENGINE_CONF_SET_QUANT_INFO, SetQuantInfo},
    {ENGINE_CONF_REFIT_CONSTANT_WEIGHTS, RefitConstantWeights},
};

void RegisterEngine(pybind11::module* m) {
    pybind11::class_<PyCudaEngine, PyEngine>(*m, "Engine")
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
                 return it->second(engine, option, args);
             });

    m->attr("ENGINE_CONF_USE_DEFAULT_ALGORITHMS") = (uint32_t)ENGINE_CONF_USE_DEFAULT_ALGORITHMS;
    m->attr("ENGINE_CONF_SET_INPUT_DIMS") = (uint32_t)ENGINE_CONF_SET_INPUT_DIMS;
    m->attr("ENGINE_CONF_SET_KERNEL_TYPE") = (uint32_t)ENGINE_CONF_SET_KERNEL_TYPE;
    m->attr("ENGINE_CONF_SET_QUANT_INFO") = (uint32_t)ENGINE_CONF_SET_QUANT_INFO;
    /*
      XXX NOTE use ENGINE_CONF_IMPORT_ALGORITHMS_FROM_BUFFER as ENGINE_CONF_IMPORT_ALGORITHMS to avoid
      conflict values.
    */
    m->attr("ENGINE_CONF_IMPORT_ALGORITHMS") = (uint32_t)ENGINE_CONF_IMPORT_ALGORITHMS_FROM_BUFFER;
    /*
      XXX NOTE use ENGINE_CONF_SET_EXPORT_ALGORITHMS_HANDLER as ENGINE_CONF_EXPORT_ALGORITHMS to avoid
      conflict values. It is impossible to set a callback function to C++ in Python.
    */
    m->attr("ENGINE_CONF_EXPORT_ALGORITHMS") = (uint32_t)ENGINE_CONF_SET_EXPORT_ALGORITHMS_HANDLER;

    m->attr("ENGINE_CONF_REFIT_CONSTANT_WEIGHTS") = (uint32_t)ENGINE_CONF_REFIT_CONSTANT_WEIGHTS;
}

}}}} // namespace ppl::nn::python::cuda
